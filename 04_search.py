#!/usr/bin/env python3
"""
PASO 4 — Búsqueda RAG híbrida.

Pipeline de retrieval:
  1. parse_query      → extrae filtros (year, doc_type, variables, sections) del NL
  2. vector recall    → HNSW sobre pgvector (top N, filtros en SQL)
  3. lexical recall   → full-text tsvector + ts_rank_cd (top N, mismos filtros)
  4. RRF fusion       → rank fusion independiente de la escala de scores
  5. MMR rerank       → penaliza redundancia (top K diverso)
  6. importance boost → pequeño re-ordenamiento final por señales curadas

Uso:
  python3 04_search.py "tasa de interés 2022" 5
  python3 04_search.py "política monetaria decisión" 10 --json
  python3 04_search.py "riesgos inflación" 5 --no-mmr
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

_MODELS_DIR = Path(__file__).parent / "models"

from taxonomy import (
    CRITICAL_VARIABLES,
    ECONOMIC_VARIABLES,
    SECTION_KEYWORDS,
    build_section_patterns,
    build_variable_patterns,
    normalize_text,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DB_NAME = os.getenv("PGDATABASE", "rag_banco")
TABLE_PREFIX = os.getenv("RAG_TABLE_PREFIX", "")


def _table_name(base: str) -> str:
    name = f"{TABLE_PREFIX}{base}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Nombre de tabla inválido generado por RAG_TABLE_PREFIX: {name!r}"
        )
    return name


DOCUMENTS_TABLE = _table_name("documents")
CHUNKS_TABLE = _table_name("chunks")

# ---------------------------------------------------------------------------
# Selección interactiva del prefijo de tablas
# ---------------------------------------------------------------------------

_KNOWN_PREFIXES = [
    ("gemma_", "modelo Gemma"),
    ("qwen_",  "modelo Qwen"),
    ("",       "sin prefijo"),
]


def _prompt_table_prefix() -> str:
    env = os.getenv("RAG_TABLE_PREFIX")
    if env is not None:
        return env
    print("\nSelecciona el conjunto de tablas a usar:")
    for i, (prefix, label) in enumerate(_KNOWN_PREFIXES, start=1):
        tbl = f"{prefix}documents / {prefix}chunks" if prefix else "documents / chunks"
        print(f"  [{i}] {tbl}  ({label})")
    while True:
        try:
            raw = input("Opción [1]: ").strip() or "1"
            idx = int(raw) - 1
            if 0 <= idx < len(_KNOWN_PREFIXES):
                return _KNOWN_PREFIXES[idx][0]
        except (ValueError, EOFError):
            pass
        print("  Opción inválida.")


def _apply_table_prefix(prefix: str) -> None:
    global TABLE_PREFIX, DOCUMENTS_TABLE, CHUNKS_TABLE
    TABLE_PREFIX = prefix
    DOCUMENTS_TABLE = _table_name("documents")
    CHUNKS_TABLE = _table_name("chunks")

# Parámetros de retrieval
RECALL_N = 50           # top-N por cada rama antes de fusionar
RRF_K = 60              # hiperparámetro estándar de RRF
MMR_LAMBDA = 0.65       # 1.0 = solo relevancia, 0.0 = solo diversidad
IMPORTANCE_BOOST = 0.15 # peso del importance_score en el re-rank final
RECENCY_WEIGHT = 0.03   # penaliza docs viejos 5% por año (tie-breaker suave)

# ---------------------------------------------------------------------------
# Parseo de query
# ---------------------------------------------------------------------------

YEAR_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")
YEAR_RANGE_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\s*[-–a]\s*(19[8-9]\d|20[0-4]\d)\b")

MONTH_NAMES: dict[str, int] = {
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "mayo": 5,
    "junio": 6, "jun": 6,
    "julio": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "sep": 9, "sept": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11,
    "diciembre": 12, "dic": 12,
}
MONTH_RE = re.compile(
    r"\b(" + "|".join(sorted(MONTH_NAMES, key=len, reverse=True)) + r")\b"
)
# Inverso para display: número → nombre largo en español
MONTH_NUM_TO_NAME: dict[int, str] = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}

# ── Patrones para fechas completas (día + mes + año) ──────────────────────────
_MONTH_ALT = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))

# "15 de marzo de 2024", "15 marzo 2024"
FULL_DATE_NAMED_RE = re.compile(
    r"\b(\d{1,2})\s+(?:de\s+)?(" + _MONTH_ALT + r")\s+(?:de\s+)?(\d{4})\b",
    re.IGNORECASE,
)
# ISO y variantes: "2024-03-15", "2024/03/15", "2024.03.15"
FULL_DATE_YMD_RE = re.compile(r"\b(20[0-4]\d)[/_.-](\d{2})[/_.-](\d{2})\b")
# Europeo/latinoamérica: "15/03/2024", "15-03-2024", "15.03.24"
FULL_DATE_DMY_RE = re.compile(r"\b(\d{1,2})[/_.-](\d{1,2})[/_.-](20[0-4]\d|\d{2})\b")
# "15 de marzo", "15 marzo" (sin año — se combina con año detectado después)
DAY_MONTH_NAMED_RE = re.compile(
    r"\b(\d{1,2})\s+(?:de\s+)?(" + _MONTH_ALT + r")\b",
    re.IGNORECASE,
)

FILENAME_DATE_RE = re.compile(
    r"(?:^|[^0-9])(?:(\d{4})[._/-](\d{2})[._/-](\d{2})|(\d{2})[._/-](\d{2})[._/-](\d{2}))(?:[^0-9]|$)"
)

# Patrones para tipo de documento — específicos y más descriptivos que los
# patrones del paso 0 (que operan sobre filepath).
DOC_TYPE_HINTS = {
    "COMUNICADO": ["comunicado", "anuncio de politica"],
    "MINUTA": ["minuta", "reunion de politica", "consejeros"],
    "FED_STATEMENT": ["fed", "federal reserve", "fomc"],
    "REPORTE_RESEARCH": ["research", "jpmorgan", "jpm", "reporte"],
    "MONITOR_PM": ["monitor pm", "monitor de mercado", "monitor financiero"],
}

VARIABLE_PATTERNS = build_variable_patterns()
SECTION_PATTERNS = build_section_patterns()


def parse_query(query: str) -> dict:
    """Extrae filtros del query de usuario. Retorna dict con:
      - raw_query: texto original
      - clean_query: texto sin tokens de filtro (años, fechas, etc.)
      - exact_date: fecha ISO "YYYY-MM-DD" si se detectó día+mes+año, si no None
      - day: día del mes (1–31) si se detectó, si no None
      - year_from, year_to: rango anual o None
      - month: mes (1–12) si se detectó, si no None
      - doc_types: lista de doc_type_category matcheados
      - variables: lista de variables económicas mencionadas
      - sections: lista de secciones solicitadas
    """
    norm = normalize_text(query)
    clean = query

    year_from = year_to = month = exact_date = day = None

    # ── 1. Fecha completa: detectar ANTES que años/meses sueltos ─────────────
    # Prioridad: named > YMD > DMY (de más específico a más ambiguo)
    fm = FULL_DATE_NAMED_RE.search(norm)
    if fm:
        d_val, m_name, y_val = int(fm.group(1)), fm.group(2), int(fm.group(3))
        m_val = MONTH_NAMES.get(m_name)
        if m_val and 1 <= d_val <= 31:
            day, month, year_from, year_to = d_val, m_val, y_val, y_val
            exact_date = f"{y_val:04d}-{m_val:02d}-{d_val:02d}"
            clean = FULL_DATE_NAMED_RE.sub("", clean)

    if not exact_date:
        fm = FULL_DATE_YMD_RE.search(norm)
        if fm:
            y_val, m_val, d_val = int(fm.group(1)), int(fm.group(2)), int(fm.group(3))
            if 1 <= m_val <= 12 and 1 <= d_val <= 31:
                day, month, year_from, year_to = d_val, m_val, y_val, y_val
                exact_date = f"{y_val:04d}-{m_val:02d}-{d_val:02d}"
                clean = FULL_DATE_YMD_RE.sub("", clean)

    if not exact_date:
        fm = FULL_DATE_DMY_RE.search(norm)
        if fm:
            d_val, m_val = int(fm.group(1)), int(fm.group(2))
            y_str = fm.group(3)
            y_val = int(y_str) if len(y_str) == 4 else 2000 + int(y_str)
            if 1 <= m_val <= 12 and 1 <= d_val <= 31:
                day, month, year_from, year_to = d_val, m_val, y_val, y_val
                exact_date = f"{y_val:04d}-{m_val:02d}-{d_val:02d}"
                clean = FULL_DATE_DMY_RE.sub("", clean)

    # ── 2. Año o rango (solo si no se capturó en fecha completa) ─────────────
    if year_from is None:
        m = YEAR_RANGE_RE.search(norm)
        if m:
            year_from, year_to = int(m.group(1)), int(m.group(2))
            if year_from > year_to:
                year_from, year_to = year_to, year_from
            clean = YEAR_RANGE_RE.sub("", clean)
        else:
            years = [int(y) for y in YEAR_RE.findall(norm)]
            if years:
                year_from, year_to = min(years), max(years)
                clean = YEAR_RE.sub("", clean)

    # ── 3. Mes suelto (solo si no fue capturado por fecha completa) ───────────
    if month is None:
        mm = MONTH_RE.search(norm)
        if mm:
            month = MONTH_NAMES[mm.group(1)]
            clean = MONTH_RE.sub("", clean)

    # ── 4. Día + mes sin año → combinar con año si es único ──────────────────
    if exact_date is None and day is None:
        dm = DAY_MONTH_NAMED_RE.search(norm)
        if dm:
            d_val = int(dm.group(1))
            m_name = dm.group(2)
            m_val = MONTH_NAMES.get(m_name)
            if m_val and 1 <= d_val <= 31:
                day = d_val
                if month is None:
                    month = m_val
                # Si hay un único año conocido, construimos exact_date
                if year_from is not None and year_from == year_to:
                    exact_date = f"{year_from:04d}-{m_val:02d}-{d_val:02d}"
                clean = DAY_MONTH_NAMED_RE.sub("", clean)

    # ── 5. Tipos de documento ─────────────────────────────────────────────────
    doc_types: list[str] = []
    for dt, keywords in DOC_TYPE_HINTS.items():
        if any(kw in norm for kw in keywords):
            doc_types.append(dt)

    # Variables
    variables = [v for v, pat in VARIABLE_PATTERNS.items() if pat.search(norm)]

    # Secciones
    sections = [s for s, pat in SECTION_PATTERNS.items() if pat.search(norm)]

    return {
        "raw_query": query,
        "clean_query": re.sub(r"\s+", " ", clean).strip() or query,
        "exact_date": exact_date,
        "day": day,
        "year_from": year_from,
        "year_to": year_to,
        "month": month,
        "doc_types": doc_types,
        "variables": variables,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

def connect():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", 5432)),
        user=os.getenv("PGUSER", os.getenv("USER", "postgres")),
        password=os.getenv("PGPASSWORD", "postgres"),
        database=DB_NAME,
    )


def get_db_embedding_dim(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
                            AND c.relname = %s
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
                        """,
                        (CHUNKS_TABLE,)
        )
        row = cur.fetchone()
    if not row:
        return None

    type_expr = row[0] or ""
    match = re.search(r"vector\((\d+)\)", type_expr)
    if not match:
        return None
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Filtros SQL — comparten WHERE entre vector y lexical
# ---------------------------------------------------------------------------

def build_filters_sql(parsed: dict) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []

    if parsed["doc_types"]:
        clauses.append("d.doc_type_category = ANY(%s)")
        params.append(parsed["doc_types"])

    # ── Filtros de fecha ───────────────────────────────────────────────────────
    # Prioridad: exact_date > month (+ year) > year solo
    if parsed.get("exact_date"):
        # Monitor PM: solo chunks de esa fecha exacta.
        # PDFs: sin restricción de chunk_date (document_date se filtra en Python).
        clauses.append("(c.chunk_date IS NULL OR c.chunk_date = %s::date)")
        params.append(parsed["exact_date"])
    else:
        # Rango de año (COALESCE: chunk_date para Monitor PM, document_year para PDFs)
        if parsed["year_from"] is not None:
            clauses.append(
                "COALESCE(EXTRACT(YEAR FROM c.chunk_date)::int, d.document_year) >= %s"
            )
            params.append(parsed["year_from"])
        if parsed["year_to"] is not None:
            clauses.append(
                "COALESCE(EXTRACT(YEAR FROM c.chunk_date)::int, d.document_year) <= %s"
            )
            params.append(parsed["year_to"])
        # Mes: aplica en SQL solo para Monitor PM (chunk_date != NULL).
        # Los PDFs sin chunk_date pasan el filtro SQL y se filtran en Python
        # por document_date vía _row_matches_month().
        if parsed.get("month") is not None:
            clauses.append(
                "(c.chunk_date IS NULL OR EXTRACT(MONTH FROM c.chunk_date)::int = %s)"
            )
            params.append(parsed["month"])

    if parsed["sections"]:
        clauses.append("c.section_type = ANY(%s)")
        params.append(parsed["sections"])
    if parsed["variables"]:
        # chunk debe mencionar AL MENOS una variable solicitada,
        # O ser una DECISION de política (que siempre es relevante aunque
        # la variable no haya sido etiquetada explícitamente en ese chunk).
        clauses.append("(c.economic_variables ?| %s OR c.section_type = 'DECISION')")
        params.append(parsed["variables"])

    sql = " AND ".join(clauses) if clauses else "TRUE"
    return sql, params


def _extract_month_from_row(row: dict) -> int | None:
    chunk_date = row.get("chunk_date")
    if chunk_date is not None:
        month = getattr(chunk_date, "month", None)
        if month is not None:
            return int(month)
        if isinstance(chunk_date, str) and len(chunk_date) >= 7 and chunk_date[4] == "-":
            try:
                return int(chunk_date[5:7])
            except ValueError:
                pass

    document_date = row.get("document_date")
    if isinstance(document_date, str):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", document_date):
            return int(document_date[5:7])
        if re.match(r"^\d{2}[._/-]\d{2}[._/-]\d{2,4}$", document_date):
            parts = re.split(r"[._/-]", document_date)
            if len(parts) == 3:
                try:
                    return int(parts[1])
                except ValueError:
                    pass

    filename = row.get("filename") or ""
    match = FILENAME_DATE_RE.search(filename)
    if match:
        month_part = match.group(2) or match.group(5)
        if month_part:
            try:
                return int(month_part)
            except ValueError:
                return None
    return None


def _row_matches_month(row: dict, month: int | None) -> bool:
    if month is None:
        return True
    return _extract_month_from_row(row) == month


# ---------------------------------------------------------------------------
# Recalls
# ---------------------------------------------------------------------------

def vector_recall(conn, query_embedding: list[float], parsed: dict, n: int) -> list[dict]:
    filter_sql, params = build_filters_sql(parsed)
    emb_str = "[" + ",".join(f"{x:.7f}" for x in query_embedding) + "]"

    sql = f"""
    SELECT
        c.chunk_id,
        c.document_id,
        c.text,
        c.page_start, c.page_end,
        c.section_type,
        c.importance_score,
        c.economic_variables,
        c.numeric_values,
        c.tags,
        c.chunk_date,
        d.filename, d.doc_type_category, d.document_date,
        1 - (c.embedding <=> %s::vector) AS vector_score,
        c.embedding
    FROM {CHUNKS_TABLE} c
    JOIN {DOCUMENTS_TABLE} d USING (document_id)
    WHERE {filter_sql}
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, [emb_str] + params + [emb_str, n])
        return cur.fetchall()


def lexical_recall(conn, query_text: str, parsed: dict, n: int) -> list[dict]:
    filter_sql, params = build_filters_sql(parsed)
    # plainto_tsquery es permisivo y convierte la query a una forma indexable.
    # ts_rank_cd usa cover density (pondera proximidad).
    sql = f"""
    SELECT
        c.chunk_id,
        c.document_id,
        c.text,
        c.page_start, c.page_end,
        c.section_type,
        c.importance_score,
        c.economic_variables,
        c.numeric_values,
        c.tags,
        c.chunk_date,
        d.filename, d.doc_type_category, d.document_date,
        ts_rank_cd(c.text_tsv, plainto_tsquery('simple', %s)) AS lexical_score,
        c.embedding
    FROM {CHUNKS_TABLE} c
    JOIN {DOCUMENTS_TABLE} d USING (document_id)
    WHERE ({filter_sql})
      AND c.text_tsv @@ plainto_tsquery('simple', %s)
    ORDER BY lexical_score DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, [query_text] + params + [query_text, n])
        return cur.fetchall()


def date_importance_fallback(conn, parsed: dict, n: int) -> list[dict]:
    filter_sql, params = build_filters_sql(parsed)
    sql = f"""
    SELECT
        c.chunk_id,
        c.document_id,
        c.text,
        c.page_start, c.page_end,
        c.section_type,
        c.importance_score,
        c.economic_variables,
        c.numeric_values,
        c.tags,
        c.chunk_date,
        d.filename, d.doc_type_category, d.document_date,
        c.embedding
    FROM {CHUNKS_TABLE} c
    JOIN {DOCUMENTS_TABLE} d USING (document_id)
    WHERE {filter_sql}
    ORDER BY c.importance_score DESC, c.page_start ASC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params + [n])
        rows = cur.fetchall()

    if parsed.get("month") is not None:
        rows = [row for row in rows if _row_matches_month(row, parsed["month"])]
    return rows


# ---------------------------------------------------------------------------
# Fusión y re-ranking
# ---------------------------------------------------------------------------

def rrf_fuse(vector_hits: list[dict], lexical_hits: list[dict], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion — combina dos rankings sin calibrar pesos.

    score(doc) = Σ 1 / (k + rank_in_list_i)
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, row in enumerate(vector_hits, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        docs[cid] = dict(row)
        docs[cid]["_rrf_vector_rank"] = rank

    for rank, row in enumerate(lexical_hits, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in docs:
            docs[cid] = dict(row)
        docs[cid]["_rrf_lexical_rank"] = rank

    ordered = sorted(docs.values(), key=lambda d: scores[d["chunk_id"]], reverse=True)
    for d in ordered:
        d["rrf_score"] = scores[d["chunk_id"]]
    return ordered


def _parse_embedding(emb) -> np.ndarray:
    """pgvector devuelve el embedding como string '[0.1,0.2,...]' o lista."""
    if isinstance(emb, str):
        emb = emb.strip("[]").split(",")
        return np.array([float(x) for x in emb], dtype=np.float32)
    return np.asarray(emb, dtype=np.float32)


def _mmr_score(
    candidate_idx: int,
    cand_vecs: list[np.ndarray],
    sims_to_query: np.ndarray,
    selected_idx: list[int],
    lambda_param: float,
) -> float:
    """Computes MMR score for one candidate given already-selected items."""
    max_redundancy = max(
        float(np.dot(cand_vecs[candidate_idx], cand_vecs[selected])) for selected in selected_idx
    )
    return lambda_param * sims_to_query[candidate_idx] - (1 - lambda_param) * max_redundancy


def mmr_select(
    candidates: list[dict],
    query_vec: np.ndarray,
    k: int,
    lambda_param: float = MMR_LAMBDA,
) -> list[dict]:
    """Maximal Marginal Relevance: balancea relevancia y diversidad.

    score_mmr(d) = λ * sim(d, q) - (1-λ) * max_selected sim(d, s)
    """
    if not candidates:
        return []

    cand_vecs = [_parse_embedding(c["embedding"]) for c in candidates]
    sims_to_query = np.array([float(np.dot(query_vec, v)) for v in cand_vecs])

    selected_idx: list[int] = []
    remaining = set(range(len(candidates)))

    while len(selected_idx) < k and remaining:
        if not selected_idx:
            best = max(remaining, key=lambda i: sims_to_query[i])
        else:
            best = max(
                remaining,
                key=lambda i: _mmr_score(i, cand_vecs, sims_to_query, selected_idx, lambda_param),
            )
        selected_idx.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_idx]


def _doc_year(hit: dict) -> int | None:
    """Extrae el año del chunk (chunk_date) o del documento (document_date)."""
    for key in ("chunk_date", "document_date"):
        v = hit.get(key)
        if v:
            m = re.match(r"(\d{4})", str(v))
            if m:
                return int(m.group(1))
    return None


def recency_factor(doc_year: int | None) -> float:
    """0.0–1.0: penaliza 5% por año de antigüedad. Docs sin año → 0.5 neutral."""
    import datetime
    if doc_year is None:
        return 0.5
    age = datetime.date.today().year - doc_year
    return max(0.0, 1.0 - age * 0.05)


def importance_boost(
    hits: list[dict],
    weight: float = IMPORTANCE_BOOST,
    recency_weight: float = RECENCY_WEIGHT,
) -> list[dict]:
    """Re-ordenamiento final: importance + recency como tie-breakers suaves sobre RRF.

    No reemplaza la relevancia del retrieval; sólo desempata entre candidatos
    con RRF similar, prefiriendo chunks curados y documentos más recientes.
    """
    for h in hits:
        rec = recency_factor(_doc_year(h))
        h["final_score"] = (
            h.get("rrf_score", 0.0)
            + weight * float(h.get("importance_score", 0))
            + recency_weight * rec
        )
    hits.sort(key=lambda h: h["final_score"], reverse=True)
    return hits


# ---------------------------------------------------------------------------
# Embedding de la query (con caché del modelo por proceso)
# ---------------------------------------------------------------------------

_MODEL = None
_MODEL_NAME = None


def load_embedding_model():
    global _MODEL, _MODEL_NAME
    if _MODEL is not None:
        return _MODEL, _MODEL_NAME
    from sentence_transformers import SentenceTransformer
    model_id = os.environ.get("RAG_EMBEDDING_MODEL", "EmbeddingGemma")
    local_path = _MODELS_DIR / model_id
    name = str(local_path) if local_path.exists() else model_id
    try:
        _MODEL = SentenceTransformer(name)
        _MODEL_NAME = name
    except Exception:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _MODEL_NAME = "all-MiniLM-L6-v2"
    return _MODEL, _MODEL_NAME


def embed_query(query_text: str) -> np.ndarray:
    model, name = load_embedding_model()
    text = query_text
    if "e5" in name.lower():
        text = "query: " + text
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def _recall_and_fuse(conn, query_vec: np.ndarray, parsed: dict) -> list[dict]:
    vec_hits = vector_recall(conn, query_vec.tolist(), parsed, n=RECALL_N)
    lex_hits = lexical_recall(conn, parsed["clean_query"], parsed, n=RECALL_N)
    return rrf_fuse(vec_hits, lex_hits)


def _mark_low_confidence(results: list[dict], threshold: float = 0.05) -> None:
    for result in results:
        result["low_confidence"] = result.get("final_score", 0) < threshold


def search(query: str, k: int = 5, use_mmr: bool = True) -> list[dict]:
    parsed = parse_query(query)
    query_vec = embed_query(parsed["clean_query"])

    with connect() as conn:
        db_dim = get_db_embedding_dim(conn)
        if db_dim is not None and query_vec.shape[0] != db_dim:
            if query_vec.shape[0] > db_dim:
                query_vec = query_vec[:db_dim]
            else:
                query_vec = np.pad(query_vec, (0, db_dim - query_vec.shape[0]))
        fused = _recall_and_fuse(conn, query_vec, parsed)

    # Filters on variables/sections can be too strict — relax them if no results.
    has_strict_filters = bool(parsed["variables"] or parsed["sections"])
    if not fused and has_strict_filters:
        relaxed_parsed = dict(parsed, variables=[], sections=[])
        with connect() as conn:
            fused = _recall_and_fuse(conn, query_vec, relaxed_parsed)

    if parsed.get("month") is not None:
        fused = [hit for hit in fused if _row_matches_month(hit, parsed["month"])]

    if not fused:
        with connect() as conn:
            fused = date_importance_fallback(conn, parsed, n=max(k * 4, 20))

    if use_mmr and fused:
        fused = mmr_select(fused, query_vec, k=min(k * 2, len(fused)))

    ranked = importance_boost(fused)
    top = ranked[:k]
    _mark_low_confidence(top)

    return top, parsed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_text_output(results: list[dict], parsed: dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"QUERY: {parsed['raw_query']}")
    if parsed["clean_query"] != parsed["raw_query"]:
        lines.append(f"  clean: {parsed['clean_query']}")
    filters = []
    if parsed.get("exact_date"):
        filters.append(f"fecha={parsed['exact_date']}")
    else:
        if parsed["year_from"] is not None:
            yr = (f"{parsed['year_from']}" if parsed["year_from"] == parsed["year_to"]
                  else f"{parsed['year_from']}–{parsed['year_to']}")
            filters.append(f"año={yr}")
        if parsed.get("month") is not None:
            filters.append(f"mes={MONTH_NUM_TO_NAME.get(parsed['month'], parsed['month'])}")
        if parsed.get("day") is not None:
            filters.append(f"día={parsed['day']}")
    if parsed["doc_types"]:
        filters.append(f"doc={','.join(parsed['doc_types'])}")
    if parsed["variables"]:
        filters.append(f"vars={','.join(parsed['variables'])}")
    if parsed["sections"]:
        filters.append(f"sec={','.join(parsed['sections'])}")
    if filters:
        lines.append(f"  filtros: {' | '.join(filters)}")
    lines.append("=" * 80)

    if not results:
        lines.append("\nSin resultados.")
        return "\n".join(lines)

    if all(r.get("low_confidence") for r in results):
        lines.append("⚠  Resultados con baja confianza — la query puede estar fuera del dominio.")


    for i, r in enumerate(results, start=1):
        vars_str = ",".join(r.get("economic_variables", {}).keys()) or "-"
        nums = r.get("numeric_values") or []
        nums_str = ", ".join(n.get("raw", "") for n in nums[:3]) or "-"
        tags_str = ",".join(r.get("tags") or []) or "-"

        lines.append(
            f"\n[{i}] {r['filename']} p.{r['page_start']}"
            + (f"-{r['page_end']}" if r['page_end'] != r['page_start'] else "")
            + f" · {r['doc_type_category']} · {r['section_type']}"
        )
        effective_date = r.get("chunk_date") or r.get("document_date") or "-"
        lines.append(
            f"    fecha: {effective_date} | imp: {r['importance_score']:.2f} | "
            f"rrf: {r.get('rrf_score',0):.3f} | final: {r.get('final_score',0):.3f}"
        )
        lines.append(f"    vars: {vars_str}")
        lines.append(f"    datos: {nums_str}")
        lines.append(f"    tags: {tags_str}")
        text = r["text"].replace("\n", " ")
        lines.append(f"    > {text[:400]}" + ("…" if len(text) > 400 else ""))

    return "\n".join(lines)


def format_json_output(results: list[dict], parsed: dict) -> str:
    def clean(r: dict) -> dict:
        out = {
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "filename": r["filename"],
            "doc_type_category": r["doc_type_category"],
            "document_date": r.get("document_date"),
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "section_type": r["section_type"],
            "importance_score": float(r["importance_score"]),
            "rrf_score": float(r.get("rrf_score", 0)),
            "final_score": float(r.get("final_score", 0)),
            "economic_variables": r.get("economic_variables", {}),
            "numeric_values": r.get("numeric_values", []),
            "tags": r.get("tags") or [],
            "text": r["text"],
        }
        return out
    payload = {
        "query": parsed["raw_query"],
        "parsed": {k: v for k, v in parsed.items() if k != "raw_query"},
        "results": [clean(r) for r in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Búsqueda RAG híbrida")
    ap.add_argument("query", help="Consulta en lenguaje natural")
    ap.add_argument("k", nargs="?", type=int, default=5, help="Número de resultados (default 5)")
    ap.add_argument("--no-mmr", action="store_true", help="Desactivar MMR (más relevancia, menos diversidad)")
    ap.add_argument("--json", action="store_true", help="Salida JSON")
    args = ap.parse_args()
    _apply_table_prefix(_prompt_table_prefix())

    results, parsed = search(args.query, k=args.k, use_mmr=not args.no_mmr)

    if args.json:
        print(format_json_output(results, parsed))
    else:
        print(format_text_output(results, parsed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
