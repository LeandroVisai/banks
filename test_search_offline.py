#!/usr/bin/env python3
"""
test_search_offline.py — Evalúa retrieval SIN PostgreSQL, leyendo chunks_enriched.json
en memoria.

Replica la rama lexical + filtros de 04_search.py:
  - BM25 sobre el campo text
  - Filtros por section_type, economic_variables, doc_type, año
  - Importance boost en el ranking final

Útil para:
  - Validar el benchmark sin necesitar PostgreSQL/pgvector
  - Medir calidad del enriquecimiento (paso 1) independiente del modelo de embedding
  - Comparar Qwen vs Gemma en términos de cobertura de chunks (cuántos del benchmark están bien clasificados)

NO replica:
  - Búsqueda vectorial (HNSW)
  - RRF fusion (no hay rama vectorial)
  - MMR

Uso:
  python3 test_search_offline.py                         # logs_qwen por default
  python3 test_search_offline.py --logs logs_gemma       # evalúa con Gemma
  python3 test_search_offline.py --top-k 5
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BASE           = Path(__file__).parent
BENCHMARK_PATH = BASE / "queries_benchmark.json"


# ── Normalización (mismo criterio que taxonomy.py) ───────────────────────────

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "y", "o", "que", "se", "su", "sus",
    "en", "a", "por", "para", "con", "sin", "sobre", "entre",
    "es", "son", "ser", "fue", "ha", "han", "hay",
    "the", "a", "an", "of", "in", "to", "for", "and", "or",
    "que", "como", "más", "mas", "muy",
})


def tokenize(text: str) -> list[str]:
    norm = normalize(text)
    return [t for t in _TOKEN_RE.findall(norm) if t not in _STOPWORDS and len(t) > 1]


# ── BM25 simple ──────────────────────────────────────────────────────────────

class BM25:
    """Implementación BM25 estándar (k1=1.5, b=0.75)."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_lens = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0
        self.df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        for tokens in corpus_tokens:
            tf = Counter(tokens)
            self.tf.append(tf)
            for term in tf:
                self.df[term] += 1
        self.idf = {
            term: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        tf = self.tf[doc_idx]
        dl = self.doc_lens[doc_idx]
        denom_norm = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            idf = self.idf.get(term, 0.0)
            score += idf * (f * (self.k1 + 1)) / (f + denom_norm)
        return score

    def top_k(self, query_tokens: list[str], k: int = 50) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]


# ── Parser ligero de query ────────────────────────────────────────────────────

YEAR_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")

MONTH_NAMES_OFFLINE: dict[str, int] = {
    "enero": 1, "ene": 1, "febrero": 2, "feb": 2, "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4, "mayo": 5, "junio": 6, "jun": 6,
    "julio": 7, "jul": 7, "agosto": 8, "ago": 8,
    "septiembre": 9, "sep": 9, "sept": 9, "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11, "diciembre": 12, "dic": 12,
}
_M_ALT = "|".join(sorted(MONTH_NAMES_OFFLINE, key=len, reverse=True))
_FULL_DATE_NAMED = re.compile(
    r"\b(\d{1,2})\s+(?:de\s+)?(" + _M_ALT + r")\s+(?:de\s+)?(\d{4})\b", re.IGNORECASE
)
_FULL_DATE_YMD = re.compile(r"\b(20[0-4]\d)[/_.-](\d{2})[/_.-](\d{2})\b")
_FULL_DATE_DMY = re.compile(r"\b(\d{1,2})[/_.-](\d{1,2})[/_.-](20[0-4]\d|\d{2})\b")
_DAY_MONTH     = re.compile(r"\b(\d{1,2})\s+(?:de\s+)?(" + _M_ALT + r")\b", re.IGNORECASE)
_MONTH_ONLY    = re.compile(r"\b(" + _M_ALT + r")\b")

DOC_TYPE_HINTS = {
    "COMUNICADO": ["comunicado"],
    "MINUTA":     ["minuta", "consejero"],
    "FED_STATEMENT": ["fed", "federal reserve", "fomc"],
    "REPORTE_RESEARCH": ["jpmorgan", "jpm", "research"],
    "MONITOR_PM": ["monitor pm"],
}

# Importa los patrones de variables/secciones de la taxonomía oficial
try:
    from taxonomy import build_variable_patterns
    _VARIABLE_PATTERNS = build_variable_patterns()
except Exception:
    _VARIABLE_PATTERNS = {}


def parse_query(query: str) -> dict[str, Any]:
    norm = normalize(query)
    exact_date = day = month = None
    years: list[int] = []

    # Fecha completa (día+mes+año) — detectar antes que años/meses sueltos
    fm = _FULL_DATE_NAMED.search(norm)
    if fm:
        d_v, m_n, y_v = int(fm.group(1)), fm.group(2), int(fm.group(3))
        m_v = MONTH_NAMES_OFFLINE.get(m_n)
        if m_v and 1 <= d_v <= 31:
            day, month, years = d_v, m_v, [y_v]
            exact_date = f"{y_v:04d}-{m_v:02d}-{d_v:02d}"
    if not exact_date:
        fm = _FULL_DATE_YMD.search(norm)
        if fm:
            y_v, m_v, d_v = int(fm.group(1)), int(fm.group(2)), int(fm.group(3))
            if 1 <= m_v <= 12 and 1 <= d_v <= 31:
                day, month, years = d_v, m_v, [y_v]
                exact_date = f"{y_v:04d}-{m_v:02d}-{d_v:02d}"
    if not exact_date:
        fm = _FULL_DATE_DMY.search(norm)
        if fm:
            d_v, m_v = int(fm.group(1)), int(fm.group(2))
            y_s = fm.group(3)
            y_v = int(y_s) if len(y_s) == 4 else 2000 + int(y_s)
            if 1 <= m_v <= 12 and 1 <= d_v <= 31:
                day, month, years = d_v, m_v, [y_v]
                exact_date = f"{y_v:04d}-{m_v:02d}-{d_v:02d}"

    if not years:
        years = [int(y) for y in YEAR_RE.findall(norm)]
    if month is None:
        mm = _MONTH_ONLY.search(norm)
        if mm:
            month = MONTH_NAMES_OFFLINE.get(mm.group(1))
    if exact_date is None and day is None:
        dm = _DAY_MONTH.search(norm)
        if dm:
            d_v = int(dm.group(1))
            m_v = MONTH_NAMES_OFFLINE.get(dm.group(2))
            if m_v and 1 <= d_v <= 31:
                day = d_v
                if month is None:
                    month = m_v
                if len(years) == 1:
                    exact_date = f"{years[0]:04d}-{m_v:02d}-{d_v:02d}"

    doc_types = [
        dt for dt, hints in DOC_TYPE_HINTS.items()
        if any(h in norm for h in hints)
    ]
    variables = [
        var_name for var_name, pat in _VARIABLE_PATTERNS.items()
        if pat.search(norm)
    ]
    return {
        "years": years, "month": month, "day": day, "exact_date": exact_date,
        "doc_types": doc_types, "variables": variables, "norm_text": norm,
    }


# ── Búsqueda offline ─────────────────────────────────────────────────────────

def chunk_year(c: dict) -> int | None:
    """Extrae año de chunk_date o document_date."""
    for key in ("chunk_date", "document_date"):
        d = c.get(key)
        if d:
            m = re.match(r"(\d{4})", str(d))
            if m:
                return int(m.group(1))
    return None


def search_offline(
    chunks: list[dict],
    bm25: BM25,
    query: str,
    k: int = 5,
    importance_boost: float = 0.20,
    max_per_doc: int = 2,
) -> list[dict]:
    parsed = parse_query(query)
    q_tokens = tokenize(query)

    # 1. BM25 recall ampliado (top 100 para reducir miss-rate en corpus grande)
    bm25_hits = bm25.top_k(q_tokens, k=100)
    hit_idxs = {idx for idx, _ in bm25_hits}
    bm25_max = max((s for _, s in bm25_hits), default=1.0)

    # 2. Inyección por variable económica: si la query nombra una variable,
    #    añade los top-15 chunks por importance que tengan esa variable,
    #    aunque no aparecieran en el BM25 pool (cubre queries bilingüe: ES→EN doc).
    injected_idxs: set[int] = set()
    injected: list[tuple[int, float]] = []
    if parsed["variables"]:
        var_pool = [
            (i, c.get("importance_score", 0.0))
            for i, c in enumerate(chunks)
            if i not in hit_idxs
            and any(v in c.get("economic_variables", {}) for v in parsed["variables"])
        ]
        var_pool.sort(key=lambda x: -x[1])
        injected = [(i, 0.0) for i, _ in var_pool[:15]]
        injected_idxs = {i for i, _ in injected}

    all_raw = list(bm25_hits) + injected

    # 3. Scoring
    candidates = []
    for idx, bm25_score in all_raw:
        c = chunks[idx]
        norm_bm25 = bm25_score / bm25_max if bm25_max > 0 else 0

        # Filtro duro por exact_date para Monitor PM (chunk_date presente)
        # Los PDFs (chunk_date=None) siempre pasan
        if parsed.get("exact_date"):
            cd = c.get("chunk_date")
            if cd is not None and str(cd)[:10] != parsed["exact_date"]:
                continue  # Monitor PM chunk de otra fecha → descartar

        # Filtro suave por mes (para Monitor PM sin exact_date)
        if parsed.get("month") and not parsed.get("exact_date"):
            cd = c.get("chunk_date")
            if cd is not None:
                m_num = re.match(r"\d{4}-(\d{2})", str(cd))
                if m_num and int(m_num.group(1)) != parsed["month"]:
                    continue  # Monitor PM de otro mes → descartar

        # Penalización por año mismatch (suave)
        year_factor = 1.0
        if parsed["years"]:
            cy = chunk_year(c)
            if cy and cy not in parsed["years"]:
                year_factor = 0.7

        # Bonus por doc_type match
        doctype_factor = 1.0
        if parsed["doc_types"] and c["doc_type_category"] in parsed["doc_types"]:
            doctype_factor = 1.15

        # Bonus por variable económica match (replica filtro de 04_search.py)
        variable_factor = 1.0
        if parsed["variables"]:
            chunk_vars = set(c.get("economic_variables", {}).keys())
            if any(v in chunk_vars for v in parsed["variables"]):
                variable_factor = 1.30
            elif c.get("section_type") == "DECISION":
                variable_factor = 1.10  # DECISION nunca se descarta (ver CLAUDE.md)

        importance = c.get("importance_score", 0.0)

        if idx in injected_idxs:
            # Chunk no encontrado por BM25 (ej. doc inglés, query en español).
            # Score sintético basado en importance, aplicando year_factor.
            final = 0.65 * importance * year_factor
        else:
            final = (norm_bm25 * year_factor * doctype_factor * variable_factor) + importance_boost * importance

        candidates.append({
            "chunk": c,
            "bm25_score": round(bm25_score, 4),
            "importance_score": importance,
            "final_score": round(final, 4),
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # 4. Diversidad de documentos: max max_per_doc chunks por documento
    #    Evita que un corpus grande (Monitor PM 6490 chunks) monopolice top-k.
    doc_count: dict[str, int] = defaultdict(int)
    diverse: list[dict] = []
    for cand in candidates:
        doc_id = cand["chunk"]["document_id"]
        if doc_count[doc_id] < max_per_doc:
            diverse.append(cand)
            doc_count[doc_id] += 1
        if len(diverse) >= k:
            break

    return diverse


# ── Métricas (reutilizadas de test_search.py) ────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: list[str], k: int = 1) -> float:
    return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    dcg = sum(
        (1.0 / math.log2(i + 2))
        for i, d in enumerate(retrieved[:k])
        if d in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate_query(bq: dict, chunks: list[dict], bm25: BM25, k: int) -> dict:
    results = search_offline(chunks, bm25, bq["query"], k=k)

    retrieved_docs: list[str] = []
    for r in results:
        doc_id = r["chunk"]["document_id"]
        if doc_id not in retrieved_docs:
            retrieved_docs.append(doc_id)

    # Match por sufijo: el benchmark usa IDs sin prefijo de carpeta (formato Qwen),
    # pero Gemma tiene IDs con prefijo (ej. "researchs_jpm_commodities_...").
    # Match si retrieved_id termina con relevant_id O es igual.
    def matches_relevant(retrieved_id: str, relevant_ids: list[str]) -> bool:
        for rel in relevant_ids:
            if retrieved_id == rel or retrieved_id.endswith("_" + rel) or rel in retrieved_id:
                return True
        return False

    def precision_suffix(retrieved: list[str], relevant: list[str], kk: int) -> float:
        return 1.0 if any(matches_relevant(d, relevant) for d in retrieved[:kk]) else 0.0

    def rr_suffix(retrieved: list[str], relevant: list[str]) -> float:
        for i, d in enumerate(retrieved, start=1):
            if matches_relevant(d, relevant):
                return 1.0 / i
        return 0.0

    def ndcg_suffix(retrieved: list[str], relevant: list[str], kk: int) -> float:
        dcg = sum(
            (1.0 / math.log2(i + 2))
            for i, d in enumerate(retrieved[:kk])
            if matches_relevant(d, relevant)
        )
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), kk)))
        return dcg / ideal if ideal > 0 else 0.0

    p1   = precision_suffix(retrieved_docs, bq["relevant_docs"], 1)
    r5   = precision_suffix(retrieved_docs, bq["relevant_docs"], 5)
    mrr  = rr_suffix(retrieved_docs, bq["relevant_docs"])
    ndcg = ndcg_suffix(retrieved_docs, bq["relevant_docs"], k)

    top1 = results[0] if results else None
    top1_section = top1["chunk"]["section_type"] if top1 else ""
    top1_importance = top1["chunk"].get("importance_score", 0) if top1 else 0
    expected_section = bq.get("expected_section", "")

    return {
        "id": bq["id"],
        "query": bq["query"],
        "relevant_docs": bq["relevant_docs"],
        "retrieved_top5": retrieved_docs[:5],
        "precision_at_1": p1,
        "recall_at_5": r5,
        "mrr": round(mrr, 4),
        "ndcg_at_5": round(ndcg, 4),
        "top1_section": top1_section,
        "top1_section_correct": (top1_section == expected_section) if expected_section else None,
        "top1_importance": round(top1_importance, 4),
    }


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "n_queries":     n,
        "precision_at_1": round(sum(r["precision_at_1"] for r in results) / n, 4),
        "recall_at_5":    round(sum(r["recall_at_5"]    for r in results) / n, 4),
        "mrr":            round(sum(r["mrr"]            for r in results) / n, 4),
        "ndcg_at_5":      round(sum(r["ndcg_at_5"]      for r in results) / n, 4),
        "section_accuracy": round(
            sum(1 for r in results if r.get("top1_section_correct"))
            / max(1, sum(1 for r in results if r.get("top1_section_correct") is not None)),
            4,
        ),
    }


def print_table(per_query: list[dict], agg: dict, model_label: str) -> None:
    W = 78
    print("\n" + "═" * W)
    print(f"  RESULTADOS OFFLINE  ({model_label})")
    print("═" * W)
    print(f"  {'ID':<5} {'P@1':>5} {'R@5':>5} {'MRR':>6} {'NDCG':>6}  {'Sec':>4}  Query")
    print("-" * W)
    for r in per_query:
        sec = ("✓" if r["top1_section_correct"] else "✗") if r["top1_section_correct"] is not None else "-"
        print(
            f"  {r['id']:<5} {r['precision_at_1']:>5.1f} {r['recall_at_5']:>5.1f}"
            f" {r['mrr']:>6.4f} {r['ndcg_at_5']:>6.4f}  {sec:>4}  {r['query'][:42]}"
        )
    print("─" * W)
    print(
        f"  {'AVG':<5} {agg['precision_at_1']:>5.4f} {agg['recall_at_5']:>5.4f}"
        f" {agg['mrr']:>6.4f} {agg['ndcg_at_5']:>6.4f}"
        f"  {agg['section_accuracy']:>4.2f}  (n={agg['n_queries']})"
    )
    print("═" * W)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación offline de retrieval (sin PostgreSQL)")
    parser.add_argument("--logs", default="logs_qwen", choices=["logs_qwen", "logs_gemma"],
                        help="Carpeta de logs a evaluar (default: logs_qwen)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K resultados")
    parser.add_argument("--query", help="Subset de queries (ej. q01,q05)")
    parser.add_argument("--both", action="store_true", help="Evalúa ambos modelos y compara")
    args = parser.parse_args()

    if not BENCHMARK_PATH.exists():
        print(f"ERROR: {BENCHMARK_PATH.name} no encontrado")
        sys.exit(1)

    benchmark = json.loads(BENCHMARK_PATH.read_text())
    if args.query:
        ids = {q.strip() for q in args.query.split(",")}
        benchmark = [q for q in benchmark if q["id"] in ids]

    folders = ["logs_qwen", "logs_gemma"] if args.both else [args.logs]
    all_aggs: dict[str, dict] = {}

    for folder in folders:
        path = BASE / folder / "chunks_enriched.json"
        if not path.exists():
            print(f"\n[skip] {path} no existe")
            continue

        print(f"\n[Cargando {folder}/chunks_enriched.json]")
        chunks = json.loads(path.read_text())
        print(f"  {len(chunks)} chunks")

        print(f"[Construyendo índice BM25]")
        corpus_tokens = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25(corpus_tokens)
        print(f"  Vocabulario: {len(bm25.df)} términos | docs: {bm25.N} | avg dl: {bm25.avgdl:.0f}")

        print(f"\n[Evaluando {len(benchmark)} queries]")
        per_query = []
        for bq in benchmark:
            res = evaluate_query(bq, chunks, bm25, args.top_k)
            per_query.append(res)
            marker = "✓" if res["precision_at_1"] == 1.0 else ("~" if res["mrr"] > 0 else "✗")
            print(f"  [{bq['id']}] {marker} P@1={res['precision_at_1']:.0f} MRR={res['mrr']:.3f}  {bq['query'][:45]}")

        agg = aggregate(per_query)
        all_aggs[folder] = agg
        print_table(per_query, agg, folder)

        out = BASE / f"test_offline_{folder}.json"
        out.write_text(json.dumps({"folder": folder, "aggregate": agg, "per_query": per_query}, ensure_ascii=False, indent=2))
        print(f"\n  Resultados: {out.name}")

    # Comparación si --both
    if args.both and len(all_aggs) == 2:
        W = 60
        print("\n" + "═" * W)
        print("  COMPARACIÓN OFFLINE: Qwen vs Gemma")
        print("═" * W)
        q = all_aggs["logs_qwen"]
        g = all_aggs["logs_gemma"]
        for m in ["precision_at_1", "recall_at_5", "mrr", "ndcg_at_5", "section_accuracy"]:
            qv, gv = q.get(m, 0), g.get(m, 0)
            winner = "qwen" if qv > gv else ("gemma" if gv > qv else "tie")
            print(f"  {m:<22} qwen={qv:.4f}  gemma={gv:.4f}  → {winner}")
        print("═" * W)


if __name__ == "__main__":
    main()
