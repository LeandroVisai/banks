"""Fusión de rankings y re-rerank: RRF + MMR + importance_boost.

Funciones puras (no requieren DB ni embedder externo), trabajan sobre
``list[dict]`` con scores como campos del dict.

- **RRF (Reciprocal Rank Fusion)**: combina dos rankings sin necesidad de
  calibrar pesos. ``score(d) = Σ 1 / (k + rank_in_list_i)``.

- **MMR (Maximal Marginal Relevance)**: balancea relevancia y diversidad.
  ``score_mmr(d) = λ * sim(d, q) - (1-λ) * max_selected sim(d, s)``.

- **Importance boost**: re-orden final tipo tie-breaker que prefiere chunks
  con mayor ``importance_score`` y documentos más recientes (sin reemplazar
  la relevancia del retrieval).
"""

from __future__ import annotations

import datetime
import re

import numpy as np

DEFAULT_RRF_K = 60
DEFAULT_MMR_LAMBDA = 0.65
DEFAULT_IMPORTANCE_BOOST = 0.15
DEFAULT_RECENCY_WEIGHT = 0.03


def rrf_fuse(
    vector_hits: list[dict],
    lexical_hits: list[dict],
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Reciprocal Rank Fusion — combina dos rankings sin calibrar pesos.

    Cada documento acumula ``1 / (k + rank)`` desde cada lista en la que aparece.
    Anota ``_rrf_vector_rank`` y ``_rrf_lexical_rank`` para debugging.
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


def parse_embedding(emb) -> np.ndarray | None:
    """Parsea un embedding de pgvector (string ``'[0.1,0.2,...]'`` o lista).

    Devuelve ``None`` si el embedding es inválido (vacío, no parseable o con
    valores no finitos), para que el caller lo descarte en vez de propagar
    ``NaN``/``inf`` o caer con ``ValueError``.
    """
    try:
        if isinstance(emb, str):
            tokens = [t for t in emb.strip("[]").split(",") if t.strip()]
            if not tokens:
                return None
            vec = np.array([float(x) for x in tokens], dtype=np.float32)
        else:
            vec = np.asarray(emb, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if vec.size == 0 or not np.isfinite(vec).all():
        return None
    return vec


def _mmr_score(
    candidate_idx: int,
    cand_vecs: list[np.ndarray],
    relevance: np.ndarray,
    selected_idx: list[int],
    lambda_param: float,
) -> float:
    max_redundancy = max(
        float(np.dot(cand_vecs[candidate_idx], cand_vecs[selected]))
        for selected in selected_idx
    )
    return lambda_param * relevance[candidate_idx] - (1 - lambda_param) * max_redundancy


def mmr_select(
    candidates: list[dict],
    query_vec: np.ndarray,
    k: int,
    lambda_param: float = DEFAULT_MMR_LAMBDA,
) -> list[dict]:
    """Maximal Marginal Relevance: balancea relevancia y diversidad.

    Args:
        candidates: rows con campo ``embedding`` (string pgvector o list[float]).
        query_vec: embedding de la query (np.ndarray L2-normalizado).
        k: cuántos elementos seleccionar.
        lambda_param: 1.0 = solo relevancia, 0.0 = solo diversidad.
    """
    if not candidates:
        return []

    # Descarta candidatos con embedding inválido (NaN, vacío, no parseable):
    # incluirlos rompería el argmax de MMR con comparaciones contra NaN.
    valid: list[dict] = []
    cand_vecs: list[np.ndarray] = []
    for c in candidates:
        vec = parse_embedding(c.get("embedding"))
        if vec is not None:
            valid.append(c)
            cand_vecs.append(vec)
    if not valid:
        return candidates[:k]
    candidates = valid

    # Relevancia para MMR: usa el ``rrf_score`` ya fusionado (incluye la señal
    # léxica BM25), normalizado a [0,1]. Cae a la similitud coseno solo si
    # algún candidato no trae rrf_score (p. ej. viene del fallback por
    # importancia, que no pasa por rrf_fuse).
    rrf_scores = [c.get("rrf_score") for c in candidates]
    if all(s is not None for s in rrf_scores):
        max_rrf = max(rrf_scores) or 1.0
        relevance = np.array([float(s) / max_rrf for s in rrf_scores])
    else:
        relevance = np.array([float(np.dot(query_vec, v)) for v in cand_vecs])

    selected_idx: list[int] = []
    remaining = set(range(len(candidates)))

    while len(selected_idx) < k and remaining:
        if not selected_idx:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(
                remaining,
                key=lambda i: _mmr_score(
                    i, cand_vecs, relevance, selected_idx, lambda_param
                ),
            )
        selected_idx.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_idx]


def doc_year(hit: dict) -> int | None:
    """Año del chunk (chunk_date) o documento (document_date)."""
    for key in ("chunk_date", "document_date"):
        v = hit.get(key)
        if v:
            m = re.match(r"(\d{4})", str(v))
            if m:
                return int(m.group(1))
    return None


_RECENCY_HALF_LIFE_YEARS = 8.0


def recency_factor(year: int | None, today_year: int | None = None) -> float:
    """0.0–1.0 — decae con la antigüedad. Sin año → 0.5 neutral.

    Usa decay exponencial (vida media de 8 años) en vez de lineal: así los
    documentos antiguos conservan un gradiente y no colapsan todos a 0 — un
    decay lineal de 5%/año igualaba todo lo anterior a ~20 años atrás.
    """
    if year is None:
        return 0.5
    today_year = today_year if today_year is not None else datetime.date.today().year
    age = max(0, today_year - year)
    return float(0.5 ** (age / _RECENCY_HALF_LIFE_YEARS))


def importance_boost(
    hits: list[dict],
    weight: float = DEFAULT_IMPORTANCE_BOOST,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
) -> list[dict]:
    """Re-orden final con ``importance_score`` y recency como tie-breakers suaves.

    No reemplaza la relevancia del retrieval; solo desempata entre candidatos
    con RRF similar, prefiriendo chunks curados y documentos más recientes.
    Modifica ``hits`` in-place agregando ``final_score`` y ordenando.
    """
    for h in hits:
        rec = recency_factor(doc_year(h))
        h["final_score"] = (
            h.get("rrf_score", 0.0)
            + weight * float(h.get("importance_score", 0))
            + recency_weight * rec
        )
    hits.sort(key=lambda h: h["final_score"], reverse=True)
    return hits


def mark_low_confidence(results: list[dict], threshold: float = 0.05) -> None:
    """Anota ``low_confidence=True`` cuando ``final_score`` está bajo umbral."""
    for result in results:
        result["low_confidence"] = result.get("final_score", 0) < threshold
