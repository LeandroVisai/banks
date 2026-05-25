"""Métricas de recuperación para el pipeline RAG.

Implementa hit-rate@k, MRR@k y nDCG@k sobre resultados de hybrid_search.
Diseñado para correr offline contra el golden set en ``data/golden_set/``.

Nota sobre ``hit_rate_at_k``: es 1.0 si hay al menos un chunk relevante en el
top-k y 0.0 si no — NO es recall (fracción de relevantes recuperada), porque
el golden set no enumera los chunk_ids relevantes. Se renombró desde
``recall_at_k`` para que el gate de CI no sugiera más de lo que mide.

Uso mínimo:
    hits = hybrid_search("query", ...)
    result = evaluate_retrieval(hits, expected_doc_types=["COMUNICADO_RPM"],
                                expected_sections=["DECISION"])
    print(result.hit_rate_at_k)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_GOLDEN_SET_PATH = Path(__file__).parents[4] / "data" / "golden_set" / "retrieval.jsonl"


@dataclass
class RetrievalResult:
    """Resultado de evaluar un conjunto de hits contra un caso del golden set."""

    query: str
    k: int
    hit_rate_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    hits_count: int
    matched_at: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Métricas agregadas sobre todos los casos del golden set."""

    n_queries: int
    hit_rate_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    k: int

    def to_dict(self) -> dict:
        return asdict(self)


def _is_hit(chunk: dict, expected_doc_types: list[str], expected_sections: list[str]) -> bool:
    """Determina si un chunk satisface los criterios del caso golden."""
    doc_type_ok = (
        not expected_doc_types
        or chunk.get("doc_type_category", "") in expected_doc_types
    )
    section_ok = (
        not expected_sections
        or chunk.get("section_type", "") in expected_sections
    )
    return doc_type_ok and section_ok


def evaluate_retrieval(
    hits: list[dict],
    *,
    expected_doc_types: list[str] | None = None,
    expected_sections: list[str] | None = None,
    min_importance: float | None = None,
    query: str = "",
    k: int | None = None,
) -> RetrievalResult:
    """Evalúa una lista de chunks recuperados contra criterios del golden set.

    Args:
        hits: chunks devueltos por hybrid_search (ya cortados a top-k).
        expected_doc_types: lista de doc_type_category válidos (OR).
        expected_sections: lista de section_type válidos (OR).
        min_importance: umbral mínimo de importance_score para contar como hit.
        query: texto de la query (solo para el reporte).
        k: tamaño del ranking evaluado; si None usa len(hits).

    Returns:
        ``RetrievalResult`` con hit-rate, MRR y nDCG.
    """
    doc_types = expected_doc_types or []
    sections = expected_sections or []
    actual_k = k if k is not None else len(hits)
    hits_to_eval = hits[:actual_k]

    relevances: list[int] = []
    matched_at: list[int] = []
    for rank, chunk in enumerate(hits_to_eval, start=1):
        hit = _is_hit(chunk, doc_types, sections)
        if min_importance is not None:
            hit = hit and chunk.get("importance_score", 0.0) >= min_importance
        rel = 1 if hit else 0
        relevances.append(rel)
        if rel:
            matched_at.append(rank)

    hit_rate = min(1.0, sum(relevances))  # 1.0 si hay ≥1 chunk relevante en top-k
    mrr = _mrr(relevances)
    ndcg = _ndcg(relevances)

    return RetrievalResult(
        query=query,
        k=actual_k,
        hit_rate_at_k=hit_rate,
        mrr_at_k=mrr,
        ndcg_at_k=ndcg,
        hits_count=len(hits_to_eval),
        matched_at=matched_at,
    )


def _mrr(relevances: list[int]) -> float:
    for i, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def _ndcg(relevances: list[int]) -> float:
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(results: list[RetrievalResult]) -> AggregateMetrics:
    """Calcula métricas promedio sobre una lista de RetrievalResult."""
    if not results:
        return AggregateMetrics(n_queries=0, hit_rate_at_k=0.0, mrr_at_k=0.0, ndcg_at_k=0.0, k=0)
    n = len(results)
    return AggregateMetrics(
        n_queries=n,
        hit_rate_at_k=sum(r.hit_rate_at_k for r in results) / n,
        mrr_at_k=sum(r.mrr_at_k for r in results) / n,
        ndcg_at_k=sum(r.ndcg_at_k for r in results) / n,
        k=results[0].k,
    )


def load_retrieval_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    """Lee el golden set de retrieval desde el JSONL."""
    p = path or _GOLDEN_SET_PATH
    if not p.exists():
        return []
    cases = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_sql_routing_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    """Lee el golden set de SQL routing desde el JSONL."""
    default = Path(__file__).parents[4] / "data" / "golden_set" / "sql_routing.jsonl"
    p = path or default
    if not p.exists():
        return []
    cases = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def evaluate_sql_routing(
    cases: list[dict[str, Any]],
    route_fn,
) -> dict[str, float]:
    """Evalúa el QueryRouter contra el golden set de SQL routing.

    Args:
        cases: lista de dicts con ``query`` y ``expected_query_id``.
        route_fn: callable(query) → RouteDecision.

    Returns:
        Dict con ``accuracy`` y ``sql_recall``.
    """
    correct_route = 0
    sql_activated = 0
    for case in cases:
        decision = route_fn(case["query"])
        expected_route = case.get("expected_route", "sql")
        if expected_route == "sql" and decision.sql:
            correct_route += 1
            sql_activated += 1
        elif expected_route == "rag" and decision.rag:
            correct_route += 1
        elif expected_route == "visual" and decision.visual:
            correct_route += 1

    n = len(cases) or 1
    return {
        "n_cases": len(cases),
        "accuracy": correct_route / n,
        "sql_recall": sql_activated / n,
    }
