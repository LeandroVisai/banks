"""Casos de uso de retrieval: parsing, filtros, fusion, hybrid search."""

from .filters import build_filters_sql
from .fusion import (
    DEFAULT_IMPORTANCE_BOOST,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_RECENCY_WEIGHT,
    DEFAULT_RRF_K,
    doc_year,
    importance_boost,
    mark_low_confidence,
    mmr_select,
    parse_embedding,
    recency_factor,
    rrf_fuse,
)
from .hybrid_search import (
    DEFAULT_RECALL_N,
    embed_query_text,
    hybrid_search,
)
from .query_parser import (
    DOC_TYPE_HINTS,
    MONTH_NAMES,
    MONTH_NUM_TO_NAME,
    extract_month_from_row,
    parse_query,
    row_matches_month,
)

__all__ = [
    "build_filters_sql",
    "DEFAULT_IMPORTANCE_BOOST",
    "DEFAULT_MMR_LAMBDA",
    "DEFAULT_RECENCY_WEIGHT",
    "DEFAULT_RRF_K",
    "DEFAULT_RECALL_N",
    "doc_year",
    "embed_query_text",
    "extract_month_from_row",
    "hybrid_search",
    "importance_boost",
    "mark_low_confidence",
    "mmr_select",
    "parse_embedding",
    "parse_query",
    "recency_factor",
    "row_matches_month",
    "rrf_fuse",
    "DOC_TYPE_HINTS",
    "MONTH_NAMES",
    "MONTH_NUM_TO_NAME",
]
