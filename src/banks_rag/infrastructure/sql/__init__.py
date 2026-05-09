"""Adaptadores SQL: recall queries (vector + lexical), futuro sql_catalog."""

from .recall_queries import (
    date_importance_fallback,
    format_pgvector,
    get_db_embedding_dim,
    lexical_recall,
    vector_recall,
)

__all__ = [
    "date_importance_fallback",
    "format_pgvector",
    "get_db_embedding_dim",
    "lexical_recall",
    "vector_recall",
]
