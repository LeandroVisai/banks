"""Adaptadores de persistencia: PostgreSQL + pgvector (sync + async)."""

from . import sql_templates
from .postgres_repo import (
    PostgresRepo,
    document_year_from_date,
    format_vector,
    safe_ident,
)

__all__ = [
    "sql_templates",
    "PostgresRepo",
    "format_vector",
    "safe_ident",
    "document_year_from_date",
]
