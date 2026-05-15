"""Resultados del pipeline de retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Salida final de ``hybrid_search``.

    ``hits`` mantiene el formato de dict legacy (compatible con
    ``RealDictCursor`` de psycopg2) para minimizar el costo de migración
    durante Fase 1E. En sub-fases futuras se podrá tipar como ``RetrievalHit``
    si se justifica.
    """

    query: str
    clean_query: str
    hits: list[dict] = field(default_factory=list)
    parsed_filters: dict | None = None  # snapshot de SearchFilters como dict para serialización

    # Trazabilidad de degradación del retrieval: si el primer recall vino vacío
    # se relajan filtros y/o se cae al fallback por importancia. Estos campos
    # permiten al agente avisar que el resultado NO respeta los filtros pedidos.
    relaxed_filters: list[str] = field(default_factory=list)  # filtros ignorados (ej. ["variables", "sections"])
    fallback_used: bool = False  # True si se usó date_importance_fallback
