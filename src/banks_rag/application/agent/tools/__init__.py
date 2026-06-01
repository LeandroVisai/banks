"""Tool registry + tools concretas del agente.

Importar este paquete inicializa el registro: cada submódulo de tool registra
su función vía decorador ``@register`` al ser importado.

Tools registradas:
  - ``search_documents``: hybrid search sobre el corpus (TEXT por default).
  - ``search_visuals``: hybrid search restringido a chunks VISUAL/TABLE.
  - ``list_documents`` / ``get_document_chunks``: exploración por tipo/año/filename.
  - ``discover_query``: descubre queries analíticas en el catálogo SQL (Fase 4).
  - ``execute_query``: ejecuta una query del catálogo sobre parquets (Fase 4).
  - ``list_parquets``: descubre datasets analíticos en el catálogo de parquets.
  - ``query_parquet``: ejecuta SQL DuckDB libre sobre cualquier parquet del catálogo.
  - ``compute_variation`` / ``compute_spread`` / ``get_series_stats`` /
    ``detect_anomaly`` / ``get_market_snapshot``: analytics sobre el catálogo
    SQL — interpretan los datos en vez de devolverlos crudos (Fase A multi-agente).
  - ``compare_meetings`` / ``get_recent_policy_decisions``: tools de reuniones
    de política monetaria a nivel de documento (Fase C multi-agente).
  - ``plot_series``: grafica series de tiempo del catálogo de parquets como
    spec Vega-Lite (la tool arma el spec; el frontend lo renderiza).
"""

from . import (  # noqa: F401
    analytics,
    discover_query,
    document_lookup,
    execute_query,
    get_series,
    list_parquets,
    meeting_lookup,
    plot_series,
    query_parquet,
    search_documents,
    search_visuals,
)
from .registry import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    dispatch,
    register,
    reset_registry,
    signature_hint,
)

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "dispatch",
    "register",
    "reset_registry",
    "signature_hint",
]
