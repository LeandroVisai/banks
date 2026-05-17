"""Tool registry + tools concretas del agente.

Importar este paquete inicializa el registro: cada submódulo de tool registra
su función vía decorador ``@register`` al ser importado.

Tools registradas:
  - ``search_documents``: hybrid search sobre el corpus (TEXT por default).
  - ``search_visuals``: hybrid search restringido a chunks VISUAL/TABLE.
  - ``list_documents`` / ``get_document_chunks``: exploración por tipo/año/filename.
  - ``list_historical_series`` / ``get_historical_series``: series macro vía DW.
  - ``discover_query``: descubre queries analíticas en el catálogo SQL (Fase 4).
  - ``execute_query``: ejecuta una query del catálogo sobre parquets (Fase 4).
  - ``compute_variation`` / ``compute_spread`` / ``get_series_stats`` /
    ``detect_anomaly`` / ``get_market_snapshot``: analytics sobre el catálogo
    SQL — interpretan los datos en vez de devolverlos crudos (Fase A multi-agente).
  - ``compare_meetings`` / ``get_recent_policy_decisions``: tools de reuniones
    de política monetaria a nivel de documento (Fase C multi-agente).
"""

from . import (  # noqa: F401
    analytics,
    discover_query,
    document_lookup,
    execute_query,
    historical_series,
    meeting_lookup,
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
