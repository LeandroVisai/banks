"""Tool registry + tools concretas del agente.

Importar este paquete inicializa el registro: cada submódulo de tool registra
su función vía decorador ``@register`` al ser importado.

Las tools concretas (search_documents, document_lookup, historical_series)
se agregan en sub-fases posteriores conforme se complete su integración con
los adaptadores nuevos (hybrid_search, postgres_repo, dw_store).
"""

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
