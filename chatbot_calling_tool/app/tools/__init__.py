"""
Catálogo de herramientas que el agente puede invocar.

Cada herramienta vive en su propio módulo y se registra en `registry.py`
mediante el decorador @register. Este __init__ importa todos los módulos
para forzar el registro al cargar el paquete.
"""
from . import (  # noqa: F401  (side-effects: registro)
    document_lookup,
    historical_series,
    search_documents,
)
from .registry import (  # noqa: F401
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    dispatch,
)
