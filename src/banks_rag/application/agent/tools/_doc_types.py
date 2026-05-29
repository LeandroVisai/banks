"""Taxonomía de tipos de documento compartida por las tools del corpus.

Los valores canónicos son los que produce ``detect_doc_type`` y que viven en
la columna ``doc_type_category`` de PostgreSQL. El LLM tiende a usar nombres
más simples ("COMUNICADO", "MINUTA", "RESEARCH", "FED STATEMENT"): este módulo
los mapea a los canónicos y acepta tanto un string como un array, de modo que
``list_documents`` / ``search_documents`` / ``search_visuals`` se comporten de
forma consistente sin reventar el SQL (bug ``text = text[]`` de los logs).
"""

from __future__ import annotations

# Valores canónicos de doc_type_category (== salida de detect_doc_type).
DOC_TYPE_VALUES: tuple[str, ...] = (
    "COMUNICADO_RPM",
    "MINUTA_RPM",
    "MINUTA_IPOM",
    "IPOM",
    "IEF",
    "FED_STATEMENT",
    "REPORTE_RESEARCH",
    "MONITOR_PM",
)

_CANON = set(DOC_TYPE_VALUES)

# Alias frecuentes del LLM → uno o varios valores canónicos. Clave en mayúsculas
# y sin espacios/guiones (ver _key).
_ALIASES: dict[str, tuple[str, ...]] = {
    "COMUNICADO": ("COMUNICADO_RPM",),
    "COMUNICADORPM": ("COMUNICADO_RPM",),
    "MINUTA": ("MINUTA_RPM", "MINUTA_IPOM"),
    "MINUTAS": ("MINUTA_RPM", "MINUTA_IPOM"),
    "RESEARCH": ("REPORTE_RESEARCH",),
    "REPORTERESEARCH": ("REPORTE_RESEARCH",),
    "JPMORGAN": ("REPORTE_RESEARCH",),
    "JPM": ("REPORTE_RESEARCH",),
    "FED": ("FED_STATEMENT",),
    "FEDSTATEMENT": ("FED_STATEMENT",),
    "MONITORPM": ("MONITOR_PM",),
    "MONITOR": ("MONITOR_PM",),
    "IPOM": ("IPOM",),
    "IPOMS": ("IPOM",),
    "IEF": ("IEF",),
}


def _key(value: str) -> str:
    """Normaliza para lookup: mayúsculas, sin espacios/guiones."""
    return value.upper().replace(" ", "").replace("-", "").replace("_", "")


def normalize_doc_types(value: str | list[str] | None) -> list[str]:
    """Normaliza ``doc_type`` (string o array) a una lista de valores canónicos.

    - ``None`` / vacío → ``[]`` (sin filtro).
    - Reconoce valores canónicos y alias del LLM ("COMUNICADO" → "COMUNICADO_RPM").
    - "MINUTA" expande a ambas minutas (RPM + IPoM).
    - Valores no reconocidos se descartan (evita filtros que no matchean nada).
    - Resultado deduplicado, preservando el orden de aparición.
    """
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)

    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        if item in _CANON:
            canon = (item,)
        else:
            canon = _ALIASES.get(_key(item), ())
        for c in canon:
            if c not in out:
                out.append(c)
    return out
