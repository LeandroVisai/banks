"""Detección de metadata desde la ruta y/o filename.

Funciones puras que infieren tipo de documento, institución, fecha y
``document_id`` a partir de la ruta relativa al directorio raíz de datos.

Política central: ``doc_type`` y ``institution`` se derivan **solo del filepath**,
nunca del contenido. Un chunk de minuta que mencione "comunicado" no se
re-clasifica como tal.
"""

from __future__ import annotations

import re
from pathlib import Path

_MONTH_ES: dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

_DATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(\d{4})[._/-](\d{1,2})[._/-](\d{1,2})"),  # ISO-ish: 2022-07-13
    re.compile(r"(\d{1,2})[._/-](\d{1,2})[._/-](\d{2,4})"),  # DMY: 13-07-2022
    re.compile(  # "13 de julio de 2022"
        r"(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|"
        r"julio|agosto|septiembre|octubre|noviembre|diciembre)"
        r"\s+de\s+(\d{4})",
        re.IGNORECASE,
    ),
]

_YEAR_FALLBACK_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def detect_doc_type(rel_path: str) -> str:
    """Infiere ``doc_type_category`` desde la ruta relativa."""
    p = rel_path.lower()
    if "comunicados_rpm" in p or "comunicado_rpm" in p or "comunicados rpm" in p:
        return "COMUNICADO_RPM"
    if "minutas_ipom" in p or "minuta_ipom" in p or "minutas ipom" in p:
        return "MINUTA_IPOM"
    if "minutas_rpm" in p or "minuta_rpm" in p or "minutas rpm" in p:
        return "MINUTA_RPM"
    if "ipom" in p:
        return "IPOM"
    if "ief" in p:
        return "IEF"
    if "/fed/" in p or p.startswith("fed/") or "reunion_fed" in p:
        return "FED_STATEMENT"
    if "monitor_pm" in p or "monitor pm" in p:
        return "MONITOR_PM"
    if "research" in p or "jpm" in p:
        return "REPORTE_RESEARCH"
    return "REPORTE_RESEARCH"  # default conservador


def detect_institution(rel_path: str) -> str:
    """Infiere institución emisora desde la ruta relativa."""
    p = rel_path.lower()
    if "jpm" in p or "jpmorgan" in p:
        return "jpmorgan"
    if "/fed/" in p or p.startswith("fed/") or "fed" in p:
        return "federal_reserve"
    if "comunicado" in p or "minuta" in p or "monitor_pm" in p or "monitor pm" in p:
        return "banco_central_chile"
    return "unknown"


def detect_date(filename: str, first_page_text: str = "") -> str | None:
    """Busca fecha en filename primero, luego en los primeros 500 chars del doc.

    Retorna ISO ``YYYY-MM-DD`` o solo el año si no se detecta día/mes.
    """
    haystack = filename + " " + first_page_text[:500]

    for pat in _DATE_PATTERNS:
        m = pat.search(haystack)
        if not m:
            continue
        groups = m.groups()
        try:
            if len(groups) == 3 and groups[1].lower() in _MONTH_ES:
                day, month_name, year = groups
                return f"{int(year):04d}-{_MONTH_ES[month_name.lower()]:02d}-{int(day):02d}"
            if len(groups[0]) == 4:
                y, mo, d = groups
                return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
            d, mo, y = groups
            year = int(y)
            if year < 100:
                year += 2000 if year < 50 else 1900
            return f"{year:04d}-{int(mo):02d}-{int(d):02d}"
        except (ValueError, KeyError):
            continue

    m = _YEAR_FALLBACK_RE.search(haystack)
    return m.group(1) if m else None


def slugify_document_id(rel_path: str) -> str:
    """Convierte una ruta relativa en slug único para usar como ``document_id``.

    Usa la ruta completa (sin extensión) en minúsculas para que archivos con el
    mismo nombre en subdirectorios distintos obtengan IDs distintos.
    """
    base = Path(rel_path).with_suffix("").as_posix().lower()
    return _SLUG_RE.sub("_", base).strip("_")
