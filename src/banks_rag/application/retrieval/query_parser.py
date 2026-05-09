"""Parseo de queries en lenguaje natural a ``ParsedQuery``.

Extrae filtros desde el texto del usuario en orden de prioridad:

  1. Fecha completa (día + mes + año): ``"15 de marzo de 2024"``, ``"2024-03-15"``,
     ``"15/03/2024"``. Detección antes que años o meses sueltos.
  2. Rango anual: ``"2022–2023"`` o ``"2022-2023"``.
  3. Año solo: ``"2022"``, ``"2023"`` (puede haber múltiples).
  4. Mes solo: ``"enero"``, ``"febrero"``, ``"ene"``, etc.
  5. Día + mes sin año: ``"15 de marzo"`` (se combina si hay un único año detectado).
  6. Tipos de documento: matching de keywords como ``"comunicado"``, ``"minuta"``, ``"fed"``.
  7. Variables económicas y secciones: vía patterns de ``taxonomy``.

El ``clean_query`` removueve los tokens consumidos por filtros.
"""

from __future__ import annotations

import re

from banks_rag.domain.retrieval import ParsedQuery, SearchFilters
from banks_rag.domain_knowledge.taxonomy import (
    build_section_patterns,
    build_variable_patterns,
    normalize_text,
)

# Patterns compilados a la importación.
_VARIABLE_PATTERNS = build_variable_patterns()
_SECTION_PATTERNS = build_section_patterns()

YEAR_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")
YEAR_RANGE_RE = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\s*[-–a]\s*(19[8-9]\d|20[0-4]\d)\b")

MONTH_NAMES: dict[str, int] = {
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "mayo": 5,
    "junio": 6, "jun": 6,
    "julio": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "sep": 9, "sept": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11,
    "diciembre": 12, "dic": 12,
}
MONTH_RE = re.compile(
    r"\b(" + "|".join(sorted(MONTH_NAMES, key=len, reverse=True)) + r")\b"
)
MONTH_NUM_TO_NAME: dict[int, str] = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}

# Patrones de fecha completa (orden importa: específico → ambiguo).
_MONTH_ALT = "|".join(sorted(MONTH_NAMES, key=len, reverse=True))

FULL_DATE_NAMED_RE = re.compile(
    r"\b(\d{1,2})\s+(?:de\s+)?(" + _MONTH_ALT + r")\s+(?:de\s+)?(\d{4})\b",
    re.IGNORECASE,
)
FULL_DATE_YMD_RE = re.compile(r"\b(20[0-4]\d)[/_.-](\d{2})[/_.-](\d{2})\b")
FULL_DATE_DMY_RE = re.compile(r"\b(\d{1,2})[/_.-](\d{1,2})[/_.-](20[0-4]\d|\d{2})\b")
DAY_MONTH_NAMED_RE = re.compile(
    r"\b(\d{1,2})\s+(?:de\s+)?(" + _MONTH_ALT + r")\b",
    re.IGNORECASE,
)

FILENAME_DATE_RE = re.compile(
    r"(?:^|[^0-9])(?:(\d{4})[._/-](\d{2})[._/-](\d{2})|"
    r"(\d{2})[._/-](\d{2})[._/-](\d{2}))(?:[^0-9]|$)"
)

# Hints para tipo de documento. Más específicos que detect_doc_type del paso 0
# (que opera sobre filepath); estos detectan menciones explícitas en la query.
DOC_TYPE_HINTS: dict[str, list[str]] = {
    "COMUNICADO": ["comunicado", "anuncio de politica"],
    "MINUTA": ["minuta", "reunion de politica", "consejeros"],
    "FED_STATEMENT": ["fed", "federal reserve", "fomc"],
    "REPORTE_RESEARCH": ["research", "jpmorgan", "jpm", "reporte"],
    "MONITOR_PM": ["monitor pm", "monitor de mercado", "monitor financiero"],
}


def parse_query(query: str) -> ParsedQuery:
    """Parsea una query en NL extrayendo filtros y devolviendo ``ParsedQuery``.

    El orden de extracción importa: fecha completa antes que año o mes sueltos.
    Los tokens consumidos se eliminan de ``clean_query``.
    """
    norm = normalize_text(query)
    clean = query

    filters = SearchFilters()

    # ── 1. Fecha completa: detectar antes que años/meses sueltos ─────────────
    # Prioridad: named > YMD > DMY (de más específico a más ambiguo).
    fm = FULL_DATE_NAMED_RE.search(norm)
    if fm:
        d_val, m_name, y_val = int(fm.group(1)), fm.group(2), int(fm.group(3))
        m_val = MONTH_NAMES.get(m_name)
        if m_val and 1 <= d_val <= 31:
            filters.day = d_val
            filters.month = m_val
            filters.year_from = y_val
            filters.year_to = y_val
            filters.exact_date = f"{y_val:04d}-{m_val:02d}-{d_val:02d}"
            clean = FULL_DATE_NAMED_RE.sub("", clean)

    if not filters.exact_date:
        fm = FULL_DATE_YMD_RE.search(norm)
        if fm:
            y_val, m_val, d_val = int(fm.group(1)), int(fm.group(2)), int(fm.group(3))
            if 1 <= m_val <= 12 and 1 <= d_val <= 31:
                filters.day = d_val
                filters.month = m_val
                filters.year_from = y_val
                filters.year_to = y_val
                filters.exact_date = f"{y_val:04d}-{m_val:02d}-{d_val:02d}"
                clean = FULL_DATE_YMD_RE.sub("", clean)

    if not filters.exact_date:
        fm = FULL_DATE_DMY_RE.search(norm)
        if fm:
            d_val, m_val = int(fm.group(1)), int(fm.group(2))
            y_str = fm.group(3)
            y_val = int(y_str) if len(y_str) == 4 else 2000 + int(y_str)
            if 1 <= m_val <= 12 and 1 <= d_val <= 31:
                filters.day = d_val
                filters.month = m_val
                filters.year_from = y_val
                filters.year_to = y_val
                filters.exact_date = f"{y_val:04d}-{m_val:02d}-{d_val:02d}"
                clean = FULL_DATE_DMY_RE.sub("", clean)

    # ── 2. Año o rango (solo si no se capturó en fecha completa) ─────────────
    if filters.year_from is None:
        m = YEAR_RANGE_RE.search(norm)
        if m:
            year_from, year_to = int(m.group(1)), int(m.group(2))
            if year_from > year_to:
                year_from, year_to = year_to, year_from
            filters.year_from = year_from
            filters.year_to = year_to
            clean = YEAR_RANGE_RE.sub("", clean)
        else:
            years = [int(y) for y in YEAR_RE.findall(norm)]
            if years:
                filters.year_from = min(years)
                filters.year_to = max(years)
                clean = YEAR_RE.sub("", clean)

    # ── 3. Mes suelto (si no fue capturado por fecha completa) ───────────────
    if filters.month is None:
        mm = MONTH_RE.search(norm)
        if mm:
            filters.month = MONTH_NAMES[mm.group(1)]
            clean = MONTH_RE.sub("", clean)

    # ── 4. Día + mes sin año → combinar con año si es único ──────────────────
    if filters.exact_date is None and filters.day is None:
        dm = DAY_MONTH_NAMED_RE.search(norm)
        if dm:
            d_val = int(dm.group(1))
            m_name = dm.group(2)
            m_val = MONTH_NAMES.get(m_name)
            if m_val and 1 <= d_val <= 31:
                filters.day = d_val
                if filters.month is None:
                    filters.month = m_val
                if filters.year_from is not None and filters.year_from == filters.year_to:
                    filters.exact_date = f"{filters.year_from:04d}-{m_val:02d}-{d_val:02d}"
                clean = DAY_MONTH_NAMED_RE.sub("", clean)

    # ── 5. Tipos de documento ────────────────────────────────────────────────
    filters.doc_types = [
        dt for dt, keywords in DOC_TYPE_HINTS.items()
        if any(kw in norm for kw in keywords)
    ]

    # ── 6. Variables económicas y secciones (desde taxonomy) ─────────────────
    filters.variables = [v for v, pat in _VARIABLE_PATTERNS.items() if pat.search(norm)]
    filters.sections = [s for s, pat in _SECTION_PATTERNS.items() if pat.search(norm)]

    cleaned = re.sub(r"\s+", " ", clean).strip() or query

    return ParsedQuery(
        raw_query=query,
        clean_query=cleaned,
        filters=filters,
    )


def extract_month_from_row(row: dict) -> int | None:
    """Extrae el mes (1-12) de un row de retrieval, en orden de prioridad:

    1. ``chunk_date`` (date object o ISO string).
    2. ``document_date`` (ISO o DD-MM-YYYY).
    3. Fecha en ``filename`` (regex).
    """
    chunk_date = row.get("chunk_date")
    if chunk_date is not None:
        month = getattr(chunk_date, "month", None)
        if month is not None:
            return int(month)
        if isinstance(chunk_date, str) and len(chunk_date) >= 7 and chunk_date[4] == "-":
            try:
                return int(chunk_date[5:7])
            except ValueError:
                pass

    document_date = row.get("document_date")
    if isinstance(document_date, str):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", document_date):
            return int(document_date[5:7])
        if re.match(r"^\d{2}[._/-]\d{2}[._/-]\d{2,4}$", document_date):
            parts = re.split(r"[._/-]", document_date)
            if len(parts) == 3:
                try:
                    return int(parts[1])
                except ValueError:
                    pass

    filename = row.get("filename") or ""
    match = FILENAME_DATE_RE.search(filename)
    if match:
        month_part = match.group(2) or match.group(5)
        if month_part:
            try:
                return int(month_part)
            except ValueError:
                return None
    return None


def row_matches_month(row: dict, month: int | None) -> bool:
    """``True`` si la fila coincide con el mes filtrado (o si no hay filtro)."""
    if month is None:
        return True
    return extract_month_from_row(row) == month
