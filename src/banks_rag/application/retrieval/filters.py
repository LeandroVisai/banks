"""Construcción de cláusulas WHERE SQL desde ``SearchFilters``.

Combina los filtros auto-detectados (años, fechas, vars, sections) con los
avanzados (years[], institutions, entities, tags, importance ranges,
exclusiones) en un único ``WHERE ... AND ...``.

El SQL emitido es agnóstico del nombre de tablas — el caller debe pasar los
nombres validados (vía ``safe_ident``) para evitar inyección.

Nota sobre fechas:
  - ``exact_date`` aplica a chunks con ``chunk_date`` (Monitor PM); los PDFs
    con ``chunk_date IS NULL`` pasan el filtro y se filtran en Python por
    ``document_date`` vía ``row_matches_month``.
  - ``month`` solo se filtra en SQL para Monitor PM; los PDFs se filtran
    post-SQL por su ``document_date``.
"""

from __future__ import annotations

from banks_rag.domain.retrieval import SearchFilters


def build_filters_sql(
    filters: SearchFilters,
    *,
    docs_table: str,
    chunks_table: str,
    chunks_alias: str | None = None,
    docs_alias: str | None = None,
) -> tuple[str, list]:
    """Construye ``(where_clause, params)`` desde un ``SearchFilters``.

    Args:
        filters: filtros declarativos.
        docs_table: nombre validado de la tabla documents.
        chunks_table: nombre validado de la tabla chunks.
        chunks_alias: alias SQL de la tabla chunks en el FROM (p.ej. ``"c"``).
            Si ``None``, usa ``chunks_table`` directamente.
        docs_alias: alias SQL de la tabla docs en el FROM (p.ej. ``"d"``).
            Si ``None``, usa ``docs_table`` directamente.

    Returns:
        ``(where_sql, params)``. ``where_sql`` es ``"TRUE"`` si no hay filtros.
    """
    c = chunks_alias if chunks_alias is not None else chunks_table
    d = docs_alias if docs_alias is not None else docs_table
    clauses: list[str] = []
    params: list = []

    # ── doc_types (auto-detectado de NL) ─────────────────────────────────────
    if filters.doc_types:
        clauses.append(f"{d}.doc_type_category = ANY(%s)")
        params.append(filters.doc_types)

    # ── Filtros de fecha auto-detectados ─────────────────────────────────────
    # Prioridad: exact_date > year_range + month
    if filters.exact_date is not None:
        clauses.append(
            f"({c}.chunk_date IS NULL OR {c}.chunk_date = %s::date)"
        )
        params.append(filters.exact_date)
    else:
        if filters.year_from is not None:
            clauses.append(
                f"COALESCE(EXTRACT(YEAR FROM {c}.chunk_date)::int, "
                f"{d}.document_year) >= %s"
            )
            params.append(filters.year_from)
        if filters.year_to is not None:
            clauses.append(
                f"COALESCE(EXTRACT(YEAR FROM {c}.chunk_date)::int, "
                f"{d}.document_year) <= %s"
            )
            params.append(filters.year_to)
        if filters.month is not None:
            clauses.append(
                f"({c}.chunk_date IS NULL OR "
                f"EXTRACT(MONTH FROM {c}.chunk_date)::int = %s)"
            )
            params.append(filters.month)

    if filters.sections:
        clauses.append(f"{c}.section_type = ANY(%s)")
        params.append(filters.sections)

    if filters.variables:
        # AL MENOS una variable o sección DECISION (que siempre es relevante).
        clauses.append(
            f"({c}.economic_variables ?| %s "
            f"OR {c}.section_type = 'DECISION')"
        )
        params.append(filters.variables)

    # ── Filtros avanzados (explícitos) ───────────────────────────────────────
    if filters.years:
        placeholders = ",".join(["%s"] * len(filters.years))
        clauses.append(
            f"COALESCE(EXTRACT(YEAR FROM {c}.chunk_date)::int, "
            f"{d}.document_year) IN ({placeholders})"
        )
        params.extend(filters.years)

    if filters.institutions:
        placeholders = ",".join(["%s"] * len(filters.institutions))
        clauses.append(f"{d}.institution IN ({placeholders})")
        params.extend(filters.institutions)

    if filters.exclude_institutions:
        placeholders = ",".join(["%s"] * len(filters.exclude_institutions))
        clauses.append(f"{d}.institution NOT IN ({placeholders})")
        params.extend(filters.exclude_institutions)

    if filters.exclude_doc_types:
        placeholders = ",".join(["%s"] * len(filters.exclude_doc_types))
        clauses.append(f"{d}.doc_type_category NOT IN ({placeholders})")
        params.extend(filters.exclude_doc_types)

    if filters.entities:
        # Una clausula `?` por cada entidad — todas deben estar presentes.
        for entity in filters.entities:
            clauses.append(f"{c}.entities ? %s")
            params.append(entity)

    if filters.tags:
        placeholders = ",".join(["%s"] * len(filters.tags))
        clauses.append(f"{c}.tags && ARRAY[{placeholders}]")
        params.extend(filters.tags)

    if filters.min_importance > 0.0 or filters.max_importance < 1.0:
        clauses.append(f"{c}.importance_score BETWEEN %s AND %s")
        params.extend([filters.min_importance, filters.max_importance])

    if filters.min_section_confidence > 0.0:
        clauses.append(f"{c}.section_confidence >= %s")
        params.append(filters.min_section_confidence)

    if filters.exclude_boilerplate:
        clauses.append(f"{c}.importance_score > %s")
        params.append(0.0)

    # Fase 2: filtro por tipo de chunk (TEXT/VISUAL/TABLE).
    if filters.kinds:
        clauses.append(f"{c}.kind = ANY(%s)")
        params.append(filters.kinds)

    where_sql = " AND ".join(clauses) if clauses else "TRUE"
    return where_sql, params
