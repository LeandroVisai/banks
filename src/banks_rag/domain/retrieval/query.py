"""Value objects de retrieval: ParsedQuery + SearchFilters.

``SearchFilters`` unifica filtros auto-detectados desde NL (años, fechas,
variables, secciones — los que producía ``04_search.parse_query``) con
filtros granulares explícitos (institutions, entities, tags, importance
ranges — los que producía ``04_advanced_search.AdvancedFilters``).

Si un campo está en su default (None / lista vacía / 0.0 / 1.0), el filtro
no se aplica. El orquestador combina ambos sets en un único WHERE SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchFilters:
    """Filtros declarativos para hybrid search.

    Campos auto-detectables desde lenguaje natural (lo extrae ``query_parser``):
        exact_date, day, year_from, year_to, month, doc_types, variables, sections

    Campos avanzados (los pasa el caller explícitamente):
        years, months (listas en lugar de scalar), institutions, entities,
        tags, min_importance, max_importance, min_section_confidence,
        exclude_boilerplate, exclude_doc_types, exclude_institutions
    """

    # Auto-detectables desde NL
    exact_date: str | None = None        # ISO YYYY-MM-DD
    day: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    month: int | None = None             # 1-12
    doc_types: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)

    # Filtros avanzados (explícitos)
    years: list[int] = field(default_factory=list)
    months: list[int] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    min_importance: float = 0.0
    max_importance: float = 1.0
    min_section_confidence: float = 0.0
    exclude_boilerplate: bool = False
    exclude_doc_types: list[str] = field(default_factory=list)
    exclude_institutions: list[str] = field(default_factory=list)

    def has_strict_filters(self) -> bool:
        """``True`` si hay filtros que pueden hacer que el retrieval no devuelva nada.

        Se usa para decidir si vale la pena reintentar relajando filtros cuando
        el primer recall viene vacío.
        """
        return bool(self.variables or self.sections)

    def active_filter_count(self) -> int:
        """Cuenta cuántos filtros están activos (los que difieren del default)."""
        count = 0
        if self.exact_date is not None:
            count += 1
        if self.year_from is not None or self.year_to is not None:
            count += 1
        if self.month is not None:
            count += 1
        for fld in (
            self.doc_types, self.variables, self.sections, self.years, self.months,
            self.institutions, self.entities, self.tags,
            self.exclude_doc_types, self.exclude_institutions,
        ):
            if fld:
                count += 1
        if self.min_importance > 0.0:
            count += 1
        if self.max_importance < 1.0:
            count += 1
        if self.min_section_confidence > 0.0:
            count += 1
        if self.exclude_boilerplate:
            count += 1
        return count


@dataclass
class ParsedQuery:
    """Resultado de parsear una query de lenguaje natural."""

    raw_query: str        # texto original del usuario
    clean_query: str      # texto sin tokens de filtro (años, fechas, etc.)
    filters: SearchFilters  # filtros auto-detectados
