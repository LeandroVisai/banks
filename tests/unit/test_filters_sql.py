"""Unit tests para build_filters_sql."""

from __future__ import annotations

import pytest

from banks_rag.application.retrieval import build_filters_sql
from banks_rag.domain.retrieval import SearchFilters


@pytest.mark.unit
class TestBuildFiltersSql:
    def test_empty_filters_returns_true(self) -> None:
        sql, params = build_filters_sql(
            SearchFilters(), docs_table="documents", chunks_table="chunks"
        )
        assert sql == "TRUE"
        assert params == []

    def test_doc_types_uses_any(self) -> None:
        f = SearchFilters(doc_types=["COMUNICADO", "MINUTA"])
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "documents.doc_type_category = ANY(%s)" in sql
        assert params == [["COMUNICADO", "MINUTA"]]

    def test_exact_date_takes_precedence_over_year(self) -> None:
        f = SearchFilters(exact_date="2024-03-15", year_from=2024, year_to=2024)
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        # exact_date se aplica; year_from/year_to NO se aplican
        assert "chunks.chunk_date = %s::date" in sql
        assert "EXTRACT(YEAR FROM" not in sql

    def test_year_range_uses_coalesce(self) -> None:
        f = SearchFilters(year_from=2022, year_to=2023)
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "COALESCE(EXTRACT(YEAR FROM chunks.chunk_date)::int, documents.document_year) >= %s" in sql
        assert 2022 in params and 2023 in params

    def test_variables_includes_decision_fallback(self) -> None:
        f = SearchFilters(variables=["TASA_INTERES"])
        sql, _ = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "section_type = 'DECISION'" in sql

    def test_advanced_years_list(self) -> None:
        f = SearchFilters(years=[2022, 2023, 2024])
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "IN (%s,%s,%s)" in sql
        assert params == [2022, 2023, 2024]

    def test_institutions_filter(self) -> None:
        f = SearchFilters(institutions=["banco_central_chile"])
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "documents.institution IN (%s)" in sql
        assert params == ["banco_central_chile"]

    def test_exclude_doc_types(self) -> None:
        f = SearchFilters(exclude_doc_types=["MONITOR_PM"])
        sql, _ = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "doc_type_category NOT IN" in sql

    def test_entities_one_clause_each(self) -> None:
        f = SearchFilters(entities=["BANCO_CENTRAL_CHILE", "FEDERAL_RESERVE"])
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert sql.count("chunks.entities ? %s") == 2
        assert "BANCO_CENTRAL_CHILE" in params and "FEDERAL_RESERVE" in params

    def test_tags_array_overlap(self) -> None:
        f = SearchFilters(tags=["DECISION_POLITICA"])
        sql, _ = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "tags && ARRAY[%s]" in sql

    def test_importance_range(self) -> None:
        f = SearchFilters(min_importance=0.7)
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "importance_score BETWEEN %s AND %s" in sql
        assert 0.7 in params and 1.0 in params

    def test_exclude_boilerplate(self) -> None:
        f = SearchFilters(exclude_boilerplate=True)
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "importance_score > %s" in sql
        assert 0.0 in params

    def test_combined_filters(self) -> None:
        f = SearchFilters(
            doc_types=["COMUNICADO"],
            year_from=2022,
            year_to=2023,
            tags=["DECISION_POLITICA"],
            min_importance=0.5,
            exclude_boilerplate=True,
        )
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        # Verifica que todas las clauses estén concatenadas
        assert sql.count(" AND ") >= 3

    def test_uses_provided_table_names(self) -> None:
        f = SearchFilters(year_from=2024)
        sql, _ = build_filters_sql(
            f, docs_table="qwen_documents", chunks_table="qwen_chunks"
        )
        assert "qwen_chunks.chunk_date" in sql
        assert "qwen_documents.document_year" in sql

    def test_kinds_filter_visual_only(self) -> None:
        f = SearchFilters(kinds=["VISUAL"])
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "chunks.kind = ANY(%s)" in sql
        assert params == [["VISUAL"]]

    def test_kinds_empty_no_filter(self) -> None:
        f = SearchFilters(kinds=[])
        sql, _ = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "kind" not in sql

    def test_kinds_multi(self) -> None:
        f = SearchFilters(kinds=["TEXT", "TABLE"])
        sql, params = build_filters_sql(
            f, docs_table="documents", chunks_table="chunks"
        )
        assert "chunks.kind = ANY(%s)" in sql
        assert params == [["TEXT", "TABLE"]]
