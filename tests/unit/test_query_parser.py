"""Unit tests para query_parser."""

from __future__ import annotations

import pytest

from banks_rag.application.retrieval import (
    extract_month_from_row,
    parse_query,
    row_matches_month,
)


@pytest.mark.unit
class TestParseQuery:
    def test_returns_parsed_query_with_clean_text(self) -> None:
        pq = parse_query("decisión política monetaria 2024")
        assert pq.raw_query == "decisión política monetaria 2024"
        assert "2024" not in pq.clean_query  # se consume como filtro
        assert pq.filters.year_from == 2024
        assert pq.filters.year_to == 2024

    def test_year_range(self) -> None:
        pq = parse_query("inflación 2022-2023")
        assert pq.filters.year_from == 2022
        assert pq.filters.year_to == 2023

    def test_year_range_inverted_swapped(self) -> None:
        pq = parse_query("2023-2022 inflación")
        assert pq.filters.year_from == 2022
        assert pq.filters.year_to == 2023

    def test_month_only(self) -> None:
        pq = parse_query("comunicado de enero")
        assert pq.filters.month == 1

    def test_full_date_iso(self) -> None:
        pq = parse_query("decisión 2024-03-15")
        assert pq.filters.exact_date == "2024-03-15"
        assert pq.filters.day == 15
        assert pq.filters.month == 3
        assert pq.filters.year_from == 2024

    def test_full_date_dmy(self) -> None:
        pq = parse_query("reunión 15/03/2024")
        assert pq.filters.exact_date == "2024-03-15"

    def test_full_date_named(self) -> None:
        pq = parse_query("acta del 15 de marzo de 2024")
        assert pq.filters.exact_date == "2024-03-15"

    def test_doc_type_detected(self) -> None:
        pq = parse_query("comunicado tasa interés")
        assert "COMUNICADO" in pq.filters.doc_types

        pq = parse_query("minuta del consejo")
        assert "MINUTA" in pq.filters.doc_types

        pq = parse_query("declaración de la fed")
        assert "FED_STATEMENT" in pq.filters.doc_types

    def test_variables_detected(self) -> None:
        pq = parse_query("evolución de la TPM")
        assert any("TASA" in v for v in pq.filters.variables)

    def test_no_match_query(self) -> None:
        pq = parse_query("hola mundo")
        assert pq.filters.year_from is None
        assert pq.filters.month is None
        assert pq.filters.exact_date is None


@pytest.mark.unit
class TestExtractMonthFromRow:
    def test_chunk_date_object(self) -> None:
        import datetime
        row = {"chunk_date": datetime.date(2024, 5, 15)}
        assert extract_month_from_row(row) == 5

    def test_chunk_date_iso_string(self) -> None:
        row = {"chunk_date": "2024-05-15"}
        assert extract_month_from_row(row) == 5

    def test_document_date_iso(self) -> None:
        row = {"document_date": "2024-03-31"}
        assert extract_month_from_row(row) == 3

    def test_document_date_dmy(self) -> None:
        row = {"document_date": "15-03-2024"}
        assert extract_month_from_row(row) == 3

    def test_filename_fallback(self) -> None:
        row = {"filename": "comunicado_2024-03-15.pdf"}
        assert extract_month_from_row(row) == 3

    def test_no_date(self) -> None:
        row = {"filename": "no_date_here.pdf"}
        assert extract_month_from_row(row) is None


@pytest.mark.unit
class TestRowMatchesMonth:
    def test_no_filter_passes_all(self) -> None:
        assert row_matches_month({"chunk_date": "2024-05-15"}, None) is True

    def test_match(self) -> None:
        assert row_matches_month({"chunk_date": "2024-05-15"}, 5) is True

    def test_no_match(self) -> None:
        assert row_matches_month({"chunk_date": "2024-05-15"}, 7) is False

    def test_no_date_no_match(self) -> None:
        assert row_matches_month({"filename": "x.pdf"}, 5) is False
