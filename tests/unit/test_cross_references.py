"""Unit tests para cross-references entre Monitor PM (daily) y PDFs (period)."""

from __future__ import annotations

import pytest

from banks_rag.domain_knowledge.cross_references import (
    build_cross_references,
    period_contains_date,
)


@pytest.mark.unit
class TestPeriodContainsDate:
    def test_same_month_monthly_doc(self) -> None:
        assert period_contains_date("2024-03-15", "2024-03-31", "COMUNICADO")
        assert period_contains_date("2024-03-01", "2024-03-15", "MINUTA")

    def test_different_months_no_match(self) -> None:
        assert not period_contains_date("2024-02-28", "2024-03-15", "COMUNICADO")

    def test_quarterly_doc_same_quarter(self) -> None:
        # IPOM Q1 2024 cubre enero, febrero, marzo
        assert period_contains_date("2024-01-15", "2024-03-15", "IPOM")
        assert period_contains_date("2024-02-15", "2024-03-15", "IPOM")
        assert period_contains_date("2024-03-15", "2024-03-15", "IPOM")

    def test_quarterly_doc_different_quarter(self) -> None:
        # IPOM Q1 2024 NO cubre abril (Q2)
        assert not period_contains_date("2024-04-15", "2024-03-15", "IPOM")
        assert not period_contains_date("2023-12-15", "2024-03-15", "IPOM")

    def test_quarterly_ief(self) -> None:
        # IEF también es trimestral
        assert period_contains_date("2024-01-15", "2024-02-28", "IEF")
        assert not period_contains_date("2024-04-15", "2024-02-28", "IEF")

    def test_invalid_dates_return_false(self) -> None:
        assert not period_contains_date("", "2024-03-15", "COMUNICADO")
        assert not period_contains_date("2024-03-15", "", "COMUNICADO")
        assert not period_contains_date("invalid", "2024-03-15", "COMUNICADO")
        assert not period_contains_date("2024", "2024-03-15", "COMUNICADO")  # < 7 chars


@pytest.mark.unit
class TestBuildCrossReferences:
    def test_empty_inputs(self) -> None:
        result = build_cross_references([], [])
        assert result == {"excel_to_period": {}, "period_to_daily": {}}

    def test_no_overlap_no_links(self) -> None:
        excel = [{
            "chunk_id": "ex1",
            "chunk_date": "2024-03-15",
            "economic_variables": {"TASA_INTERES": {}},
        }]
        pdf = [{
            "chunk_id": "pdf1",
            "document_date": "2024-05-15",  # mes distinto
            "doc_type_category": "COMUNICADO",
            "economic_variables": {"TASA_INTERES": {}},
        }]
        result = build_cross_references(excel, pdf)
        assert result["excel_to_period"] == {}
        assert result["period_to_daily"] == {}

    def test_match_same_month_shared_var(self) -> None:
        excel = [{
            "chunk_id": "ex1",
            "chunk_date": "2024-03-15",
            "economic_variables": {"TASA_INTERES": {}, "INFLACION": {}},
        }]
        pdf = [{
            "chunk_id": "pdf1",
            "document_date": "2024-03-31",
            "doc_type_category": "COMUNICADO",
            "economic_variables": {"TASA_INTERES": {}, "TIPO_CAMBIO": {}},
        }]
        result = build_cross_references(excel, pdf)
        assert "ex1" in result["excel_to_period"]
        assert result["excel_to_period"]["ex1"][0]["record_id"] == "pdf1"
        assert result["excel_to_period"]["ex1"][0]["pointer_type"] == "period_context"
        assert result["excel_to_period"]["ex1"][0]["shared_vars"] == ["TASA_INTERES"]
        # Bidireccional
        assert "pdf1" in result["period_to_daily"]
        assert result["period_to_daily"]["pdf1"][0]["record_id"] == "ex1"
        assert result["period_to_daily"]["pdf1"][0]["pointer_type"] == "daily_example"

    def test_no_shared_vars_no_link(self) -> None:
        excel = [{
            "chunk_id": "ex1",
            "chunk_date": "2024-03-15",
            "economic_variables": {"TASA_INTERES": {}},
        }]
        pdf = [{
            "chunk_id": "pdf1",
            "document_date": "2024-03-31",
            "doc_type_category": "COMUNICADO",
            "economic_variables": {"INFLACION": {}},  # variable distinta
        }]
        result = build_cross_references(excel, pdf)
        assert result["excel_to_period"] == {}
