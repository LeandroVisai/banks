"""Unit tests para detección de doc_type, institution, fecha y slug."""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.extractors.doc_metadata import (
    detect_date,
    detect_doc_type,
    detect_institution,
    slugify_document_id,
)


@pytest.mark.unit
class TestDetectDocType:
    @pytest.mark.parametrize(
        "rel_path, expected",
        [
            ("Comunicados/comunicado1.pdf", "COMUNICADO"),
            ("Minutas/minuta_03_2024.pdf", "MINUTA"),
            ("Fed/fed_statement_2024.pdf", "FED_STATEMENT"),
            ("Researchs/jpm/report_q1.pdf", "REPORTE_RESEARCH"),
            ("Researchs/research/report.pdf", "REPORTE_RESEARCH"),
            ("Monitor PM/textos_monitor_pm.xlsx", "MONITOR_PM"),
            ("monitor_pm/file.xlsx", "MONITOR_PM"),
            ("desconocido/archivo.pdf", "REPORTE_RESEARCH"),  # default
        ],
    )
    def test_classification(self, rel_path: str, expected: str) -> None:
        assert detect_doc_type(rel_path) == expected


@pytest.mark.unit
class TestDetectInstitution:
    @pytest.mark.parametrize(
        "rel_path, expected",
        [
            ("Comunicados/comunicado1.pdf", "banco_central_chile"),
            ("Minutas/minuta_03_2024.pdf", "banco_central_chile"),
            ("Monitor PM/textos.xlsx", "banco_central_chile"),
            ("Fed/fed_2024.pdf", "federal_reserve"),
            ("Researchs/jpm/report.pdf", "jpmorgan"),
            ("Researchs/jpmorgan/x.pdf", "jpmorgan"),
            ("Researchs/Other/x.pdf", "unknown"),
        ],
    )
    def test_classification(self, rel_path: str, expected: str) -> None:
        assert detect_institution(rel_path) == expected


@pytest.mark.unit
class TestDetectDate:
    def test_iso_format(self) -> None:
        assert detect_date("Comunicado_2022-07-13.pdf") == "2022-07-13"

    def test_dmy_dotted(self) -> None:
        assert detect_date("Reporte 13.07.2022.pdf") == "2022-07-13"

    def test_two_digit_year(self) -> None:
        assert detect_date("Comunicado_31-01-24.pdf") == "2024-01-31"

    def test_spanish_natural(self) -> None:
        # Patrón "13 de julio de 2022"
        assert detect_date("", "Reunión del 13 de julio de 2022") == "2022-07-13"

    def test_year_only_fallback(self) -> None:
        # Sin día/mes pero con año aislado por word boundary (espacio o guión).
        # Underscores no cuentan como boundary porque \w incluye _.
        assert detect_date("research 2023 summary.pdf") == "2023"
        assert detect_date("research-2023-summary.pdf") == "2023"

    def test_year_only_no_word_boundary(self) -> None:
        # Underscores rompen word boundaries — comportamiento legacy preservado.
        assert detect_date("research_2023_summary.pdf") is None

    def test_returns_none_when_no_date(self) -> None:
        assert detect_date("documento_sin_fecha.pdf") is None


@pytest.mark.unit
class TestSlugify:
    def test_lowercases_and_replaces_separators(self) -> None:
        assert slugify_document_id("Comunicados/Comunicado1.pdf") == "comunicados_comunicado1"

    def test_distinct_slugs_for_same_filename_in_different_paths(self) -> None:
        a = slugify_document_id("Comunicados/comunicado1.pdf")
        b = slugify_document_id("Minutas/comunicado1.pdf")
        assert a != b

    def test_strips_trailing_separators(self) -> None:
        assert slugify_document_id("a/b///") == "a_b"
