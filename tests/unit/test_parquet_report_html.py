"""Unit tests del renderer HTML del informe de parquets (determinista, sin LLM)."""

from __future__ import annotations

import pytest

from banks_rag.application.reporting.html_render import (
    DISCLAIMER,
    render_parquet_report_html,
)
from banks_rag.application.reporting.parquet_report import DatasetSection, ParquetReport


def _section(
    dataset_id: str,
    *,
    chart_type: str = "line",
    status: str = "ok",
    paragraph: str = "Texto del párrafo.",
    name: str | None = None,
) -> DatasetSection:
    return DatasetSection(
        dataset_id=dataset_id, name=name or dataset_id, chart_type=chart_type,
        unit="US$ Mill.", segment="ffmm", last_date="2026-04-30",
        paragraph=paragraph, status=status,
    )


def _report(sections: list[DatasetSection], **overrides) -> ParquetReport:
    kwargs = dict(
        title="Informe descriptivo — segmento ffmm",
        selector_label="ffmm",
        selector_desc="segmento ffmm",
        generated_at="2026-06-11 10:00",
        windows=("7d", "30d"),
        overview_md="Resumen del segmento.\n\n- punto destacado",
        sections=sections,
        missing_ids=(),
    )
    kwargs.update(overrides)
    return ParquetReport(**kwargs)


@pytest.mark.unit
class TestRenderParquetReportHtml:
    def test_sections_carry_dataset_metadata_for_phase2_charts(self) -> None:
        html_doc = render_parquet_report_html(_report([
            _section("flujos_ffmm", chart_type="grouped_bar"),
            _section("allocation", chart_type="stacked_area", status="no_data"),
        ]))
        assert 'data-dataset-id="flujos_ffmm"' in html_doc
        assert 'data-chart-type="grouped_bar"' in html_doc
        assert 'data-status="ok"' in html_doc
        assert 'data-dataset-id="allocation"' in html_doc
        assert 'data-chart-type="stacked_area"' in html_doc
        assert 'data-status="no_data"' in html_doc
        # Un placeholder de gráfico por dataset, listo para fase 2.
        assert html_doc.count('class="chart-placeholder"') == 2

    def test_sections_render_in_report_order_and_numbered(self) -> None:
        html_doc = render_parquet_report_html(_report([
            _section("flujos_ffmm", name="Flujos semanales"),
            _section("duracion_ffmm", name="Duración"),
        ]))
        first = html_doc.index("1. Flujos semanales")
        second = html_doc.index("2. Duración")
        assert first < second

    def test_paragraph_and_attributes_are_escaped(self) -> None:
        html_doc = render_parquet_report_html(_report([
            _section("flujos_ffmm", paragraph='Valor < 5 & <script>alert("x")</script>'),
        ]))
        assert "<script>" not in html_doc
        assert "&lt;script&gt;" in html_doc
        assert "Valor &lt; 5 &amp;" in html_doc

    def test_overview_and_references_sections(self) -> None:
        html_doc = render_parquet_report_html(_report(
            [_section("a"), _section("b", status="ungrounded")],
            missing_ids=("no_existe",),
        ))
        assert '<section class="overview">' in html_doc
        assert "Resumen del segmento." in html_doc
        assert '<ul class="bullet-list">' in html_doc  # viñeta de la síntesis
        assert '<section class="references">' in html_doc
        assert "ok: 1" in html_doc and "sin evidencia: 1" in html_doc
        assert "no_existe" in html_doc
        assert "7d, 30d" in html_doc

    def test_title_and_subtitle(self) -> None:
        report = _report([_section("a")], title="Informe <X> & Co")
        html_doc = render_parquet_report_html(report)
        assert "<h1>Informe &lt;X&gt; &amp; Co</h1>" in html_doc
        assert DISCLAIMER in html_doc
        custom = render_parquet_report_html(report, subtitle="Solo uso interno")
        assert "Solo uso interno" in custom

    def test_dataset_meta_line(self) -> None:
        html_doc = render_parquet_report_html(_report([_section("flujos_ffmm")]))
        assert '<p class="dataset-meta">' in html_doc
        assert "datos hasta 2026-04-30" in html_doc
