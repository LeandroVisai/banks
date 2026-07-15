"""Unit tests de la versión editable del informe (puerto de "Guardar editable")."""

from __future__ import annotations

import pytest

from banks_rag.application.reporting.editable_html import (
    make_editable_html,
    strip_editable_chrome,
)
from banks_rag.application.reporting.html_render import render_parquet_report_html
from banks_rag.application.reporting.parquet_report import DatasetSection, ParquetReport


def _report() -> ParquetReport:
    sec = DatasetSection(
        dataset_id="flujos_ffmm", name="Flujos FFMM", chart_type="grouped_bar",
        unit="US$ Mill.", segment="ffmm", last_date="2026-05-30",
        paragraph="Los flujos subieron.", status="ok",
    )
    return ParquetReport(
        title="Informe FFMM", selector_label="ffmm", selector_desc="segmento ffmm",
        generated_at="2026-07-14", windows=("7d", "30d"),
        overview_md="- Punto uno\n- Punto dos", sections=[sec], missing_ids=(),
    )


@pytest.mark.unit
class TestEditableMarkers:
    def test_report_carries_editable_markers(self) -> None:
        html = render_parquet_report_html(_report())
        assert "data-synthesis-body" in html
        assert 'data-text-slot="flujos_ffmm"' in html
        assert 'class="section-text"' in html

    def test_markers_do_not_break_existing_structure(self) -> None:
        html = render_parquet_report_html(_report())
        # se conservan los marcadores que ya usaban otros consumidores
        assert '<section class="overview">' in html
        assert '<p class="dataset-meta">' in html
        assert 'data-dataset-id="flujos_ffmm"' in html


@pytest.mark.unit
class TestMakeEditable:
    def test_injects_panel_before_body(self) -> None:
        html = render_parquet_report_html(_report())
        ed = make_editable_html(html)
        assert 'id="cb-editable-script"' in ed
        assert "cb-save-widget" in ed and "💾 Guardar" in ed and "📄 Versión final" in ed
        assert ed.index('id="cb-editable-script"') < ed.index("</body>")

    def test_idempotent(self) -> None:
        ed = make_editable_html(render_parquet_report_html(_report()))
        assert make_editable_html(ed) == ed

    def test_no_body_tag_appends(self) -> None:
        ed = make_editable_html("<div>x</div>")
        assert 'id="cb-editable-script"' in ed


@pytest.mark.unit
class TestStripEditableChrome:
    def test_removes_all_edit_chrome(self) -> None:
        html = render_parquet_report_html(_report())
        ed = make_editable_html(html)
        # simula el HTML tras editar y guardar con 💾 (runtime aplica los atributos)
        ed = ed.replace(
            '<div class="section-text" data-text-slot="flujos_ffmm">',
            '<div class="section-text" data-text-slot="flujos_ffmm" contenteditable="true" '
            'style="outline: 1px dashed rgb(203, 213, 225); outline-offset: 2px;">',
        ).replace("</body>", '<div id="cb-save-widget" data-cb-chrome=""><button>x</button></div></body>')

        clean = strip_editable_chrome(ed)
        assert "cb-editable-script" not in clean
        assert "cb-save-widget" not in clean
        assert "contenteditable" not in clean
        assert "outline" not in clean
        assert 'style=""' not in clean  # el style vacío tras quitar el contorno se elimina
        # el CONTENIDO del informe sobrevive
        assert 'data-text-slot="flujos_ffmm"' in clean
        assert "Los flujos subieron." in clean
