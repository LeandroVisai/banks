"""Unit tests de la inyección de gráficos en el HTML (proceso separado).

El catálogo y los parquets se inyectan (parquets de prueba con DuckDB en
tmp_path); no se toca el catálogo real ni hay BD/modelos.
"""

from __future__ import annotations

import duckdb
import pytest

from banks_rag.application.reporting import (
    DatasetSection,
    ParquetReport,
    inject_charts_into_html,
    render_parquet_report_html,
)
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset


def _write(path, select_sql: str) -> None:
    duckdb.sql(f"COPY ({select_sql}) TO '{path.as_posix()}' (FORMAT parquet)")


def _ds(file: str, **kw) -> ParquetDataset:
    return ParquetDataset(
        id=kw["id"], file=file, name=kw.get("name", "Dataset"), description="",
        segment="ffmm", unit=kw.get("unit", "US$"), date_range=None, columns=[],
        chart_type=kw.get("chart_type", "line"),
    )


def _report(sections: list[DatasetSection]) -> ParquetReport:
    return ParquetReport(
        title="T", selector_label="ffmm", selector_desc="seg ffmm",
        generated_at="2026-06-13 10:00", windows=("7d",), overview_md="ov", sections=sections,
    )


def _section(dataset_id: str, *, chart_type="line", status="ok") -> DatasetSection:
    return DatasetSection(
        dataset_id=dataset_id, name=dataset_id, chart_type=chart_type, unit="US$",
        segment="ffmm", last_date="2026-04-30", paragraph="p", status=status,
    )


@pytest.mark.unit
class TestInjectCharts:
    def test_injects_svg_for_ok_section(self, tmp_path):
        p = tmp_path / "s.parquet"
        _write(p, "SELECT * FROM (VALUES (DATE '2026-04-26', 5.0), (DATE '2026-04-27', 7.0)) t(Fecha, V)")
        entries = [_ds("s.parquet", id="s")]
        html = render_parquet_report_html(_report([_section("s")]))
        assert "<svg" not in html  # el informe de texto NO trae gráficos
        out, stats = inject_charts_into_html(html, entries=entries, parquet_dir=tmp_path)
        assert "<svg" in out
        assert stats.charted == 1
        assert ".report-chart" in out  # CSS inyectado

    def test_skips_non_ok_section(self, tmp_path):
        entries = [_ds("s.parquet", id="s")]
        html = render_parquet_report_html(_report([_section("s", status="no_data")]))
        out, stats = inject_charts_into_html(html, entries=entries, parquet_dir=tmp_path)
        assert stats.skipped == 1
        assert stats.charted == 0
        assert "<svg" not in out

    def test_unknown_id_counted(self, tmp_path):
        html = render_parquet_report_html(_report([_section("fantasma")]))
        out, stats = inject_charts_into_html(html, entries=[], parquet_dir=tmp_path)
        assert stats.unknown_id == 1
        assert "<svg" not in out

    def test_missing_parquet_is_no_series(self, tmp_path):
        entries = [_ds("ausente.parquet", id="s")]
        html = render_parquet_report_html(_report([_section("s")]))
        out, stats = inject_charts_into_html(html, entries=entries, parquet_dir=tmp_path)
        assert stats.no_series == 1
        assert "<svg" not in out

    def test_table_family_falls_back_to_html_table(self, tmp_path):
        p = tmp_path / "m.parquet"
        _write(p, "SELECT * FROM (VALUES (DATE '2026-04-26', 5.0), (DATE '2026-04-27', 7.0)) t(Fecha, V)")
        entries = [_ds("m.parquet", id="m", chart_type="market_monitor_table")]
        html = render_parquet_report_html(_report([_section("m", chart_type="market_monitor_table")]))
        out, stats = inject_charts_into_html(html, entries=entries, parquet_dir=tmp_path)
        assert stats.tabled == 1
        assert "chart-fallback" in out

    def test_idempotent_marker_blocks_double_css(self, tmp_path):
        p = tmp_path / "s.parquet"
        _write(p, "SELECT * FROM (VALUES (DATE '2026-04-26', 5.0), (DATE '2026-04-27', 7.0)) t(Fecha, V)")
        entries = [_ds("s.parquet", id="s")]
        html = render_parquet_report_html(_report([_section("s")]))
        out1, _ = inject_charts_into_html(html, entries=entries, parquet_dir=tmp_path)
        # re-inyectar sobre una salida ya con .report-chart no duplica el bloque CSS
        out2, _ = inject_charts_into_html(out1, entries=entries, parquet_dir=tmp_path)
        assert out2.count("@media print { .report-chart") == out1.count("@media print { .report-chart")
