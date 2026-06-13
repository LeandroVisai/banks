"""Unit tests del renderer SVG (Python puro, sin dependencias ni I/O)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from banks_rag.application.reporting.parquet_facts import PlotData, PlotSeries
from banks_rag.application.reporting.svg_chart import (
    _fmt_date,
    _fmt_num,
    render_mini_table_html,
    render_plot_svg,
)


def _ts(family="line", series=None) -> PlotData:
    series = series or [PlotSeries("A", [("2026-01-01", 1.0), ("2026-02-01", 3.0), ("2026-03-01", 2.0)])]
    return PlotData("ds", family, "timeseries", "US$", series)


def _snap(family="pie") -> PlotData:
    return PlotData("ds", family, "snapshot", "%", [PlotSeries("cat", [("DAP", 60.0), ("BB", 40.0)])])


@pytest.mark.unit
class TestFormat:
    def test_es_cl_number(self):
        assert _fmt_num(1234.5) == "1.234"          # >=1000: sin decimales, miles con punto
        assert _fmt_num(12.34) == "12,3"            # decimal con coma
        assert _fmt_num(1.234) == "1,23"
        assert _fmt_num(-5.0) == "-5,00"

    def test_date_cl(self):
        assert _fmt_date("2026-04-30") == "30-04-26"
        assert _fmt_date("2026-04-30T00:00:00") == "30-04-26"


@pytest.mark.unit
class TestRenderSvg:
    def test_timeseries_is_wellformed_svg(self):
        svg = render_plot_svg(_ts())
        ET.fromstring(svg)  # no levanta → XML bien formado
        assert "<polyline" in svg
        assert "nan" not in svg.lower() and "inf" not in svg.lower()

    def test_snapshot_is_wellformed_hbars(self):
        svg = render_plot_svg(_snap())
        ET.fromstring(svg)
        assert svg.count("<rect") >= 3  # fondo + 2 barras
        assert "60,0" in svg  # share de DAP (60/100) renderizado en formato es-CL

    def test_table_family_returns_none(self):
        assert render_plot_svg(_ts(family="table")) is None

    def test_empty_plot_returns_none(self):
        assert render_plot_svg(PlotData("ds", "line", "timeseries", "", [])) is None

    def test_single_point_uses_marker(self):
        svg = render_plot_svg(_ts(series=[PlotSeries("A", [("2026-01-01", 5.0)])]))
        ET.fromstring(svg)
        assert "<circle" in svg

    def test_multi_series_uses_palette(self):
        plot = _ts(series=[
            PlotSeries("A", [("2026-01-01", 1.0), ("2026-02-01", 2.0)]),
            PlotSeries("B", [("2026-01-01", 5.0), ("2026-02-01", 6.0)]),
        ])
        svg = render_plot_svg(plot)
        assert svg.count("<polyline") == 2


@pytest.mark.unit
class TestMiniTable:
    def test_snapshot_table(self):
        html = render_mini_table_html(_snap())
        assert "DAP" in html and "BB" in html
        assert "<table" in html

    def test_timeseries_table_shows_dates(self):
        html = render_mini_table_html(_ts())
        assert "01-03-26" in html  # última fecha formateada es-CL
