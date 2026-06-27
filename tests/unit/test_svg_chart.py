"""Unit tests del renderer SVG (Python puro, sin dependencias ni I/O)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from banks_rag.application.reporting.parquet_facts import PlotData, PlotSeries
from banks_rag.application.reporting.svg_chart import (
    _fmt_date,
    _fmt_num,
    render_mini_table_html,
    render_plot_svg,
    renders_natively,
)


def _grouped(family="grouped_bar") -> PlotData:
    return PlotData("g", family, "grouped", "%", [
        PlotSeries("Δ7d", [("Tipo 1", 0.1), ("Tipo 2", -0.05)]),
        PlotSeries("Δ30d", [("Tipo 1", 0.33), ("Tipo 2", 0.24)]),
    ])


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
class TestNewChartTypes:
    def test_grouped_bar_wellformed(self):
        svg = render_plot_svg(_grouped(), chart="grouped_bar")
        ET.fromstring(svg)
        assert svg.count("<rect") >= 4  # fondo + 2 cat x 2 series (con valores no nulos)
        assert "nan" not in svg.lower()

    def test_stacked_bar_wellformed(self):
        svg = render_plot_svg(_grouped(), chart="stacked_bar")
        ET.fromstring(svg)
        assert "<rect" in svg

    def test_stacked_area_wellformed(self):
        plot = _ts(series=[
            PlotSeries("BB", [("2026-01-31", 100.0), ("2026-02-28", 120.0)]),
            PlotSeries("PDBC", [("2026-01-31", 50.0), ("2026-02-28", 40.0)]),
        ])
        svg = render_plot_svg(plot, chart="stacked_area")
        ET.fromstring(svg)
        assert "<polygon" in svg

    def test_pie_wellformed(self):
        svg = render_plot_svg(_snap(), chart="pie")
        ET.fromstring(svg)
        assert "<path" in svg or "<circle" in svg

    def test_renders_natively_mapping(self):
        assert renders_natively("grouped", "grouped_bar")
        assert renders_natively("grouped", "stacked_bar")
        assert renders_natively("timeseries", "stacked_area")
        assert renders_natively("snapshot", "pie")
        # tipo objetivo que NO matchea el kind → no nativo (vista preliminar)
        assert not renders_natively("timeseries", "grouped_bar")
        assert not renders_natively("snapshot", "line")


@pytest.mark.unit
class TestOverlayAndDiverging:
    """Apilados DIVERGENTES (pos arriba / neg abajo) + serie "Neto" superpuesta."""

    def test_diverging_stacked_area_has_negative_band(self):
        # Una serie positiva y otra negativa → el dominio del eje debe cruzar el 0.
        plot = PlotData("spc", "stacked_area", "timeseries", "US$ Mill.", [
            PlotSeries("Pagan", [("2026-01-01", 5.0), ("2026-02-01", 6.0)]),
            PlotSeries("Reciben", [("2026-01-01", -8.0), ("2026-02-01", -4.0)]),
            PlotSeries("Neto", [("2026-01-01", -3.0), ("2026-02-01", 2.0)]),
        ], overlay=("Neto",))
        svg = render_plot_svg(plot, chart="stacked_area")
        ET.fromstring(svg)
        # Neto va como línea (polyline), no como banda apilada; en color de overlay.
        assert "<polyline" in svg and "#c8102e" in svg
        # Un eje con tick negativo (la banda "Reciben" baja del 0).
        assert "-" in svg

    def test_overlay_excluded_from_stack_palette(self):
        # Con overlay, las bandas usan la paleta sin rojo (reservado al Neto).
        plot = PlotData("spc", "stacked_area", "timeseries", "x", [
            PlotSeries("A", [("2026-01-01", 5.0), ("2026-02-01", 6.0)]),
            PlotSeries("Neto", [("2026-01-01", 5.0), ("2026-02-01", 6.0)]),
        ], overlay=("Neto",))
        svg = render_plot_svg(plot, chart="stacked_area")
        # el rojo de overlay aparece como línea (stroke) pero NO como banda opaca
        assert 'fill="#c8102e" fill-opacity="0.85"' not in svg  # ninguna banda en rojo
        assert 'stroke="#c8102e"' in svg                        # la línea Neto sí

    def test_stacked_bar_neto_drawn_as_points(self):
        plot = PlotData("deriv", "stacked_bar", "grouped", "US$ Mill.", [
            PlotSeries("Suscripcion", [("1 a 90", 450.0), ("91 a 360", 2050.0)]),
            PlotSeries("Vencimiento", [("1 a 90", -1083.0), ("91 a 360", -1172.0)]),
            PlotSeries("Neto", [("1 a 90", -633.0), ("91 a 360", 878.0)]),
        ], overlay=("Neto",))
        svg = render_plot_svg(plot, chart="stacked_bar")
        ET.fromstring(svg)
        # Neto = un punto (circle) por categoría, en rojo; no una barra.
        assert svg.count("<circle") >= 2
        assert "#c8102e" in svg

    def test_plain_stacked_area_still_positive(self):
        # Sin overlay ni negativos: comportamiento clásico (todo apilado sobre 0).
        plot = _ts(series=[
            PlotSeries("A", [("2026-01-01", 1.0), ("2026-02-01", 2.0)]),
            PlotSeries("B", [("2026-01-01", 3.0), ("2026-02-01", 1.0)]),
        ])
        svg = render_plot_svg(plot, chart="stacked_area")
        ET.fromstring(svg)
        assert "<polygon" in svg and "<polyline" not in svg  # sin línea Neto


@pytest.mark.unit
class TestRestackData:
    """``data-stack``: datos crudos + geometría que el JS usa para re-apilar las
    series VISIBLES al ocultar una en la leyenda (área/barra apilada)."""

    @staticmethod
    def _stack_of(svg: str) -> dict:
        root = ET.fromstring(svg)
        raw = root.get("data-stack")
        assert raw, "el SVG apilado debe exponer data-stack"
        return json.loads(raw)

    def test_stacked_area_embeds_raw_series(self):
        plot = _ts(series=[
            PlotSeries("BB", [("2026-01-31", 100.0), ("2026-02-28", 120.0)]),
            PlotSeries("PDBC", [("2026-01-31", 50.0), ("2026-02-28", 40.0)]),
        ])
        st = self._stack_of(render_plot_svg(plot, chart="stacked_area"))
        assert st["k"] == "area"
        assert len(st["x"]) == 2                       # dos fechas
        assert [s["i"] for s in st["s"]] == [0, 1]     # índices = data-si de los polígonos
        assert st["s"][0]["v"] == [100.0, 120.0]       # valores crudos SIN acumular
        assert st["s"][1]["v"] == [50.0, 40.0]
        assert st["ymax"] > st["ymin"]                 # geometría del eje fija presente

    def test_stacked_area_excludes_overlay_from_stack(self):
        # El "Neto" es overlay (línea), NO se re-apila → no va en data-stack.
        plot = PlotData("spc", "stacked_area", "timeseries", "x", [
            PlotSeries("A", [("2026-01-01", 5.0), ("2026-02-01", 6.0)]),
            PlotSeries("Neto", [("2026-01-01", 5.0), ("2026-02-01", 6.0)]),
        ], overlay=("Neto",))
        st = self._stack_of(render_plot_svg(plot, chart="stacked_area"))
        assert len(st["s"]) == 1 and st["s"][0]["i"] == 0  # solo la banda apilable

    def test_stacked_bar_embeds_data_ci(self):
        st = self._stack_of(render_plot_svg(_grouped(), chart="stacked_bar"))
        assert st["k"] == "bar"
        # cada serie trae un valor por categoría (mismo largo para todas)
        ncat = len(st["s"][0]["v"])
        assert all(len(s["v"]) == ncat for s in st["s"])
        # las rects apiladas se mapean por data-ci (robusto a barras ~0 no dibujadas)
        svg = render_plot_svg(_grouped(), chart="stacked_bar")
        assert 'data-ci="0"' in svg

    def test_line_chart_has_no_stack_data(self):
        # Series temporal (línea): ocultar es el comportamiento estándar, sin re-apilar.
        root = ET.fromstring(render_plot_svg(_ts()))
        assert root.get("data-stack") is None

    def test_grouped_bar_has_no_stack_data(self):
        # Barras AGRUPADAS (no apiladas): tampoco se re-apilan.
        root = ET.fromstring(render_plot_svg(_grouped(), chart="grouped_bar"))
        assert root.get("data-stack") is None


@pytest.mark.unit
class TestMiniTable:
    def test_snapshot_table(self):
        html = render_mini_table_html(_snap())
        assert "DAP" in html and "BB" in html
        assert "<table" in html

    def test_timeseries_table_shows_dates(self):
        html = render_mini_table_html(_ts())
        assert "01-03-26" in html  # última fecha formateada es-CL
