"""Unit tests del informe curado por familia (spec + transforms + builder + HTML).

Parquets de prueba con DuckDB en ``tmp_path`` (sin BD ni modelos ni internet),
mismo patrón que test_compute_series.
"""

from __future__ import annotations

import duckdb
import pytest

from banks_rag.application.reporting import (
    build_curated_report,
    render_curated_html,
)
from banks_rag.application.reporting.curated_report import (
    _scale_plot,
    fill_synthesis_slot,
    fill_text_slots,
)
from banks_rag.application.reporting.parquet_facts import HtmlTable, PlotData, PlotSeries
from banks_rag.application.reporting.report_spec import (
    STATUS_EXP,
    STATUS_MVP,
    STATUS_SKIP,
    FamilyReportSpec,
    ReportBlock,
)
from banks_rag.application.reporting.series_transforms import (
    accumulated_series,
    allocation_by_fund,
    composition_by_bucket,
    dcv_cut_dates,
    dcv_heatmap,
    filter_fund,
    get_transform,
    is_known,
    monthly_var_alloc,
    window_returns,
)
from banks_rag.application.reporting.specs import available_families, get_spec
from banks_rag.application.reporting.specs.ffmm_spec import FFMM_SPEC
from banks_rag.application.reporting.svg_chart import _nice_ticks, render_plot_svg
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset


def _write(path, select_sql: str) -> None:
    duckdb.sql(f"COPY ({select_sql}) TO '{path.as_posix()}' (FORMAT parquet)")


def _ds(file: str, **kw) -> ParquetDataset:
    return ParquetDataset(
        id=kw["id"], file=file, name=kw.get("name", "DS"), description="",
        segment="ffmm", unit=kw.get("unit", "US$"), date_range=None, columns=[],
        chart_type=kw.get("chart_type", "line"), value_scale=kw.get("value_scale", 1.0),
    )


def _categorical_parquet(path) -> None:
    rows = []
    for d in ["2026-04-26", "2026-04-27", "2026-04-28"]:
        for tipo, base in [("Tipo 1", 1.0), ("Tipo 2", 2.0), ("Tipo 3", 3.0), ("Tipo 6", 6.0)]:
            rows.append(f"(DATE '{d}', '{tipo}', {base})")
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, tipo_fondo, Duracion)")


# ── Spec real de ffmm ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFfmmSpec:
    def test_registered_and_listed(self):
        assert get_spec("ffmm") is FFMM_SPEC
        assert "ffmm" in available_families()
        assert get_spec("inexistente") is None

    def test_has_all_sections_in_order(self):
        secs = FFMM_SPEC.sections()
        assert secs[0] == "Flujos y Retornos"
        assert "Portafolio DCV" in secs
        assert secs[-1] == "Mercado cambiario"
        assert "Carteras DCV" not in secs  # sección eliminada (su único gráfico era duplicado)
        assert len(FFMM_SPEC.blocks) == 27

    def test_has_mvp_and_exp_blocks(self):
        counts = FFMM_SPEC.status_counts()
        assert counts.get(STATUS_MVP, 0) >= 5
        assert counts.get(STATUS_EXP, 0) >= 1

    def test_no_duplicate_charts(self):
        import json
        sigs = [
            (b.source_id, b.transform, json.dumps(b.params or {}, sort_keys=True))
            for b in FFMM_SPEC.blocks
        ]
        assert len(sigs) == len(set(sigs)), "gráfico repetido (mismo source+transform+params)"
        titles = [b.title for b in FFMM_SPEC.blocks]
        assert len(titles) == len(set(titles)), "título de bloque repetido"

    def test_mvp_blocks_point_to_implemented_transforms(self):
        for b in FFMM_SPEC.blocks:
            if b.status == STATUS_MVP:
                assert get_transform(b.transform) is not None, b.title


# ── Transforms ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTransforms:
    def test_filter_fund_restricts_and_orders(self, tmp_path):
        p = tmp_path / "dur.parquet"
        _categorical_parquet(p)
        ds = _ds("dur.parquet", id="duracion_ffmm")
        plot = filter_fund(ds, tmp_path, {"funds": ["Tipo 1", "Tipo 2"]})
        labels = [s.label for s in plot.series]
        assert set(labels) == {"Tipo 1", "Tipo 2"}  # solo los pedidos

    def test_dcv_transforms_are_implemented(self):
        assert get_transform("dcv_cut_dates") is not None
        assert get_transform("dcv_heatmap") is not None
        assert is_known("dcv_cut_dates") and is_known("dcv_heatmap")

    def test_unknown_transform(self):
        assert not is_known("no_existe")
        assert get_transform("no_existe") is None

    def test_allocation_by_fund_filters_fund_series_by_instrument(self, tmp_path):
        # parquet con DOS categoricas: instrumento y tipo de fondo.
        p = tmp_path / "alloc.parquet"
        rows = []
        for d in ["2026-01-31", "2026-02-28", "2026-03-31"]:
            for inst, base in [("BB", 100.0), ("PDBC", 50.0)]:
                for fund, mult in [("Tipo 1", 1.0), ("Tipo 2", 2.0)]:
                    rows.append(f"(DATE '{d}', '{inst}', '{fund}', {base * mult})")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows)
               + ") t(Fecha, Tipo_instrumento, Tipo_fondo, Monto_USD)")
        ds = _ds("alloc.parquet", id="var_cartera_mensual_ffmm", chart_type="stacked_area")
        plot = allocation_by_fund(ds, tmp_path, {"fund": "Tipo 1"})
        # series por instrumento (no por fondo), solo del fondo pedido
        assert {s.label for s in plot.series} == {"BB", "PDBC"}
        # Tipo 1 → BB=100 (no 100+200)
        bb = next(s for s in plot.series if s.label == "BB")
        assert bb.points[-1][1] == 100.0

    def test_monthly_var_alloc_grouped_mes_ytd(self, tmp_path):
        p = tmp_path / "alloc.parquet"
        vals = [("2026-01-31", 100.0), ("2026-02-28", 130.0), ("2026-03-31", 120.0)]
        rows = [f"(DATE '{d}', 'BB', 'Tipo 1', {v})" for d, v in vals]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows)
               + ") t(Fecha, Tipo_instrumento, Tipo_fondo, Monto_USD)")
        ds = _ds("alloc.parquet", id="var_cartera_mensual_ffmm")
        plot = monthly_var_alloc(ds, tmp_path, {"fund": "Tipo 1"})
        assert plot.kind == "grouped"
        assert {s.label for s in plot.series} == {"Mes", "YtD"}
        mes = dict(next(s for s in plot.series if s.label == "Mes").points)
        ytd = dict(next(s for s in plot.series if s.label == "YtD").points)
        assert round(mes["BB"], 1) == -10.0   # 120 - 130
        assert round(ytd["BB"], 1) == 20.0     # 120 - 100 (inicio del año)


    def test_dcv_cut_dates_returns_html_table(self, tmp_path):
        p = tmp_path / "stock.parquet"
        rows = []
        for d in ["2026-01-05", "2026-05-22", "2026-05-29"]:
            for tipo, base in [("BB", 100.0), ("BTP", 50.0)]:
                rows.append(f"(DATE '{d}', '{tipo}', {base})")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Tipo, Stock_USD)")
        ds = _ds("stock.parquet", id="stock_nivel_ffmm", unit="US$ Mill.")
        result = dcv_cut_dates(ds, tmp_path, {})
        assert isinstance(result, HtmlTable)
        assert "BB" in result.html and "BTP" in result.html
        assert "T-5" in result.html and "T-20" in result.html

    def test_dcv_heatmap_returns_html_table(self, tmp_path):
        p = tmp_path / "var.parquet"
        rows = []
        for i, d in enumerate(["2026-01-05", "2026-05-22", "2026-05-29"]):
            for bucket in ["Menor a 1Y", "Entre 1 y 2Y"]:
                for tipo, v in [("BB", 100.0 + i * 10), ("DAP", 50.0 - i * 3)]:
                    rows.append(f"(DATE '{d}', '{bucket}', '{tipo}', 'CLP', {v})")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Bucket, Tipo, Moneda, Stock_USD)")
        ds = _ds("var.parquet", id="variacion_stock_ffmm", unit="US$ Mill.")
        result = dcv_heatmap(ds, tmp_path, {})
        assert isinstance(result, HtmlTable)
        assert "BB" in result.html and "DAP" in result.html
        assert "Menor a 1Y" in result.html
        assert "T-5" in result.html

    def test_dcv_transforms_return_none_for_wrong_structure(self, tmp_path):
        # parquet de una sola categórica (duracion) no tiene la estructura DCV
        p = tmp_path / "dur.parquet"
        _categorical_parquet(p)
        ds = _ds("dur.parquet", id="duracion_ffmm")
        assert dcv_heatmap(ds, tmp_path, {}) is None  # <2 cat cols → None


@pytest.mark.unit
class TestGroupedTransforms:
    def test_window_returns_is_index_change(self, tmp_path):
        # retornos = ÍNDICE de retorno acumulado (no diario): Δ = final - inicio.
        p = tmp_path / "ret.parquet"
        vals = {"Tipo 1": [50.0, 52.0], "Tipo 2": [30.0, 29.5]}
        rows = []
        for i, d in enumerate(["2026-06-08", "2026-06-10"]):
            for t, vv in vals.items():
                rows.append(f"(DATE '{d}', '{t}', {vv[i]})")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Tipo, Retorno)")
        ds = _ds("ret.parquet", id="retornos_fondo_ffmm", chart_type="grouped_bar")
        plot = window_returns(ds, tmp_path, {"windows": ["7d"], "funds": ["Tipo 1", "Tipo 2"]})
        assert plot.kind == "grouped"
        d7 = dict(plot.series[0].points)
        assert round(d7["Tipo 1"], 2) == 2.0    # 52 - 50
        assert round(d7["Tipo 2"], 2) == -0.5   # 29.5 - 30

    def test_composition_by_bucket_stacked(self, tmp_path):
        # plazo (Bucket) x instrumento (Tipo): composición al corte
        p = tmp_path / "vs.parquet"
        rows = []
        for b in ["Menor a 1Y", "Entre 2 y 5Y"]:
            for t, v in [("DAP", 100.0), ("BB", 50.0)]:
                rows.append(f"(DATE '2026-06-09', '{b}', '{t}', 'CLP', {v})")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Bucket, Tipo, Moneda, Stock_USD)")
        ds = _ds("vs.parquet", id="variacion_stock_ffmm", chart_type="stacked_bar")
        plot = composition_by_bucket(ds, tmp_path, {})
        assert plot.kind == "grouped"
        # categorías = plazos ordenados (Menor a 1Y antes que Entre 2 y 5Y)
        cats = [c for c, _ in plot.series[0].points]
        assert cats == ["Menor a 1Y", "Entre 2 y 5Y"]
        assert {s.label for s in plot.series} == {"DAP", "BB"}


@pytest.mark.unit
class TestAccumulated:
    def test_flow_series_gets_cumsum(self, tmp_path):
        # serie de FLUJO (oscila, con negativos) → suma acumulada
        p = tmp_path / "flow.parquet"
        vals = [("2026-01-02", 10.0), ("2026-01-03", -4.0), ("2026-01-06", 6.0)]
        rows = [f"(DATE '{d}', {v})" for d, v in vals]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Spot)")
        ds = _ds("flow.parquet", id="flujos_spot_ffmm", unit="US$ Mill.")
        plot = accumulated_series(ds, tmp_path, {"window": "ytd"})
        pts = plot.series[0].points
        # cumsum: 10, 6, 12
        assert [round(v, 1) for _, v in pts] == [10.0, 6.0, 12.0]

    def test_level_series_gets_rebased(self, tmp_path):
        # serie de NIVEL (positiva, sin negativos) → valor - base (var. acumulada)
        p = tmp_path / "level.parquet"
        vals = [("2026-01-02", 100.0), ("2026-01-03", 130.0), ("2026-01-06", 90.0)]
        rows = [f"(DATE '{d}', 'DAP', {v})" for d, v in vals]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, instr, mmus)")
        ds = _ds("level.parquet", id="dap_pdbc_ffmm")
        plot = accumulated_series(ds, tmp_path, {"window": "ytd"})
        dap = next(s for s in plot.series if s.label == "DAP")
        # rebase: 0, +30, -10
        assert [round(v, 1) for _, v in dap.points] == [0.0, 30.0, -10.0]

    def test_ytd_window_drops_prior_year(self, tmp_path):
        p = tmp_path / "flow.parquet"
        rows = ["(DATE '2025-12-30', 100.0)", "(DATE '2026-01-02', 5.0)", "(DATE '2026-01-03', 5.0)"]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Spot)")
        ds = _ds("flow.parquet", id="flujos_spot_ffmm")
        plot = accumulated_series(ds, tmp_path, {"window": "ytd", "accumulate": "cumsum"})
        pts = plot.series[0].points
        # 2025 excluido; cumsum 2026: 5, 10 (no arrastra el 100 de dic-2025)
        assert [round(v, 1) for _, v in pts] == [5.0, 10.0]


# ── Builder + HTML ───────────────────────────────────────────────────────────

def _spec_for_build() -> FamilyReportSpec:
    return FamilyReportSpec(
        family="t", title="Informe T",
        blocks=(
            # nativo (línea) → gráfico no-preliminar
            ReportBlock(section="S1", title="Línea", chart="line", status=STATUS_MVP,
                        source_id="dur", transform="filter_fund",
                        params={"funds": ["Tipo 1", "Tipo 2"]}, text_slot="t:s1"),
            # objetivo barras SIN transform propia → se dibuja como VISTA PRELIMINAR
            # (serie natural → línea, porque grouped_bar no es nativo de timeseries)
            ReportBlock(section="S1", title="Barras (prelim)", chart="grouped_bar", status=STATUS_EXP,
                        source_id="dur", transform=None),
            # objetivo tabla → placeholder (no se aproxima como serie)
            ReportBlock(section="S2", title="Heatmap", chart="heatmap_table", status=STATUS_EXP,
                        source_id="dur", transform="dcv_heatmap"),
            # SKIP → tarjeta "sin datos"
            ReportBlock(section="S2", title="Sin parquet", chart="line", status=STATUS_SKIP),
        ),
    )


@pytest.mark.unit
class TestBuildCuratedReport:
    def test_build_renders_everything_graphable(self, tmp_path):
        p = tmp_path / "dur.parquet"
        _categorical_parquet(p)
        entries = [_ds("dur.parquet", id="dur")]
        report = build_curated_report(_spec_for_build(), entries=entries, parquet_dir=tmp_path)
        kinds = [b.render_kind for b in report.blocks]
        # línea + barras(preliminar) = chart; heatmap = placeholder; SKIP = skip
        assert kinds == ["chart", "chart", "placeholder", "skip"]
        assert report.blocks[0].preliminary is False   # objetivo nativo
        assert report.blocks[1].preliminary is True     # objetivo barras → preliminar

    def test_missing_dataset_becomes_placeholder(self, tmp_path):
        report = build_curated_report(_spec_for_build(), entries=[], parquet_dir=tmp_path)
        # sin dataset 'dur' en el catálogo → placeholder, no revienta
        assert report.blocks[0].render_kind == "placeholder"

    def test_render_html_structure(self, tmp_path):
        p = tmp_path / "dur.parquet"
        _categorical_parquet(p)
        entries = [_ds("dur.parquet", id="dur")]
        report = build_curated_report(_spec_for_build(), entries=entries, parquet_dir=tmp_path)
        html = render_curated_html(report)
        assert "Informe T" in html
        assert html.count("section-banner") >= 2 + 2  # 2 secciones + 2 en CSS
        assert "<svg" in html                            # bloques graficados
        assert 'data-text-slot="t:s1"' in html           # slot de texto presente
        assert "placeholder-card" in html                # heatmap + SKIP
        assert 'data-chart="heatmap_table"' in html     # bloque heatmap presente
        assert "vista preliminar" in html                # nota del bloque de barras
        assert '<meta charset="UTF-8">' in html          # acentos correctos al imprimir

    def test_text_slots_are_empty(self, tmp_path):
        p = tmp_path / "dur.parquet"
        _categorical_parquet(p)
        entries = [_ds("dur.parquet", id="dur")]
        report = build_curated_report(_spec_for_build(), entries=entries, parquet_dir=tmp_path)
        html = render_curated_html(report)
        # el slot va vacío (el texto se llena después con el LLM)
        assert 'data-text-slot="t:s1"></div>' in html


# ── Ejes redondos + base 0, tooltips, escala, texto/síntesis (mejoras gerencia) ─


@pytest.mark.unit
class TestNiceAxes:
    def test_round_ticks_and_zero_base_when_positive(self):
        ticks, lo, _hi = _nice_ticks(4326704.0, 25879410.0)
        assert lo == 0.0  # base 0 cuando todo es positivo
        diffs = {round(ticks[i + 1] - ticks[i], 6) for i in range(len(ticks) - 1)}
        assert len(diffs) == 1  # paso constante (ticks equiespaciados)
        step = diffs.pop()
        assert step in (5_000_000.0, 10_000_000.0)  # número redondo, no el máximo crudo

    def test_zero_in_range_when_negatives(self):
        ticks, lo, hi = _nice_ticks(-1844.0, 1065.0)
        assert lo < 0 < hi
        assert 0.0 in ticks  # el eje cruza por 0


@pytest.mark.unit
class TestTooltips:
    def test_line_has_hover_points(self):
        plot = PlotData("d", "line", "timeseries", "US$ Mill.",
                        [PlotSeries("Tipo 1", [("2026-01-01", 10.0), ("2026-02-01", 20.0)])])
        svg = render_plot_svg(plot, chart="line")
        assert 'class="tip-pt"' in svg
        assert 'data-s="Tipo 1"' in svg              # serie
        assert 'data-k="01-01-26"' in svg            # fecha formateada
        assert 'data-v="10,0 US$ Mill."' in svg      # valor + unidad (es-CL)

    def test_bars_have_hover_data(self):
        plot = PlotData("d", "grouped_bar", "grouped", "%",
                        [PlotSeries("Δ7d", [("Tipo 1", 1.0), ("Tipo 2", -2.0)])])
        svg = render_plot_svg(plot, chart="grouped_bar")
        assert svg.count('class="tip-pt"') >= 2
        assert 'data-k="Tipo 1"' in svg and 'data-s="Δ7d"' in svg

    def test_pie_has_hover_data(self):
        plot = PlotData("d", "pie", "snapshot", "MM USD",
                        [PlotSeries("c", [("DAP", 28.0), ("BB", 23.0)])])
        svg = render_plot_svg(plot, chart="pie")
        assert 'class="tip-pt"' in svg and 'data-s="DAP"' in svg

    def test_shell_ships_interactive_tooltip(self, tmp_path):
        p = tmp_path / "dur.parquet"
        _categorical_parquet(p)
        report = build_curated_report(
            _spec_for_build(), entries=[_ds("dur.parquet", id="dur")], parquet_dir=tmp_path,
        )
        html = render_curated_html(report)
        assert 'id="chart-tip"' in html              # caja del tooltip
        assert "getElementById('chart-tip')" in html  # el JS quedó inyectado
        assert 'class="tip-pt"' in html               # los puntos llevan la clase


@pytest.mark.unit
class TestScale:
    def test_scale_plot_multiplies_and_keeps_original(self):
        p = PlotData("d", "line", "timeseries", "Mill US$.",
                     [PlotSeries("BB", [("2026-01-01", 1000.0)])])
        s = _scale_plot(p, 0.001)
        assert s.series[0].points[0][1] == 1.0
        assert p.series[0].points[0][1] == 1000.0  # el original no se muta

    def test_catalog_value_scale_scales_chart_and_facts(self, tmp_path):
        # value_scale del catálogo escala TANTO el gráfico (builder) como el
        # texto (compute_facts), para que prosa y curva coincidan.
        from banks_rag.application.reporting.parquet_facts import compute_facts
        p = tmp_path / "alloc.parquet"
        _categorical_parquet(p)  # Tipo 6 = 6.0
        ds = _ds("alloc.parquet", id="dur", value_scale=10.0)
        spec = FamilyReportSpec(family="t", title="T", blocks=(
            ReportBlock(section="S", title="L", chart="line", status=STATUS_MVP,
                        source_id="dur", transform="filter_fund", params={"funds": ["Tipo 6"]}),
        ))
        rep = build_curated_report(spec, entries=[ds], parquet_dir=tmp_path)
        assert "60" in rep.blocks[0].body_html  # 6.0 x 10 en el tooltip del grafico
        facts = compute_facts(ds, tmp_path, [("última semana", 7), ("último mes", 30)])
        cats = {c["categoria"]: c["ultimo_valor"] for c in facts["por_categoria"]}
        assert cats["Tipo 6"] == 60.0  # 6.0 x 10 en los hechos del texto


@pytest.mark.unit
class TestTextAndSynthesis:
    def test_fill_text_slots_renders_two_paragraphs(self):
        html = '<div class="section-text" data-text-slot="ffmm:x"></div>'
        out = fill_text_slots(html, {"ffmm:x": "Mensual.\n\nSemanal."})
        assert out.count("<p>") == 2
        assert "Mensual." in out and "Semanal." in out

    def test_fill_synthesis_bullets_and_bold(self):
        html = '<div class="synthesis-body" data-synthesis-body></div>'
        out = fill_synthesis_slot(html, "Intro.\n\n**Mes:**\n- a\n- b")
        assert "<strong>Mes:</strong>" in out
        assert out.count("<li>") == 2

    def test_fill_synthesis_empty_is_noop(self):
        html = '<div class="synthesis-body" data-synthesis-body></div>'
        assert fill_synthesis_slot(html, "") == html
