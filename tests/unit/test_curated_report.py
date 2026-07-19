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
    category_series,
    composition_by_bucket,
    daily_wide_stacked,
    dcv_bucket_table,
    dcv_cut_dates,
    dcv_heatmap,
    dcv_portfolio_table,
    dcv_snapshot_stacked,
    filter_fund,
    fx_agent_delta_table,
    fx_sector_flow_table,
    fx_tasas_scatter,
    gbi_rendimiento_range,
    get_transform,
    is_known,
    latest_snapshot,
    monthly_var_alloc,
    snapshot_grouped,
    snapshot_stacked,
    stacked_by_bucket,
    wide_lines,
    wide_monthly_bars,
    wide_row_stacked,
    wide_window_bars,
    window_grouped,
    window_grouped_long,
    window_returns,
    window_stacked_by_cat,
    window_stacked_two_cat,
)
from banks_rag.application.reporting.specs import available_families, get_spec
from banks_rag.application.reporting.specs.afp_spec import AFP_SPEC
from banks_rag.application.reporting.specs.dcv_spec import DCV_SPEC
from banks_rag.application.reporting.specs.ffmm_spec import FFMM_SPEC
from banks_rag.application.reporting.specs.fx_spec import FX_SPEC
from banks_rag.application.reporting.specs.nr_spec import NR_SPEC
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
        # Informe semanal: Rentabilidad y Mercado cambiario van justo después de Flujos.
        assert secs[:3] == ["Flujos", "Rentabilidad", "Mercado cambiario"]
        assert secs.index("Mercado cambiario") < secs.index("Portafolio DCV")
        assert secs[-1] == "Variación en allocation carteras mensuales"
        assert "Carteras DCV" not in secs  # sección eliminada (su único gráfico era duplicado)
        assert len(FFMM_SPEC.blocks) == 26

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


# ── Specs reales de nr / afp ─────────────────────────────────────────────────


@pytest.mark.unit
class TestNrSpec:
    def test_registered_and_listed(self):
        assert get_spec("nr") is NR_SPEC
        assert "nr" in available_families()

    def test_sections_in_order(self):
        secs = NR_SPEC.sections()
        assert secs == ["Mercado de Derivados", "Mercado de Renta Fija",
                        "Mercado SPC (tasas)", "Flujos Spot", "Gráficos Extras"]

    def test_blocks_reference_known_transforms(self):
        for b in NR_SPEC.blocks:
            assert get_transform(b.transform) is not None, b.title

    def test_no_duplicate_charts_or_titles(self):
        import json
        sigs = [(b.source_id, b.transform, json.dumps(b.params or {}, sort_keys=True)) for b in NR_SPEC.blocks]
        assert len(sigs) == len(set(sigs)), "gráfico repetido"
        titles = [b.title for b in NR_SPEC.blocks]
        assert len(titles) == len(set(titles)), "título repetido"


@pytest.mark.unit
class TestAfpSpec:
    def test_registered_and_listed(self):
        assert get_spec("afp") is AFP_SPEC
        assert "afp" in available_families()

    def test_sections_in_order(self):
        secs = AFP_SPEC.sections()
        assert secs[0] == "Allocation y patrimonio"
        assert secs[-1] == "Atribución de retorno"
        assert "Mercado cambiario" in secs

    def test_blocks_reference_known_transforms(self):
        for b in AFP_SPEC.blocks:
            assert get_transform(b.transform) is not None, b.title

    def test_dual_axis_block_declares_right_axis(self):
        dual = [b for b in AFP_SPEC.blocks if b.chart == "dual_axis"]
        assert dual, "el informe afp debe tener al menos un bloque de doble eje"
        for b in dual:
            assert b.params.get("right_axis"), b.title

    def test_no_duplicate_titles(self):
        titles = [b.title for b in AFP_SPEC.blocks]
        assert len(titles) == len(set(titles))


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
        assert "T-7" in result.html and "T-30" in result.html

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
        assert "T-7" in result.html

    def test_dcv_transforms_return_none_for_wrong_structure(self, tmp_path):
        # parquet de una sola categórica (duracion) no tiene la estructura DCV
        p = tmp_path / "dur.parquet"
        _categorical_parquet(p)
        ds = _ds("dur.parquet", id="duracion_ffmm")
        assert dcv_heatmap(ds, tmp_path, {}) is None  # <2 cat cols → None


@pytest.mark.unit
class TestGroupedTransforms:
    def test_window_returns_geometric(self, tmp_path):
        # retornos_fondo = ÍNDICE de retorno acumulado en % (39,28 ↔ i=0,3928):
        # la ventana es el retorno COMPUESTO (1+i_end)/(1+i_start)-1, no la resta.
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
        assert round(d7["Tipo 1"], 2) == 1.33   # (1.52/1.50-1)*100, no 52-50=2.0
        assert round(d7["Tipo 2"], 2) == -0.38  # (1.295/1.30-1)*100, no 29.5-30=-0.5

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
class TestGenericTransforms:
    """Transforms genéricas reutilizables (NR / AFP)."""

    def test_category_series_casts_varchar_value(self, tmp_path):
        # Valor como VARCHAR (números como texto) → detect_roles no lo ve, pero
        # category_series fuerza la columna y castea.
        p = tmp_path / "pos.parquet"
        rows = []
        for d in ["2026-06-09", "2026-06-10"]:
            for plz, v in [("1 a 90 dias", "100.5"), ("Mayor a 360 dias", "200.0")]:
                rows.append(f"(DATE '{d}', '{plz}', '{v}')")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Plazo, Valor)")
        ds = _ds("pos.parquet", id="posicion_nr_derivados", chart_type="stacked_area")
        plot = category_series(ds, tmp_path, {"category": "Plazo", "value": "Valor",
                                              "order": ["1 a 90 dias", "Mayor a 360 dias"]})
        assert [s.label for s in plot.series] == ["1 a 90 dias", "Mayor a 360 dias"]
        assert dict(plot.series[0].points)["2026-06-10"] == 100.5

    def test_category_series_filter_and_cumsum(self, tmp_path):
        p = tmp_path / "spot.parquet"
        rows = [
            "(DATE '2026-01-02', 'Total', 'No', 10.0)",
            "(DATE '2026-01-03', 'Total', 'No', 5.0)",
            "(DATE '2026-01-02', 'BBVA', 'No', 99.0)",  # se filtra fuera
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Institucion, Afecto_derivado, Net)")
        ds = _ds("spot.parquet", id="spot_acumulado_agente", chart_type="line")
        plot = category_series(ds, tmp_path, {
            "filter_col": "Institucion", "filter_val": "Total",
            "category": "Afecto_derivado", "value": "Net",
            "accumulate": "cumsum", "window": "ytd"})
        pts = dict(plot.series[0].points)
        assert pts["2026-01-03"] == 15.0  # cumsum: 10 + 5

    def test_wide_lines_include_excludes_other_cols(self, tmp_path):
        p = tmp_path / "w.parquet"
        _write(p, "SELECT * FROM (VALUES "
               "(DATE '2026-06-09', 1.0, 2.0, 9.0), (DATE '2026-06-10', 3.0, 4.0, 9.0)) "
               "t(Fecha, A, B, Neto)")
        ds = _ds("w.parquet", id="nr_var_posicion_spc", chart_type="line")
        plot = wide_lines(ds, tmp_path, {"include": ["A", "B"]})
        assert [s.label for s in plot.series] == ["A", "B"]  # 'Neto' excluido

    def test_window_grouped_last_window_with_net(self, tmp_path):
        p = tmp_path / "vd.parquet"
        rows = [
            "('2026-06-10', '1 a 90 días', 100.0, -40.0)",
            "('2026-06-09', '1 a 90 días', 50.0, -10.0)",
            "('2020-01-01', '1 a 90 días', 999.0, 999.0)",  # fuera de ventana
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Plazo, Suscripcion, Vencimiento)")
        ds = _ds("vd.parquet", id="var_pos_derivados", chart_type="grouped_bar")
        plot = window_grouped(ds, tmp_path, {"group": "Plazo", "values": ["Suscripcion", "Vencimiento"],
                                             "window_days": 7, "include_net": True})
        assert plot.kind == "grouped"
        labels = [s.label for s in plot.series]
        assert labels == ["Suscripcion", "Vencimiento", "Neto"]
        susc = dict(plot.series[0].points)["1 a 90 días"]
        neto = dict(plot.series[2].points)["1 a 90 días"]
        assert susc == 150.0          # 100 + 50 (la fila de 2020 queda fuera)
        assert neto == 100.0          # 150 + (-50)

    def test_snapshot_grouped_no_date(self, tmp_path):
        p = tmp_path / "cb.parquet"
        _write(p, "SELECT * FROM (VALUES "
               "('Habitat', 10.0, 20.0, 30.0), ('Provida', 5.0, -2.0, 3.0)) "
               "t(Sector_contraparte, Spot, Forward, Neto)")
        ds = _ds("cb.parquet", id="cambiario_afp", chart_type="grouped_bar")
        plot = snapshot_grouped(ds, tmp_path, {"group": "Sector_contraparte",
                                               "values": ["Spot", "Forward", "Neto"]})
        assert plot.kind == "grouped"
        assert [s.label for s in plot.series] == ["Spot", "Forward", "Neto"]
        assert dict(plot.series[0].points)["Habitat"] == 10.0

    def test_snapshot_stacked_two_categoricals(self, tmp_path):
        p = tmp_path / "at.parquet"
        rows = [
            "('A', 'RVI', 1.0)", "('A', 'RFN', 2.0)",
            "('B', 'RVI', 3.0)", "('B', 'RFN', 4.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(fondo, Clase, Valor)")
        ds = _ds("at.parquet", id="attribution", chart_type="stacked_bar")
        plot = snapshot_stacked(ds, tmp_path, {"x": "fondo", "series": "Clase",
                                               "value": "Valor", "x_order": ["A", "B"]})
        assert plot.kind == "grouped"
        cats = [c for c, _ in plot.series[0].points]
        assert cats == ["A", "B"]
        assert {s.label for s in plot.series} == {"RVI", "RFN"}

    def test_latest_snapshot_uses_last_date(self, tmp_path):
        p = tmp_path / "sn.parquet"
        rows = [
            "(DATE '2026-06-09', 'BTP', 10.0)", "(DATE '2026-06-09', 'BTU', 20.0)",
            "(DATE '2026-06-10', 'BTP', 30.0)", "(DATE '2026-06-10', 'BTU', 40.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Tipo, Stock_USD)")
        ds = _ds("sn.parquet", id="stock_nivel_afp", chart_type="pie")
        plot = latest_snapshot(ds, tmp_path, {"category": "Tipo", "value": "Stock_USD"})
        assert plot.kind == "snapshot"
        pts = dict(plot.series[0].points)
        assert pts == {"BTU": 40.0, "BTP": 30.0}  # solo el último día

    def test_category_series_net_auto_adds_overlay(self, tmp_path):
        # net="auto" agrega una serie "Neto" = suma de categorías por fecha, marcada
        # como overlay (apilado divergente + línea Neto).
        p = tmp_path / "spc.parquet"
        rows = []
        for d in ["2026-06-09", "2026-06-10"]:
            for plz, v in [("1 a 90 dias", -30.0), ("Mayor a 2Y", 10.0)]:
                rows.append(f"(DATE '{d}', '{plz}', {v})")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Plazos_D, Monto_USD)")
        ds = _ds("spc.parquet", id="posicion_nr_spc", chart_type="stacked_area")
        plot = category_series(ds, tmp_path, {"category": "Plazos_D", "value": "Monto_USD",
                                              "net": "auto", "order": ["1 a 90 dias", "Mayor a 2Y"]})
        assert plot.overlay == ("Neto",)
        neto = next(s for s in plot.series if s.label == "Neto")
        assert dict(neto.points)["2026-06-10"] == -20.0  # -30 + 10

    def test_wide_lines_overlay_marks_neto(self, tmp_path):
        p = tmp_path / "w.parquet"
        _write(p, "SELECT * FROM (VALUES "
               "(DATE '2026-06-09', 1.0, 2.0, 3.0), (DATE '2026-06-10', 3.0, 4.0, 7.0)) "
               "t(Fecha, A, B, Neto)")
        ds = _ds("w.parquet", id="nr_var_posicion_spc", chart_type="stacked_area")
        plot = wide_lines(ds, tmp_path, {"overlay": ["Neto"]})
        assert plot.overlay == ("Neto",)
        assert "Neto" in [s.label for s in plot.series]

    def test_window_grouped_negate_and_net_overlay(self, tmp_path):
        # Vencimiento se NEGA → Neto = Suscripción - Vencimiento; Neto como overlay.
        p = tmp_path / "vd.parquet"
        rows = [
            "('2026-06-10', '1 a 90 días', 100.0, 40.0)",
            "('2026-06-09', '1 a 90 días', 50.0, 10.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Plazo, Suscripcion, Vencimiento)")
        ds = _ds("vd.parquet", id="var_pos_derivados", chart_type="grouped_bar")
        plot = window_grouped(ds, tmp_path, {"group": "Plazo", "values": ["Suscripcion", "Vencimiento"],
                                             "window_days": 7, "include_net": True,
                                             "net_as_overlay": True, "negate": ["Vencimiento"]})
        assert plot.overlay == ("Neto",)
        venc = dict(next(s for s in plot.series if s.label == "Vencimiento").points)["1 a 90 días"]
        neto = dict(next(s for s in plot.series if s.label == "Neto").points)["1 a 90 días"]
        assert venc == -50.0          # -(40 + 10) negado
        assert neto == 100.0          # 150 (susc) - 50 (venc) = 150 + (-50)

    def test_wide_window_bars_diff_last_n(self, tmp_path):
        # NIVELES → diff día-a-día; últimos 2 días; Neto como overlay-punto.
        p = tmp_path / "lv.parquet"
        rows = [
            "(DATE '2026-06-08', 10.0, 100.0)",
            "(DATE '2026-06-09', 13.0, 105.0)",
            "(DATE '2026-06-10', 14.0, 108.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, A, Neto)")
        ds = _ds("lv.parquet", id="nr_var_posicion_spc", chart_type="stacked_bar")
        plot = wide_window_bars(ds, tmp_path, {"overlay": ["Neto"], "last_n": 2, "diff": True})
        assert plot.kind == "grouped" and plot.overlay == ("Neto",)
        a = dict(next(s for s in plot.series if s.label == "A").points)
        # diff: 13-10=3 (09), 14-13=1 (10); last_n=2 → ambos
        assert a["09-06"] == 3.0 and a["10-06"] == 1.0

    def test_window_stacked_by_cat_daily_bars(self, tmp_path):
        # Parquet LARGO (fecha+fondo+flujo) → barras apiladas por día, últimos N.
        p = tmp_path / "mov.parquet"
        rows = []
        for d in ["2026-06-08", "2026-06-09"]:
            for f, v in [("A", 10.0), ("B", -5.0)]:
                rows.append(f"(DATE '{d}', '{f}', {v})")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(fecha, fondo, flujos_usd)")
        ds = _ds("mov.parquet", id="movimientos_fondos", chart_type="stacked_bar")
        plot = window_stacked_by_cat(ds, tmp_path, {"category": "fondo", "value": "flujos_usd",
                                                    "order": ["A", "B"], "last_n": 2})
        assert plot.kind == "grouped"
        cats = [c for c, _ in plot.series[0].points]
        assert cats == ["08-06", "09-06"]                       # eje X = días
        assert {s.label for s in plot.series} == {"A", "B"}
        assert dict(plot.series[0].points)["09-06"] == 10.0     # fondo A

    def test_snapshot_grouped_overlay_marks_neto(self, tmp_path):
        p = tmp_path / "cb.parquet"
        _write(p, "SELECT * FROM (VALUES "
               "('Habitat', 10.0, 20.0, 30.0), ('Modelo', 5.0, -8.0, -3.0)) "
               "t(Sector_contraparte, Spot, Forward, Neto)")
        ds = _ds("cb.parquet", id="cambiario_afp", chart_type="grouped_bar")
        plot = snapshot_grouped(ds, tmp_path, {"group": "Sector_contraparte",
                                               "values": ["Spot", "Forward", "Neto"],
                                               "overlay": ["Neto"]})
        assert plot.overlay == ("Neto",)
        assert dict(next(s for s in plot.series if s.label == "Neto").points)["Modelo"] == -3.0

    def test_snapshot_stacked_total_overlay(self, tmp_path):
        p = tmp_path / "at.parquet"
        rows = ["('A', 'RVI', 3.0)", "('A', 'RFN', -1.0)", "('B', 'RVI', 2.0)", "('B', 'RFN', 1.0)"]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(fondo, Clase, Valor)")
        ds = _ds("at.parquet", id="attribution", chart_type="stacked_bar")
        plot = snapshot_stacked(ds, tmp_path, {"x": "fondo", "series": "Clase",
                                               "value": "Valor", "x_order": ["A", "B"],
                                               "total_overlay": True})
        assert plot.overlay == ("Total",)
        tot = dict(next(s for s in plot.series if s.label == "Total").points)
        assert tot["A"] == 2.0 and tot["B"] == 3.0             # suma de clases por fondo

    def test_stacked_by_bucket_net_overlay(self, tmp_path):
        p = tmp_path / "vs.parquet"
        rows = []
        for d in ["2026-06-02", "2026-06-09"]:
            for bucket, tipo, v in [("Menor a 1Y", "PDBC", 100.0 if d == "2026-06-02" else 80.0),
                                    ("Menor a 1Y", "BTP", 10.0 if d == "2026-06-02" else 25.0)]:
                rows.append(f"(DATE '{d}', '{bucket}', '{tipo}', 'CLP', {v})")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows)
               + ") t(Fecha, Bucket, Tipo, Moneda, Stock_USD)")
        ds = _ds("vs.parquet", id="variacion_stock_afp", chart_type="stacked_bar")
        plot = stacked_by_bucket(ds, tmp_path, {"window": "7d", "net_as_overlay": True})
        assert plot.overlay == ("Neto",)
        neto = dict(next(s for s in plot.series if s.label == "Neto").points)["Menor a 1Y"]
        assert neto == -5.0    # PDBC (80-100=-20) + BTP (25-10=+15) = -5

    def test_window_grouped_long_signed_with_net(self, tmp_path):
        # Formato LARGO con columna Tipo (Suscripción/Vencimiento): apila susc (+) y
        # vcto (-), Neto = susc - vcto, top_n por |neto|, excluye 'Spot'.
        p = tmp_path / "ag.parquet"
        rows = [
            "(DATE '2026-06-10', 'JP Morgan', 'Suscripción', 100.0)",
            "(DATE '2026-06-10', 'JP Morgan', 'Vencimiento', 30.0)",
            "(DATE '2026-06-10', 'JP Morgan', 'Spot', 999.0)",     # excluido
            "(DATE '2026-06-10', 'BBVA', 'Suscripción', 5.0)",
            "(DATE '2020-01-01', 'JP Morgan', 'Suscripción', 999.0)",  # fuera de ventana
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Institucion, Tipo, Monto)")
        ds = _ds("ag.parquet", id="spot_susc_vcto_agente", chart_type="stacked_bar")
        plot = window_grouped_long(ds, tmp_path, {"group": "Institucion", "type_col": "Tipo",
                                                  "value": "Monto", "pos": "Suscripción",
                                                  "neg": "Vencimiento", "exclude_types": ["Spot"],
                                                  "window_days": 7, "top_n": 1})
        assert plot.overlay == ("Neto",)
        cats = [c for c, _ in plot.series[0].points]
        assert cats == ["JP Morgan"]                       # top_n=1 por |neto|
        venc = dict(next(s for s in plot.series if s.label == "Vencimiento").points)["JP Morgan"]
        neto = dict(next(s for s in plot.series if s.label == "Neto").points)["JP Morgan"]
        assert venc == -30.0 and neto == 70.0              # 100 - 30, Spot ignorado

    def test_wide_monthly_bars_sums_by_month(self, tmp_path):
        p = tmp_path / "wm.parquet"
        rows = [
            "(DATE '2026-05-10', 1.0, 9.0)", "(DATE '2026-05-20', 2.0, 9.0)",  # may: A=3
            "(DATE '2026-06-05', 4.0, 9.0)",                                    # jun: A=4
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, A, Neto)")
        ds = _ds("wm.parquet", id="nr_var_posicion_derivados", chart_type="stacked_bar")
        plot = wide_monthly_bars(ds, tmp_path, {"include": ["A"], "overlay": ["Neto"], "months": 12})
        assert plot.kind == "grouped" and plot.overlay == ("Neto",)
        a = dict(next(s for s in plot.series if s.label == "A").points)
        assert a["may26"] == 3.0 and a["jun26"] == 4.0     # suma dentro del mes

    def test_dual_axis_render_has_two_axes(self):
        plot = PlotData(
            dataset_id="allocation_int_nac", family="line", kind="timeseries", unit="%",
            series=[
                PlotSeries(label="Nacional", points=[("2026-01-01", 60.0), ("2026-02-01", 55.0)]),
                PlotSeries(label="AUM", points=[("2026-01-01", 200000.0), ("2026-02-01", 250000.0)]),
            ],
        )
        svg = render_plot_svg(plot, chart="dual_axis", right_axis=["AUM"], right_unit="US$ Mill.")
        assert svg is not None
        assert "doble eje" in svg
        assert "AUM (eje der.)" in svg  # leyenda marca el eje derecho
        assert "US$ Mill." in svg       # unidad del eje derecho


@pytest.mark.unit
class TestGbiRendimientoRangeTransform:
    """``gbi_rendimiento_range``: caja [mín,máx] + promedio + "hoy" (kind='range',
    réplica de "Rendimiento monedas" GBI)."""

    def _write_gbi(self, tmp_path):
        p = tmp_path / "gbi.parquet"
        rows = [
            "(DATE '2026-01-02', 'CLP | A', 100.0)",
            "(DATE '2026-03-01', 'CLP | A', 103.0)",
            "(DATE '2026-07-13', 'CLP | A', 102.93)",
            "(DATE '2026-01-02', 'BRL | BB-', 100.0)",
            "(DATE '2026-03-01', 'BRL | BB-', 90.0)",
            "(DATE '2026-07-13', 'BRL | BB-', 93.53)",
            "(DATE '2025-12-01', 'CLP | A', 999.0)",  # fuera de ventana YTD: no debe pesar
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Tenor, Valor)")
        return p

    def test_rebases_to_100_and_computes_range_mean_hoy(self, tmp_path):
        self._write_gbi(tmp_path)
        ds = _ds("gbi.parquet", id="gbi_index", chart_type="hist_range", unit="Índice (base 100)")
        plot = gbi_rendimiento_range(ds, tmp_path, {"order": ["BRL | BB-", "CLP | A"], "window": "ytd"})
        assert plot.kind == "range"
        assert [c for c, _ in plot.series[0].points] == ["BRL | BB-", "CLP | A"]  # respeta 'order'
        by_label = {s.label: dict(s.points) for s in plot.series}
        assert by_label["Hoy"]["CLP | A"] == pytest.approx(102.93)
        assert by_label["Hoy"]["BRL | BB-"] == pytest.approx(93.53)
        assert by_label["Mínimo"]["BRL | BB-"] == pytest.approx(90.0)
        assert by_label["Máximo"]["BRL | BB-"] == pytest.approx(100.0)
        assert "Índice base 100" in plot.date_note

    def test_default_order_is_alphabetical_when_not_given(self, tmp_path):
        self._write_gbi(tmp_path)
        ds = _ds("gbi.parquet", id="gbi_index", chart_type="hist_range")
        plot = gbi_rendimiento_range(ds, tmp_path, {})
        cats = [c for c, _ in plot.series[0].points]
        assert cats == sorted(cats)

    def test_missing_parquet_returns_none(self, tmp_path):
        ds = _ds("nope.parquet", id="gbi_index", chart_type="hist_range")
        assert gbi_rendimiento_range(ds, tmp_path, {}) is None

    def test_renders_as_native_hist_range_via_render_plot_svg(self, tmp_path):
        self._write_gbi(tmp_path)
        ds = _ds("gbi.parquet", id="gbi_index", chart_type="hist_range")
        plot = gbi_rendimiento_range(ds, tmp_path, {"order": ["BRL | BB-", "CLP | A"]})
        svg = render_plot_svg(plot, chart="hist_range")
        assert svg is not None and "93,5" in svg


@pytest.mark.unit
class TestFxTasasScatterTransform:
    """``fx_tasas_scatter``: dispersión retorno FX vs. retorno tasas (kind='scatter',
    réplica de "Retorno FX y tasas GBI"). Esquema real confirmado contra el
    catálogo (id ``retorno_fx_tasas``, chart_type ``fx_retorno_scatter_interactive``):
    columnas PREFIJADAS ``fx_{PAIS}``/``rates_{PAIS}``, país en MAYÚSCULA."""

    def _write_fx(self, tmp_path):
        p = tmp_path / "fx.parquet"
        rows = [
            "(DATE '2026-01-02', 800.0, 5.0, 4.20, 3.0)",
            "(DATE '2026-04-01', 820.0, 5.5, 4.10, 3.2)",
            "(DATE '2026-07-13', 810.0, 5.8, 4.00, 3.5)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") "
                  "t(Fecha, fx_CLP, rates_CLP, fx_USD, rates_USD)")
        return p

    def test_computes_additive_cumulative_return_per_country(self, tmp_path):
        self._write_fx(tmp_path)
        ds = _ds("fx.parquet", id="retorno_fx_tasas", chart_type="point", unit="Porcentaje")
        plot = fx_tasas_scatter(ds, tmp_path, {"window": "ytd", "highlight": "clp"})
        assert plot.kind == "scatter"
        assert plot.unit == "Porcentaje"  # usa dataset.unit del catálogo, no hardcodeado
        by_label = {s.label: s.points[0] for s in plot.series}
        assert set(by_label) == {"CLP", "USD"}
        # fx_CLP: 800->820->810 desde 2026-01-02; pct diarios +2.5%, -1.2195..% ; suma=1.2805;
        # negado (convención apreciación) => -1.2805
        x_clp, y_clp = by_label["CLP"]
        assert float(x_clp) == pytest.approx(-1.28048780487805, rel=1e-6)
        assert y_clp == pytest.approx(15.454545454545451, rel=1e-6)  # rates_CLP: +10% +5.4545%

    def test_defaults_to_percent_when_dataset_unit_missing(self, tmp_path):
        self._write_fx(tmp_path)
        ds = _ds("fx.parquet", id="retorno_fx_tasas", chart_type="point", unit="")
        plot = fx_tasas_scatter(ds, tmp_path, {})
        assert plot.unit == "%"

    def test_highlight_sets_overlay(self, tmp_path):
        self._write_fx(tmp_path)
        ds = _ds("fx.parquet", id="retorno_fx_tasas", chart_type="point")
        plot = fx_tasas_scatter(ds, tmp_path, {"highlight": "clp"})
        assert plot.overlay == ("CLP",)

    def test_no_highlight_leaves_overlay_empty(self, tmp_path):
        self._write_fx(tmp_path)
        ds = _ds("fx.parquet", id="retorno_fx_tasas", chart_type="point")
        plot = fx_tasas_scatter(ds, tmp_path, {})
        assert plot.overlay == ()

    def test_negate_fx_false_keeps_raw_sign(self, tmp_path):
        self._write_fx(tmp_path)
        ds = _ds("fx.parquet", id="retorno_fx_tasas", chart_type="point")
        plot = fx_tasas_scatter(ds, tmp_path, {"negate_fx": False})
        by_label = {s.label: s.points[0] for s in plot.series}
        x_clp, _ = by_label["CLP"]
        assert float(x_clp) == pytest.approx(1.28048780487805, rel=1e-6)

    def test_country_missing_one_side_is_excluded(self, tmp_path):
        # fx_BRL sin su par rates_BRL → BRL no debe aparecer (requiere AMBOS lados)
        p = tmp_path / "fx2.parquet"
        _write(p, "SELECT * FROM (VALUES "
               "(DATE '2026-01-02', 800.0, 5.0, 4.0), "
               "(DATE '2026-07-13', 810.0, 5.8, 4.2)) "
               "t(Fecha, fx_CLP, rates_CLP, fx_BRL)")
        ds = _ds("fx2.parquet", id="retorno_fx_tasas", chart_type="point")
        plot = fx_tasas_scatter(ds, tmp_path, {})
        assert {s.label for s in plot.series} == {"CLP"}

    def test_custom_prefixes_override_defaults(self, tmp_path):
        p = tmp_path / "fx3.parquet"
        _write(p, "SELECT * FROM (VALUES "
               "(DATE '2026-01-02', 800.0, 5.0), (DATE '2026-07-13', 810.0, 5.8)) "
               "t(Fecha, spot_CLP, yield_CLP)")
        ds = _ds("fx3.parquet", id="retorno_fx_tasas", chart_type="point")
        plot = fx_tasas_scatter(ds, tmp_path, {"fx_prefix": "spot_", "rate_prefix": "yield_"})
        assert {s.label for s in plot.series} == {"CLP"}

    def test_missing_parquet_returns_none(self, tmp_path):
        ds = _ds("nope.parquet", id="retorno_fx_tasas", chart_type="point")
        assert fx_tasas_scatter(ds, tmp_path, {}) is None

    def test_renders_as_native_scatter_via_render_plot_svg(self, tmp_path):
        self._write_fx(tmp_path)
        ds = _ds("fx.parquet", id="retorno_fx_tasas", chart_type="point")
        plot = fx_tasas_scatter(ds, tmp_path, {"highlight": "clp"})
        svg = render_plot_svg(plot, chart="point", x_label="Retorno FX", y_label="Retorno tasas")
        assert svg is not None and "CLP" in svg and "USD" in svg


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


@pytest.mark.unit
class TestWeeklyAnchorAndFunds:
    """Corte semanal acotado a Flujos+DCV y fondos 1/2/3/6 en los bloques ffmm."""

    def test_anchor_source_ids_are_flujos_and_dcv(self):
        from banks_rag.application.reporting.curated_report import weekly_anchor_source_ids

        anchor = weekly_anchor_source_ids(FFMM_SPEC)
        # Solo datasets de las secciones Flujos y Portafolio DCV definen el corte.
        assert "flujos_acum_ffmm" in anchor        # Flujos
        assert "stock_nivel_ffmm" in anchor        # Portafolio DCV
        assert "variacion_stock_ffmm" in anchor
        # Rentabilidad / Mercado cambiario NO anclan (usan su propio máximo).
        assert "retorno_acum_ffmm" not in anchor
        assert "retornos_fondo_ffmm" not in anchor
        assert "flujos_spot_ffmm" not in anchor

    def test_spec_without_anchor_sections_uses_all(self):
        from banks_rag.application.reporting.curated_report import (
            _spec_source_ids,
            weekly_anchor_source_ids,
        )

        # Un spec que comparte corte y NO declara weekly_anchor_sections → corte
        # global (todos los source_ids).
        spec = FamilyReportSpec(
            family="demo", title="Demo",
            blocks=(
                ReportBlock(section="S", title="A", chart="line", status=STATUS_MVP,
                            source_id="ds_a", transform="straight_series"),
                ReportBlock(section="S", title="B", chart="line", status=STATUS_MVP,
                            source_id="ds_b", transform="straight_series"),
            ),
        )
        assert not spec.weekly_anchor_sections
        assert spec.share_weekly_cutoff  # default
        assert weekly_anchor_source_ids(spec) == set(_spec_source_ids(spec))

    def test_afp_disables_shared_cutoff(self, tmp_path):
        # AFP fija share_weekly_cutoff=False: cada parquet se ancla a su propio
        # máximo (no hay corte común que arrastre todo a la fecha más vieja).
        from banks_rag.application.reporting.curated_report import (
            compute_weekly_cutoff,
            weekly_anchor_source_ids,
        )

        assert AFP_SPEC.share_weekly_cutoff is False
        assert weekly_anchor_source_ids(AFP_SPEC) == set()
        # Con anchor vacío el corte común es None (no toca parquets reales).
        assert compute_weekly_cutoff(AFP_SPEC, [], tmp_path) is None

    @pytest.mark.parametrize("spec", [AFP_SPEC, NR_SPEC, FX_SPEC, DCV_SPEC])
    def test_specs_that_want_the_latest_datum_of_each_parquet(self, spec, tmp_path):
        """afp, nr, fx y dcv muestran el ÚLTIMO dato de CADA parquet.

        El corte común es el mín de los máximos: basta un parquet atrasado para
        arrastrar todo el informe a su fecha y esconder datos que sí existen en
        los demás. Estas cuatro familias lo desactivan a propósito."""
        from banks_rag.application.reporting.curated_report import (
            compute_weekly_cutoff,
            weekly_anchor_source_ids,
        )

        assert spec.share_weekly_cutoff is False, spec.family
        assert weekly_anchor_source_ids(spec) == set(), spec.family
        assert compute_weekly_cutoff(spec, [], tmp_path) is None, spec.family

    def test_funds_1_2_3_6_in_key_blocks(self):
        by_id = {(b.source_id, b.transform): b for b in FFMM_SPEC.blocks}
        wanted = ["Tipo 1", "Tipo 2", "Tipo 3", "Tipo 6"]
        # Variación Patrimonio efectivo (flujos) y rentabilidad mensual: fondos 1/2/3/6.
        assert by_id[("flujos_acum_ffmm", "monthly_sum_by_fund")].params["funds"] == wanted
        assert by_id[("retornos_fondo_ffmm", "monthly_returns")].params["funds"] == wanted
        # La rentabilidad ACUMULADA, en cambio, muestra TODOS los fondos disponibles.
        assert by_id[("retorno_acum_ffmm", "ytd_return_geom")].params["funds"] == "all"

    def test_rentabilidad_acumulada_uses_geometric_transform(self):
        block = next(b for b in FFMM_SPEC.blocks if b.source_id == "retorno_acum_ffmm")
        assert block.transform == "ytd_return_geom"  # ya no "accumulated"+rebase


# ── Familia fx (Informe Flujos Cambiarios) ───────────────────────────────────

@pytest.mark.unit
class TestFxSpec:
    def test_registered_and_listed(self):
        assert get_spec("fx") is FX_SPEC
        assert "fx" in available_families()

    def test_segment_alias_resolves(self):
        # el segmento del catálogo (fx_diferencial) debe caer en el spec fx
        assert get_spec("fx_diferencial") is FX_SPEC

    def test_sections_follow_the_email_order(self):
        assert FX_SPEC.sections() == [
            "RESUMEN GENERAL", "Flujos SDR Forward FX USD - CLP", "NO RESIDENTES",
        ]

    def test_blocks_replicate_the_screenshots_one_to_one(self):
        """Orden y TÍTULOS literales de las capturas de
        ``data_pipeline/Tipos de informe/Informe Flujos Cambiario`` (1.jpg … 7.jpg),
        que fueron tomadas en el orden del correo. Este test es el contrato con ese
        original: 22 bloques (2 tablas + 20 gráficos) en esta secuencia exacta."""
        assert [b.title for b in FX_SPEC.blocks] == [
            # 1.jpg — RESUMEN GENERAL
            "Resumen de flujos por sector (US$ MM)",
            # 2.jpg
            "Gráfico N°1: Spot acumulado por sectores (US$ MM)",
            "Gráfico N°2: Derivados acumulado por sectores (US$ MM)",
            "Gráfico N°3: Suscripciones netas derivados (US$ MM)",
            "Gráfico N°4: Vencimientos netos derivados (US$ MM)",
            # 3.jpg
            "Gráfico N°5: Spot por tramo de precio (US$ MM)",
            "Gráfico N°6: Suscripciones por tramo de precio y vencimientos (US$ MM)",
            "Gráfico N°7: Próximo fixing por banco - NDF (US$ MM)",
            "Gráfico N°8: Últimos y próximos fixing de la banca - NDF (US$ MM)",
            # 4.jpg
            "Gráfico N°7.1. Próximo fixing según tipo de Instrumento",
            "Gráfico N°8.1 Fixing NDF según banco",
            "Monto transado según bucket (usd)",
            "Monto transado según fecha de vencimiento (usd)",
            "Precio promedio transacciones",
            # 5.jpg — NO RESIDENTES
            "Gráficos N°9: Posición derivados (US$ MM)",
            "Tabla N°1: Posición derivados (US$ MM)",
            "Gráfico N°10: Suscripciones brutas derivados (US$ MM)",
            # 6.jpg
            "Gráfico N°11: Posición por plazo derivados (US$ MM)",
            "Gráfico N°12: Posición NR todos los derivados (US$)",
            "Gráfico N°13: Posición acumulada por plazos derivados (US$ MM)",
            # 7.jpg
            "Gráfico N°14: Último fixing de la banca por agente - NDF (US$ MM)",
            "Gráfico N°15: Próximo fixing de la banca por agente - NDF (US$ MM)",
        ]

    def test_counts_match_the_original(self):
        tablas = [b for b in FX_SPEC.blocks if b.chart == "heatmap_table"]
        assert len(FX_SPEC.blocks) == 22
        assert len(tablas) == 2                       # RESUMEN GENERAL + Tabla N°1
        assert len(FX_SPEC.blocks) - len(tablas) == 20  # gráficos N°1..N°15 + 7.1/8.1 + SDR

    def test_unit_lives_in_the_title_not_in_the_unit_field(self):
        """El original lleva la unidad DENTRO del título ("… (US$ MM)"); dejar
        también ``unit`` la haría aparecer dos veces en el bloque."""
        for b in FX_SPEC.blocks:
            assert not b.unit, b.title

    def test_blocks_with_source_reference_known_transforms(self):
        for b in FX_SPEC.blocks:
            if b.status == STATUS_SKIP:
                continue
            assert get_transform(b.transform) is not None, b.title

    def test_skip_blocks_say_which_parquet_is_missing(self):
        # un SKIP sin explicación es indistinguible de un bug: la nota debe decir qué falta
        skipped = [b for b in FX_SPEC.blocks if b.status == STATUS_SKIP]
        assert skipped, "el informe fx documenta bloques sin parquet"
        for b in skipped:
            assert b.note.startswith("Falta parquet:"), b.title
            assert b.source_id is None, b.title

    def test_no_duplicate_charts_or_titles(self):
        import json
        sigs = [(b.source_id, b.transform, json.dumps(b.params or {}, sort_keys=True))
                for b in FX_SPEC.blocks if b.source_id]
        assert len(sigs) == len(set(sigs)), "gráfico repetido"
        titles = [b.title for b in FX_SPEC.blocks]
        assert len(titles) == len(set(titles)), "título repetido"

    def test_wide_parquets_do_not_use_category_series(self):
        """``posicion_nr_derivados`` es ANCHO: con ``category_series`` DuckDB no
        encuentra la columna "Plazo" y el bloque cae a placeholder en silencio."""
        for spec in (FX_SPEC, NR_SPEC):
            for b in spec.blocks:
                if b.source_id == "posicion_nr_derivados":
                    assert b.transform == "wide_lines", b.title

    def test_overlay_is_drawn_even_when_not_in_include(self, tmp_path):
        """``overlay`` se resuelve aparte de ``include`` (misma convención que
        ``wide_monthly_bars``): pedir el Neto superpuesto no obliga a listarlo entre
        las series apiladas. Antes se perdía la línea del Neto en silencio."""
        p = tmp_path / "w.parquet"
        rows = ["(DATE '2026-07-09', 1.0, 2.0, 3.0)", "(DATE '2026-07-10', 4.0, 5.0, 9.0)"]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, A, B, Neto)")
        ds = _ds("w.parquet", id="w", chart_type="stacked_area")
        plot = wide_lines(ds, tmp_path, {"include": ["A", "B"], "overlay": ["Neto"]})
        assert plot.overlay == ("Neto",)
        assert "Neto" in {s.label for s in plot.series}


@pytest.mark.unit
class TestWideRowStacked:
    def _write_fixing(self, tmp_path):
        p = tmp_path / "fixing.parquet"
        rows = [
            "('Santander', 51.0, 27.0, -5.0, 73.0)",
            "('BCI', 65.0, 25.0, -3.0, 87.0)",
            "('Total', 116.0, 52.0, -8.0, 160.0)",  # fila agregada: se excluye
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Institucion, AFP, BCCh, NR, Neto)")
        return p

    def test_rows_become_x_axis_and_columns_become_stack(self, tmp_path):
        self._write_fixing(tmp_path)
        ds = _ds("fixing.parquet", id="fixing_banca_sector", chart_type="stacked_bar")
        plot = wide_row_stacked(ds, tmp_path, {
            "row": "Institucion", "overlay": ["Neto"], "exclude_rows": ["Total"],
        })
        assert plot.kind == "grouped"
        assert [c for c, _ in plot.series[0].points] == ["Santander", "BCI"]
        assert plot.overlay == ("Neto",)
        # el Neto va como serie superpuesta, no apilada
        neto = next(s for s in plot.series if s.label == "Neto")
        assert dict(neto.points)["Santander"] == pytest.approx(73.0)

    def test_excluded_row_is_not_plotted(self, tmp_path):
        self._write_fixing(tmp_path)
        ds = _ds("fixing.parquet", id="f", chart_type="stacked_bar")
        plot = wide_row_stacked(ds, tmp_path, {"row": "Institucion", "exclude_rows": ["Total"]})
        assert "Total" not in [c for c, _ in plot.series[0].points]


@pytest.mark.unit
class TestWindowStackedTwoCat:
    def _write_susc(self, tmp_path):
        p = tmp_path / "susc.parquet"
        rows = [
            # último día (2026-07-13): lo que debe entrar con window_days=1
            "(DATE '2026-07-13', 'HSBC', 'FWD', 'Suscripción', -60.0)",
            "(DATE '2026-07-13', 'HSBC', 'FXS', 'Suscripción', -5.0)",
            "(DATE '2026-07-13', 'Santander', 'FWD', 'Suscripción', 30.0)",
            "(DATE '2026-07-13', 'HSBC', 'FWD', 'Vencimiento', 999.0)",  # otro Tipo: fuera
            # día anterior: fuera de la ventana de 1 día
            "(DATE '2026-07-10', 'HSBC', 'FWD', 'Suscripción', 500.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) +
               ") t(Fecha, Institucion, Instrumento, Tipo, Monto)")
        return p

    def test_filters_type_and_window_then_stacks(self, tmp_path):
        self._write_susc(tmp_path)
        ds = _ds("susc.parquet", id="susc_vcto_agente_instrumento", chart_type="stacked_bar")
        plot = window_stacked_two_cat(ds, tmp_path, {
            "group": "Institucion", "series": "Instrumento", "value": "Monto",
            "filter_col": "Tipo", "filter_val": "Suscripción", "window_days": 1,
            "labels": {"FWD": "Forward", "FXS": "FX swap"},
            "total_label": "Total",
        })
        by_label = {s.label: dict(s.points) for s in plot.series}
        assert by_label["Forward"]["HSBC"] == pytest.approx(-60.0)   # sin el 500 del día previo
        assert by_label["FX swap"]["HSBC"] == pytest.approx(-5.0)
        assert by_label["Neto"]["HSBC"] == pytest.approx(-65.0)      # suma de instrumentos
        assert by_label["Forward"]["Total"] == pytest.approx(-30.0)  # -60 + 30
        assert plot.overlay == ("Neto",)

    def test_labels_rename_the_legend(self, tmp_path):
        self._write_susc(tmp_path)
        ds = _ds("susc.parquet", id="s", chart_type="stacked_bar")
        plot = window_stacked_two_cat(ds, tmp_path, {
            "group": "Institucion", "series": "Instrumento", "value": "Monto",
            "filter_col": "Tipo", "filter_val": "Suscripción",
            "labels": {"FWD": "Forward"},
        })
        assert "Forward" in {s.label for s in plot.series}
        assert "FWD" not in {s.label for s in plot.series}


@pytest.mark.unit
class TestDailyWideStacked:
    def test_negated_column_subtracts_and_net_is_the_bar_height(self, tmp_path):
        p = tmp_path / "pos.parquet"
        rows = [
            "(DATE '2026-07-09', 153.0, 323.0)",
            "(DATE '2026-07-10', -124.0, 34.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Suscripcion, Vencimiento)")
        ds = _ds("pos.parquet", id="var_pos_derivados", chart_type="stacked_bar")
        plot = daily_wide_stacked(ds, tmp_path, {
            "include": ["Suscripcion", "Vencimiento"], "negate": ["Vencimiento"],
            "last_n": 5, "labels": {"Suscripcion": "Suscripciones", "Vencimiento": "Vencimientos"},
        })
        by_label = {s.label: dict(s.points) for s in plot.series}
        key = next(iter(by_label["Suscripciones"]))  # eje X viene formateado DD-MM-AA
        assert by_label["Vencimientos"][key] == pytest.approx(-323.0)  # negado
        assert by_label["Neto"][key] == pytest.approx(153.0 - 323.0)
        assert plot.overlay == ("Neto",)


@pytest.mark.unit
class TestFxTables:
    def _write_flujo(self, tmp_path):
        p = tmp_path / "flujo.parquet"
        rows = [
            "(DATE '2026-07-10', 'AFP', -135.1, 82.0)",
            "(DATE '2026-07-10', 'NR', 64.6, -90.9)",
            "(DATE '2026-07-09', 'AFP', 10.0, 5.0)",
            "(DATE '2026-07-09', 'NR', 20.0, 1.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Sector, Spot, Forward)")
        return p

    def test_summary_table_splits_day_and_window(self, tmp_path):
        self._write_flujo(tmp_path)
        ds = _ds("flujo.parquet", id="flujo_cambiario", unit="US$ Mill.", chart_type="line")
        out = fx_sector_flow_table(ds, tmp_path, {
            "sector": "Sector", "spot": "Spot", "deriv": "Forward", "days": 2,
            "labels": {"AFP": "AFP", "NR": "NO RESIDENTES"}, "order": ["AFP", "NO RESIDENTES"],
        })
        assert isinstance(out, HtmlTable)
        # día: solo 2026-07-10 · acumulado 2 días: suma de ambos días
        assert "-135,1" in out.html and "82,0" in out.html
        assert "-125,1" in out.html   # AFP spot acumulado = -135,1 + 10
        assert "NO RESIDENTES" in out.html and "TOTAL" in out.html

    def test_agent_delta_table_nets_subscription_minus_maturity(self, tmp_path):
        p = tmp_path / "ag.parquet"
        rows = [
            "(DATE '2026-07-13', 'Santander', 'Suscripción', 300.0)",
            "(DATE '2026-07-13', 'Santander', 'Vencimiento', 62.0)",
            "(DATE '2026-07-10', 'Santander', 'Suscripción', 100.0)",
            "(DATE '2026-07-13', 'Total', 'Suscripción', 999.0)",  # excluido
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Institucion, Tipo, Monto)")
        ds = _ds("ag.parquet", id="susc_vcto_agente_instrumento", unit="US$ Mill.", chart_type="line")
        out = fx_agent_delta_table(ds, tmp_path, {
            "agent": "Institucion", "value": "Monto", "type_col": "Tipo",
            "pos": "Suscripción", "neg": "Vencimiento", "windows": [1, 2],
            "exclude_agents": ["Total"],
        })
        assert isinstance(out, HtmlTable)
        assert "238" in out.html          # ΔT-1 = 300 - 62
        assert "338" in out.html          # ΔT-2 = 238 + 100
        assert ">999<" not in out.html    # la fila agregada no entra


@pytest.mark.unit
class TestCategorySeriesFxParams:
    """Params que el informe de flujos cambiarios agregó a ``category_series``."""

    def _write(self, tmp_path):
        p = tmp_path / "pos.parquet"
        rows = [
            # A opera los 3 días; B recién el 2º (para probar el ancla común)
            "(DATE '2026-07-01', 'A', 100.0, 40.0)",
            "(DATE '2026-07-02', 'A', 50.0, 10.0)",
            "(DATE '2026-07-03', 'A', -20.0, 5.0)",
            "(DATE '2026-07-02', 'B', 200.0, 100.0)",
            "(DATE '2026-07-03', 'B', 10.0, 0.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) +
               ") t(Fecha, Plazo, Suscripcion, Vencimiento)")
        return p

    def _run(self, tmp_path, **extra):
        self._write(tmp_path)
        ds = _ds("pos.parquet", id="posicion_derivados_plazo", chart_type="line")
        params = {"category": "Plazo", "value": "Suscripcion", "order": ["A", "B"]}
        params.update(extra)
        return category_series(ds, tmp_path, params)

    def test_value_neg_subtracts_the_second_column(self, tmp_path):
        """Sin ``value_neg`` el 'acumulado de posición' solo sumaría suscripciones."""
        plot = self._run(tmp_path, value_neg="Vencimiento", accumulate="cumsum")
        a = dict(plot.series[0].points)
        assert a["2026-07-01"] == pytest.approx(60.0)    # 100 - 40
        assert a["2026-07-03"] == pytest.approx(75.0)    # 60 + 40 + (-25)

    def test_abs_makes_the_flow_gross(self, tmp_path):
        plot = self._run(tmp_path, abs=True)
        a = dict(plot.series[0].points)
        assert a["2026-07-03"] == pytest.approx(20.0)    # |-20|, no -20

    def test_anchor_zero_uses_a_common_date_not_each_series_first_point(self, tmp_path):
        """B no operó el 1-jul: si se anclara en SU primer punto perdería el flujo
        del 2-jul y dejaría de ser comparable con A."""
        plot = self._run(tmp_path, accumulate="cumsum", anchor_zero=True)
        a = dict(plot.series[0].points)
        b = dict(plot.series[1].points)
        assert a["2026-07-01"] == pytest.approx(0.0)     # ancla común
        assert a["2026-07-03"] == pytest.approx(30.0)    # 50 + (-20), sin el 1-jul
        assert b["2026-07-03"] == pytest.approx(210.0)   # 200 + 10 íntegros

    def test_mean_overlay_is_the_average_of_the_daily_total(self, tmp_path):
        plot = self._run(tmp_path, mean_overlay=True)
        prom = next(s for s in plot.series if s.label == "Promedio")
        # totales diarios: 100 · 250 · -10  → promedio 113,33
        assert prom.points[0][1] == pytest.approx((100 + 250 - 10) / 3)
        assert "Promedio" in plot.overlay
        assert len({v for _, v in prom.points}) == 1     # línea horizontal


@pytest.mark.unit
class TestBuildFamilyReportLayout:
    """``build_family_report.py`` escribe una carpeta POR FAMILIA."""

    def _load_script(self):
        import importlib.util
        import pathlib
        path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_family_report.py"
        spec = importlib.util.spec_from_file_location("build_family_report_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod

    def test_each_family_writes_into_its_own_subfolder(self, tmp_path, monkeypatch):
        mod = self._load_script()
        captured = {}

        def fake_build(spec):
            captured["family"] = spec.family
            return "REPORT"

        monkeypatch.setattr(mod, "build_curated_report", fake_build)
        monkeypatch.setattr(mod, "render_curated_html", lambda r: "<html>x</html>")
        monkeypatch.setattr(mod, "make_editable_html", lambda h: h + "<!--ed-->")
        # summary() se usa en el print del CLI
        monkeypatch.setattr(mod, "build_curated_report",
                            lambda spec: type("R", (), {"summary": lambda self: "ok"})())

        assert mod.build_one("fx", tmp_path, editable=True) is True
        fam_dir = tmp_path / "fx"
        assert fam_dir.is_dir(), "el informe debe caer en <out>/<familia>/"
        names = sorted(p.name for p in fam_dir.iterdir())
        assert len(names) == 2 and all(n.startswith("fx_") for n in names)
        assert any(n.endswith("_editable.html") for n in names)
        # nada suelto en la raíz
        assert [p.name for p in tmp_path.iterdir()] == ["fx"]

    def test_unknown_family_reports_failure(self, tmp_path):
        mod = self._load_script()
        assert mod.build_one("no_existe", tmp_path, editable=False) is False


# ── Informe DCV (Stocks Depósito Central de Valores) ─────────────────────────


def _dcv_parquet(path) -> None:
    """Parquet maestro del informe DCV, con la forma REAL de
    ``variacion_stock_todos``: Fecha x Bucket x Tipo x Sector x Moneda x Stock_USD.

    Dos instrumentos con apertura de moneda (DAP en CLP y UF) y uno de una sola
    moneda (PDBC), para ejercitar la regla de etiquetado de filas."""
    rows = []
    for d in ["2026-07-13", "2026-07-14"]:
        for sector, mult in [("Bancos", 1.0), ("AFP", 2.0)]:
            for tipo, moneda, bucket, base in [
                ("PDBC", "CLP", "Menor a 1Y", 100.0),
                ("DAP", "CLP", "Menor a 1Y", 40.0),
                ("DAP", "UF", "Entre 2 y 5Y", 10.0),
                ("BTP", "CLP", "Mayor a 10Y", 25.0),
            ]:
                rows.append(f"(DATE '{d}', '{bucket}', '{tipo}', '{sector}', '{moneda}', {base * mult})")
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) +
           ") t(Fecha, Bucket, Tipo, Sector, Moneda, Stock_USD)")


@pytest.mark.unit
class TestDcvTransforms:
    @pytest.fixture
    def ds(self, tmp_path):
        _dcv_parquet(tmp_path / "dcv.parquet")
        return _ds("dcv.parquet", id="variacion_stock_todos", unit="US$ Mill.")

    def test_portfolio_table_totals_match_the_parquet(self, ds, tmp_path):
        html = dcv_portfolio_table(ds, tmp_path, {}).html
        # Último corte: Bancos = 175, AFP = 350, total = 525.
        assert "175,0" in html and "350,0" in html and "525,0" in html
        assert "14-jul-2026" in html  # el corte que se cita es el último con dato

    def test_portfolio_table_splits_only_multi_currency_instruments(self, ds, tmp_path):
        html = dcv_portfolio_table(ds, tmp_path, {})
        assert "DAP $" in html.html and "DAP UF" in html.html
        # PDBC y BTP existen en una sola moneda: van sin sufijo.
        assert ">PDBC<" in html.html and ">BTP<" in html.html

    def test_portfolio_table_uses_email_agent_names(self, ds, tmp_path):
        html = dcv_portfolio_table(ds, tmp_path, {}).html
        assert "FP y AFC" in html and ">AFP<" not in html

    def test_bucket_table_filters_by_sector(self, ds, tmp_path):
        todos = dcv_bucket_table(ds, tmp_path, {}).html
        bancos = dcv_bucket_table(ds, tmp_path, {"sector": "Bancos"}).html
        assert "525,0" in todos      # total de los dos agentes
        assert "175,0" in bancos and "525,0" not in bancos

    def test_bucket_table_orders_buckets_short_to_long(self, ds, tmp_path):
        html = dcv_bucket_table(ds, tmp_path, {}).html
        assert html.index("Menor a 1Y") < html.index("Entre 2 y 5Y") < html.index("Mayor a 10Y")

    def test_snapshot_stacked_by_agent(self, ds, tmp_path):
        plot = dcv_snapshot_stacked(ds, tmp_path, {"x": "Tipo", "series": "Sector"})
        assert plot.kind == "grouped"
        assert [s.label for s in plot.series] == ["Bancos", "FP y AFC"]
        # Eje X en el orden del correo: PDBC, DAP $, DAP UF, BTP.
        assert [c for c, _ in plot.series[0].points] == ["PDBC", "DAP $", "DAP UF", "BTP"]

    def test_snapshot_stacked_by_bucket_for_one_agent(self, ds, tmp_path):
        plot = dcv_snapshot_stacked(
            ds, tmp_path, {"x": "Bucket", "series": "Tipo", "sector": "Bancos"},
        )
        assert [c for c, _ in plot.series[0].points] == [
            "Menor a 1Y", "Entre 2 y 5Y", "Mayor a 10Y",
        ]
        assert sum(v for s in plot.series for _, v in s.points) == pytest.approx(175.0)

    def test_missing_parquet_returns_none(self, tmp_path):
        ds = _ds("no_existe.parquet", id="variacion_stock_todos")
        assert dcv_portfolio_table(ds, tmp_path, {}) is None
        assert dcv_bucket_table(ds, tmp_path, {}) is None
        assert dcv_snapshot_stacked(ds, tmp_path, {}) is None


@pytest.mark.unit
class TestDcvSpec:
    def test_registered_and_listed(self):
        assert get_spec("dcv") is DCV_SPEC
        assert "dcv" in available_families()

    def test_sections_follow_the_screenshot_order(self):
        """Banners del correo en el orden de las capturas de
        ``data_pipeline/Tipos de informe/Informe DCV`` (1.jpg … 6.jpg)."""
        assert DCV_SPEC.sections() == [
            "Portafolio por agente",
            "Próximos Vencimientos",
            "Todos los instrumentos",
            "Bancos",
            "Fondos de Pensiones y AFC",
            "Fondos Mutuos",
            "Compañías de Seguros",
            "Mandantes y Depósitos de Valores",
            "Corredores de Bolsa y Bolsa de Valores",
            "Otros",
        ]

    def test_every_agent_section_has_table_then_chart(self):
        """Cada agente sale como el original: primero la tabla instrumento x tramo,
        después su apilado por tramo."""
        for section in DCV_SPEC.sections()[2:]:
            blocks = [b for b in DCV_SPEC.blocks if b.section == section]
            assert [b.chart for b in blocks] == ["heatmap_table", "stacked_bar"], section

    def test_blocks_with_source_reference_known_transforms(self):
        for b in DCV_SPEC.blocks:
            if b.status == STATUS_SKIP:
                continue
            assert get_transform(b.transform) is not None, b.title

    def test_skip_blocks_say_which_parquet_is_missing(self):
        # un SKIP sin explicación es indistinguible de un bug: la nota debe decir qué falta
        skipped = [b for b in DCV_SPEC.blocks if b.status == STATUS_SKIP]
        # 2 duraciones + tabla de vencimientos por agente + 2 agentes x 2 bloques.
        assert len(skipped) == 7
        for b in skipped:
            assert b.note.startswith("Falta parquet:"), b.title

    def test_no_common_weekly_cutoff(self):
        """El informe es un CORTE de stocks, no ventanas semanales: forzar corte
        común arrastraría las tablas a la fecha del perfil de vencimientos, que
        vive en el futuro."""
        assert DCV_SPEC.share_weekly_cutoff is False

    def test_charts_do_not_duplicate_the_text(self):
        """Un párrafo por agente: el apilado acompaña a la tabla y no la comenta,
        porque ambos son el MISMO corte visto de dos formas.

        "Vencimientos Totales" es la excepción y sí comenta: la tabla de su
        sección no tiene parquet (SKIP), así que es el único bloque con dato ahí."""
        charts = [
            b for b in DCV_SPEC.blocks
            if b.chart == "stacked_bar" and b.source_id and b.title != "Vencimientos Totales"
        ]
        assert charts and all(b.no_text for b in charts)


# ── Recorte de fechas por bloque (date_from / date_to del spec) ──────────────


@pytest.mark.unit
class TestBlockDateFilter:
    """``date_from``/``date_to`` acotan el PARQUET antes de que la transform lo lea,
    así la "última fecha" del gráfico y todo lo que se deriva de ella (ventanas,
    acumulados, corte citado) respetan el recorte."""

    @pytest.fixture
    def ds(self, tmp_path):
        rows = ", ".join(
            f"(DATE '2026-0{m}-15', 'A', {m * 10.0})" for m in range(1, 7)
        )
        _write(tmp_path / "s.parquet", f"SELECT * FROM (VALUES {rows}) t(Fecha, Tipo, Valor)")
        return _ds("s.parquet", id="serie", unit="US$")

    def _dates(self, ds, tmp_path, date_from="", date_to=""):
        from banks_rag.application.reporting.curated_report import date_filtered_dir
        from banks_rag.application.reporting.parquet_facts import compute_series

        with date_filtered_dir(ds, tmp_path, date_from, date_to) as pdir:
            plot = compute_series(ds, pdir)
        return [iso for s in plot.series for iso, _ in s.points]

    def test_without_filter_yields_the_original_dir(self, ds, tmp_path):
        from banks_rag.application.reporting.curated_report import date_filtered_dir

        with date_filtered_dir(ds, tmp_path) as pdir:
            assert pdir is tmp_path  # sin recorte no se copia nada
        assert len(self._dates(ds, tmp_path)) == 6

    def test_date_from_trims_the_start(self, ds, tmp_path):
        assert self._dates(ds, tmp_path, date_from="2026-04-01") == [
            "2026-04-15", "2026-05-15", "2026-06-15",
        ]

    def test_date_to_trims_the_end(self, ds, tmp_path):
        assert self._dates(ds, tmp_path, date_to="2026-03-31") == [
            "2026-01-15", "2026-02-15", "2026-03-15",
        ]

    def test_both_bounds_are_inclusive(self, ds, tmp_path):
        assert self._dates(ds, tmp_path, date_from="2026-02-15", date_to="2026-04-15") == [
            "2026-02-15", "2026-03-15", "2026-04-15",
        ]

    def test_last_date_of_the_chart_follows_the_cut(self, ds, tmp_path):
        """El punto que importa: recortando, el gráfico cambia cuál es su ÚLTIMA
        fecha. Si el recorte se aplicara después de calcular, el "último dato"
        seguiría siendo junio y las ventanas se medirían contra días invisibles."""
        assert self._dates(ds, tmp_path, date_to="2026-03-31")[-1] == "2026-03-15"

    def test_empty_result_keeps_the_original_data(self, ds, tmp_path):
        # Un rango sin filas dejaría el bloque como "sin serie", indistinguible de
        # un parquet ausente: se ignora el recorte y se avisa por log.
        assert len(self._dates(ds, tmp_path, date_from="2030-01-01")) == 6

    def test_parquet_without_date_column_is_left_alone(self, tmp_path):
        from banks_rag.application.reporting.curated_report import date_filtered_dir

        _write(tmp_path / "flat.parquet", "SELECT * FROM (VALUES ('A', 1.0)) t(Tipo, Valor)")
        ds = _ds("flat.parquet", id="flat")
        with date_filtered_dir(ds, tmp_path, "2026-01-01", "2026-02-01") as pdir:
            assert pdir is tmp_path

    def test_missing_parquet_is_left_alone(self, tmp_path):
        from banks_rag.application.reporting.curated_report import date_filtered_dir

        ds = _ds("no_existe.parquet", id="x")
        with date_filtered_dir(ds, tmp_path, "2026-01-01", "") as pdir:
            assert pdir is tmp_path

    def test_block_filter_reaches_the_transform(self, ds, tmp_path):
        """El recorte declarado en el bloque llega al gráfico dibujado."""
        from banks_rag.application.reporting.curated_report import _process_block
        from banks_rag.application.reporting.report_spec import STATUS_MVP

        block = ReportBlock(
            section="S", title="T", chart="line", status=STATUS_MVP,
            source_id="serie", transform="straight_series", date_to="2026-03-31",
        )
        cb = _process_block(block, entries=[ds], parquet_dir=tmp_path)
        assert cb.render_kind == "chart"
        assert cb.plot.series[0].points[-1][0] == "2026-03-15"

    def test_text_inherits_the_filter_of_the_block_it_comments(self):
        """``spec_date_filters``: el párrafo hereda el recorte del bloque que POSEE
        su slot de texto, no el de un bloque ``no_text`` que use la misma fuente."""
        from banks_rag.application.reporting.curated_report import spec_date_filters
        from banks_rag.application.reporting.report_spec import STATUS_MVP

        spec = FamilyReportSpec(
            family="t", title="T",
            blocks=(
                ReportBlock(section="S", title="con texto", chart="line", status=STATUS_MVP,
                            source_id="serie", transform="straight_series",
                            date_from="2026-02-01", date_to="2026-04-30"),
                ReportBlock(section="S", title="sin texto", chart="line", status=STATUS_MVP,
                            source_id="serie", transform="straight_series",
                            date_from="2020-01-01", no_text=True),
            ),
        )
        assert spec_date_filters(spec) == {"serie": ("2026-02-01", "2026-04-30")}

    def test_blocks_without_filter_are_not_listed(self):
        from banks_rag.application.reporting.curated_report import spec_date_filters

        for spec in (FFMM_SPEC, NR_SPEC, AFP_SPEC, FX_SPEC, DCV_SPEC):
            # Hoy ningún spec recorta fechas: el diccionario vacío confirma que la
            # función es opt-in y no altera los informes actuales.
            assert spec_date_filters(spec) == {}, spec.family
