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
    dcv_cut_dates,
    dcv_heatmap,
    filter_fund,
    get_transform,
    is_known,
    latest_snapshot,
    monthly_var_alloc,
    snapshot_grouped,
    snapshot_stacked,
    stacked_by_bucket,
    wide_lines,
    wide_monthly_bars,
    wide_window_bars,
    window_grouped,
    window_grouped_long,
    window_returns,
    window_stacked_by_cat,
)
from banks_rag.application.reporting.specs import available_families, get_spec
from banks_rag.application.reporting.specs.afp_spec import AFP_SPEC
from banks_rag.application.reporting.specs.ffmm_spec import FFMM_SPEC
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
                        "Mercado SPC (tasas)", "Flujos Spot"]

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

        # AFP no declara weekly_anchor_sections → corte global (todos los source_ids).
        assert not AFP_SPEC.weekly_anchor_sections
        assert weekly_anchor_source_ids(AFP_SPEC) == set(_spec_source_ids(AFP_SPEC))

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
