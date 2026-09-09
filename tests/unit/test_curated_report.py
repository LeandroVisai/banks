"""Unit tests del informe curado por familia (spec + transforms + builder + HTML).

Parquets de prueba con DuckDB en ``tmp_path`` (sin BD ni modelos ni internet),
mismo patrón que test_compute_series.
"""

from __future__ import annotations

import re

import duckdb
import pytest

from banks_rag.application.reporting import (
    build_curated_report,
    render_curated_html,
)
from banks_rag.application.reporting.curated_report import (
    _scale_plot,
    _wide_block_ids,
    fill_synthesis_slot,
    fill_text_slots,
    section_slot_ids,
    section_text_slots,
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
    dcv_duration_scatter,
    dcv_heatmap,
    dcv_portfolio_table,
    dcv_snapshot_stacked,
    dcv_upcoming_maturities_table,
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

    def test_every_source_id_is_in_the_real_catalog(self):
        """Un ``source_id`` que no exista en el catálogo real sale como
        placeholder silencioso en el informe (bug real: "Posición SPC en
        dólares" apuntaba a ``afp_posicion_swap_usd``, que nunca existió — el
        parquet y la entrada del catálogo son ``afp_posicion_spc_usd``)."""
        from banks_rag.infrastructure.sql.parquet_catalog_loader import (
            get_dataset,
            load_parquet_catalog,
        )

        entries = load_parquet_catalog()
        for b in AFP_SPEC.blocks:
            if b.source_id:
                assert get_dataset(entries, b.source_id) is not None, b.source_id


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

    def _retorno_mensual_afp_parquet(self, path) -> None:
        """Réplica de la forma REAL de ``retorno_mensual_afp`` (Fecha x AFP x Fondo
        → Retorno, ya el retorno MENSUAL, no un índice a diferenciar). Incluye una
        fila de dic-2025 (año previo) para ejercitar el recorte YTD."""
        rows = [
            "(DATE '2025-12-31', 'Capital', 'A', 5.0)",
            # ene-2026
            "(DATE '2026-01-31', 'Total',   'A',  1.0)", "(DATE '2026-01-31', 'Total',   'B',  2.0)",
            "(DATE '2026-01-31', 'Capital', 'A',  1.0)", "(DATE '2026-01-31', 'Capital', 'B',  2.0)",
            "(DATE '2026-01-31', 'Cuprum',  'A',  1.5)", "(DATE '2026-01-31', 'Cuprum',  'B',  2.5)",
            # feb-2026
            "(DATE '2026-02-28', 'Total',   'A', -1.0)", "(DATE '2026-02-28', 'Total',   'B',  0.5)",
            "(DATE '2026-02-28', 'Capital', 'A', -1.0)", "(DATE '2026-02-28', 'Capital', 'B',  0.5)",
            "(DATE '2026-02-28', 'Cuprum',  'A',  0.0)", "(DATE '2026-02-28', 'Cuprum',  'B',  1.0)",
            # mar-2026 (último mes)
            "(DATE '2026-03-31', 'Total',   'A',  2.0)", "(DATE '2026-03-31', 'Total',   'B', -0.5)",
            "(DATE '2026-03-31', 'Capital', 'A',  2.0)", "(DATE '2026-03-31', 'Capital', 'B', -0.5)",
            "(DATE '2026-03-31', 'Cuprum',  'A',  1.0)", "(DATE '2026-03-31', 'Cuprum',  'B',  0.5)",
        ]
        _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, AFP, Fondo, Retorno)")

    def test_monthly_bars_by_cat_filters_and_groups_by_month(self, tmp_path):
        """``monthly_returns`` (índice ANCHO a diferenciar) no sirve para un parquet
        LARGO donde el valor YA es el retorno del mes — por eso esta transform
        aparte: filtra AFP=Total y desagrupa por Fondo, X=mes."""
        from banks_rag.application.reporting.series_transforms import monthly_bars_by_cat

        p = tmp_path / "retorno_mensual_afp.parquet"
        self._retorno_mensual_afp_parquet(p)
        ds = _ds("retorno_mensual_afp.parquet", id="retorno_mensual_afp", unit="%")
        plot = monthly_bars_by_cat(ds, tmp_path, {
            "category": "Fondo", "value": "Retorno",
            "filter_col": "AFP", "filter_val": "Total",
            "order": ["A", "B"], "months": 10,
        })
        assert plot.kind == "grouped"
        assert [s.label for s in plot.series] == ["A", "B"]  # orden pedido, Capital/Cuprum afuera
        a = dict(next(s for s in plot.series if s.label == "A").points)
        b = dict(next(s for s in plot.series if s.label == "B").points)
        assert list(a.values()) == pytest.approx([1.0, -1.0, 2.0])   # ene, feb, mar (Total)
        assert list(b.values()) == pytest.approx([2.0, 0.5, -0.5])
        assert len(a) == 3  # dic-2025 no se cuela (era Capital, no Total)

    def test_ytd_grouped_by_cat_compounds_current_year_excludes_total(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import ytd_grouped_by_cat

        p = tmp_path / "retorno_mensual_afp.parquet"
        self._retorno_mensual_afp_parquet(p)
        ds = _ds("retorno_mensual_afp.parquet", id="retorno_mensual_afp", unit="%")
        plot = ytd_grouped_by_cat(ds, tmp_path, {
            "category": "AFP", "group": "Fondo", "value": "Retorno",
            "exclude": ["Total"], "order": ["A", "B"],
        })
        assert plot.kind == "grouped"
        assert set(s.label for s in plot.series) == {"Capital", "Cuprum"}  # Total excluido
        capital = dict(next(s for s in plot.series if s.label == "Capital").points)
        cuprum = dict(next(s for s in plot.series if s.label == "Cuprum").points)
        # Capital A: dic-2025 (5.0) NO cuenta — solo ene/feb/mar 2026: 1.0,-1.0,2.0
        assert capital["A"] == pytest.approx(((1.01 * 0.99 * 1.02) - 1) * 100, abs=1e-6)
        assert capital["B"] == pytest.approx(((1.02 * 1.005 * 0.995) - 1) * 100, abs=1e-6)
        assert cuprum["A"] == pytest.approx(((1.015 * 1.0 * 1.01) - 1) * 100, abs=1e-6)
        assert cuprum["B"] == pytest.approx(((1.025 * 1.01 * 1.005) - 1) * 100, abs=1e-6)

    def test_wide_monthly_diff_bars_diffs_cumulative_levels_not_sums_them(self, tmp_path):
        """``afp_variacion_spc`` viene como NIVEL acumulado por tramo, con cortes
        irregulares (no siempre fin de mes calendario) — la barra de cada mes es
        el corte vs. el corte previo, NO la suma de valores dentro del mes
        (wide_monthly_bars daría un número sin sentido sobre un acumulado)."""
        from banks_rag.application.reporting.series_transforms import wide_monthly_diff_bars

        p = tmp_path / "afp_variacion_spc.parquet"
        rows = [
            "(DATE '2026-01-15', 100.0, 50.0, 150.0)",
            "(DATE '2026-02-20', 120.0, 40.0, 160.0)",
            "(DATE '2026-03-10',  90.0, 70.0, 160.0)",
        ]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows)
               + ') t(Fecha, "1 a 90 dias", "91 a 360 dias", Neto)')
        ds = _ds("afp_variacion_spc.parquet", id="afp_variacion_spc", unit="Millones de USD")
        plot = wide_monthly_diff_bars(ds, tmp_path, {
            "include": ["1 a 90 dias", "91 a 360 dias"], "overlay": ["Neto"], "months": 10,
        })
        assert plot.kind == "grouped"
        assert plot.overlay == ("Neto",)
        assert [s.label for s in plot.series] == ["1 a 90 dias", "91 a 360 dias", "Neto"]
        b1 = dict(next(s for s in plot.series if s.label == "1 a 90 dias").points)
        b2 = dict(next(s for s in plot.series if s.label == "91 a 360 dias").points)
        neto = dict(next(s for s in plot.series if s.label == "Neto").points)
        # ene-2026 (primer corte) no tiene previo con qué diferenciar: no aparece.
        assert list(b1.keys()) == ["20feb26", "10mar26"]
        assert list(b1.values()) == pytest.approx([20.0, -30.0])   # 120-100, 90-120
        assert list(b2.values()) == pytest.approx([-10.0, 30.0])   # 40-50, 70-40
        assert list(neto.values()) == pytest.approx([10.0, 0.0])   # 160-150, 160-160

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
        # un slot de texto por TÓPICO (banner de sección), no uno por gráfico
        assert 'data-text-slot="t:sec:s1"' in html
        assert 'data-text-slot="t:sec:s2"' in html
        assert html.count("data-text-slot") == 2
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
        assert 'data-text-slot="t:sec:s1"></div>' in html

    def test_topic_slot_sits_right_below_its_banner(self, tmp_path):
        """El texto editable va debajo del banner del tópico, ANTES de su primer
        gráfico — no debajo del título de cada gráfico."""
        p = tmp_path / "dur.parquet"
        _categorical_parquet(p)
        entries = [_ds("dur.parquet", id="dur")]
        report = build_curated_report(_spec_for_build(), entries=entries, parquet_dir=tmp_path)
        html = render_curated_html(report)
        banner = html.index('<div class="section-banner">S1</div>')
        slot = html.index('data-text-slot="t:sec:s1"')
        title = html.index('<div class="block-title">Línea</div>')
        assert banner < slot < title


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
class TestTopicTextSlots:
    """Un slot por TÓPICO: id estable desde el nombre de la sección y párrafos del
    LLM (uno por dataset) concatenados dentro del tópico al que pertenecen."""

    def _spec(self) -> FamilyReportSpec:
        return FamilyReportSpec(
            family="t", title="Informe T",
            blocks=(
                ReportBlock(section="Renta Fija (DCV)", title="A", chart="line",
                            status=STATUS_MVP, source_id="a"),
                ReportBlock(section="Renta Fija (DCV)", title="B", chart="line",
                            status=STATUS_MVP, source_id="b"),
                ReportBlock(section="Mercado cambiario", title="C", chart="line",
                            status=STATUS_MVP, source_id="c"),
            ),
        )

    def test_slot_id_is_slug_of_section(self):
        ids = section_slot_ids(self._spec())
        assert ids["Renta Fija (DCV)"] == "t:sec:renta_fija_dcv"
        assert ids["Mercado cambiario"] == "t:sec:mercado_cambiario"

    def test_slot_ids_are_unique_when_slugs_collide(self):
        spec = FamilyReportSpec(
            family="t", title="T",
            blocks=(
                ReportBlock(section="Cobre", title="A", chart="line", status=STATUS_MVP, source_id="a"),
                ReportBlock(section="COBRE!", title="B", chart="line", status=STATUS_MVP, source_id="b"),
            ),
        )
        assert len(set(section_slot_ids(spec).values())) == 2

    def test_paragraphs_are_grouped_by_topic_in_spec_order(self):
        slots = section_text_slots(self._spec(), {"a": "Uno.", "b": "Dos.", "c": "Tres."})
        assert slots["t:sec:renta_fija_dcv"] == "Uno.\n\nDos."
        assert slots["t:sec:mercado_cambiario"] == "Tres."

    def test_missing_paragraphs_are_skipped(self):
        slots = section_text_slots(self._spec(), {"b": "Solo B."})
        assert slots == {"t:sec:renta_fija_dcv": "Solo B."}

    def test_filled_topic_renders_one_p_per_dataset_paragraph(self):
        html = '<div class="section-text" data-text-slot="t:sec:renta_fija_dcv"></div>'
        slots = section_text_slots(self._spec(), {"a": "Uno.", "b": "Dos."})
        out = fill_text_slots(html, slots)
        assert out.count("<p>") == 2


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

        out_dir = tmp_path / "fx" / "no_editable"
        editable_dir = tmp_path / "fx" / "Editable"
        assert mod.build_one("fx", out_dir, editable_dir, editable=True) is True
        assert out_dir.is_dir() and editable_dir.is_dir()
        names = sorted(p.name for p in out_dir.iterdir())
        assert len(names) == 1 and names[0].startswith("fx_") and not names[0].endswith("_editable.html")
        ed_names = sorted(p.name for p in editable_dir.iterdir())
        assert len(ed_names) == 1 and ed_names[0].endswith("_editable.html")

    def test_unknown_family_reports_failure(self, tmp_path):
        mod = self._load_script()
        assert mod.build_one("no_existe", tmp_path, None, editable=False) is False

    def test_resolve_out_dir_formats_familia_placeholder(self, tmp_path):
        mod = self._load_script()
        template = str(tmp_path / "{familia}" / "Editable")
        assert mod._resolve_out_dir(template, "fx") == tmp_path / "fx" / "Editable"

    def test_resolve_out_dir_without_placeholder_appends_family(self, tmp_path):
        mod = self._load_script()
        assert mod._resolve_out_dir(str(tmp_path), "fx") == tmp_path / "fx"


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


def _dcv_duration_parquets(dir_path) -> None:
    """``duracion_iif.parquet`` (PDBC, DAP) + ``duracion_rf.parquet`` (BTP):
    mismos instrumentos/agentes/monedas que ``_dcv_parquet``, para poder cruzar
    Monto x Duración en la tabla de portafolio. Snapshot de una sola fecha (como
    el dato real: "duración hoy", sin histórico)."""
    iif_rows = [
        "(DATE '2026-07-20', 'PDBC', 'Bancos', 'CLP', 0.5)",
        "(DATE '2026-07-20', 'PDBC', 'AFP', 'CLP', 1.5)",
        "(DATE '2026-07-20', 'DAP', 'Bancos', 'CLP', 2.0)",
        "(DATE '2026-07-20', 'DAP', 'AFP', 'CLP', 3.0)",
        "(DATE '2026-07-20', 'DAP', 'Bancos', 'UF', 4.0)",
        "(DATE '2026-07-20', 'DAP', 'AFP', 'UF', 5.0)",
    ]
    _write(dir_path / "duracion_iif.parquet",
           "SELECT * FROM (VALUES " + ", ".join(iif_rows) + ") t(Fecha, Tipo, Sector, Moneda, Duracion)")
    rf_rows = [
        "(DATE '2026-07-20', 'BTP', 'Bancos', 'CLP', 6.0)",
        "(DATE '2026-07-20', 'BTP', 'AFP', 'CLP', 7.0)",
    ]
    _write(dir_path / "duracion_rf.parquet",
           "SELECT * FROM (VALUES " + ", ".join(rf_rows) + ") t(Fecha, Tipo, Sector, Moneda, Duracion)")


def _dcv_maturity_parquets(dir_path) -> None:
    """4 parquets de "Próximos Vencimientos": 3 snapshots (T/T+1/Acum 5d.) +
    ``vencimientos_futuros_instrumento`` (mensual), con filas de agosto-2026,
    septiembre-2026 y agosto-2025 (para probar que el filtro de mes en curso
    -``as_of``- solo toma el mes/año pedido)."""
    hoy_rows = [
        "(DATE '2026-08-10', 'PDBC', 'Bancos', 'CLP', 100.0)",
        "(DATE '2026-08-10', 'PDBC', 'AFP', 'CLP', 50.0)",
        "(DATE '2026-08-10', 'DAP', 'Bancos', 'CLP', 20.0)",
        "(DATE '2026-08-10', 'DAP', 'AFP', 'UF', 10.0)",
        "(DATE '2026-08-10', 'Otros', 'Bancos', 'CLP', 30.0)",
    ]
    _write(dir_path / "vencimientos_hoy.parquet",
           "SELECT * FROM (VALUES " + ", ".join(hoy_rows) + ") t(Vencimiento, Tipo, Sector, Moneda, Stock_USD)")

    t1_rows = [
        "(DATE '2026-08-11', 'PDBC', 'Bancos', 'CLP', 40.0)",
        "(DATE '2026-08-11', 'Otros', 'AFP', 'CLP', 15.0)",
    ]
    _write(dir_path / "vencimientos_t_mas_uno.parquet",
           "SELECT * FROM (VALUES " + ", ".join(t1_rows) + ") t(Vencimiento, Tipo, Sector, Moneda, Stock_USD)")

    c5_rows = [
        "(DATE '2026-08-06', 'PDBC', 'Bancos', 'CLP', 200.0)",
        "(DATE '2026-08-06', 'DAP', 'Bancos', 'UF', 25.0)",
    ]
    _write(dir_path / "vencimientos_cinco_dias.parquet",
           "SELECT * FROM (VALUES " + ", ".join(c5_rows) + ") t(Fecha, Tipo, Sector, Moneda, Stock_USD)")

    mensual_rows = [
        "('2026', 'ago', 'PDBC', 'Bancos', 'CLP', 500.0)",
        "('2026', 'sep', 'PDBC', 'Bancos', 'CLP', 999.0)",   # otro mes: no debe entrar
        "('2026', 'ago', 'BB', 'Bancos', 'CLP', 300.0)",     # -> RF
        "('2025', 'ago', 'PDBC', 'Bancos', 'CLP', 111.0)",   # otro año: no debe entrar
    ]
    _write(dir_path / "vencimientos_futuros_instrumento.parquet",
           "SELECT * FROM (VALUES " + ", ".join(mensual_rows)
           + ') t("Año", Mes_label, Tipo, Sector, Moneda, Stock_USD)')


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

    def test_portfolio_table_without_duration_files_omits_the_column(self, ds, tmp_path):
        """Comportamiento histórico intacto: sin duracion_iif/duracion_rf en el
        directorio, la tabla queda igual que antes (solo Monto y % port.)."""
        html = dcv_portfolio_table(ds, tmp_path, {}).html
        assert "Dur." not in html

    def test_portfolio_table_adds_weighted_average_duration(self, ds, tmp_path):
        """La columna Dur. cruza Monto (variacion_instrumento_todos_plazo) con
        Duración (duracion_iif/duracion_rf, parquets aparte, mismo grano). El
        Total de fila/columna es un PROMEDIO PONDERADO por Monto, no una suma."""
        _dcv_duration_parquets(tmp_path)
        html = dcv_portfolio_table(ds, tmp_path, {}).html
        assert "Dur." in html
        # Bancos: PDBC=100·0,5 + DAP$=40·2,0 + DAP UF=10·4,0 + BTP=25·6,0 = 320 / 175
        assert "1,83" in html
        # AFP: 200·1,5 + 80·3,0 + 20·5,0 + 50·7,0 = 990 / 350
        assert "2,83" in html
        # Columna Total, fila PDBC: 100·0,5 + 200·1,5 = 350 / 300 (Monto total PDBC)
        assert "1,17" in html

    def test_dcv_duration_scatter_builds_categorical_points_by_agent(self, tmp_path):
        _dcv_duration_parquets(tmp_path)
        ds_iif = _ds("duracion_iif.parquet", id="duracion_iif", unit="Años")
        plot = dcv_duration_scatter(ds_iif, tmp_path, {})
        assert plot.kind == "grouped"
        assert {s.label for s in plot.series} == {"Bancos", "FP y AFC"}
        bancos = dict(next(s for s in plot.series if s.label == "Bancos").points)
        assert bancos == {"PDBC": pytest.approx(0.5), "DAP $": pytest.approx(2.0),
                          "DAP UF": pytest.approx(4.0)}
        # eje X en el orden del correo (PDBC antes que DAP)
        cats = [c for c, _ in next(s for s in plot.series if s.label == "Bancos").points]
        assert cats == ["PDBC", "DAP $", "DAP UF"]

    def test_dcv_duration_scatter_missing_parquet_returns_none(self, tmp_path):
        ds = _ds("no_existe.parquet", id="duracion_iif")
        assert dcv_duration_scatter(ds, tmp_path, {}) is None

    def test_render_plot_svg_draws_dots_not_bars_for_point_chart(self, tmp_path):
        """``chart='point'`` sobre un PlotData ``kind='grouped'`` (duración por
        agente) dibuja puntos, no barras — a diferencia de ``grouped_bar``."""
        from banks_rag.application.reporting.svg_chart import renders_natively

        _dcv_duration_parquets(tmp_path)
        ds_iif = _ds("duracion_iif.parquet", id="duracion_iif", unit="Años")
        plot = dcv_duration_scatter(ds_iif, tmp_path, {})
        svg = render_plot_svg(plot, chart="point")
        # 2 agentes x 3 instrumentos (PDBC, DAP $, DAP UF) = 6 puntos, sin barras
        # (los <rect> que aparecen son el fondo blanco + los swatches de leyenda).
        assert svg.count("<circle") == 6
        assert "Bancos" in svg and "FP y AFC" in svg
        assert renders_natively("grouped", "point") is True

    def test_real_dcv_duration_blocks_render_natively_end_to_end(self, tmp_path):
        """Extremo a extremo con los bloques REALES de DCV_SPEC (no sintéticos)."""
        _dcv_parquet(tmp_path / "variacion_instrumento_todos_plazo.parquet")
        _dcv_duration_parquets(tmp_path)
        wanted = {"Portafolio por agente", "Duración agentes IIF", "Duración agentes RF"}
        blocks = tuple(b for b in DCV_SPEC.blocks if b.title in wanted)
        assert len(blocks) == 3
        entries = [
            _ds("variacion_instrumento_todos_plazo.parquet",
                id="variacion_instrumento_todos_plazo", unit="US$ Mill."),
            _ds("duracion_iif.parquet", id="duracion_iif", unit="Años"),
            _ds("duracion_rf.parquet", id="duracion_rf", unit="Años"),
        ]
        spec = FamilyReportSpec(family="t", title="T", blocks=blocks)
        report = build_curated_report(spec, entries=entries, parquet_dir=tmp_path)
        for cb in report.blocks:
            assert cb.render_kind == "chart", cb.block.title
            assert cb.preliminary is False, cb.block.title
        portafolio = next(cb for cb in report.blocks if cb.block.title == "Portafolio por agente")
        assert "Dur." in portafolio.body_html

    def test_dcv_maturity_label_collapses_non_pdbc_dap_to_rf(self):
        from banks_rag.application.reporting.series_transforms import _dcv_maturity_label

        assert _dcv_maturity_label("PDBC", "CLP") == "PDBC"
        assert _dcv_maturity_label("DAP", "CLP") == "DAP $"
        assert _dcv_maturity_label("DAP", "UF") == "DAP UF"
        assert _dcv_maturity_label("Otros", "CLP") == "RF"
        # instrumento más fino del parquet mensual (vencimientos_futuros_instrumento
        # trae BB/BCCh/BE/BTP/BTU/Letras MdH): igual colapsa, por consistencia con
        # los 3 snapshots (que solo declaran "Otros").
        assert _dcv_maturity_label("BB", "CLP") == "RF"

    def test_upcoming_maturities_table_builds_grid_with_month_filter(self, tmp_path):
        _dcv_maturity_parquets(tmp_path)
        ds = _ds("vencimientos_hoy.parquet", id="vencimientos_hoy", unit="Millones de USD")
        html = dcv_upcoming_maturities_table(ds, tmp_path, {"as_of": "2026-08-15"}).html
        assert "Totales" in html and "Bancos" in html and "FP y AFC" in html
        assert "100,0" in html      # T · PDBC · Bancos
        assert "40,0" in html       # T+1 · PDBC · Bancos
        assert "200,0" in html      # Acum 5d. · PDBC · Bancos
        assert "500,0" in html      # Mes · PDBC · Bancos (agosto-2026, el mes de as_of)
        assert "999,0" not in html  # septiembre-2026: otro mes, no entra
        assert "111,0" not in html  # agosto-2025: otro año, no entra
        assert "RF" in html         # Otros (T/T+1) y BB (Mes) colapsan ahí
        assert "DAP $" in html and "DAP UF" in html

    def test_upcoming_maturities_table_missing_all_parquets_returns_none(self, tmp_path):
        ds = _ds("vencimientos_hoy.parquet", id="vencimientos_hoy")
        assert dcv_upcoming_maturities_table(ds, tmp_path, {}) is None

    def test_real_dcv_maturities_block_renders_natively_end_to_end(self, tmp_path):
        """Extremo a extremo con el bloque REAL de DCV_SPEC. Sin ``as_of``
        explícito (el bloque real tampoco lo fija, usa ``date.today()``): solo
        se verifican las columnas T/T+1/Acum 5d. (snapshots, no dependen del mes
        en curso), no la columna Mes."""
        _dcv_maturity_parquets(tmp_path)
        block = next(b for b in DCV_SPEC.blocks if b.title.startswith("Próximos vencimientos por agente"))
        entries = [_ds("vencimientos_hoy.parquet", id="vencimientos_hoy", unit="Millones de USD")]
        spec = FamilyReportSpec(family="t", title="T", blocks=(block,))
        report = build_curated_report(spec, entries=entries, parquet_dir=tmp_path)
        cb = report.blocks[0]
        assert cb.render_kind == "chart"
        assert cb.preliminary is False
        assert "100,0" in cb.body_html
        assert "Acum 5d." in cb.body_html


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
        # sin bloques SKIP: duración y próximos vencimientos ya tienen parquet
        # (duracion_iif/duracion_rf, vencimientos_hoy/t_mas_uno/cinco_dias +
        # vencimientos_futuros_instrumento) y pasaron a MVP.
        assert len(skipped) == 0
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


# ── Informe Cambiario AM (familia cambiarioam) ───────────────────────────────
#
# El informe todavía no tiene sus parquets en data_pipeline/parquet (los genera
# scripts/build_cambiario_parquets.py desde el Excel del DOMA + el DW), así que
# acá se arman a mano con el mismo esquema declarado en el catálogo.

def _cam_ohlc_parquet(path) -> None:
    """20 sesiones OHLC: alternan cierre al alza y a la baja para cubrir ambos
    colores de vela."""
    rows = []
    for i in range(20):
        day = f"2026-07-{i + 1:02d}"
        op = 900.0 + i
        cl = op + (4.0 if i % 2 == 0 else -4.0)
        rows.append(f"(DATE '{day}', {op}, {max(op, cl) + 2}, {min(op, cl) - 2}, {cl}, {500.0 + i})")
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "Apertura", "Máximo", "Mínimo", "Cierre", "Monto transado")')


def _cam_wide_parquet(path, col: str, values: list[float], *, start_day: int = 1) -> None:
    rows = [
        f"(DATE '2026-07-{start_day + i:02d}', {v})"
        for i, v in enumerate(values)
    ]
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + f') t(Fecha, "{col}")')


@pytest.mark.unit
class TestCambiarioAmSpec:
    def test_registered_and_listed(self):
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        assert get_spec("cambiarioam") is CAMBIARIOAM_SPEC
        assert "cambiarioam" in available_families()

    def test_sections_follow_the_dashboard_order(self):
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        assert CAMBIARIOAM_SPEC.sections() == [
            "Drivers del día", "CLP · Análisis", "No Residentes", "Monedas & Carry",
            "Cobre", "Dólar · DXY", "Expectativas", "Otros",
        ]

    def test_blocks_reference_known_transforms(self):
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        for b in CAMBIARIOAM_SPEC.blocks:
            assert get_transform(b.transform) is not None, b.title

    def test_every_source_id_is_in_the_catalog(self):
        """Un source_id que no esté en el catálogo sale como placeholder silencioso:
        el spec y el catálogo tienen que moverse juntos."""
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC
        from banks_rag.infrastructure.sql.parquet_catalog_loader import (
            get_dataset,
            load_parquet_catalog,
        )

        entries = load_parquet_catalog()
        for b in CAMBIARIOAM_SPEC.blocks:
            assert get_dataset(entries, b.source_id) is not None, b.source_id

    def test_dual_axis_blocks_declare_right_axis(self):
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        dual = [b for b in CAMBIARIOAM_SPEC.blocks if b.chart == "dual_axis"]
        assert len(dual) >= 8
        for b in dual:
            assert b.params.get("right_axis"), b.title

    def test_sr_tables_pair_with_their_chart_and_carry_no_text(self):
        """Cada tabla de percentiles va pegada a su gráfico, con la MISMA fuente y
        ventana, y sin párrafo propio (el comentario se escribe en el gráfico)."""
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        blocks = list(CAMBIARIOAM_SPEC.blocks)
        tables = [(i, b) for i, b in enumerate(blocks) if b.transform == "cam_sr_table"]
        assert len(tables) == 4  # CLP, NR, cobre y DXY
        for i, table in tables:
            chart = blocks[i - 1]
            assert chart.transform == "cam_sr_percentiles"
            assert chart.source_id == table.source_id
            assert chart.params["months"] == table.params["months"]
            assert table.no_text

    def test_no_duplicate_charts_or_titles(self):
        import json

        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        sigs = [
            (b.source_id, b.transform, json.dumps(b.params or {}, sort_keys=True, default=str))
            for b in CAMBIARIOAM_SPEC.blocks
        ]
        assert len(sigs) == len(set(sigs)), "gráfico repetido"
        titles = [b.title for b in CAMBIARIOAM_SPEC.blocks]
        assert len(titles) == len(set(titles)), "título repetido"


@pytest.mark.unit
class TestCambiarioTransforms:
    def test_candlestick_returns_the_four_ohlc_series(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_candlestick

        _cam_ohlc_parquet(tmp_path / "ohlc.parquet")
        ds = _ds("ohlc.parquet", id="cam_clp_ohlc", unit="CLP/USD")
        plot = cam_candlestick(ds, tmp_path, {"months": 12})
        assert [s.label for s in plot.series] == ["Apertura", "Máximo", "Mínimo", "Cierre"]
        # El monto transado del parquet NO entra: vive en su propio gráfico.
        assert all(s.label != "Monto transado" for s in plot.series)

    def test_candlestick_svg_colors_by_direction(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_candlestick
        from banks_rag.application.reporting.svg_chart import render_plot_svg, renders_natively

        _cam_ohlc_parquet(tmp_path / "ohlc.parquet")
        ds = _ds("ohlc.parquet", id="cam_clp_ohlc", unit="CLP/USD")
        svg = render_plot_svg(cam_candlestick(ds, tmp_path, {"months": 12}), chart="candlestick")
        assert svg.count("#0a8a5f") >= 10 and svg.count("#c8102e") >= 10  # alza y baja
        assert renders_natively("timeseries", "candlestick")

    def test_candle_table_soporte_resistencia_are_p25_p75(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_candle_table

        _cam_ohlc_parquet(tmp_path / "ohlc.parquet")
        ds = _ds("ohlc.parquet", id="cam_clp_ohlc", unit="CLP/USD")
        table = cam_candle_table(ds, tmp_path, {"months": 12})
        assert isinstance(table, HtmlTable)
        for label in ("Máximo del día", "Mínimo del día", "Soporte", "Resistencia"):
            assert label in table.html

    def test_sr_percentiles_draw_flat_levels_matching_the_table(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import (
            cam_sr_percentiles,
            cam_sr_table,
        )

        values = [float(v) for v in range(100, 121)]
        _cam_wide_parquet(tmp_path / "sr.parquet", "CLP Cierre", values)
        ds = _ds("sr.parquet", id="cam_clp_sr", unit="CLP/USD")
        plot = cam_sr_percentiles(ds, tmp_path, {"months": 120})

        assert plot.series[0].label == "CLP Cierre"
        levels = plot.series[1:]
        assert [s.label.split(" ·")[0] for s in levels] == ["P10", "P25", "P50", "P75", "P90", "P100"]
        for s in levels:  # cada nivel es una horizontal de extremo a extremo
            assert len(s.points) == 2
            assert s.points[0][1] == s.points[1][1]
        # La mediana de 100..120 es 110: el gráfico y la tabla citan el mismo nivel.
        p50 = next(s for s in levels if s.label.startswith("P50"))
        assert p50.points[0][1] == pytest.approx(110.0)
        assert "110" in cam_sr_table(ds, tmp_path, {"months": 120}).html

    def test_bollinger_bands_wrap_the_moving_average(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_bollinger

        _cam_wide_parquet(tmp_path / "bb.parquet", "CLP Cierre",
                          [900.0 + (i % 5) for i in range(25)])
        ds = _ds("bb.parquet", id="cam_clp_bollinger", unit="CLP/USD")
        plot = cam_bollinger(ds, tmp_path, {"period": 10})
        labels = [s.label for s in plot.series]
        assert labels == ["Banda superior (10d)", "Media móvil 10d", "Banda inferior (10d)", "CLP Cierre"]
        upper, ma, lower = (dict(s.points) for s in plot.series[:3])
        for iso in ma:
            assert lower[iso] < ma[iso] < upper[iso]

    def test_rsi_bands_add_the_70_30_thresholds(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_rsi_bands

        _cam_wide_parquet(tmp_path / "rsi.parquet", "RSI Cobre", [40.0, 55.0, 72.0, 61.0])
        ds = _ds("rsi.parquet", id="cam_rsi_cobre", unit="RSI")
        plot = cam_rsi_bands(ds, tmp_path, {"levels": [70, 30]})
        assert [s.label for s in plot.series] == [
            "RSI Cobre", "Sobrecompra · 70", "Sobreventa · 30",
        ]
        assert [v for _d, v in plot.series[1].points] == [70.0, 70.0]

    def test_base100_rebases_each_series_on_its_first_point(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_base100

        rows = [f"(DATE '2026-07-{i + 1:02d}', {800.0 + i * 8}, {20.0 + i})" for i in range(10)]
        _write(tmp_path / "mon.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "CLP", "MXN")')
        ds = _ds("mon.parquet", id="cam_monedas_latam", unit="paridad")
        plot = cam_base100(ds, tmp_path, {"months": 120})
        for s in plot.series:  # toda serie arranca EXACTAMENTE en 100
            assert s.points[0][1] == pytest.approx(100.0)
        assert plot.unit == "Índice base 100"

    def test_signed_bars_split_by_sign_so_each_bar_gets_its_color(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_signed_bars

        _write(tmp_path / "var.parquet",
               "SELECT * FROM (VALUES ('Chile', 1.5), ('Brasil', -2.0), ('Peru', 0.5)) t(Pais, Variacion)")
        ds = _ds("var.parquet", id="cam_monedas_variacion_dia", unit="%")
        plot = cam_signed_bars(ds, tmp_path, {"category": "Pais", "value": "Variacion"})
        assert plot.kind == "grouped"
        pos, neg = plot.series
        assert dict(pos.points)["Chile"] == pytest.approx(1.5)
        assert dict(neg.points)["Brasil"] == pytest.approx(-2.0)
        # Orden descendente por valor, como el gráfico original.
        assert [c for c, _v in pos.points] == ["Chile", "Peru", "Brasil"]

    def test_spread_expanding_computes_bp_and_running_average(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_spread_expanding

        rows = [
            "(DATE '2026-07-01', 5.0, 4.0)",
            "(DATE '2026-07-02', 5.0, 3.0)",
        ]
        _write(tmp_path / "tpm.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "TPM Chile 1Y", "TPM US 1Y")')
        ds = _ds("tpm.parquet", id="cam_tpm_spread", unit="%")
        plot = cam_spread_expanding(ds, tmp_path, {
            "minuend": "TPM Chile 1Y", "subtrahend": "TPM US 1Y", "scale": 100.0,
        })
        spread = dict(plot.series[2].points)
        avg = dict(plot.series[3].points)
        assert spread["2026-07-01"] == pytest.approx(100.0)   # (5-4)*100 pb
        assert spread["2026-07-02"] == pytest.approx(200.0)
        assert avg["2026-07-02"] == pytest.approx(150.0)      # promedio acumulado

    def test_fixing_stacked_anchors_on_the_last_day_with_data(self, tmp_path):
        """El original filtraba por ``Timestamp.today()`` y quedaba vacío en feriados;
        acá el corte es el último día CON dato."""
        from banks_rag.application.reporting.series_transforms import cam_fixing_stacked

        rows = [
            "(DATE '2026-07-01', 'Santander', 'AFP', 10.0)",
            "(DATE '2026-07-02', 'Total', 'AFP', 30.0)",
            "(DATE '2026-07-02', 'Santander', 'AFP', 20.0)",
            "(DATE '2026-07-02', 'Santander', 'FFMM', 5.0)",
        ]
        _write(tmp_path / "fix.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Informante, Sector, Pos_neta)")
        ds = _ds("fix.parquet", id="cam_fixing_bancos", unit="MM USD")
        plot = cam_fixing_stacked(ds, tmp_path, {})
        assert "02-jul-2026" in plot.date_note
        assert plot.overlay == ("Total agente",)
        total = dict(next(s for s in plot.series if s.label == "Total agente").points)
        assert total["Santander"] == pytest.approx(25.0)  # 20 AFP + 5 FFMM
        # "Total" va primero en el eje X, como en el dashboard original.
        assert plot.series[0].points[0][0] == "Total"

    def test_market_table_flags_direction_per_driver(self, tmp_path):
        """Cobre al alza APRECIA el peso; el DXY al alza lo DEPRECIA."""
        from banks_rag.application.reporting.series_transforms import cam_market_table

        rows = [
            "(DATE '2026-07-01', 400.0, 100.0)",
            "(DATE '2026-07-02', 410.0, 101.0)",
        ]
        _write(tmp_path / "drv.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "Cobre", "DXY")')
        ds = _ds("drv.parquet", id="cam_drivers_snapshot", unit="niveles")
        table = cam_market_table(ds, tmp_path, {"indicators": [
            {"column": "Cobre", "label": "Cobre", "mode": "pct", "clp": +1},
            {"column": "DXY", "label": "DXY", "mode": "pct", "clp": -1},
        ]})
        assert isinstance(table, HtmlTable)
        assert "Aprecia" in table.html and "Deprecia" in table.html

    def test_gamma_heatmap_orders_strikes_descending(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_gamma_heatmap

        rows = [
            "(DATE '2026-07-10', 930.0, 5.0)",
            "(DATE '2026-07-10', 935.0, 9.0)",
            "(DATE '2026-07-17', 930.0, 2.0)",
        ]
        _write(tmp_path / "gam.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Vencimiento, Strike, Gamma)")
        ds = _ds("gam.parquet", id="cam_gamma_heatmap", unit="MM USD")
        plot = cam_gamma_heatmap(ds, tmp_path, {"row": "Strike", "value": "Gamma"})
        assert plot.kind == "heatmap"
        assert [s.label for s in plot.series] == ["935,0", "930,0"]  # strike descendente

    def test_intraday_lines_keep_the_time_component(self, tmp_path):
        """Sin la hora, los puntos de un mismo día colapsan sobre la misma abscisa
        y el gráfico intradía sale como una barra vertical."""
        from banks_rag.application.reporting.series_transforms import cam_lines

        rows = [
            "(TIMESTAMP '2026-07-10 09:00:00', 100.0)",
            "(TIMESTAMP '2026-07-10 12:30:00', 101.0)",
        ]
        _write(tmp_path / "intra.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "CLP")')
        ds = _ds("intra.parquet", id="cam_clp_intradia", unit="Índice base 100")
        plot = cam_lines(ds, tmp_path, {"keep_time": True})
        isos = [iso for iso, _v in plot.series[0].points]
        assert isos == ["2026-07-10T09:00", "2026-07-10T12:30"]

        from banks_rag.application.reporting.svg_chart import _date_ord

        assert _date_ord(isos[0]) != _date_ord(isos[1])


@pytest.mark.unit
class TestIpcTreemapTransform:
    """``ipc_treemap``: división (exterior) → grupo (interior), área=Ponderacion,
    color=Variacion. Guard de regresión del bug real que produjo esta prueba: el
    extractor (``scripts/ingest/from_excel.py``) renombraba ``Glosa``→``Grupo``
    con ``DataFrame.rename()`` sobre un frame que YA tenía una columna "Grupo"
    (el código numérico), y pandas no fusiona duplicados — el parquet salía con
    dos columnas "Grupo" y DuckDB desambiguaba la segunda como "Grupo_1", que la
    transform nunca leía."""

    def _write_treemap(self, tmp_path):
        rows = [
            ("'Alimentos'", "'Alimentos'", 20.0, 1.5),
            ("'Alimentos'", "'Bebidas'", 2.0, 0.9),
            ("'Vivienda'", "'Arriendo'", 10.0, -0.4),
        ]
        values = ", ".join(f"({d}, {g}, {p}, {v})" for d, g, p, v in rows)
        _write(tmp_path / "treemap.parquet",
              f'SELECT * FROM (VALUES {values}) t(Division, Grupo, Ponderacion, Variacion)')
        return _ds("treemap.parquet", id="ipc_canasta_treemap", unit="%")

    def test_one_series_per_division_points_are_grupo_and_ponderacion(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import ipc_treemap

        ds = self._write_treemap(tmp_path)
        plot = ipc_treemap(ds, tmp_path, {})
        assert plot.kind == "hierarchy"
        by_label = {s.label: s.points for s in plot.series}
        assert set(by_label) == {"Alimentos", "Vivienda"}
        assert dict(by_label["Alimentos"]) == {"Alimentos": 20.0, "Bebidas": 2.0}
        assert dict(by_label["Vivienda"]) == {"Arriendo": 10.0}

    def test_node_color_aligned_by_position_with_points(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import ipc_treemap

        ds = self._write_treemap(tmp_path)
        plot = ipc_treemap(ds, tmp_path, {})
        alimentos = next(s for s in plot.series if s.label == "Alimentos")
        # mismo orden que .points: [Alimentos=1.5, Bebidas=0.9]
        assert [leaf for leaf, _v in alimentos.points] == ["Alimentos", "Bebidas"]
        assert plot.node_color["Alimentos"] == [1.5, 0.9]

    def test_renders_to_svg_without_error(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import ipc_treemap
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        ds = self._write_treemap(tmp_path)
        svg = render_plot_svg(ipc_treemap(ds, tmp_path, {}), chart="treemap")
        assert svg is not None and "<svg" in svg

    def test_zero_or_negative_ponderacion_is_dropped(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import ipc_treemap

        _write(tmp_path / "treemap.parquet",
              "SELECT * FROM (VALUES ('Alimentos', 'Alimentos', 0.0, 1.5), "
              "('Alimentos', 'Bebidas', 2.0, 0.9)) t(Division, Grupo, Ponderacion, Variacion)")
        ds = _ds("treemap.parquet", id="ipc_canasta_treemap", unit="%")
        plot = ipc_treemap(ds, tmp_path, {})
        assert [leaf for leaf, _v in plot.series[0].points] == ["Bebidas"]

    def test_missing_parquet_returns_none(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import ipc_treemap

        ds = _ds("nope.parquet", id="ipc_canasta_treemap", unit="%")
        assert ipc_treemap(ds, tmp_path, {}) is None


@pytest.mark.unit
class TestAxisScaling:
    """El eje Y de una serie temporal: base 0 para flujos, ajustado para niveles."""

    def _svg(self, values: list[float], *, zero_base: bool) -> str:
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        pts = [(f"2026-07-{i + 1:02d}", v) for i, v in enumerate(values)]
        plot = PlotData("t", "line", "timeseries", "CLP/USD",
                        [PlotSeries(label="CLP", points=pts)], zero_base=zero_base)
        return render_plot_svg(plot, chart="line")

    def test_default_still_anchors_the_axis_on_zero(self):
        """Comportamiento histórico intacto: los informes de flujos siguen viendo el 0."""
        assert "0,00" in self._svg([930.0, 935.0, 928.0], zero_base=True)

    def test_zero_base_false_fits_the_axis_to_the_data(self):
        """Un tipo de cambio en 930 con base 0 queda aplastado contra el borde."""
        svg = self._svg([930.0, 935.0, 928.0], zero_base=False)
        assert "0,00" not in svg
        assert "930" in svg or "935" in svg

    def test_intraday_ticks_show_the_hour(self):
        """Una serie de una sola jornada repetía la misma fecha en los cinco ticks."""
        from banks_rag.application.reporting.svg_chart import _fmt_date

        assert _fmt_date("2026-07-10") == "10-07-26"
        assert _fmt_date("2026-07-10T14:35") == "10-07 14:35"
        # Medianoche exacta = parquet diario guardado como TIMESTAMP: sin hora.
        assert _fmt_date("2026-07-10T00:00:00") == "10-07-26"


# ── Layout "grid" (opt-in por spec) ──────────────────────────────────────────

@pytest.mark.unit
class TestGridLayout:
    def test_existing_families_default_to_stack(self):
        """El campo es nuevo: si alguna familia existente lo declarara sin
        querer, su HTML cambiaría de un día para otro."""
        for spec in (FFMM_SPEC, NR_SPEC, AFP_SPEC, FX_SPEC, DCV_SPEC):
            assert spec.layout == "stack", spec.family

    def test_cambiarioam_opts_into_grid(self):
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        assert CAMBIARIOAM_SPEC.layout == "grid"

    def test_stack_render_has_no_grid_markup(self, tmp_path):
        """Con layout por defecto, el HTML no debe contener el MARKUP de grilla
        (la CSS del sistema de grilla es compartida y siempre se emite, muerta
        si no se usa; lo que no debe aparecer son los elementos)."""
        spec = _spec_for_build()
        report = build_curated_report(spec, entries=[], parquet_dir=tmp_path)
        html = render_curated_html(report)
        assert '<div class="cards-grid">' not in html
        assert '<div class="card"' not in html
        assert '<div class="card-companion">' not in html

    def test_group_cards_merges_only_matching_companion(self):
        from banks_rag.application.reporting.curated_report import CuratedBlock, _group_cards

        chart_a = ReportBlock(section="S", title="A", chart="line", status=STATUS_MVP, source_id="x")
        table_a = ReportBlock(section="S", title="A tabla", chart="heatmap_table", status=STATUS_MVP,
                              source_id="x", no_text=True)
        chart_b = ReportBlock(section="S", title="B", chart="line", status=STATUS_MVP, source_id="y")
        # no_text que NO acompaña a nada (fuente distinta a la del bloque anterior):
        # no debe fundirse.
        orphan = ReportBlock(section="S", title="Huérfano", chart="line", status=STATUS_MVP,
                             source_id="z", no_text=True)

        blocks = [
            CuratedBlock(chart_a, "chart", "<svg-a/>"),
            CuratedBlock(table_a, "chart", "<table-a/>"),
            CuratedBlock(chart_b, "chart", "<svg-b/>"),
            CuratedBlock(orphan, "chart", "<svg-orphan/>"),
        ]
        cards = _group_cards(blocks)
        assert len(cards) == 3
        assert cards[0].main.block.title == "A" and cards[0].companion.block.title == "A tabla"
        assert cards[1].main.block.title == "B" and cards[1].companion is None
        assert cards[2].main.block.title == "Huérfano" and cards[2].companion is None

    def test_grid_render_wraps_section_and_widens_the_odd_card(self):
        from banks_rag.application.reporting.curated_report import CuratedReport

        blocks = [
            ReportBlock(section="S", title="Uno", chart="line", status=STATUS_MVP, source_id="a"),
            ReportBlock(section="S", title="Dos", chart="line", status=STATUS_MVP, source_id="b"),
            ReportBlock(section="S", title="Tres", chart="line", status=STATUS_MVP, source_id="c"),
        ]
        spec = FamilyReportSpec(family="t", title="T", blocks=tuple(blocks), layout="grid")
        cbs = [_curated_block_of(b) for b in blocks]
        report = CuratedReport(spec=spec, generated_at="now", blocks=cbs)
        html = render_curated_html(report)

        assert html.count('<div class="cards-grid">') == 1
        assert html.count('data-chart="line"') == 3  # 3 tarjetas, ninguna fundida
        card_divs = re.findall(r'<div class="card(?: card-wide)?"', html)
        assert len(card_divs) == 3
        # Solo la ÚLTIMA (impar) se ensancha, no las tres.
        assert card_divs.count('<div class="card card-wide"') == 1

    def test_grid_companion_table_lives_inside_the_chart_card(self):
        from banks_rag.application.reporting.curated_report import CuratedReport

        chart = ReportBlock(section="S", title="Gráfico", chart="line", status=STATUS_MVP, source_id="a")
        table = ReportBlock(section="S", title="Niveles", chart="heatmap_table", status=STATUS_MVP,
                            source_id="a", no_text=True)
        spec = FamilyReportSpec(family="t", title="T", blocks=(chart, table), layout="grid")
        cbs = [_curated_block_of(chart), _curated_block_of(table)]
        report = CuratedReport(spec=spec, generated_at="now", blocks=cbs)
        html = render_curated_html(report)

        card_divs = re.findall(r'<div class="card(?: card-wide)?"', html)
        assert len(card_divs) == 1  # UNA sola tarjeta para las dos piezas
        assert html.count('data-source="a"') == 1  # un solo elemento de tarjeta física
        assert '<div class="card-companion">' in html
        assert 'class="block-title">Gráfico<' in html
        # El título de la tabla NO se repite (show_title=False en la compañera).
        assert 'class="block-title">Niveles<' not in html


def _curated_block_of(block):
    from banks_rag.application.reporting.curated_report import CuratedBlock

    return CuratedBlock(block, "chart", f"<svg data-t='{block.title}'/>")


# ── Resaltado de series (PlotData.emphasis / .muted) ─────────────────────────

@pytest.mark.unit
class TestSeriesEmphasis:
    def test_defaults_are_empty(self):
        plot = PlotData("t", "line", "timeseries", "u", [PlotSeries(label="X", points=[("2026-01-01", 1.0)])])
        assert plot.emphasis == {}
        assert plot.muted == ()

    def test_emphasis_overrides_palette_color(self):
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        pts = [("2026-01-01", 1.0), ("2026-01-02", 2.0)]
        plot = PlotData("t", "line", "timeseries", "u",
                        [PlotSeries(label="Base", points=pts), PlotSeries(label="MA50", points=pts)],
                        emphasis={"MA50": "#8a6d3b"})
        svg = render_plot_svg(plot, chart="line")
        assert 'stroke="#8a6d3b"' in svg
        assert 'stroke-width="2.4"' in svg  # resaltada: más gruesa que el default 1.8

    def test_muted_renders_thin_and_gray(self):
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        pts = [("2026-01-01", 1.0), ("2026-01-02", 2.0)]
        plot = PlotData("t", "line", "timeseries", "u",
                        [PlotSeries(label="Base", points=pts), PlotSeries(label="MA10", points=pts)],
                        muted=("MA10",))
        svg = render_plot_svg(plot, chart="line")
        assert 'stroke="#9aa2b1" stroke-width="0.9"' in svg


@pytest.mark.unit
class TestCamMovingAverages:
    def test_highlights_50_and_200_mutes_the_rest(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_moving_averages

        rows = []
        for i in range(15):
            day = f"2026-07-{i + 1:02d}"
            rows.append(
                f"(DATE '{day}', {930.0 + i}, {928.0 + i}, {929.0 + i}, {931.0 + i}, {700.0 + i})"
            )
        _write(
            tmp_path / "ma.parquet",
            "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "CLP Cierre", '
            '"CLP Media móvil 10", "CLP Media móvil 50", "CLP Media móvil 200", "Monto transado (MM5d)")',
        )
        ds = _ds("ma.parquet", id="cam_clp_medias_moviles", unit="CLP/USD")
        plot = cam_moving_averages(ds, tmp_path, {
            "base": "CLP Cierre", "right": "Monto transado (MM5d)",
        })
        labels = [s.label for s in plot.series]
        assert labels == [
            "CLP Cierre", "CLP Media móvil 10", "CLP Media móvil 50",
            "CLP Media móvil 200", "Monto transado (MM5d)",
        ]
        assert plot.emphasis == {"CLP Media móvil 50": "#8a6d3b", "CLP Media móvil 200": "#c8102e"}
        assert plot.muted == ("CLP Media móvil 10",)

    def test_returns_none_without_any_moving_average_column(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_moving_averages

        _write(tmp_path / "flat.parquet",
               "SELECT * FROM (VALUES (DATE '2026-07-01', 930.0)) t(Fecha, \"CLP Cierre\")")
        ds = _ds("flat.parquet", id="cam_clp_medias_moviles", unit="CLP/USD")
        assert cam_moving_averages(ds, tmp_path, {}) is None


@pytest.mark.unit
class TestCategoryLabelRotation:
    """Etiquetas del eje X de barras: rotan cuando son largas O cuando no caben."""

    def _svg(self, cats: list[str]) -> str:
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        plot = PlotData("t", "grouped_bar", "grouped", "%",
                        [PlotSeries(label="s", points=[(c, 1.0) for c in cats])])
        return render_plot_svg(plot, chart="grouped_bar")

    def test_few_short_labels_stay_horizontal(self):
        assert "rotate(-45)" not in self._svg(["Chile", "Peru", "Brasil"])

    def test_long_labels_rotate(self):
        assert "rotate(-45)" in self._svg(["Itaú-Corpbanca", "Scotiabank"])

    def test_many_short_labels_rotate_because_they_do_not_fit(self):
        """37 monedas de nombre corto: ninguna supera los 9 caracteres, pero a
        ~25px por categoría se pisaban entre sí."""
        cats = [f"Pais{i:02d}" for i in range(37)]
        assert max(len(c) for c in cats) <= 9   # no entra por la regla de largo
        assert "rotate(-45)" in self._svg(cats)  # sí entra por la de ancho


@pytest.mark.unit
class TestCambiarioFixingAnchor:
    def test_spec_anchors_the_clp_section(self):
        """El fixing se abre por fecha de VENCIMIENTO (futura): sin el corte
        común el gráfico mostraría los forwards a dos meses en vez de la sesión."""
        from banks_rag.application.reporting.curated_report import weekly_anchor_source_ids
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        assert CAMBIARIOAM_SPEC.share_weekly_cutoff is True
        assert CAMBIARIOAM_SPEC.weekly_anchor_sections == ("CLP · Análisis",)
        assert "cam_fixing_bancos" in weekly_anchor_source_ids(CAMBIARIOAM_SPEC)

    def test_fixing_transform_respects_the_anchor_over_the_parquet_max(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_fixing_stacked

        rows = [
            "(DATE '2026-07-07', 'Santander', 'AFP', 10.0)",
            # Fixing FUTURO: es el máximo del parquet pero no la sesión del informe.
            "(DATE '2026-09-03', 'Santander', 'AFP', 999.0)",
        ]
        _write(tmp_path / "fx.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Informante, Sector, Pos_neta)")
        ds = _ds("fx.parquet", id="cam_fixing_bancos", unit="MM USD")

        anchored = cam_fixing_stacked(ds, tmp_path, {"weekly_asof": "2026-07-07"})
        assert "07-jul-2026" in anchored.date_note
        assert dict(anchored.series[0].points)["Santander"] == pytest.approx(10.0)

        # Sin ancla cae al máximo del parquet (el vencimiento futuro).
        assert "03-sep-2026" in cam_fixing_stacked(ds, tmp_path, {}).date_note


@pytest.mark.unit
class TestGridWidths:
    """Reparto de anchos de la grilla: misma regla que ``_annotate_layout`` del
    dashboard original (hero declarado + paridad automática)."""

    def _cards(self, blocks):
        from banks_rag.application.reporting.curated_report import (
            CuratedBlock,
            _assign_widths,
            _group_cards,
            _wide_block_ids,
        )

        spec = FamilyReportSpec(family="t", title="T", blocks=tuple(blocks), layout="grid")
        cards = _group_cards([CuratedBlock(b, "chart", "<svg/>") for b in blocks])
        _assign_widths(cards, _wide_block_ids(spec))
        return cards

    def _block(self, title, *, full_width=False, source_id=None, no_text=False):
        return ReportBlock(section="S", title=title, chart="line", status=STATUS_MVP,
                           source_id=source_id or title, full_width=full_width, no_text=no_text)

    def test_full_width_block_is_wide(self):
        cards = self._cards([self._block("Hero", full_width=True), self._block("A"), self._block("B")])
        assert [c.wide for c in cards] == [True, False, False]

    def test_odd_leftover_widens_the_last_one(self):
        cards = self._cards([self._block("A"), self._block("B"), self._block("C")])
        assert [c.wide for c in cards] == [False, False, True]

    def test_even_leftover_widens_nothing(self):
        cards = self._cards([self._block("A"), self._block("B")])
        assert [c.wide for c in cards] == [False, False]

    def test_hero_does_not_count_for_the_parity_rule(self):
        """El hero sale del conteo: con hero + 3 automáticas, la última se
        ensancha (3 es impar), igual que en el original."""
        cards = self._cards([
            self._block("Hero", full_width=True),
            self._block("A"), self._block("B"), self._block("C"),
        ])
        assert [c.wide for c in cards] == [True, False, False, True]

    def test_merged_companion_counts_as_one_card(self):
        """Un gráfico + su tabla son UNA tarjeta: la paridad cuenta tarjetas, no
        bloques (si contara bloques, 3 bloques = 2 tarjetas daría par y dejaría
        una tarjeta huérfana)."""
        cards = self._cards([
            self._block("Gráfico", source_id="x"),
            self._block("Tabla", source_id="x", no_text=True),
            self._block("Otro", source_id="y"),
        ])
        assert len(cards) == 2
        assert [c.wide for c in cards] == [False, False]


@pytest.mark.unit
class TestCambiarioAmWidths:
    def test_layout_matches_the_original_dashboard(self):
        """Las cuatro secciones con ``hero: True`` en el nav_groups del original
        más la tabla de portada son las que abren a fila completa. Gamma Proxy y
        su heatmap NO están acá: el heatmap se dibuja en SVG (auto-ajustable,
        ``_render_heatmap``) y comparte fila con Gamma Proxy como en el
        original, sin necesitar ancho completo."""
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        wide = {b.title for b in CAMBIARIOAM_SPEC.blocks if b.full_width}
        assert wide == {
            "Snapshot de Mercado · Drivers",  # tabla de portada (fuera de la grilla)
            "Derivados",                      # hero de No Residentes
            "Monedas LATAM",                  # hero de Monedas & Carry
            "Cobre vs CLP",                   # hero de Cobre
            "DXY vs CLP",                     # hero de Dólar · DXY
        }

    def test_drivers_section_puts_the_two_small_charts_side_by_side(self):
        from banks_rag.application.reporting.curated_report import (
            CuratedBlock,
            _assign_widths,
            _group_cards,
            _wide_block_ids,
        )
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        blocks = [b for b in CAMBIARIOAM_SPEC.blocks if b.section == "Drivers del día"]
        cards = _group_cards([CuratedBlock(b, "chart", "<svg/>") for b in blocks])
        _assign_widths(cards, _wide_block_ids(CAMBIARIOAM_SPEC))
        layout = [(c.main.block.title, c.wide) for c in cards]
        assert layout == [
            ("Snapshot de Mercado · Drivers", True),
            ("Variación Monedas en el día", False),
            ("Monedas Intradía", False),
            ("Evolución tipo de cambio", True),  # impar → cierra la fila
        ]
        # La tabla de la vela viaja DENTRO de la tarjeta del candlestick.
        assert cards[-1].companion.block.title.endswith("resumen de la sesión")

    def test_moving_averages_keep_the_full_history(self):
        """El dashboard original dibuja las medias desde 2019; acotarlas rompe la
        comparación con el informe que reemplaza."""
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        ma = next(b for b in CAMBIARIOAM_SPEC.blocks if b.title == "Medias Móviles")
        assert "months" not in ma.params


@pytest.mark.unit
class TestDualAxisRightStyle:
    """La serie del eje derecho: área tenue (default) o línea con su color."""

    def _plot(self):
        pts = [("2026-07-01", 1.0), ("2026-07-02", 2.0), ("2026-07-03", 1.5)]
        rpts = [("2026-07-01", 900.0), ("2026-07-02", 950.0), ("2026-07-03", 930.0)]
        return PlotData("t", "line", "timeseries", "USD/lb", [
            PlotSeries(label="Cobre", points=pts),
            PlotSeries(label="CLP", points=rpts),
        ])

    def _svg(self, **kw):
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        return render_plot_svg(self._plot(), chart="dual_axis", right_axis=["CLP"], **kw)

    def test_default_keeps_the_area(self):
        """Comportamiento histórico intacto: afp/nr dibujan el AUM como área."""
        svg = self._svg()
        assert "<polygon" in svg
        assert 'fill="#c8ccd4"' in svg  # relleno gris del área

    def test_line_style_draws_a_polyline_with_its_palette_color(self):
        svg = self._svg(right_style="line")
        assert "<polygon" not in svg          # ya no es área
        assert svg.count("<polyline") == 2    # las dos series son líneas
        assert 'fill="#c8ccd4"' not in svg    # sin relleno gris (el eje sí lo usa)
        assert "CLP (eje der.)" in svg        # sigue marcada como eje derecho

    def test_unknown_style_falls_back_to_area(self):
        assert "<polygon" in self._svg(right_style="zigzag")


def _polyline_points_for_index(svg: str, idx: int) -> list[tuple[float, float]]:
    """``[(x, y), ...]`` de la ``<polyline data-si="idx">`` (asume ``style="line"``)."""
    m = re.search(rf'<polyline points="([^"]+)"[^>]*data-si="{idx}"', svg)
    assert m, svg
    return [(float(x), float(y)) for x, y in (p.split(",") for p in m.group(1).split())]


@pytest.mark.unit
class TestDualAxisInvertAndLeftStyle:
    """``right_invert`` (réplica de "Posición cambiaria (eje inv.)") y
    ``left_style="area"`` (réplica de "AUM Renta Fija internacional")."""

    def _plot(self):
        # CLP (eje derecho, índice 1): 900 → 950 → 930, para leer sin ambigüedad
        # quién quedó "arriba" (y de SVG más chico) según se invierta o no.
        pts = [("2026-07-01", 1.0), ("2026-07-02", 2.0), ("2026-07-03", 1.5)]
        rpts = [("2026-07-01", 900.0), ("2026-07-02", 950.0), ("2026-07-03", 930.0)]
        return PlotData("t", "line", "timeseries", "USD/lb", [
            PlotSeries(label="Cobre", points=pts),
            PlotSeries(label="CLP", points=rpts),
        ])

    def _svg(self, **kw):
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        return render_plot_svg(self._plot(), chart="dual_axis", right_axis=["CLP"], **kw)

    def test_without_invert_the_highest_value_lands_near_the_top(self):
        svg = self._svg(right_style="line")
        by_val = {v: y for (_x, y), v in zip(_polyline_points_for_index(svg, 1),
                                              [900.0, 950.0, 930.0], strict=True)}
        assert by_val[950.0] < by_val[900.0]  # más alto = y de SVG más chico = arriba

    def test_right_invert_flips_it_the_lowest_value_lands_near_the_top(self):
        svg = self._svg(right_style="line", right_invert=True)
        by_val = {v: y for (_x, y), v in zip(_polyline_points_for_index(svg, 1),
                                              [900.0, 950.0, 930.0], strict=True)}
        assert by_val[900.0] < by_val[950.0]  # invertido: más bajo queda arriba

    def test_right_invert_marks_the_legend_as_inverted(self):
        assert "CLP (eje inv.)" in self._svg(right_invert=True)

    def test_without_invert_legend_says_plain_eje_der(self):
        svg = self._svg()
        assert "CLP (eje der.)" in svg and "eje inv." not in svg

    def test_left_style_area_draws_the_left_series_as_area_too(self):
        svg = self._svg(left_style="area")
        assert svg.count("<polygon") == 2      # eje der. (área default) + eje izq. (área)
        assert 'fill="#a9c4e0"' in svg          # celeste del área izquierda
        assert "Cobre (eje der.)" not in svg    # Cobre sigue en el eje IZQUIERDO

    def test_left_style_defaults_to_line_unchanged(self):
        assert self._svg().count("<polygon") == 1  # comportamiento histórico: solo el área derecha

    def test_real_afp_block_renders_inverted_without_legend_truncation(self, tmp_path):
        """Extremo a extremo con el bloque REAL de AFP_SPEC (no uno sintético): la
        leyenda trunca a 22 caracteres (_legend_row) — "Posicion" (8) + el sufijo
        corto entra bien, cosa que "(eje inv. | eje der.)" del tablero real no
        lograba ni con una serie de 3 letras (ver los tests de arriba)."""
        block = next(b for b in AFP_SPEC.blocks if b.source_id == "afp_aum_posicion_cambiaria")
        assert block.params.get("right_invert") is True
        assert block.params.get("left_style") == "area"

        rows = [
            "(DATE '2026-01-31', -10000.0, 30000.0)",
            "(DATE '2026-02-28', -15000.0, 32000.0)",
            "(DATE '2026-03-31', -12000.0, 34000.0)",
        ]
        _write(tmp_path / "afp_aum_posicion_cambiaria.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Posicion, AUM)")
        entries = [_ds("afp_aum_posicion_cambiaria.parquet", id="afp_aum_posicion_cambiaria",
                       unit="Millones de USD")]
        spec = FamilyReportSpec(family="t", title="T", blocks=(block,))
        report = build_curated_report(spec, entries=entries, parquet_dir=tmp_path)
        cb = report.blocks[0]
        assert cb.render_kind == "chart"
        assert "Posicion (eje inv.)" in cb.body_html   # sin truncar
        assert 'fill="#a9c4e0"' in cb.body_html        # AUM como área celeste en el eje izquierdo


@pytest.mark.unit
class TestLeftUnitOverride:
    def test_block_can_relabel_the_left_axis(self, tmp_path):
        """En Inventarios el eje izquierdo lleva el PRECIO, no la unidad del
        dataset: sin el override el eje diría 'Toneladas' sobre valores USD/lb."""
        rows = [f"(DATE '2026-07-{i + 1:02d}', {100.0 + i}, {5.0 + i / 10})" for i in range(5)]
        _write(tmp_path / "inv.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "Inventarios", "Precio")')
        entries = [_ds("inv.parquet", id="inv", unit="Toneladas")]
        spec = FamilyReportSpec(
            family="t", title="T",
            blocks=(ReportBlock(
                section="S", title="Inventarios", chart="dual_axis", status=STATUS_MVP,
                source_id="inv", transform="cam_lines",
                params={"right_axis": ["Inventarios"], "right_unit": "Toneladas",
                        "left_unit": "USD/lb"},
            ),),
        )
        report = build_curated_report(spec, entries=entries, parquet_dir=tmp_path)
        assert report.blocks[0].plot.unit == "USD/lb"


@pytest.mark.unit
class TestCambiarioDualAxisStyles:
    def test_only_volume_series_stay_as_area(self):
        """El original dibuja línea contra línea salvo cuando la serie del eje
        derecho es un VOLUMEN (monto transado, inventarios)."""
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        area = {
            b.title for b in CAMBIARIOAM_SPEC.blocks
            if b.chart == "dual_axis" and b.params.get("right_style", "area") == "area"
        }
        assert area == {"Medias Móviles", "Inventarios COMEX", "Inventarios Londres"}

    def test_the_rest_are_line_against_line(self):
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        line = [b for b in CAMBIARIOAM_SPEC.blocks
                if b.chart == "dual_axis" and b.params.get("right_style") == "line"]
        assert len(line) == 10
        for b in line:
            assert b.params.get("right_axis"), b.title


@pytest.mark.unit
class TestWideChartCanvas:
    """El viewBox de un bloque ancho es el DOBLE, no el mismo estirado por CSS —
    si no, el navegador escala todo (ejes, leyenda, grosor) y se ve con zoom."""

    def _spec(self, *, wide_title: str) -> FamilyReportSpec:
        # Dos bloques NORMALES además del ancho (par): así "B" no se ensancha
        # también por la regla de paridad automática y el test aísla el efecto
        # de ``full_width``.
        blocks = (
            ReportBlock(section="S", title=wide_title, chart="line", status=STATUS_MVP,
                       source_id="a", transform="filter_fund", full_width=True),
            ReportBlock(section="S", title="B", chart="line", status=STATUS_MVP,
                       source_id="a", transform="filter_fund"),
            ReportBlock(section="S", title="C", chart="line", status=STATUS_MVP,
                       source_id="a", transform="filter_fund"),
        )
        return FamilyReportSpec(family="t", title="T", blocks=blocks, layout="grid")

    def test_wide_block_gets_the_double_viewbox(self, tmp_path):
        import re

        _categorical_parquet(tmp_path / "dur.parquet")
        entries = [_ds("dur.parquet", id="a")]
        report = build_curated_report(self._spec(wide_title="Ancho"), entries=entries, parquet_dir=tmp_path)
        by = {b.block.title: b for b in report.blocks}

        m_wide = re.search(r'viewBox="0 0 ([\d.]+)', by["Ancho"].body_html)
        m_normal = re.search(r'viewBox="0 0 ([\d.]+)', by["B"].body_html)
        assert float(m_wide.group(1)) == float(m_normal.group(1)) * 2

    def test_html_render_stays_consistent_with_the_chosen_width(self, tmp_path):
        """La tarjeta que la grilla ensancha con CSS es la MISMA que recibió el
        viewBox ancho — no pueden desincronizarse (misma fuente: _wide_block_ids)."""
        _categorical_parquet(tmp_path / "dur.parquet")
        entries = [_ds("dur.parquet", id="a")]
        report = build_curated_report(self._spec(wide_title="Ancho"), entries=entries, parquet_dir=tmp_path)
        html = render_curated_html(report)
        # La tarjeta "card-wide" es la que contiene el SVG de 1520 (2x760).
        idx_wide_card = html.index('class="card card-wide"')
        idx_next_card = html.index('<div class="card"', idx_wide_card)
        segment = html[idx_wide_card:idx_next_card]
        assert 'viewBox="0 0 1520' in segment
        assert 'viewBox="0 0 760' not in segment


@pytest.mark.unit
class TestCamLinesAlignFrom:
    def test_clips_all_series_to_the_latest_start(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_lines

        rows = [f"(DATE '2019-07-{i + 1:02d}', {900.0 + i}, NULL)" for i in range(5)]
        rows += [f"(DATE '2022-01-{i + 1:02d}', {900.0 + i}, {5.0 + i})" for i in range(5)]
        _write(tmp_path / "al.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "CLP", "NR")')
        ds = _ds("al.parquet", id="cam_pos_no_residentes", unit="CLP/USD")

        plot = cam_lines(ds, tmp_path, {"align_from": ["NR"]})
        clp_isos = [iso for iso, _v in next(s for s in plot.series if s.label == "CLP").points]
        nr_isos = [iso for iso, _v in next(s for s in plot.series if s.label == "NR").points]
        assert min(clp_isos) == "2022-01-01"  # recortado a donde arranca NR
        assert min(nr_isos) == "2022-01-01"

    def test_without_align_from_keeps_full_history(self, tmp_path):
        from banks_rag.application.reporting.series_transforms import cam_lines

        rows = [f"(DATE '2019-07-{i + 1:02d}', {900.0 + i}, NULL)" for i in range(5)]
        rows += [f"(DATE '2022-01-{i + 1:02d}', {900.0 + i}, {5.0 + i})" for i in range(5)]
        _write(tmp_path / "al2.parquet",
               "SELECT * FROM (VALUES " + ", ".join(rows) + ') t(Fecha, "CLP", "NR")')
        ds = _ds("al2.parquet", id="cam_pos_no_residentes", unit="CLP/USD")

        plot = cam_lines(ds, tmp_path, {})
        clp_isos = [iso for iso, _v in next(s for s in plot.series if s.label == "CLP").points]
        assert min(clp_isos) == "2019-07-01"  # sin recorte: historia completa


@pytest.mark.unit
class TestCambiarioDerivadosAlignment:
    def test_spec_aligns_derivados_from_the_nr_series(self):
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        b = next(x for x in CAMBIARIOAM_SPEC.blocks if x.title == "Derivados")
        assert b.params.get("align_from") == ["CLPBODM Index"]

    def test_otros_section_has_six_paired_cards(self):
        """6 bloques (par): se acomodan de a dos por fila, ninguno se ensancha
        por la regla de paridad — incluye Gamma Proxy + su heatmap, que ahora
        comparten fila (el heatmap es SVG auto-ajustable, no tabla HTML fija)."""
        from banks_rag.application.reporting.curated_report import _wide_block_ids
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        wide = _wide_block_ids(CAMBIARIOAM_SPEC)
        otros = [b for b in CAMBIARIOAM_SPEC.blocks if b.section == "Otros"]
        assert [b.title in wide for b in otros] == [False] * 6
        assert otros[0].title == "Gamma Proxy"
        assert otros[1].title == "Heatmap Gamma Proxy"


@pytest.mark.unit
class TestHeatmapSvg:
    """``_render_heatmap``: mapa de calor en SVG (auto-ajustable, sin JS) —
    reemplaza la vieja tabla HTML de celdas de ancho fijo. Convención de
    ``PlotData``: una serie POR FILA, ``points = [(columna, valor)]``."""

    def _plot(self, rows: dict[str, list[tuple[str, float]]], unit: str = "MM USD") -> PlotData:
        return PlotData("t", "heatmap", "heatmap", unit,
                        [PlotSeries(label=r, points=pts) for r, pts in rows.items()])

    def test_renders_a_rect_per_cell_with_hover_data(self):
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        svg = render_plot_svg(self._plot({
            "940": [("10-ago-2026", 42.5)], "920": [("12-ago-2026", 5.0)],
        }), chart="heatmap_table")
        assert svg.count("<rect") >= 2
        assert 'data-k="940"' in svg and 'data-k="920"' in svg  # fila en el tooltip
        assert 'data-s="10-ago-2026"' in svg                    # columna en el tooltip

    def test_columns_sort_chronologically_not_by_appearance(self):
        """Bug real: el parquet no viene ordenado por fecha, y cada fila trae
        SUS fechas en su propio orden — sin ordenar, las columnas salían
        mezcladas (25-ago antes que 10-ago)."""
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        svg = render_plot_svg(self._plot({
            "940": [("25-ago-2026", 1.0), ("10-ago-2026", 2.0)],
            "920": [("03-sep-2026", 3.0), ("12-ago-2026", 4.0)],
        }), chart="heatmap_table")
        order = [svg.index(f'text-anchor="start">{d}<') for d in
                 ("10-ago-2026", "12-ago-2026", "25-ago-2026", "03-sep-2026")]
        assert order == sorted(order)

    def test_non_date_columns_keep_appearance_order(self):
        """Columnas que NO parsean como fecha (p.ej. instrumentos): se dejan tal
        cual, sin intentar ordenarlas ni reventar."""
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        svg = render_plot_svg(self._plot({
            "940": [("Bono", 1.0), ("DAP", 2.0)],
        }), chart="heatmap_table")
        assert svg.index('>Bono<') < svg.index('>DAP<')

    def test_legend_bar_spans_the_data_range(self):
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        svg = render_plot_svg(self._plot({
            "940": [("10-ago-2026", 100.0)], "920": [("10-ago-2026", 0.0)],
        }), chart="heatmap_table")
        assert "100,0" in svg  # marca del máximo en la barra de escala
        assert "0,00" in svg   # marca del mínimo

    def test_missing_cells_draw_no_rect(self):
        """Fila con menos columnas que otra: la celda faltante no dibuja nada
        (fondo blanco), como un heatmap sobre datos dispersos. Se cuentan solo
        las celdas de DATO (``class="tip-pt"``, la marca del hover), no las
        franjas de la barra de escala — esas también son ``<rect>``."""
        from banks_rag.application.reporting.svg_chart import render_plot_svg

        svg = render_plot_svg(self._plot({
            "940": [("10-ago-2026", 1.0), ("12-ago-2026", 2.0)],
            "920": [("10-ago-2026", 3.0)],  # sin dato para 12-ago
        }), chart="heatmap_table")
        assert svg.count('class="tip-pt"') == 3  # 940x10ago, 940x12ago, 920x10ago (no 920x12ago)

    def test_shares_the_wide_viewbox_mechanism(self, tmp_path):
        """El heatmap usa el MISMO ``render_plot_svg(width=...)`` que cualquier
        otro gráfico: si el bloque pide ancho, sale con el viewBox ancho."""
        import re

        from banks_rag.application.reporting.svg_chart import render_plot_svg

        plot = self._plot({"940": [("10-ago-2026", 1.0)]})
        wide = render_plot_svg(plot, chart="heatmap_table", width=1520)
        normal = render_plot_svg(plot, chart="heatmap_table", width=760)
        assert re.search(r'viewBox="0 0 1520', wide)
        assert re.search(r'viewBox="0 0 760', normal)


@pytest.mark.unit
class TestCardSubgridAlignment:
    """El gráfico de dos tarjetas en la misma fila arranca a la MISMA altura,
    aunque una traiga nota/ventana de fechas y la otra no (CSS subgrid:
    .card-head se empareja antes de que arranque .card-body)."""

    def test_card_splits_into_head_and_body_rows(self):
        from banks_rag.application.reporting.curated_report import CuratedReport

        with_note = ReportBlock(section="S", title="Con nota", chart="line", status=STATUS_MVP,
                                source_id="a", note="una nota")
        without_note = ReportBlock(section="S", title="Sin nota", chart="line", status=STATUS_MVP,
                                   source_id="b")
        spec = FamilyReportSpec(family="t", title="T", blocks=(with_note, without_note), layout="grid")
        cbs = [_curated_block_of(with_note), _curated_block_of(without_note)]
        report = CuratedReport(spec=spec, generated_at="now", blocks=cbs)
        html = render_curated_html(report)

        assert html.count('<div class="card-head">') == 2
        assert html.count('<div class="card-body">') == 2
        # La nota vive en card-head, no en card-body.
        assert '<div class="card-head"><div class="block-title">Con nota</div>' in html
        assert 'block-note">una nota</div>' in html.split('<div class="card-head">')[1]

    def test_head_contains_the_variable_height_content(self):
        """Título, unidad, nota y ventana de fechas van en card-head; el gráfico
        (y el aviso de preliminar) van en card-body — es la separación que
        permite al subgrid emparejar las alturas."""
        from banks_rag.application.reporting.curated_report import (
            CuratedBlock,
            _card_head_body_html,
        )

        block = ReportBlock(section="S", title="T", chart="line", status=STATUS_MVP,
                            source_id="a", unit="CLP/USD", note="nota")
        cb = CuratedBlock(block, "chart", "<svg-x/>", date_note="ventana", preliminary=True)
        head, body = _card_head_body_html(cb)
        head_html, body_html = "".join(head), "".join(body)
        for marker in ("block-title", "block-unit", "block-note", "block-dates"):
            assert marker in head_html, marker
        assert "prelim-note" in body_html
        assert "<svg-x/>" in body_html
        assert "block-title" not in body_html and "block-dates" not in body_html

    def test_stack_layout_output_is_unaffected(self, tmp_path):
        """El layout ``stack`` (todas las familias salvo cambiarioam) sigue
        concatenando encabezado + cuerpo en el MISMO ``.block``, sin dividir en
        card-head/card-body (esa separación es solo para la grilla)."""
        spec = _spec_for_build()
        report = build_curated_report(spec, entries=[], parquet_dir=tmp_path)
        html = render_curated_html(report)
        assert '<div class="card-head">' not in html
        assert '<div class="card-body">' not in html


@pytest.mark.unit
class TestPerSectionGrid:
    """``FamilyReportSpec.grid_sections``: solo ALGUNAS secciones de un informe
    por lo demás apilado usan la grilla de 2 columnas (caso real: "Allocation y
    patrimonio" en afp), sin pasar el informe ENTERO a "grid"."""

    def _spec(self, *, grid_sections=frozenset()):
        blocks = (
            ReportBlock(section="Grilla", title="G1", chart="line", status=STATUS_MVP, source_id="a"),
            ReportBlock(section="Grilla", title="G2", chart="line", status=STATUS_MVP, source_id="b"),
            ReportBlock(section="Apilada", title="S1", chart="line", status=STATUS_MVP, source_id="c"),
            ReportBlock(section="Apilada", title="S2", chart="line", status=STATUS_MVP, source_id="d"),
        )
        return FamilyReportSpec(family="t", title="T", blocks=blocks, grid_sections=grid_sections)

    def test_default_grid_sections_is_empty_and_changes_nothing(self):
        """Ninguna familia existente declara esto: default vacío = comportamiento
        histórico intacto."""
        for spec in (FFMM_SPEC, NR_SPEC, FX_SPEC, DCV_SPEC):
            assert spec.grid_sections == frozenset()

    def test_only_the_named_section_becomes_grid(self):
        from banks_rag.application.reporting.curated_report import CuratedReport

        spec = self._spec(grid_sections=frozenset({"Grilla"}))
        cbs = [_curated_block_of(b) for b in spec.blocks]
        report = CuratedReport(spec=spec, generated_at="now", blocks=cbs)
        html = render_curated_html(report)

        assert html.count('<div class="cards-grid">') == 1  # solo "Grilla"
        # "Apilada" sigue con bloques .block sueltos, no tarjetas.
        assert html.count('<div class="block" data-block-status') == 2

    def test_no_grid_sections_behaves_exactly_like_stack(self):
        from banks_rag.application.reporting.curated_report import CuratedReport

        spec = self._spec()  # grid_sections vacío, layout default "stack"
        cbs = [_curated_block_of(b) for b in spec.blocks]
        report = CuratedReport(spec=spec, generated_at="now", blocks=cbs)
        html = render_curated_html(report)
        assert '<div class="cards-grid">' not in html

    def test_spec_wide_grid_still_covers_every_section_without_declaring_it(self):
        """cambiarioam (``layout="grid"``) no necesita nombrar sus 8 secciones en
        ``grid_sections``: la condición OR ya las cubre todas."""
        from banks_rag.application.reporting.curated_report import _is_grid_section
        from banks_rag.application.reporting.specs import CAMBIARIOAM_SPEC

        assert CAMBIARIOAM_SPEC.grid_sections == frozenset()
        for sec in CAMBIARIOAM_SPEC.sections():
            assert _is_grid_section(CAMBIARIOAM_SPEC, sec)


@pytest.mark.unit
class TestAfpAllocationGrid:
    def test_allocation_section_is_grid_the_rest_is_not(self):
        from banks_rag.application.reporting.curated_report import _is_grid_section
        from banks_rag.application.reporting.specs.afp_spec import AFP_SPEC

        assert AFP_SPEC.layout == "stack"  # el informe sigue apilado por defecto
        assert _is_grid_section(AFP_SPEC, "Allocation y patrimonio")
        for sec in AFP_SPEC.sections():
            if sec != "Allocation y patrimonio":
                assert not _is_grid_section(AFP_SPEC, sec), sec


class TestFullWidthBlocks:
    """``ReportBlock.full_width`` fuera de la grilla: el gráfico llega al borde de la
    página en vez de topar en los 760px de ``.report-chart``."""

    @staticmethod
    def _spec(**kwargs) -> FamilyReportSpec:
        return FamilyReportSpec(
            family="x", title="X",
            blocks=(
                ReportBlock(section="S", title="normal", chart="line", status=STATUS_MVP),
                ReportBlock(section="S", title="ancho", chart="stacked_bar",
                            status=STATUS_MVP, full_width=True),
            ),
            **kwargs,
        )

    def test_marca_el_bloque_de_una_seccion_apilada(self):
        # Antes solo se miraban las secciones en grilla, así que en un informe
        # apilado (dcv) marcar full_width no hacía absolutamente nada.
        assert _wide_block_ids(self._spec()) == {"ancho"}

    def test_no_arrastra_a_los_demas_bloques_de_la_seccion(self):
        assert "normal" not in _wide_block_ids(self._spec())

    def test_sin_full_width_no_hay_bloques_anchos_en_stack(self):
        spec = FamilyReportSpec(
            family="x", title="X",
            blocks=(ReportBlock(section="S", title="a", chart="line", status=STATUS_MVP),),
        )
        assert _wide_block_ids(spec) == set()

    def test_el_html_libera_al_grafico_ancho_del_tope_de_760(self, tmp_path):
        # El viewBox ancho y la CSS tienen que ir juntos: un SVG dibujado a 1520
        # dentro de un contenedor de 760px se ve igual de chico pero con la letra
        # más finita que el resto del informe.
        report = build_curated_report(self._spec(), entries=[], parquet_dir=tmp_path)
        html = render_curated_html(report)
        assert ".chart-wide .report-chart { max-width:100%; }" in html
