"""Tests para las analytics tools (post-switch al parquet_catalog).

Dos bloques:
- funciones puras de ``infrastructure/sql/series_analytics`` (sin I/O, no
  cambian con el switch);
- tools registradas, con ``fetch_rows_from_dataset`` mockeado (intercepta
  toda la cadena loader+SQL+DuckDB, sin tocar archivos reales).
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from banks_rag.application.agent.tools import analytics
from banks_rag.domain.agent import AgentState
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ColumnSpec,
    ParquetDataset,
)
from banks_rag.infrastructure.sql.series_analytics import (
    anomaly_check,
    clean_series,
    composition,
    composition_wide,
    descriptive_stats,
    spread,
    variation,
)

_MODULE = "banks_rag.application.agent.tools.analytics"


# ─────────────────────────────────────────────────────────────────────────────
# Funciones puras de series_analytics (sin I/O — sin cambios desde el switch)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSeriesAnalyticsPure:
    def test_clean_series_drops_nulls_and_sorts(self) -> None:
        rows = [
            {"fecha": "2024-03-01", "v": 10.0},
            {"fecha": "2024-01-01", "v": None},
            {"fecha": "2024-02-01", "v": "no-num"},
            {"fecha": "2024-01-15", "v": 5},
        ]
        series = clean_series(rows, "fecha", "v")
        assert series == [("2024-01-15", 5.0), ("2024-03-01", 10.0)]

    def test_variation_basic(self) -> None:
        series = [("2024-01-01", 100.0), ("2024-02-01", 90.0), ("2024-03-01", 110.0)]
        var = variation(series)
        assert var["valor_inicio"] == 100.0
        assert var["valor_fin"] == 110.0
        assert var["cambio_absoluto"] == 10.0
        assert var["variacion_pct"] == 10.0
        assert var["cambio_bps"] == 1000.0
        assert var["minimo"] == 90.0
        assert var["maximo"] == 110.0
        assert var["n_observaciones"] == 3

    def test_variation_needs_two_points(self) -> None:
        assert variation([("2024-01-01", 1.0)]) is None
        assert variation([]) is None

    def test_variation_pct_none_when_start_zero(self) -> None:
        var = variation([("2024-01-01", 0.0), ("2024-02-01", 5.0)])
        assert var["variacion_pct"] is None
        assert var["cambio_absoluto"] == 5.0

    def test_descriptive_stats_and_percentile(self) -> None:
        series = [(f"2024-01-{i:02d}", float(i)) for i in range(1, 11)]
        stats = descriptive_stats(series)
        assert stats["n_observaciones"] == 10
        assert stats["media"] == 5.5
        assert stats["minimo"] == 1.0
        assert stats["maximo"] == 10.0
        assert stats["ultimo_valor"] == 10.0
        assert stats["percentil_ultimo_valor"] == 90.0

    def test_descriptive_stats_empty(self) -> None:
        assert descriptive_stats([]) is None

    def test_anomaly_flags_outlier(self) -> None:
        series = [
            (f"2024-01-{i:02d}", 10.0 + (0.5 if i % 2 else -0.5))
            for i in range(1, 10)
        ]
        series.append(("2024-01-10", 100.0))
        check = anomaly_check(series, threshold_stds=2.0)
        assert check["es_anomalia"] is True
        assert check["z_score"] > 2.0
        assert "inusual" in check["interpretacion"].lower()

    def test_anomaly_flat_history_then_jump(self) -> None:
        series = [(f"2024-01-{i:02d}", 10.0) for i in range(1, 10)]
        series.append(("2024-01-10", 100.0))
        check = anomaly_check(series, threshold_stds=2.0)
        assert check["es_anomalia"] is True
        assert check["z_score"] is None
        assert check["desviacion_ventana"] == 0.0

    def test_anomaly_normal_value(self) -> None:
        series = [(f"2024-01-{i:02d}", 10.0 + (i % 2)) for i in range(1, 11)]
        check = anomaly_check(series, threshold_stds=2.0)
        assert check["es_anomalia"] is False

    def test_anomaly_needs_three_points(self) -> None:
        assert anomaly_check([("2024-01-01", 1.0), ("2024-01-02", 2.0)]) is None

    def test_spread_aligns_by_date(self) -> None:
        a = [("2024-01-01", 5.0), ("2024-01-02", 6.0), ("2024-01-03", 7.0)]
        b = [("2024-01-01", 2.0), ("2024-01-03", 3.0)]
        spr = spread(a, b)
        assert spr["n_observaciones_comunes"] == 2
        assert spr["spread_inicio"] == 3.0
        assert spr["spread_fin"] == 4.0
        assert spr["cambio_spread"] == 1.0

    def test_spread_no_common_dates(self) -> None:
        a = [("2024-01-01", 5.0)]
        b = [("2024-02-01", 2.0)]
        assert spread(a, b) is None


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_date (helper pura)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestResolveDate:
    def test_hoy(self) -> None:
        assert analytics._resolve_date("hoy") == date.today().isoformat()

    def test_today_english(self) -> None:
        assert analytics._resolve_date("today") == date.today().isoformat()

    def test_negative_days(self) -> None:
        expected = (date.today() - timedelta(days=90)).isoformat()
        assert analytics._resolve_date("-90d") == expected

    def test_negative_months(self) -> None:
        expected = (date.today() - timedelta(days=12 * 30)).isoformat()
        assert analytics._resolve_date("-12m") == expected

    def test_iso_passthrough(self) -> None:
        assert analytics._resolve_date("2024-06-15") == "2024-06-15"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            analytics._resolve_date("garbage")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de ParquetDataset + mock de fetch_rows_from_dataset
# ─────────────────────────────────────────────────────────────────────────────


def _dataset(
    id_: str,
    columns: list[ColumnSpec] | None = None,
    unit: str = "% anual",
) -> ParquetDataset:
    cols = columns or [
        ColumnSpec("Fecha", "TIMESTAMP"),
        ColumnSpec("Valor", "DOUBLE"),
    ]
    return ParquetDataset(
        id=id_,
        file=f"{id_}.parquet",
        name=f"Dataset {id_}",
        description="desc",
        segment="seg",
        unit=unit,
        date_range=["2024-01-01", "2024-12-31"],
        columns=cols,
    )


def _mock_fetch(
    dataset: ParquetDataset,
    rows: list[dict],
    date_col: str = "Fecha",
) -> AsyncMock:
    """AsyncMock que retorna la tupla esperada por las analytics."""
    select_cols = [date_col] + [c.name for c in dataset.columns if c.name != date_col]
    return AsyncMock(return_value=(dataset, rows, date_col, select_cols))


# ─────────────────────────────────────────────────────────────────────────────
# compute_variation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeVariation:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        state = AgentState()
        ds = _dataset("usdclp_test", unit="CLP por USD")
        rows = [
            {"Fecha": "2024-03-01", "Valor": 990.0},
            {"Fecha": "2024-01-01", "Valor": 900.0},
        ]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.compute_variation(
                state=state, dataset_id="usdclp_test", column="Valor",
            )
        assert result["valor_inicio"] == 900.0
        assert result["valor_fin"] == 990.0
        assert result["variacion_pct"] == 10.0
        assert result["unit"] == "CLP por USD"
        assert "usdclp_test:Valor" in state.series_used

    @pytest.mark.asyncio
    async def test_unknown_dataset_id(self) -> None:
        with patch(
            f"{_MODULE}.fetch_rows_from_dataset",
            new=AsyncMock(side_effect=ValueError("dataset_id desconocido: 'no'")),
        ):
            result = await analytics.compute_variation(
                state=AgentState(), dataset_id="no", column="Valor",
            )
        assert "error" in result
        assert "desconocido" in result["error"]

    @pytest.mark.asyncio
    async def test_bad_column(self) -> None:
        ds = _dataset("x")
        rows = [{"Fecha": "2024-01-01", "Valor": 1.0}]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.compute_variation(
                state=AgentState(), dataset_id="x", column="Inexistente",
            )
        assert "error" in result
        assert "available_columns" in result

    @pytest.mark.asyncio
    async def test_insufficient_observations(self) -> None:
        ds = _dataset("x")
        rows = [{"Fecha": "2024-01-01", "Valor": 1.0}]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.compute_variation(
                state=AgentState(), dataset_id="x", column="Valor",
            )
        assert "error" in result
        assert "2 observaciones" in result["error"]

    @pytest.mark.asyncio
    async def test_long_format_with_filter(self) -> None:
        """Para btp_curva-like (long format) se pasa filters={Tenor: 10Y}."""
        ds = _dataset("btp_curva", columns=[
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Tenor", "VARCHAR", values=["2Y", "10Y"]),
            ColumnSpec("Valor", "DOUBLE"),
        ], unit="% anual")
        rows = [
            {"Fecha": "2024-03-01", "Valor": 6.0},
            {"Fecha": "2024-01-01", "Valor": 5.5},
        ]
        state = AgentState()
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.compute_variation(
                state=state, dataset_id="btp_curva", column="Valor",
                filters={"Tenor": "10Y"},
            )
        assert result["filters"] == {"Tenor": "10Y"}
        # series_id incluye el filtro para no colisionar con otros tenores
        assert "btp_curva:Valor:Tenor=10Y" in state.series_used


# ─────────────────────────────────────────────────────────────────────────────
# compute_spread
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeSpread:
    @pytest.mark.asyncio
    async def test_inter_dataset_spread(self) -> None:
        """Dos datasets distintos → dos llamadas a fetch (no reusa)."""
        ds_a = _dataset("btp_curva")
        ds_b = _dataset("btu_curva")
        rows_a = [{"Fecha": "2024-02-01", "Valor": 6.5},
                  {"Fecha": "2024-01-01", "Valor": 6.0}]
        rows_b = [{"Fecha": "2024-02-01", "Valor": 2.0},
                  {"Fecha": "2024-01-01", "Valor": 1.5}]
        fetched = AsyncMock(side_effect=[
            (ds_a, rows_a, "Fecha", ["Fecha", "Valor"]),
            (ds_b, rows_b, "Fecha", ["Fecha", "Valor"]),
        ])
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=fetched):
            result = await analytics.compute_spread(
                state=AgentState(),
                dataset_id_a="btp_curva", column_a="Valor",
                dataset_id_b="btu_curva", column_b="Valor",
            )
        assert result["spread_inicio"] == 4.5    # 6.0 - 1.5
        assert result["spread_fin"] == 4.5       # 6.5 - 2.0
        assert fetched.call_count == 2

    @pytest.mark.asyncio
    async def test_intra_dataset_with_same_filters_reuses(self) -> None:
        """Mismo dataset_id + mismos filtros → una sola llamada."""
        ds = _dataset("curva", columns=[
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("col_a", "DOUBLE"),
            ColumnSpec("col_b", "DOUBLE"),
        ])
        rows = [
            {"Fecha": "2024-02-01", "col_a": 6.5, "col_b": 5.0},
            {"Fecha": "2024-01-01", "col_a": 6.0, "col_b": 5.0},
        ]
        fetched = _mock_fetch(ds, rows)
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=fetched):
            result = await analytics.compute_spread(
                state=AgentState(),
                dataset_id_a="curva", column_a="col_a",
                dataset_id_b="curva", column_b="col_b",
            )
        assert result["definicion"] == "spread = col_a - col_b"
        assert fetched.call_count == 1

    @pytest.mark.asyncio
    async def test_no_common_dates(self) -> None:
        ds = _dataset("x", columns=[
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("a", "DOUBLE"),
            ColumnSpec("b", "DOUBLE"),
        ])
        rows = [
            {"Fecha": "2024-01-01", "a": 5.0, "b": None},
            {"Fecha": "2024-02-01", "a": None, "b": 2.0},
        ]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.compute_spread(
                state=AgentState(),
                dataset_id_a="x", column_a="a",
                dataset_id_b="x", column_b="b",
            )
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# get_series_stats
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGetSeriesStats:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        ds = _dataset("x")
        rows = [{"Fecha": f"2024-01-{i:02d}", "Valor": float(i)}
                for i in range(1, 11)]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.get_series_stats(
                state=AgentState(), dataset_id="x", column="Valor",
            )
        assert result["n_observaciones"] == 10
        assert result["media"] == 5.5
        assert "periodo" in result

    @pytest.mark.asyncio
    async def test_empty_series(self) -> None:
        ds = _dataset("x")
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, [])):
            result = await analytics.get_series_stats(
                state=AgentState(), dataset_id="x", column="Valor",
            )
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# detect_anomaly
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDetectAnomaly:
    @pytest.mark.asyncio
    async def test_flags_anomaly(self) -> None:
        ds = _dataset("x")
        rows = [
            {"Fecha": f"2024-01-{i:02d}", "Valor": 10.0 + (0.5 if i % 2 else -0.5)}
            for i in range(1, 10)
        ]
        rows.append({"Fecha": "2024-01-10", "Valor": 100.0})
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.detect_anomaly(
                state=AgentState(), dataset_id="x", column="Valor",
            )
        assert result["es_anomalia"] is True
        assert result["lookback_days"] == 180

    @pytest.mark.asyncio
    async def test_insufficient_points(self) -> None:
        ds = _dataset("x")
        rows = [{"Fecha": "2024-01-01", "Valor": 1.0}]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.detect_anomaly(
                state=AgentState(), dataset_id="x", column="Valor",
            )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_clamps_params(self) -> None:
        ds = _dataset("x")
        rows = [{"Fecha": f"2024-01-{i:02d}", "Valor": 10.0} for i in range(1, 6)]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            result = await analytics.detect_anomaly(
                state=AgentState(), dataset_id="x", column="Valor",
                lookback_days=99999, umbral_desviaciones=99.0,
            )
        assert result["lookback_days"] == 1095        # cap superior
        assert result["umbral_desviaciones"] == 5.0   # cap superior


# ─────────────────────────────────────────────────────────────────────────────
# get_market_snapshot
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGetMarketSnapshot:
    @pytest.mark.asyncio
    async def test_returns_indicators_with_derived_bei(self) -> None:
        """Devuelve un indicador por entry de _SNAPSHOT_INDICATORS y deriva el BEI."""
        # Construye un dataset por cada indicador. La columna varía; el mock
        # devuelve filas con esa columna específica.
        snapshot_indicators = analytics._SNAPSHOT_INDICATORS
        last_values = {
            "usdclp": 950.0, "cobre": 620.0, "btp_10y": 5.8, "btu_10y": 2.3,
            "tib_spread": 5.0, "exp_tpm_3m": 12.0,
        }

        def _build_return(indicator: tuple) -> tuple:
            clave, _name, ds_id, column, _filters = indicator
            cols = [ColumnSpec("Fecha", "TIMESTAMP"), ColumnSpec(column, "DOUBLE")]
            ds = _dataset(ds_id, columns=cols)
            row = {"Fecha": "2026-05-15", column: last_values[clave]}
            return (ds, [row], "Fecha", ["Fecha", column])

        fetched = AsyncMock(
            side_effect=[_build_return(ind) for ind in snapshot_indicators],
        )
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=fetched):
            result = await analytics.get_market_snapshot(state=AgentState())

        claves = {ind["clave"] for ind in result["indicadores"]}
        # 6 indicadores configurados + BEI derivado
        assert "usdclp" in claves
        assert "btp_10y" in claves
        assert "btu_10y" in claves
        assert "bei_10y" in claves
        bei = next(i for i in result["indicadores"] if i["clave"] == "bei_10y")
        assert bei["valor"] == pytest.approx(3.5)   # 5.8 - 2.3


# ─────────────────────────────────────────────────────────────────────────────
# composition / composition_wide (puras)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCompositionPure:
    def test_long_shares_at_latest_date(self) -> None:
        rows = [
            {"Fecha": "2026-04-30", "Cat": "A", "Monto": 60.0},
            {"Fecha": "2026-04-30", "Cat": "B", "Monto": 40.0},
            {"Fecha": "2026-03-31", "Cat": "A", "Monto": 10.0},  # fecha anterior → ignorada
        ]
        comp = composition(rows, "Fecha", "Cat", "Monto")
        assert comp["fecha"] == "2026-04-30"
        assert comp["total"] == 100.0
        shares = {b["categoria"]: b["share_pct"] for b in comp["breakdown"]}
        assert shares == {"A": 60.0, "B": 40.0}
        # Ordenado desc por valor.
        assert comp["breakdown"][0]["categoria"] == "A"

    def test_long_aggregates_same_category(self) -> None:
        rows = [
            {"Fecha": "2026-04-30", "Cat": "A", "Monto": 30.0},
            {"Fecha": "2026-04-30", "Cat": "A", "Monto": 30.0},
            {"Fecha": "2026-04-30", "Cat": "B", "Monto": 40.0},
        ]
        comp = composition(rows, "Fecha", "Cat", "Monto")
        shares = {b["categoria"]: b["share_pct"] for b in comp["breakdown"]}
        assert shares["A"] == 60.0 and shares["B"] == 40.0

    def test_long_empty_returns_none(self) -> None:
        assert composition([], "Fecha", "Cat", "Monto") is None

    def test_long_share_none_when_total_zero(self) -> None:
        rows = [
            {"Fecha": "2026-04-30", "Cat": "A", "Monto": 5.0},
            {"Fecha": "2026-04-30", "Cat": "B", "Monto": -5.0},
        ]
        comp = composition(rows, "Fecha", "Cat", "Monto")
        assert comp["total"] == 0.0
        assert all(b["share_pct"] is None for b in comp["breakdown"])

    def test_wide_shares(self) -> None:
        rows = [
            {"fecha": "2026-04-30", "Nacional": 46.43, "Extranjero": 53.57},
            {"fecha": "2026-03-31", "Nacional": 50.0, "Extranjero": 50.0},  # ignorada
        ]
        comp = composition_wide(rows, "fecha", ["Nacional", "Extranjero"])
        assert comp["fecha"] == "2026-04-30"
        shares = {b["categoria"]: b["share_pct"] for b in comp["breakdown"]}
        assert shares["Extranjero"] == pytest.approx(53.57, abs=0.01)
        assert shares["Nacional"] == pytest.approx(46.43, abs=0.01)

    def test_wide_empty_returns_none(self) -> None:
        assert composition_wide([], "fecha", ["A", "B"]) is None


# ─────────────────────────────────────────────────────────────────────────────
# compute_composition (tool, fetch mockeado)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeComposition:
    @pytest.mark.asyncio
    async def test_wide_mode(self) -> None:
        ds = _dataset("allocation_int_nac", columns=[
            ColumnSpec("fecha", "TIMESTAMP"),
            ColumnSpec("Nacional", "DOUBLE"),
            ColumnSpec("Extranjero", "DOUBLE"),
        ], unit="%")
        rows = [{"fecha": "2026-04-30", "Nacional": 46.43, "Extranjero": 53.57}]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows, date_col="fecha")):
            state = AgentState()
            result = await analytics.compute_composition(
                state=state, dataset_id="allocation_int_nac",
                value_columns=["Nacional", "Extranjero"],
            )
        assert result["mode"] == "wide"
        shares = {b["categoria"]: b["share_pct"] for b in result["breakdown"]}
        assert shares["Extranjero"] == pytest.approx(53.57, abs=0.01)
        # Se registró como serie CATEGÓRICA → barra (no línea temporal).
        assert state.series_used
        s = next(iter(state.series_used.values()))
        assert s["chart_type"] == "bar"
        assert {p[0] for p in s["points"]} == {"Nacional", "Extranjero"}
        assert s["first_date"] is None  # categórica: sin eje de fechas

    @pytest.mark.asyncio
    async def test_long_mode(self) -> None:
        ds = _dataset("allocation", columns=[
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Tipo_instrumento", "VARCHAR"),
            ColumnSpec("Monto_USD", "DOUBLE"),
        ], unit="MM USD")
        rows = [
            {"Fecha": "2026-04-30", "Tipo_instrumento": "BTP", "Monto_USD": 70.0},
            {"Fecha": "2026-04-30", "Tipo_instrumento": "BTU", "Monto_USD": 30.0},
        ]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            state = AgentState()
            result = await analytics.compute_composition(
                state=state, dataset_id="allocation",
                category_column="Tipo_instrumento", value_column="Monto_USD",
            )
        assert result["mode"] == "long"
        shares = {b["categoria"]: b["share_pct"] for b in result["breakdown"]}
        assert shares == {"BTP": 70.0, "BTU": 30.0}
        # Serie categórica → barra, puntos [categoría, valor] ordenados desc.
        s = next(iter(state.series_used.values()))
        assert s["chart_type"] == "bar"
        assert [p[0] for p in s["points"]] == ["BTP", "BTU"]
        assert s["points"][0][1] == pytest.approx(70.0)

    @pytest.mark.asyncio
    async def test_requires_a_mode(self) -> None:
        state = AgentState()
        result = await analytics.compute_composition(state=state, dataset_id="x")
        assert "error" in result
        assert "modo" in result["error"].lower()


@pytest.mark.unit
class TestComputeAggregate:
    @pytest.mark.asyncio
    async def test_grouped_total(self) -> None:
        ds = _dataset("dv01_spc_afp", columns=[
            ColumnSpec("fecha", "TIMESTAMP"),
            ColumnSpec("moneda", "VARCHAR"),
            ColumnSpec("dv01", "DOUBLE"),
        ], unit="MM USD por bp")
        rows = [
            {"fecha": "2026-05-20", "moneda": "US$", "dv01": 96.0},
            {"fecha": "2026-05-20", "moneda": "CLP", "dv01": 9.0},
        ]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows, date_col="fecha")):
            state = AgentState()
            r = await analytics.compute_aggregate(
                state=state, dataset_id="dv01_spc_afp", value_column="dv01", group_by="moneda",
            )
        assert r["mode"] == "grouped"
        assert r["total"] == pytest.approx(105.0)
        grupos = {g["grupo"]: g["valor"] for g in r["por_grupo"]}
        assert grupos == {"US$": 96.0, "CLP": 9.0}

    @pytest.mark.asyncio
    async def test_net_of_columns_with_sign(self) -> None:
        ds = _dataset("flujo_cambiario", columns=[
            ColumnSpec("Fecha", "VARCHAR"),
            ColumnSpec("Grupo_Sector", "VARCHAR"),
            ColumnSpec("Spot", "DOUBLE"),
            ColumnSpec("Forward", "DOUBLE"),
        ], unit="MM USD")
        rows = [{"Fecha": "2026-05-19", "Grupo_Sector": "AFP", "Spot": 0.0, "Forward": -495.0}]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            state = AgentState()
            r = await analytics.compute_aggregate(
                state=state, dataset_id="flujo_cambiario",
                value_columns=["Spot", "Forward"], filters={"Grupo_Sector": "AFP"},
            )
        assert r["mode"] == "net"
        assert r["neto"] == pytest.approx(-495.0)  # neto con signo, no %
        comps = {c["columna"]: c["valor"] for c in r["componentes"]}
        assert comps == {"Spot": 0.0, "Forward": -495.0}

    @pytest.mark.asyncio
    async def test_plain_total(self) -> None:
        ds = _dataset("x", columns=[
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Stock", "DOUBLE"),
        ], unit="MM CLP")
        rows = [
            {"Fecha": "2026-05-19", "Stock": 100.0},
            {"Fecha": "2026-05-19", "Stock": 50.0},
        ]
        with patch(f"{_MODULE}.fetch_rows_from_dataset", new=_mock_fetch(ds, rows)):
            state = AgentState()
            r = await analytics.compute_aggregate(
                state=state, dataset_id="x", value_column="Stock",
            )
        assert r["mode"] == "total"
        assert r["total"] == pytest.approx(150.0)

    @pytest.mark.asyncio
    async def test_requires_a_mode(self) -> None:
        state = AgentState()
        r = await analytics.compute_aggregate(state=state, dataset_id="x")
        assert "error" in r
