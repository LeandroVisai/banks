"""Tests para las analytics tools del agente (Fase A multi-agente).

Dos bloques:
  - funciones puras de ``infrastructure/sql/series_analytics`` (sin I/O);
  - tools registradas, con ``load_catalog`` y ``run_duckdb`` mockeados.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from banks_rag.application.agent.tools import analytics
from banks_rag.domain.agent import AgentState
from banks_rag.infrastructure.sql.catalog_loader import CatalogEntry, ParamSpec
from banks_rag.infrastructure.sql.series_analytics import (
    anomaly_check,
    clean_series,
    descriptive_stats,
    spread,
    variation,
)

_MODULE = "banks_rag.application.agent.tools.analytics"


# ─────────────────────────────────────────────────────────────────────────────
# Funciones puras
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
        series = [(f"2024-01-{i:02d}", float(i)) for i in range(1, 11)]  # 1..10
        stats = descriptive_stats(series)
        assert stats["n_observaciones"] == 10
        assert stats["media"] == 5.5
        assert stats["minimo"] == 1.0
        assert stats["maximo"] == 10.0
        assert stats["ultimo_valor"] == 10.0
        # 9 de 10 valores son < 10 → percentil 90
        assert stats["percentil_ultimo_valor"] == 90.0

    def test_descriptive_stats_empty(self) -> None:
        assert descriptive_stats([]) is None

    def test_anomaly_flags_outlier(self) -> None:
        # Ventana con varianza pequeña (~10 ±0.5) + un salto a 100 al final.
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
        # Ventana perfectamente plana → z-score indefinido; el salto es anómalo.
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
        b = [("2024-01-01", 2.0), ("2024-01-03", 3.0)]  # falta 2024-01-02
        spr = spread(a, b)
        assert spr["n_observaciones_comunes"] == 2
        assert spr["spread_inicio"] == 3.0   # 5 - 2
        assert spr["spread_fin"] == 4.0      # 7 - 3
        assert spr["cambio_spread"] == 1.0

    def test_spread_no_common_dates(self) -> None:
        a = [("2024-01-01", 5.0)]
        b = [("2024-02-01", 2.0)]
        assert spread(a, b) is None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers para mockear el catálogo
# ─────────────────────────────────────────────────────────────────────────────


def _entry(query_id: str, columns: list[str]) -> CatalogEntry:
    """CatalogEntry mínima con SQL renderizable (run_duckdb se mockea aparte)."""
    return CatalogEntry(
        query_id=query_id,
        name=f"Serie {query_id}",
        description="serie de prueba",
        segment="seg",
        tags=["t"],
        parquet="x.parquet",
        sql="SELECT * FROM t WHERE date >= '{fecha_inicio}' "
            "AND date <= '{fecha_fin}' LIMIT {limit}",
        params=[
            ParamSpec("fecha_inicio", "date", "-90d"),
            ParamSpec("fecha_fin", "date", "hoy"),
            ParamSpec("limit", "int", "500"),
        ],
        unit="% anual",
        frequency="diario",
        columns=columns,
    )


def _patch(entries: list[CatalogEntry], rows: list[dict]):
    """Parchea load_catalog + run_duckdb del módulo analytics."""
    return (
        patch(f"{_MODULE}.load_catalog", return_value=entries),
        patch(f"{_MODULE}.run_duckdb", return_value=rows),
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute_variation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeVariation:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        state = AgentState()
        entry = _entry("usdclp_test", ["fecha", "usdclp"])
        rows = [
            {"fecha": "2024-03-01", "usdclp": 990.0},
            {"fecha": "2024-01-01", "usdclp": 900.0},
        ]
        p1, p2 = _patch([entry], rows)
        with p1, p2:
            result = await analytics.compute_variation(
                state=state, query_id="usdclp_test", column="usdclp",
            )
        assert result["valor_inicio"] == 900.0
        assert result["valor_fin"] == 990.0
        assert result["variacion_pct"] == 10.0
        assert result["unit"] == "% anual"
        assert len(state.series_used) == 1

    @pytest.mark.asyncio
    async def test_unknown_query_id(self) -> None:
        entry = _entry("otra", ["fecha", "v"])
        p1, p2 = _patch([entry], [])
        with p1, p2:
            result = await analytics.compute_variation(
                state=AgentState(), query_id="no_existe", column="v",
            )
        assert "error" in result
        assert "available_query_ids" in result

    @pytest.mark.asyncio
    async def test_bad_column(self) -> None:
        entry = _entry("q", ["fecha", "usdclp"])
        p1, p2 = _patch([entry], [{"fecha": "2024-01-01", "usdclp": 1.0}])
        with p1, p2:
            result = await analytics.compute_variation(
                state=AgentState(), query_id="q", column="inexistente",
            )
        assert "error" in result
        assert result["available_columns"] == ["usdclp"]

    @pytest.mark.asyncio
    async def test_insufficient_observations(self) -> None:
        entry = _entry("q", ["fecha", "v"])
        p1, p2 = _patch([entry], [{"fecha": "2024-01-01", "v": 1.0}])
        with p1, p2:
            result = await analytics.compute_variation(
                state=AgentState(), query_id="q", column="v",
            )
        assert "error" in result
        assert "2 observaciones" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# compute_spread
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeSpread:
    @pytest.mark.asyncio
    async def test_intra_query_spread(self) -> None:
        """Ambas columnas en la misma query → una sola lectura."""
        state = AgentState()
        entry = _entry("curva", ["fecha", "btp_2y", "btp_10y"])
        rows = [
            {"fecha": "2024-01-01", "btp_2y": 5.0, "btp_10y": 6.0},
            {"fecha": "2024-02-01", "btp_2y": 5.0, "btp_10y": 6.5},
        ]
        p1, p2 = _patch([entry], rows)
        with p1, p2:
            result = await analytics.compute_spread(
                state=state,
                query_id_a="curva", column_a="btp_10y",
                query_id_b="curva", column_b="btp_2y",
            )
        assert result["spread_inicio"] == 1.0   # 6.0 - 5.0
        assert result["spread_fin"] == 1.5      # 6.5 - 5.0
        assert result["definicion"] == "spread = btp_10y - btp_2y"

    @pytest.mark.asyncio
    async def test_no_common_dates(self) -> None:
        entry = _entry("curva", ["fecha", "a", "b"])
        # 'a' tiene valor solo donde 'b' es None y viceversa → sin par válido.
        rows = [
            {"fecha": "2024-01-01", "a": 5.0, "b": None},
            {"fecha": "2024-02-01", "a": None, "b": 2.0},
        ]
        p1, p2 = _patch([entry], rows)
        with p1, p2:
            result = await analytics.compute_spread(
                state=AgentState(),
                query_id_a="curva", column_a="a",
                query_id_b="curva", column_b="b",
            )
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# get_series_stats
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGetSeriesStats:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        state = AgentState()
        entry = _entry("q", ["fecha", "v"])
        rows = [{"fecha": f"2024-01-{i:02d}", "v": float(i)} for i in range(1, 11)]
        p1, p2 = _patch([entry], rows)
        with p1, p2:
            result = await analytics.get_series_stats(
                state=state, query_id="q", column="v",
            )
        assert result["n_observaciones"] == 10
        assert result["media"] == 5.5
        assert "periodo" in result

    @pytest.mark.asyncio
    async def test_empty_series(self) -> None:
        entry = _entry("q", ["fecha", "v"])
        p1, p2 = _patch([entry], [])
        with p1, p2:
            result = await analytics.get_series_stats(
                state=AgentState(), query_id="q", column="v",
            )
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# detect_anomaly
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDetectAnomaly:
    @pytest.mark.asyncio
    async def test_flags_anomaly(self) -> None:
        entry = _entry("q", ["fecha", "v"])
        rows = [
            {"fecha": f"2024-01-{i:02d}", "v": 10.0 + (0.5 if i % 2 else -0.5)}
            for i in range(1, 10)
        ]
        rows.append({"fecha": "2024-01-10", "v": 100.0})
        p1, p2 = _patch([entry], rows)
        with p1, p2:
            result = await analytics.detect_anomaly(
                state=AgentState(), query_id="q", column="v",
            )
        assert result["es_anomalia"] is True
        assert result["lookback_days"] == 180

    @pytest.mark.asyncio
    async def test_insufficient_points(self) -> None:
        entry = _entry("q", ["fecha", "v"])
        p1, p2 = _patch([entry], [{"fecha": "2024-01-01", "v": 1.0}])
        with p1, p2:
            result = await analytics.detect_anomaly(
                state=AgentState(), query_id="q", column="v",
            )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_clamps_params(self) -> None:
        entry = _entry("q", ["fecha", "v"])
        rows = [{"fecha": f"2024-01-{i:02d}", "v": 10.0} for i in range(1, 6)]
        p1, p2 = _patch([entry], rows)
        with p1, p2:
            result = await analytics.detect_anomaly(
                state=AgentState(), query_id="q", column="v",
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
        # Una entry por cada query_id de _SNAPSHOT_INDICATORS, con su columna.
        entries = [
            _entry(query_id, ["fecha", column])
            for _, _, query_id, column in analytics._SNAPSHOT_INDICATORS
        ]
        # run_duckdb mockeado devuelve la misma fila con todas las columnas.
        row = {
            "fecha": "2026-05-15",
            "usdclp": 950.0, "cobre_usd_lb": 4.2,
            "btp_10y": 5.8, "btu_10y": 2.3,
            "tib_tasa": 5.0, "ust_10y": 4.4, "spread_mipr_3m": 0.1,
        }
        p1, p2 = _patch(entries, [row])
        with p1, p2:
            result = await analytics.get_market_snapshot(state=AgentState())

        claves = {ind["clave"] for ind in result["indicadores"]}
        assert "usdclp" in claves
        assert "bei_10y" in claves  # derivado BTP 10A - BTU 10A
        bei = next(i for i in result["indicadores"] if i["clave"] == "bei_10y")
        assert bei["valor"] == pytest.approx(3.5)  # 5.8 - 2.3
