"""Tests de la tool plot_series: construcción del spec Vega-Lite (sin BD/LLM).

El acceso al parquet (fetch_rows_from_dataset) se mockea; se valida que la tool
arma un spec Vega-Lite v5 correcto según la forma de los datos y lo registra en
el AgentState."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from banks_rag.application.agent.tools.plot_series import plot_series
from banks_rag.domain.agent import AgentState
from banks_rag.infrastructure.sql.parquet_catalog_loader import ColumnSpec, ParquetDataset


def _dataset(id_: str, columns: list[ColumnSpec]) -> ParquetDataset:
    return ParquetDataset(
        id=id_, file=f"{id_}.parquet", name=f"Dataset {id_}",
        description="x", segment="seg", unit="%",
        date_range=["2024-01-01", "2026-05-20"], columns=columns,
    )


def _patch_fetch(dataset, rows, date_col, select_cols):
    """Parchea fetch_rows_from_dataset (async) para devolver datos fijos."""
    async def _fake(dataset_id, **kwargs):
        return dataset, rows, date_col, select_cols
    return patch(
        "banks_rag.application.agent.tools.plot_series.fetch_rows_from_dataset",
        side_effect=_fake,
    )


@pytest.mark.unit
class TestPlotSeries:
    @pytest.mark.asyncio
    async def test_long_format_colors_by_category(self) -> None:
        # Fecha + 1 numérica + categórica -> color por categoría.
        ds = _dataset("btp", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Valor", "DOUBLE"),
            ColumnSpec("Variable", "VARCHAR"),
        ])
        rows = [
            {"Fecha": "2026-05-20T00:00:00", "Valor": 5.68, "Variable": "10Y"},
            {"Fecha": "2026-05-19T00:00:00", "Valor": 5.60, "Variable": "10Y"},
        ]
        state = AgentState()
        with _patch_fetch(ds, rows, "Fecha", ["Fecha", "Valor", "Variable"]):
            out = await plot_series(state, "btp", chart_type="line")

        assert "error" not in out
        assert out["chart_id"] == 1
        assert len(state.charts) == 1
        spec = state.charts[0]["spec"]
        assert spec["$schema"].endswith("v5.json")
        assert spec["mark"]["type"] == "line"
        assert spec["encoding"]["x"]["type"] == "temporal"
        assert spec["encoding"]["y"]["field"] == "Valor"
        assert spec["encoding"]["color"]["field"] == "Variable"
        assert "transform" not in spec  # forma larga, sin fold
        assert len(spec["data"]["values"]) == 2

    @pytest.mark.asyncio
    async def test_wide_format_folds_numeric_columns(self) -> None:
        # Fecha + 2 numéricas -> fold a (serie, valor).
        ds = _dataset("clp", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("CLP", "DOUBLE"),
            ColumnSpec("Monto", "BIGINT"),
        ])
        rows = [{"Fecha": "2026-05-20T00:00:00", "CLP": 940.0, "Monto": 1200}]
        state = AgentState()
        with _patch_fetch(ds, rows, "Fecha", ["Fecha", "CLP", "Monto"]):
            out = await plot_series(state, "clp")

        spec = state.charts[0]["spec"]
        assert spec["transform"][0]["fold"] == ["CLP", "Monto"]
        assert spec["encoding"]["color"]["field"] == "serie"
        assert spec["encoding"]["y"]["field"] == "valor"

    @pytest.mark.asyncio
    async def test_snapshot_without_date_errors(self) -> None:
        ds = _dataset("snap", [ColumnSpec("Bucket", "VARCHAR"), ColumnSpec("MM_USD", "DOUBLE")])
        rows = [{"Bucket": "0-2Y", "MM_USD": 10.0}]
        state = AgentState()
        with _patch_fetch(ds, rows, None, ["Bucket", "MM_USD"]):
            out = await plot_series(state, "snap")
        assert "error" in out and "snapshot" in out["error"].lower()
        assert state.charts == []

    @pytest.mark.asyncio
    async def test_no_numeric_columns_errors(self) -> None:
        ds = _dataset("cats", [ColumnSpec("Fecha", "TIMESTAMP"), ColumnSpec("Tipo", "VARCHAR")])
        rows = [{"Fecha": "2026-05-20T00:00:00", "Tipo": "A"}]
        state = AgentState()
        with _patch_fetch(ds, rows, "Fecha", ["Fecha", "Tipo"]):
            out = await plot_series(state, "cats")
        assert "error" in out and "numéric" in out["error"].lower()

    @pytest.mark.asyncio
    async def test_chart_type_respected_and_default(self) -> None:
        ds = _dataset("x", [ColumnSpec("Fecha", "TIMESTAMP"), ColumnSpec("V", "DOUBLE")])
        rows = [{"Fecha": "2026-05-20T00:00:00", "V": 1.0}]
        state = AgentState()
        with _patch_fetch(ds, rows, "Fecha", ["Fecha", "V"]):
            out = await plot_series(state, "x", chart_type="bar")
        assert state.charts[0]["spec"]["mark"]["type"] == "bar"
        # tipo inválido -> default line
        state2 = AgentState()
        with _patch_fetch(ds, rows, "Fecha", ["Fecha", "V"]):
            await plot_series(state2, "x", chart_type="pie")
        assert state2.charts[0]["spec"]["mark"]["type"] == "line"
