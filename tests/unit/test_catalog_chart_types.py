"""Tests del chart_type canónico del parquet_catalog y su propagación.

Cubre:
- ``domain.agent.chart_types``: reducción chart_type → familia base / marca Vega.
- Coherencia del catálogo REAL: todo dataset declara un chart_type conocido.
- ``parquet_catalog_loader``: parseo de ``chart_type`` (default "line").
- ``infer_chart_type`` con hint del catálogo (área apilada, barras, etc.).
- ``plot_series``: el default de la marca sale del catálogo, no del LLM.
- ``execute_query``: las series registradas heredan la familia del dataset.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from banks_rag.domain.agent import AgentState
from banks_rag.domain.agent.agent_state import infer_chart_type
from banks_rag.domain.agent.chart_types import (
    BASE_FAMILIES,
    CHART_FAMILY,
    VEGA_MARK_BY_FAMILY,
    chart_family,
    vega_mark,
)
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ColumnSpec,
    ParquetDataset,
    load_parquet_catalog,
)

# ─────────────────────────────────────────────────────────────────────────────
# chart_family / vega_mark
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestChartFamily:
    def test_known_families(self) -> None:
        assert chart_family("stacked_area") == "area"
        assert chart_family("grouped_bar") == "grouped_bar"
        assert chart_family("stacked_bar_interactive") == "stacked_bar"
        assert chart_family("bar_neto_cum_interactive") == "stacked_bar"
        assert chart_family("market_monitor_table") == "table"
        assert chart_family("scatter") == "point"
        assert chart_family("spc_curve") == "line"
        assert chart_family("pie") == "pie"

    def test_unknown_or_empty_degrades_to_line(self) -> None:
        assert chart_family("tipo_que_no_existe") == "line"
        assert chart_family("") == "line"
        assert chart_family(None) == "line"

    def test_every_mapping_targets_a_base_family(self) -> None:
        assert set(CHART_FAMILY.values()) <= BASE_FAMILIES
        assert set(VEGA_MARK_BY_FAMILY) == BASE_FAMILIES

    def test_vega_mark(self) -> None:
        assert vega_mark("stacked_area") == "area"
        assert vega_mark("grouped_bar") == "bar"
        assert vega_mark("pos_delta_interactive") == "bar"
        assert vega_mark("scatter") == "point"
        assert vega_mark("line") == "line"
        # No graficables como serie degradan a una marca neutra/simple.
        assert vega_mark("market_monitor_table") == "line"
        assert vega_mark("pie") == "bar"
        assert vega_mark("desconocido") == "line"


# ─────────────────────────────────────────────────────────────────────────────
# Coherencia del catálogo real
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRealCatalogChartTypes:
    def test_all_datasets_declare_known_chart_type(self) -> None:
        # Si un chart_type del YAML no está en CHART_FAMILY, el render
        # degradaría a línea en silencio: este guard lo hace visible.
        entries = load_parquet_catalog()
        unknown = sorted({
            e.chart_type for e in entries if e.chart_type not in CHART_FAMILY
        })
        assert not unknown, f"chart_type sin mapping a familia: {unknown}"

    def test_catalog_ids_unique(self) -> None:
        entries = load_parquet_catalog()
        ids = [e.id for e in entries]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        assert not dupes, f"ids duplicados en parquet_catalog: {dupes}"

    def test_monitor_mercados_datasets_present(self) -> None:
        entries = load_parquet_catalog()
        # Segmento canónico del nuevo catálogo: "color_mercados"
        cm = {e.id for e in entries if e.segment == "color_mercados"}
        assert {"cm_bolsas", "cm_fx", "cm_riesgo", "cm_tasas"} <= cm
        assert all(
            e.chart_type == "market_monitor_table"
            for e in entries if e.id in cm
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loader: parseo de chart_type
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestLoaderChartType:
    def test_parses_and_defaults(self, tmp_path: Path) -> None:
        content = """
version: "1.0"
parquet_dir: data_pipeline/parquet

datasets:
  - id: con_tipo
    file: con_tipo.parquet
    chart_type: "stacked_area"
    name: "Con tipo"
    description: "x"
    segment: seg
    unit: u
    date_range: ["2024-01-01", "2024-12-31"]
    columns:
      - {name: Fecha, type: TIMESTAMP}
      - {name: Valor, type: DOUBLE}

  - id: sin_tipo
    file: sin_tipo.parquet
    name: "Sin tipo"
    description: "x"
    segment: seg
    unit: u
    date_range: ["2024-01-01", "2024-12-31"]
    columns:
      - {name: Fecha, type: TIMESTAMP}
      - {name: Valor, type: DOUBLE}
"""
        p = tmp_path / "parquet_catalog.yaml"
        p.write_text(content, encoding="utf-8")
        entries = load_parquet_catalog(p)
        by_id = {e.id: e for e in entries}
        assert by_id["con_tipo"].chart_type == "stacked_area"
        assert by_id["sin_tipo"].chart_type == "line"

    def test_to_dict_includes_chart_type(self, tmp_path: Path) -> None:
        ds = ParquetDataset(
            id="x", file="x.parquet", name="X", description="d",
            segment="s", unit="u", date_range=None,
            columns=[ColumnSpec("Fecha", "TIMESTAMP")],
            chart_type="grouped_bar",
        )
        assert ds.to_dict()["chart_type"] == "grouped_bar"


# ─────────────────────────────────────────────────────────────────────────────
# infer_chart_type con hint del catálogo
# ─────────────────────────────────────────────────────────────────────────────


def _temporal_points(n: int) -> list[list]:
    return [[f"2026-01-{i + 1:02d}", float(i)] for i in range(n)]


@pytest.mark.unit
class TestInferChartTypeWithHint:
    def test_area_hint_wins_for_temporal(self) -> None:
        pts = _temporal_points(10)
        assert infer_chart_type(pts, x_is_date=True, hint="stacked_area") == "area"

    def test_bar_hint_with_few_points_is_bar(self) -> None:
        pts = _temporal_points(8)
        assert infer_chart_type(pts, x_is_date=True, hint="grouped_bar") == "bar"

    def test_bar_hint_with_many_points_degrades_to_line(self) -> None:
        pts = _temporal_points(120)
        assert infer_chart_type(pts, x_is_date=True, hint="stacked_bar") == "line"

    def test_categorical_is_bar_regardless_of_hint(self) -> None:
        pts = [["A", 1.0], ["B", 2.0], ["C", 3.0]]
        assert infer_chart_type(pts, x_is_date=False, hint="stacked_area") == "bar"

    def test_too_few_points_is_bar_regardless_of_hint(self) -> None:
        pts = _temporal_points(2)
        assert infer_chart_type(pts, x_is_date=True, hint="line") == "bar"

    def test_line_and_unknown_hints_keep_line(self) -> None:
        pts = _temporal_points(10)
        assert infer_chart_type(pts, x_is_date=True, hint="line") == "line"
        assert infer_chart_type(pts, x_is_date=True, hint="market_monitor_table") == "line"
        assert infer_chart_type(pts, x_is_date=True, hint=None) == "line"

    def test_add_series_uses_hint(self) -> None:
        state = AgentState()
        rows = [{"date": f"2026-01-{i + 1:02d}", "value": float(i)} for i in range(10)]
        state.add_series("ds:v", {"series_name": "V"}, rows, chart_hint="stacked_area")
        assert state.series_used["ds:v"]["chart_type"] == "area"

    def test_add_series_explicit_overrides_hint(self) -> None:
        state = AgentState()
        rows = [{"date": f"2026-01-{i + 1:02d}", "value": float(i)} for i in range(10)]
        state.add_series(
            "ds:v", {"series_name": "V"}, rows,
            chart_type="grouped_bar", chart_hint="stacked_area",
        )
        assert state.series_used["ds:v"]["chart_type"] == "grouped_bar"


# ─────────────────────────────────────────────────────────────────────────────
# plot_series: default desde el catálogo
# ─────────────────────────────────────────────────────────────────────────────


def _dataset(id_: str, chart_type: str = "line") -> ParquetDataset:
    return ParquetDataset(
        id=id_, file=f"{id_}.parquet", name=f"Dataset {id_}",
        description="x", segment="seg", unit="%",
        date_range=["2024-01-01", "2026-05-20"],
        columns=[
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Bucket", "VARCHAR", values=["A", "B"]),
            ColumnSpec("Valor", "DOUBLE"),
        ],
        chart_type=chart_type,
    )


def _patch_plot_fetch(dataset, rows, date_col, select_cols):
    async def _fake(dataset_id, **kwargs):
        return dataset, rows, date_col, select_cols
    return patch(
        "banks_rag.application.agent.tools.plot_series.fetch_rows_from_dataset",
        side_effect=_fake,
    )


@pytest.mark.unit
class TestPlotSeriesCatalogDefault:
    @pytest.mark.asyncio
    async def test_default_mark_comes_from_catalog(self) -> None:
        from banks_rag.application.agent.tools.plot_series import plot_series
        ds = _dataset("stock", chart_type="stacked_area")
        rows = [
            {"Fecha": "2026-05-20T00:00:00", "Bucket": "A", "Valor": 1.0},
            {"Fecha": "2026-05-19T00:00:00", "Bucket": "B", "Valor": 2.0},
        ]
        state = AgentState()
        with _patch_plot_fetch(ds, rows, "Fecha", ["Fecha", "Bucket", "Valor"]):
            out = await plot_series(state, "stock")  # sin chart_type explícito
        assert out["chart_type"] == "stacked_area"   # canónico del catálogo
        assert out["mark"] == "area"                 # marca Vega derivada
        chart = state.charts[0]
        assert chart["chart_type"] == "stacked_area"
        assert chart["spec"]["mark"]["type"] == "area"

    @pytest.mark.asyncio
    async def test_explicit_simple_mark_overrides_catalog(self) -> None:
        from banks_rag.application.agent.tools.plot_series import plot_series
        ds = _dataset("stock", chart_type="stacked_area")
        rows = [{"Fecha": "2026-05-20T00:00:00", "Bucket": "A", "Valor": 1.0}]
        state = AgentState()
        with _patch_plot_fetch(ds, rows, "Fecha", ["Fecha", "Bucket", "Valor"]):
            out = await plot_series(state, "stock", chart_type="bar")
        assert out["chart_type"] == "bar"
        assert state.charts[0]["spec"]["mark"]["type"] == "bar"

    @pytest.mark.asyncio
    async def test_interactive_type_reduces_to_bar_mark(self) -> None:
        from banks_rag.application.agent.tools.plot_series import plot_series
        ds = _dataset("flujo", chart_type="bar_neto_cum_interactive")
        rows = [{"Fecha": "2026-05-20T00:00:00", "Bucket": "A", "Valor": 1.0}]
        state = AgentState()
        with _patch_plot_fetch(ds, rows, "Fecha", ["Fecha", "Bucket", "Valor"]):
            out = await plot_series(state, "flujo")
        assert out["chart_type"] == "bar_neto_cum_interactive"
        assert out["mark"] == "bar"
        assert state.charts[0]["spec"]["mark"]["type"] == "bar"


# ─────────────────────────────────────────────────────────────────────────────
# execute_query: las series heredan la familia del dataset
# ─────────────────────────────────────────────────────────────────────────────

_EXEC_MODULE = "banks_rag.application.agent.tools.execute_query"


@pytest.mark.unit
class TestExecuteQuerySeriesChartType:
    @pytest.mark.asyncio
    async def test_series_inherit_catalog_family(self) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query
        ds = _dataset("stock", chart_type="stacked_area")
        rows = [
            {"Fecha": f"2026-05-{20 - i:02d}", "Valor": float(i)} for i in range(5)
        ]
        state = AgentState()
        with patch(
            f"{_EXEC_MODULE}.fetch_rows_from_dataset",
            new=AsyncMock(return_value=(ds, rows, "Fecha", ["Fecha", "Valor"])),
        ):
            await execute_query(state=state, dataset_id="stock")
        assert state.series_used["stock:Valor"]["chart_type"] == "area"
