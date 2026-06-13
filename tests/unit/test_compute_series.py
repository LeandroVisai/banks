"""Unit tests de compute_series (serie a graficar leída del parquet REAL).

Mismo patrón que test_parquet_facts: parquets de prueba creados con DuckDB
(``COPY (VALUES …) TO``), sin pandas/pyarrow, en ``tmp_path``. Sin BD ni modelos.
"""

from __future__ import annotations

import duckdb
import pytest

from banks_rag.application.reporting.parquet_facts import (
    PlotData,
    _downsample,
    compute_series,
)
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset


def _ds(file: str, **kw) -> ParquetDataset:
    return ParquetDataset(
        id=kw.get("id", "ds"), file=file, name=kw.get("name", "Dataset"),
        description="", segment=kw.get("segment", "ffmm"), unit=kw.get("unit", "US$ Mill."),
        date_range=kw.get("date_range"), columns=kw.get("columns", []),
        chart_type=kw.get("chart_type", "line"),
    )


def _write(path, select_sql: str) -> None:
    duckdb.sql(f"COPY ({select_sql}) TO '{path.as_posix()}' (FORMAT parquet)")


@pytest.mark.unit
class TestComputeSeries:
    def test_single_series(self, tmp_path):
        p = tmp_path / "single.parquet"
        rows = [f"(DATE '2026-04-{d:02d}', {float(v)})" for d, v in [(26, 50), (27, 52), (28, 51), (29, 55)]]
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Valor)")
        plot = compute_series(_ds("single.parquet"), tmp_path)
        assert isinstance(plot, PlotData)
        assert plot.kind == "timeseries"
        assert len(plot.series) == 1
        assert plot.series[0].points[0] == ("2026-04-26", 50.0)
        assert plot.series[0].points[-1] == ("2026-04-29", 55.0)

    def test_categorical_multi_series(self, tmp_path):
        p = tmp_path / "cat.parquet"
        rows = []
        for d in ["2026-04-26", "2026-04-27", "2026-04-28"]:
            rows.append(f"(DATE '{d}', 'A', 100.0)")
            rows.append(f"(DATE '{d}', 'B', 10.0)")
        _write(p, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, tipo, Monto)")
        plot = compute_series(_ds("cat.parquet"), tmp_path)
        assert plot.kind == "timeseries"
        labels = {s.label for s in plot.series}
        assert labels == {"A", "B"}
        # ordenadas por |nivel| último: A (100) antes que B (10)
        assert plot.series[0].label == "A"

    def test_snapshot_composition(self, tmp_path):
        p = tmp_path / "snap.parquet"
        _write(p, "SELECT * FROM (VALUES ('DAP', 60.0), ('BB', 40.0)) t(Tipo, mmusd)")
        plot = compute_series(_ds("snap.parquet", chart_type="pie"), tmp_path)
        assert plot.kind == "snapshot"
        assert plot.family == "pie"
        pts = plot.series[0].points
        assert pts[0] == ("DAP", 60.0)  # ordenado desc
        assert pts[1] == ("BB", 40.0)

    def test_missing_parquet_returns_none(self, tmp_path):
        assert compute_series(_ds("no_existe.parquet"), tmp_path) is None

    def test_non_iso_varchar_date_dropped(self, tmp_path):
        # Fecha VARCHAR no-ISO no castea a DATE → fila descartada (serie recortada,
        # señal VISIBLE del problema de datos, no error silencioso).
        p = tmp_path / "bad_date.parquet"
        _write(
            p,
            "SELECT * FROM (VALUES ('2026-04-26', 5.0), ('31/12/2026', 9.0)) t(fecha, Valor)",
        )
        plot = compute_series(_ds("bad_date.parquet"), tmp_path)
        # solo la fila ISO sobrevive
        assert plot is not None
        all_pts = [pt for s in plot.series for pt in s.points]
        assert all_pts == [("2026-04-26", 5.0)]


@pytest.mark.unit
class TestDownsample:
    def test_caps_and_keeps_endpoints(self):
        pts = [(str(i), float(i)) for i in range(1000)]
        out = _downsample(pts, max_points=50)
        assert len(out) <= 50
        assert out[0] == ("0", 0.0)
        assert out[-1] == ("999", 999.0)

    def test_below_cap_unchanged(self):
        pts = [("a", 1.0), ("b", 2.0)]
        assert _downsample(pts, max_points=200) == pts
