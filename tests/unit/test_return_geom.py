"""Rentabilidad acumulada GEOMÉTRICA (return_index) — despeje del retorno diario,
acumulado YTD y retorno compuesto por ventana para el párrafo.

El índice de retorno acumulado viene en FRACCIÓN (1+i = ∏(1+r)). La variación de
una ventana es el retorno COMPUESTO ``(1+i_end)/(1+i_start)-1``, no la resta del
índice. Parquets de prueba con DuckDB (sin pandas).
"""

from __future__ import annotations

import duckdb
import pytest

from banks_rag.application.reporting.parquet_facts import (
    compute_facts,
    facts_to_text,
    geom_return,
    geom_ytd,
)
from banks_rag.application.reporting.series_transforms import ytd_return_geom
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset


def _ds(file: str, **kw) -> ParquetDataset:
    return ParquetDataset(
        id=kw.get("id", "retorno_acum_ffmm"), file=file, name="Rentabilidad", description="",
        segment="ffmm", unit="%", date_range=None, columns=[], chart_type="line",
        value_scale=kw.get("value_scale", 100.0), value_kind=kw.get("value_kind", "return_index"),
    )


def _write(path, select_sql: str) -> None:
    duckdb.sql(f"COPY ({select_sql}) TO '{path.as_posix()}' (FORMAT parquet)")


# Índice acumulado en FRACCIÓN: arranca 2025, sigue en 2026. Crecimiento ~compuesto.
def _index_parquet(path) -> None:
    rows = [
        ("2025-12-29", 0.100, 0.200),
        ("2025-12-30", 0.101, 0.198),  # base YTD (último de 2025)
        ("2026-01-02", 0.1050, 0.1950),
        ("2026-06-01", 0.1200, 0.2100),
        ("2026-06-08", 0.1230, 0.2050),  # t-7
        ("2026-06-15", 0.1265, 0.2010),  # last
    ]
    vals = ", ".join(f"(DATE '{d}', {a}, {b})" for d, a, b in rows)
    _write(path, f"SELECT * FROM (VALUES {vals}) t(Fecha, \"Tipo 1\", \"Tipo 2\")")


@pytest.mark.unit
class TestGeomHelpers:
    def test_geom_return_compound(self):
        pts = [("2026-01-01", 0.0), ("2026-01-08", 0.10), ("2026-01-15", 0.21)]
        # full: (1.21)/(1.0)-1 = 0.21 ; semana 08→15: (1.21)/(1.10)-1 = 0.10
        assert geom_return(pts, "2026-01-01", "2026-01-15") == pytest.approx(0.21)
        assert geom_return(pts, "2026-01-08", "2026-01-15") == pytest.approx(0.10)

    def test_geom_return_at_or_before(self):
        pts = [("2026-06-01", 0.12), ("2026-06-08", 0.123), ("2026-06-15", 0.1265)]
        # start 2026-06-09 no existe → usa el ≤ (08). end 2026-06-20 → usa el ≤ (15).
        r = geom_return(pts, "2026-06-09", "2026-06-20")
        assert r == pytest.approx((1.1265) / (1.123) - 1)

    def test_geom_return_none_empty(self):
        assert geom_return([], "2026-01-01", "2026-01-08") is None

    def test_geom_ytd_rebases_from_prior_year_close(self):
        pts = [("2025-12-30", 0.101), ("2026-01-02", 0.105), ("2026-06-15", 0.1265)]
        ytd = geom_ytd(pts, "2026-01-01")
        assert [iso for iso, _ in ytd] == ["2026-01-02", "2026-06-15"]  # solo 2026
        # base = 0.101 (cierre 2025). último YTD = (1.1265)/(1.101)-1.
        assert ytd[-1][1] == pytest.approx((1.1265) / (1.101) - 1)


@pytest.mark.unit
class TestYtdReturnGeomTransform:
    def test_funds_and_geometric_ytd(self, tmp_path):
        _index_parquet(tmp_path / "r.parquet")
        ds = _ds("r.parquet")
        plot = ytd_return_geom(ds, tmp_path, {"funds": ["Tipo 1", "Tipo 2"]})
        labels = [s.label for s in plot.series]
        assert labels == ["Tipo 1", "Tipo 2"]
        t1 = next(s for s in plot.series if s.label == "Tipo 1")
        # YTD Tipo 1 = (1+0.1265)/(1+0.101)-1, en FRACCIÓN (curated x100 → %).
        assert t1.points[-1][1] == pytest.approx((1.1265) / (1.101) - 1)
        assert all(iso >= "2026-01-01" for iso, _ in t1.points)  # solo YTD

    def test_default_funds_when_unspecified(self, tmp_path):
        _index_parquet(tmp_path / "r.parquet")
        # Sin funds → default Tipo 1/2/3/6; el parquet solo tiene 1 y 2 → esos.
        plot = ytd_return_geom(_ds("r.parquet"), tmp_path, {})
        assert [s.label for s in plot.series] == ["Tipo 1", "Tipo 2"]

    def test_funds_all_includes_every_column(self, tmp_path):
        # funds="all" → todos los fondos disponibles del parquet (aquí Tipo 1 y 2).
        _index_parquet(tmp_path / "r.parquet")
        plot = ytd_return_geom(_ds("r.parquet"), tmp_path, {"funds": "all"})
        assert [s.label for s in plot.series] == ["Tipo 1", "Tipo 2"]


@pytest.mark.unit
class TestReturnIndexFacts:
    def test_compute_facts_geometric_windows(self, tmp_path):
        _index_parquet(tmp_path / "r.parquet")
        facts = compute_facts(_ds("r.parquet"), tmp_path, [("última semana", 7), ("último mes", 30)])
        assert facts["shape"] == "return_index"
        assert facts["aggregatable"] is False
        t1 = next(f for f in facts["por_fondo"] if f["fondo"] == "Tipo 1")
        wk = next(v for v in t1["ventanas"] if v["label"] == "última semana")
        # semanal Tipo 1 (last 15 vs t-7 08): (1.1265)/(1.123)-1, x100 (value_scale).
        esperado_pct = ((1.1265) / (1.123) - 1) * 100
        assert wk["retorno_pct"] == pytest.approx(round(esperado_pct, 4))
        # índice acumulado al cierre en %: 0.1265 x100 = 12.65%.
        assert t1["indice_acum_pct"] == pytest.approx(12.65)

    def test_text_states_direction_and_magnitude(self, tmp_path):
        _index_parquet(tmp_path / "r.parquet")
        facts = compute_facts(_ds("r.parquet"), tmp_path, [("última semana", 7)])
        text = facts_to_text(facts)
        assert "RETORNO COMPUESTO" in text
        assert "rentó positivo" in text   # Tipo 1 sube en la semana
        assert "rentó negativo" in text   # Tipo 2 baja en la semana
