"""Unit tests de parquet_facts (cómputo en Python sin LLM ni tools).

Los parquets de prueba se crean con DuckDB (``COPY (VALUES …) TO``), evitando
pyarrow/pandas. Cada test arma un parquet pequeño en ``tmp_path`` y verifica la
detección de roles y los hechos calculados.
"""

from __future__ import annotations

import duckdb
import pytest

from banks_rag.application.reporting.parquet_facts import (
    compute_facts,
    detect_roles,
    facts_to_text,
)
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset

_W = [("última semana", 7), ("último mes", 30)]


def _ds(file: str, **kw) -> ParquetDataset:
    return ParquetDataset(
        id=kw.get("id", "ds"), file=file, name=kw.get("name", "Dataset"),
        description="", segment=kw.get("segment", "ffmm"), unit=kw.get("unit", "US$ Mill."),
        date_range=kw.get("date_range"), columns=kw.get("columns", []),
        chart_type=kw.get("chart_type", "line"),
    )


def _write(path, select_sql: str) -> None:
    duckdb.sql(f"COPY ({select_sql}) TO '{path.as_posix()}' (FORMAT parquet)")


def _categorical_parquet(path) -> None:
    # Serie diaria con 2 categorías sobre 5 días consecutivos.
    rows = []
    for i, d in enumerate(["2026-04-26", "2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30"]):
        rows.append(f"(DATE '{d}', 'A', {100.0 + i})")
        rows.append(f"(DATE '{d}', 'B', {10.0 + i})")
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, tipo, Monto)")


def _single_parquet(path) -> None:
    rows = [f"(DATE '2026-04-{d:02d}', {float(v)})" for d, v in
            [(26, 50), (27, 52), (28, 51), (29, 55), (30, 60)]]
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Valor)")


def _snapshot_parquet(path) -> None:
    _write(path, "SELECT * FROM (VALUES ('DAP', 60.0), ('BB', 40.0)) t(Tipo_Instrumento, mmusd)")


@pytest.mark.unit
class TestDetectRoles:
    def test_categorical(self, tmp_path):
        p = tmp_path / "cat.parquet"
        _categorical_parquet(p)
        roles = detect_roles(p)
        assert roles.date_col == "Fecha"
        assert roles.category_cols == ["tipo"]
        assert roles.value_cols == ["Monto"]

    def test_snapshot_has_no_date(self, tmp_path):
        p = tmp_path / "snap.parquet"
        _snapshot_parquet(p)
        roles = detect_roles(p)
        assert roles.date_col is None
        assert roles.category_cols == ["Tipo_Instrumento"]
        assert roles.value_cols == ["mmusd"]


@pytest.mark.unit
class TestComputeFacts:
    def test_missing_parquet_returns_none(self, tmp_path):
        ds = _ds("no_existe.parquet")
        assert compute_facts(ds, tmp_path, _W) is None

    def test_categorical_facts(self, tmp_path):
        _categorical_parquet(tmp_path / "cat.parquet")
        facts = compute_facts(_ds("cat.parquet"), tmp_path, _W)
        assert facts["shape"] == "timeseries_categorical"
        assert facts["last_date"] == "2026-04-30"
        # Total última semana: 26-abr (A100+B10=110) → 30-abr (A104+B14=118).
        wk = next(w for w in facts["total_ventanas"] if w["label"] == "última semana")
        assert wk["variacion"]["valor_inicio"] == 110.0
        assert wk["variacion"]["valor_fin"] == 118.0
        # Composición al corte: A=104, B=14 → A ~88%.
        comp = facts["composicion_corte"]["breakdown"]
        assert comp[0]["categoria"] == "A"
        assert comp[0]["share_pct"] == pytest.approx(88.14, abs=0.1)
        # Detalle por categoría: top por nivel = A.
        assert facts["por_categoria"][0]["categoria"] == "A"
        # Señales nuevas: drivers del movimiento + clasificación de tendencia.
        assert facts.get("contribuciones")
        drivers = facts["contribuciones"][0]["drivers"]
        assert drivers[0]["categoria"] in ("A", "B")
        assert facts["tendencia"]["clasificacion"] in (
            "aceleración", "desaceleración", "reversión", "estable", "en línea con la tendencia",
        )

    def test_single_facts(self, tmp_path):
        _single_parquet(tmp_path / "s.parquet")
        facts = compute_facts(_ds("s.parquet"), tmp_path, _W)
        assert facts["shape"] == "timeseries_single"
        assert facts["estadisticas"]["ultimo_valor"] == 60.0
        wk = next(w for w in facts["windows_variation"] if w["label"] == "última semana")
        assert wk["variacion"]["valor_inicio"] == 50.0
        assert wk["variacion"]["valor_fin"] == 60.0

    def test_snapshot_facts(self, tmp_path):
        _snapshot_parquet(tmp_path / "snap.parquet")
        facts = compute_facts(_ds("snap.parquet"), tmp_path, _W)
        assert facts["shape"] == "snapshot"
        assert facts["last_date"] is None
        bd = facts["composition"]["breakdown"]
        assert facts["composition"]["total"] == 100.0
        assert bd[0]["categoria"] == "DAP" and bd[0]["share_pct"] == 60.0


@pytest.mark.unit
class TestFactsToText:
    def test_renders_numbers_no_crash(self, tmp_path):
        _categorical_parquet(tmp_path / "cat.parquet")
        text = facts_to_text(compute_facts(_ds("cat.parquet", unit="US$ Mill."), tmp_path, _W))
        assert "Composición" in text
        assert "última semana" in text
        assert "US$ Mill." in text

    def test_snapshot_text(self, tmp_path):
        _snapshot_parquet(tmp_path / "snap.parquet")
        text = facts_to_text(compute_facts(_ds("snap.parquet"), tmp_path, _W))
        assert "Corte transversal" in text
        assert "DAP" in text
