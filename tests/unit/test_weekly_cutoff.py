"""Corte semanal común + variación semanal canónica (informe ffmm semanal).

Cubre los 4 puntos del cambio: clasificación de alta frecuencia, ``weekly_cutoff``
(mín de máximos de los datasets diarios/semanales, excluyendo mensuales),
``weekly_delta`` (resolución única at-or-before) y la CONSISTENCIA entre la celda
del gráfico (``stacked_by_bucket``) y la del texto (``compute_facts``).

Parquets de prueba con DuckDB (``COPY (VALUES …) TO``), sin pandas/pyarrow.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pytest

from banks_rag.application.reporting.parquet_facts import (
    compute_facts,
    is_high_frequency,
    weekly_cutoff,
    weekly_delta,
)
from banks_rag.application.reporting.series_transforms import stacked_by_bucket
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset


def _ds(file: str, **kw) -> ParquetDataset:
    return ParquetDataset(
        id=kw.get("id", file.split(".")[0]), file=file, name=kw.get("name", "Dataset"),
        description="", segment="ffmm", unit="US$ Mill.",
        date_range=None, columns=[], chart_type="line",
    )


def _write(path, select_sql: str) -> None:
    duckdb.sql(f"COPY ({select_sql}) TO '{path.as_posix()}' (FORMAT parquet)")


def _daily(path, last_iso: str, n: int = 12) -> None:
    """Serie diaria simple (Fecha, Valor) que termina en ``last_iso``."""
    end = date.fromisoformat(last_iso)
    rows = [
        f"(DATE '{(end - timedelta(days=n - 1 - i)).isoformat()}', {float(100 + i)})"
        for i in range(n)
    ]
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Valor)")


def _weekly(path, last_iso: str, n: int = 8) -> None:
    """Serie SEMANAL (gap 7 días) que termina en ``last_iso``."""
    end = date.fromisoformat(last_iso)
    rows = [
        f"(DATE '{(end - timedelta(days=7 * (n - 1 - i))).isoformat()}', {float(50 + i)})"
        for i in range(n)
    ]
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Valor)")


def _monthly(path, last_iso: str, n: int = 6) -> None:
    """Serie MENSUAL (gap ~30 días) que termina en ``last_iso``."""
    end = date.fromisoformat(last_iso)
    rows = [
        f"(DATE '{(end - timedelta(days=30 * (n - 1 - i))).isoformat()}', {float(10 + i)})"
        for i in range(n)
    ]
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Valor)")


@pytest.mark.unit
class TestHighFrequency:
    def test_daily_is_high_freq(self):
        ds = ["2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18"]
        assert is_high_frequency(ds)

    def test_weekly_is_high_freq(self):
        ds = ["2026-05-25", "2026-06-01", "2026-06-08", "2026-06-15"]
        assert is_high_frequency(ds)  # gap 7 ≤ umbral 10

    def test_monthly_is_not_high_freq(self):
        ds = ["2026-02-28", "2026-03-31", "2026-04-30", "2026-05-31"]
        assert not is_high_frequency(ds)  # gap ~30

    def test_single_date_is_not_high_freq(self):
        assert not is_high_frequency(["2026-06-21"])


@pytest.mark.unit
class TestWeeklyCutoff:
    def test_min_of_high_freq_maxes_excludes_monthly(self, tmp_path):
        # A diaria→21, B semanal→22, C diaria→22, M mensual→30-abr.
        _daily(tmp_path / "a.parquet", "2026-06-21")
        _weekly(tmp_path / "b.parquet", "2026-06-22")
        _daily(tmp_path / "c.parquet", "2026-06-22")
        _monthly(tmp_path / "m.parquet", "2026-04-30")
        datasets = [_ds("a.parquet"), _ds("b.parquet"), _ds("c.parquet"), _ds("m.parquet")]
        # El mensual (abril) NO arrastra el corte: queda el mín de {21,22,22} = 21.
        assert weekly_cutoff(datasets, tmp_path) == "2026-06-21"

    def test_none_when_no_high_freq(self, tmp_path):
        _monthly(tmp_path / "m.parquet", "2026-04-30")
        assert weekly_cutoff([_ds("m.parquet")], tmp_path) is None

    def test_ignores_missing_parquet(self, tmp_path):
        _daily(tmp_path / "a.parquet", "2026-06-21")
        assert weekly_cutoff([_ds("a.parquet"), _ds("falta.parquet")], tmp_path) == "2026-06-21"


@pytest.mark.unit
class TestWeeklyDelta:
    def test_levels_at_or_before_both_ends(self):
        # Niveles diarios; asof=30, semana = [23,30]. valor_asof(30)=... ; base at-or-before(23).
        series = [(f"2026-04-{d:02d}", float(v)) for d, v in
                  [(20, 90), (23, 100), (27, 110), (30, 120)]]
        wd = weekly_delta(series, "2026-04-30", days=7)
        # base = valor en o antes de 23 → (23,100); fin = valor en o antes de 30 → 120.
        assert wd["valor_inicio"] == 100.0
        assert wd["valor_fin"] == 120.0
        assert wd["cambio_absoluto"] == 20.0

    def test_levels_fallback_to_first_when_no_point_before_start(self):
        series = [("2026-04-26", 110.0), ("2026-04-30", 118.0)]
        wd = weekly_delta(series, "2026-04-30", days=7)  # start=23, sin punto ≤ 23
        assert wd["valor_inicio"] == 110.0  # cae al primer punto
        assert wd["valor_fin"] == 118.0

    def test_flows_sum_over_window(self):
        series = [(f"2026-04-{d:02d}", float(v)) for d, v in
                  [(24, 5), (26, 10), (28, -3), (30, 8)]]
        wd = weekly_delta(series, "2026-04-30", days=7, is_flow=True)
        assert wd["is_flow_sum"] is True
        assert wd["flujo_periodo"] == 20.0  # 5+10-3+8

    def test_equivalent_to_old_chart_formula_when_asof_is_last(self):
        # Garantía de no-regresión: weekly_delta(asof=last) == la fórmula vieja de
        # stacked_by_bucket (s[-1] menos valor at-or-before(start)).
        series = [(f"2026-06-{d:02d}", float(v)) for d, v in
                  [(8, 200), (12, 210), (15, 205), (19, 230), (22, 240)]]
        last = series[-1][0]
        days = 7
        start = (date.fromisoformat(last) - timedelta(days=days)).isoformat()
        base = next((v for iso, v in reversed(series) if iso <= start), series[0][1])
        old = series[-1][1] - base
        new = weekly_delta(series, last, days=days)["cambio_absoluto"]
        assert new == old


# Serie DCV diaria con un SALTO el último día (no lineal): así anclar a 21 vs 22
# da variaciones distintas (incluir/excluir el salto). 12 días hasta el 2026-06-22.
_DCV_LAST = "2026-06-22"
_DCV_VALS = [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1300]


def _dcv_series() -> list[tuple[str, float]]:
    end = date.fromisoformat(_DCV_LAST)
    n = len(_DCV_VALS)
    return [((end - timedelta(days=n - 1 - i)).isoformat(), float(v)) for i, v in enumerate(_DCV_VALS)]


def _dcv_parquet(path) -> None:
    """Parquet DCV (Fecha, Bucket, Tipo, Stock_USD): 1 plazo x 1 instrumento, diario."""
    rows = [f"(DATE '{iso}', '0-3m', 'BB', {v})" for iso, v in _dcv_series()]
    _write(path, "SELECT * FROM (VALUES " + ", ".join(rows) + ") t(Fecha, Bucket, Tipo, Stock_USD)")


def _bb_cell(ds, tmp_path, asof: str) -> float:
    plot = stacked_by_bucket(ds, tmp_path, {"window": "7d", "weekly_asof": asof})
    bb = next(s for s in plot.series if s.label == "BB")
    return dict(bb.points)["0-3m"]


@pytest.mark.unit
class TestChartTextConsistency:
    def test_stacked_bar_cell_equals_text_weekly_delta(self, tmp_path):
        # MISMA fecha de corte (21) en gráfico y texto → MISMA variación semanal.
        _dcv_parquet(tmp_path / "dcv.parquet")
        ds = _ds("dcv.parquet", id="variacion_stock_ffmm")
        asof = "2026-06-21"  # corte común (un día antes del máximo del parquet)
        cell = _bb_cell(ds, tmp_path, asof)
        wd = weekly_delta(_dcv_series(), asof, days=7)  # misma serie reconstruida
        assert cell == pytest.approx(wd["cambio_absoluto"])

    def test_cutoff_shifts_weekly_value(self, tmp_path):
        # Anclar a 21 vs 22 da valores distintos (incluir/excluir el salto del 22).
        _dcv_parquet(tmp_path / "dcv.parquet")
        ds = _ds("dcv.parquet", id="variacion_stock_ffmm")
        assert _bb_cell(ds, tmp_path, "2026-06-21") != _bb_cell(ds, tmp_path, "2026-06-22")


@pytest.mark.unit
class TestComputeFactsWeeklyAsof:
    def test_cutoff_anchors_all_windows(self, tmp_path):
        # weekly_asof ancla TODAS las ventanas (semanal Y mensual) al MISMO corte,
        # para que Flujos y DCV compartan T, T-7 y T-30 (los mismos días).
        _daily(tmp_path / "s.parquet", "2026-06-22", n=40)
        facts = compute_facts(
            _ds("s.parquet"), tmp_path, [("última semana", 7), ("último mes", 30)],
            weekly_asof="2026-06-21",
        )
        wins = {w["label"]: w for w in facts["windows_variation"]}
        assert wins["última semana"]["hasta"] == "2026-06-21"
        assert wins["último mes"]["hasta"] == "2026-06-21"  # mensual TAMBIÉN al corte

    def test_no_cutoff_uses_own_max(self, tmp_path):
        # Sin weekly_asof, cada ventana se ancla al máximo del propio parquet.
        _daily(tmp_path / "s.parquet", "2026-06-22", n=40)
        facts = compute_facts(_ds("s.parquet"), tmp_path, [("última semana", 7), ("último mes", 30)])
        wins = {w["label"]: w for w in facts["windows_variation"]}
        assert wins["última semana"]["hasta"] == "2026-06-22"
        assert wins["último mes"]["hasta"] == "2026-06-22"
