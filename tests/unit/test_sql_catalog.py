"""Tests del catálogo SQL agentic (Fase 4).

Cubre:
- catalog_loader: carga YAML, parse de entries, render_sql, resolve_date.
- discover_query: scoring, filtro por segmento, top_k.
- execute_query: entry no encontrada, ejecución real con DuckDB (si parquets existen).
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from banks_rag.infrastructure.sql.catalog_loader import (
    CatalogEntry,
    ParamSpec,
    _resolve_date,
    get_entry,
    load_catalog,
    render_sql,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_catalog_yaml(tmp_path: Path) -> Path:
    """Catálogo mínimo con 2 entries para tests."""
    content = """
version: "1.0"
snapshots_dir: data_pipeline/snapshots
queries:
  - query_id: usdclp_test
    name: "Tipo de cambio USD/CLP"
    description: "Serie diaria del dólar en pesos chilenos."
    segment: mercado_cambiario
    tags: [dolar, USD, CLP, tipo de cambio]
    parquet: fx.parquet
    sql: |
      SELECT strftime(date, '%Y-%m-%d') AS fecha, value AS usdclp
      FROM read_parquet('{snapshots_dir}/fx.parquet')
      WHERE series_id = 'usdclp'
        AND date >= '{fecha_inicio}' AND date <= '{fecha_fin}'
      ORDER BY date DESC LIMIT {limit}
    params:
      - name: fecha_inicio
        type: date
        default: "-365d"
      - name: fecha_fin
        type: date
        default: "hoy"
      - name: limit
        type: int
        default: 200
    unit: "CLP por USD"
    frequency: diario
    columns: [fecha, usdclp]

  - query_id: curva_btp_test
    name: "Curva BTP CLP"
    description: "Tasas bonos BTP en pesos chilenos."
    segment: renta_fija_chile
    tags: [BTP, bonos, pesos, CLP, curva soberana]
    parquet: bonos_clp.parquet
    sql: |
      SELECT fecha, btp_5y FROM read_parquet('{snapshots_dir}/bonos_clp.parquet')
      WHERE date >= '{fecha_inicio}' LIMIT {limit}
    params:
      - name: fecha_inicio
        type: date
        default: "-365d"
      - name: fecha_fin
        type: date
        default: "hoy"
      - name: limit
        type: int
        default: 100
    unit: "% anual"
    frequency: diario
    columns: [fecha, btp_5y]
"""
    p = tmp_path / "catalog.yaml"
    p.write_text(content)
    return p


@pytest.fixture
def entries(minimal_catalog_yaml: Path) -> list[CatalogEntry]:
    return load_catalog(minimal_catalog_yaml)


# ── Tests catalog_loader ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestCatalogLoader:
    def test_loads_entries(self, entries: list[CatalogEntry]) -> None:
        assert len(entries) == 2

    def test_entry_fields(self, entries: list[CatalogEntry]) -> None:
        e = entries[0]
        assert e.query_id == "usdclp_test"
        assert e.name == "Tipo de cambio USD/CLP"
        assert e.segment == "mercado_cambiario"
        assert "dolar" in e.tags
        assert e.unit == "CLP por USD"
        assert e.frequency == "diario"
        assert "fecha" in e.columns

    def test_params_parsed(self, entries: list[CatalogEntry]) -> None:
        e = entries[0]
        assert len(e.params) == 3
        p = e.params[0]
        assert p.name == "fecha_inicio"
        assert p.type == "date"
        assert p.default == "-365d"

    def test_get_entry_found(self, entries: list[CatalogEntry]) -> None:
        e = get_entry(entries, "curva_btp_test")
        assert e is not None
        assert e.query_id == "curva_btp_test"

    def test_get_entry_missing(self, entries: list[CatalogEntry]) -> None:
        assert get_entry(entries, "no_existe") is None

    def test_render_sql_substitutes_snapshots_dir(self, entries: list[CatalogEntry]) -> None:
        sql = render_sql(entries[0], "/data/snaps", {})
        assert "/data/snaps/fx.parquet" in sql

    def test_render_sql_substitutes_limit(self, entries: list[CatalogEntry]) -> None:
        sql = render_sql(entries[0], "/snaps", {"limit": 50})
        assert "50" in sql
        assert "{limit}" not in sql

    def test_render_sql_fecha_fin_hoy(self, entries: list[CatalogEntry]) -> None:
        sql = render_sql(entries[0], "/snaps", {})
        assert date.today().isoformat() in sql

    def test_render_sql_fecha_inicio_relative(self, entries: list[CatalogEntry]) -> None:
        sql = render_sql(entries[0], "/snaps", {"fecha_inicio": "-30d"})
        expected = (date.today() - timedelta(days=30)).isoformat()
        assert expected in sql

    def test_to_discovery_dict(self, entries: list[CatalogEntry]) -> None:
        d = entries[0].to_discovery_dict()
        assert d["query_id"] == "usdclp_test"
        assert "description" in d
        assert "params" in d
        assert isinstance(d["params"], list)


@pytest.mark.unit
class TestResolveDate:
    def test_hoy(self) -> None:
        assert _resolve_date("hoy") == date.today().isoformat()

    def test_today_english(self) -> None:
        assert _resolve_date("today") == date.today().isoformat()

    def test_negative_days(self) -> None:
        expected = (date.today() - timedelta(days=90)).isoformat()
        assert _resolve_date("-90d") == expected

    def test_negative_months(self) -> None:
        expected = (date.today() - timedelta(days=12 * 30)).isoformat()
        assert _resolve_date("-12m") == expected

    def test_iso_passthrough(self) -> None:
        assert _resolve_date("2023-06-15") == "2023-06-15"

    def test_invalid_falls_back_to_today(self) -> None:
        assert _resolve_date("garbage") == date.today().isoformat()


# ── Tests discover_query (tool) ───────────────────────────────────────────────

@pytest.mark.unit
class TestDiscoverQuery:
    def _run(self, coro) -> object:
        return asyncio.run(coro)

    def _mock_catalog(self, entries):
        return patch(
            "banks_rag.application.agent.tools.discover_query.load_catalog",
            return_value=entries,
        )

    def test_returns_relevant_entry(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        state = MagicMock()
        with self._mock_catalog(entries):
            result = self._run(discover_query(state=state, query="tipo de cambio dólar CLP"))
        assert result["n_results"] >= 1
        ids = [r["query_id"] for r in result["results"]]
        assert "usdclp_test" in ids

    def test_segment_filter(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        state = MagicMock()
        with self._mock_catalog(entries):
            result = self._run(
                discover_query(state=state, query="bonos", segment="renta_fija_chile")
            )
        assert all(r["segment"] == "renta_fija_chile" for r in result["results"])

    def test_unknown_segment_returns_error(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        state = MagicMock()
        with self._mock_catalog(entries):
            result = self._run(
                discover_query(state=state, query="test", segment="no_existe")
            )
        assert "error" in result

    def test_top_k_limits_results(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        state = MagicMock()
        with self._mock_catalog(entries):
            result = self._run(discover_query(state=state, query="datos", top_k=1))
        assert result["n_results"] <= 1

    def test_result_has_hint(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        state = MagicMock()
        with self._mock_catalog(entries):
            result = self._run(discover_query(state=state, query="dólar"))
        assert "hint" in result


# ── Tests execute_query (tool) ────────────────────────────────────────────────

@pytest.mark.unit
class TestExecuteQuery:
    def _run(self, coro) -> object:
        return asyncio.run(coro)

    def _mock_catalog(self, entries):
        return patch(
            "banks_rag.application.agent.tools.execute_query.load_catalog",
            return_value=entries,
        )

    def test_unknown_query_id_returns_error(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query
        state = MagicMock()
        with self._mock_catalog(entries):
            result = self._run(execute_query(state=state, query_id="no_existe"))
        assert "error" in result
        assert "available_query_ids" in result

    def test_duckdb_error_returns_error_dict(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query
        state = MagicMock()
        with self._mock_catalog(entries):
            with patch(
                "banks_rag.application.agent.tools.execute_query._run_duckdb",
                side_effect=RuntimeError("parquet not found"),
            ):
                result = self._run(execute_query(state=state, query_id="usdclp_test"))
        assert "error" in result
        assert "usdclp_test" in result["error"]

    def test_success_result_structure(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query
        state = MagicMock()
        fake_rows = [{"fecha": "2024-01-01", "usdclp": 900.5}]
        with self._mock_catalog(entries):
            with patch(
                "banks_rag.application.agent.tools.execute_query._run_duckdb",
                return_value=fake_rows,
            ):
                result = self._run(execute_query(state=state, query_id="usdclp_test"))
        assert result["query_id"] == "usdclp_test"
        assert result["rows"] == fake_rows
        assert result["n_rows"] == 1
        assert result["unit"] == "CLP por USD"
        assert result["truncated"] is False

    def test_limit_capped_at_500(self, entries: list[CatalogEntry]) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query, _MAX_ROWS
        state = MagicMock()
        captured_sql: list[str] = []

        def fake_run(sql: str) -> list[dict]:
            captured_sql.append(sql)
            return []

        with self._mock_catalog(entries):
            with patch(
                "banks_rag.application.agent.tools.execute_query._run_duckdb",
                side_effect=fake_run,
            ):
                self._run(execute_query(state=state, query_id="usdclp_test", limit=9999))
        assert str(_MAX_ROWS) in captured_sql[0]


# ── Test integración real con DuckDB (skip si no hay parquets) ────────────────

@pytest.mark.unit
class TestExecuteQueryIntegration:
    """Ejecuta contra parquets reales — se salta si no existen."""

    SNAPSHOTS = Path(__file__).parents[2] / "data_pipeline" / "snapshots"

    def _run(self, coro) -> object:
        return asyncio.run(coro)

    @pytest.mark.skipif(
        not (Path(__file__).parents[2] / "data_pipeline" / "snapshots" / "fx.parquet").exists(),
        reason="Parquets de DW no disponibles en este entorno",
    )
    def test_usdclp_real(self) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query
        state = MagicMock()
        result = self._run(
            execute_query(
                state=state,
                query_id="usdclp_historico",
                fecha_inicio="2024-01-01",
                fecha_fin="2024-03-31",
                limit=10,
            )
        )
        assert "error" not in result
        assert result["n_rows"] > 0
        assert "usdclp" in result["columns"]
