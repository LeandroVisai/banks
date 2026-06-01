"""Tests del catálogo parquet + tools del agente (post-switch SQL→parquet).

Cubre:
- ``parquet_catalog_loader``: carga del YAML, parseo, search_datasets, get_dataset.
- ``_parquet_query``: ``date_column``, ``build_fetch_sql`` (forma + validación),
  ``_normalize_filters``.
- ``discover_query`` tool: scoring, filtro por segmento, top_k.
- ``execute_query`` tool: dataset desconocido, parquet faltante, happy path con
  registro de series en ``state.series_used``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from banks_rag.application.agent.tools._parquet_query import (
    build_fetch_sql,
    date_column,
    find_date_column,
    resolve_columns,
    resolve_dataset_id,
)
from banks_rag.domain.agent import AgentState
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ColumnSpec,
    ParquetDataset,
    get_dataset,
    load_parquet_catalog,
    search_datasets,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_parquet_catalog_yaml(tmp_path: Path) -> Path:
    """Catálogo mínimo con 2 datasets (uno simple, uno long-format con enum)."""
    content = """
version: "1.0"
parquet_dir: data_pipeline/parquet

datasets:
  - id: usdclp_test
    file: clp_monto_test.parquet
    name: "USD/CLP test"
    description: "Serie diaria del dólar"
    segment: mercado_cambiario
    unit: "CLP por USD"
    date_range: ["2024-01-01", "2024-12-31"]
    columns:
      - {name: Fecha, type: TIMESTAMP}
      - {name: CLP, type: DOUBLE}

  - id: btp_test
    file: btp_test.parquet
    name: "Curva BTP test"
    description: "Curva soberana CLP por tenor"
    segment: renta_fija_chile
    unit: "% anual"
    date_range: ["2024-01-01", "2024-12-31"]
    columns:
      - {name: Fecha, type: TIMESTAMP}
      - {name: Tenor, type: VARCHAR, values: ["2Y", "5Y", "10Y"]}
      - {name: Valor, type: DOUBLE}
"""
    p = tmp_path / "parquet_catalog.yaml"
    # encoding utf-8 explícito: en Windows el default es cp1252 y los acentos
    # del contenido (ó, í…) se escribirían en cp1252, pero load_parquet_catalog
    # lee utf-8 → UnicodeDecodeError. En POSIX el default ya es utf-8.
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def datasets(minimal_parquet_catalog_yaml: Path) -> list[ParquetDataset]:
    return load_parquet_catalog(minimal_parquet_catalog_yaml)


def _dataset(
    id_: str,
    columns: list[ColumnSpec] | None = None,
    segment: str = "seg",
    unit: str = "u",
) -> ParquetDataset:
    """Helper para construir un ParquetDataset en memoria."""
    cols = columns or [
        ColumnSpec("Fecha", "TIMESTAMP"),
        ColumnSpec("Valor", "DOUBLE"),
    ]
    return ParquetDataset(
        id=id_,
        file=f"{id_}.parquet",
        name=f"Dataset {id_}",
        description=f"desc {id_}",
        segment=segment,
        unit=unit,
        date_range=["2024-01-01", "2024-12-31"],
        columns=cols,
    )


# ─────────────────────────────────────────────────────────────────────────────
# parquet_catalog_loader
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestParquetCatalogLoader:
    def test_loads_datasets(self, datasets: list[ParquetDataset]) -> None:
        assert len(datasets) == 2
        ids = {d.id for d in datasets}
        assert {"usdclp_test", "btp_test"} == ids

    def test_columns_parsed(self, datasets: list[ParquetDataset]) -> None:
        btp = get_dataset(datasets, "btp_test")
        assert btp is not None
        tenor = next(c for c in btp.columns if c.name == "Tenor")
        assert tenor.type == "VARCHAR"
        assert tenor.values == ["2Y", "5Y", "10Y"]

    def test_get_dataset_missing(self, datasets: list[ParquetDataset]) -> None:
        assert get_dataset(datasets, "no_existe") is None

    def test_search_returns_relevant(self, datasets: list[ParquetDataset]) -> None:
        results = search_datasets(datasets, "tipo de cambio dolar usd clp")
        assert results[0].id == "usdclp_test"

    def test_search_segment_filter(self, datasets: list[ParquetDataset]) -> None:
        results = search_datasets(datasets, "bonos", segment="renta_fija_chile")
        assert all(d.segment == "renta_fija_chile" for d in results)

    def test_to_dict_shape(self, datasets: list[ParquetDataset]) -> None:
        d = get_dataset(datasets, "btp_test").to_dict()
        assert d["id"] == "btp_test"
        assert any(c["name"] == "Tenor" and c.get("values") == ["2Y", "5Y", "10Y"]
                   for c in d["columns"])


# ─────────────────────────────────────────────────────────────────────────────
# _parquet_query: date_column + build_fetch_sql + validación
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDateColumn:
    def test_prefers_timestamp(self) -> None:
        ds = _dataset("x", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Valor", "DOUBLE"),
        ])
        assert date_column(ds) == "Fecha"

    def test_fallback_to_name(self) -> None:
        # Sin TIMESTAMP, cae a nombres convencionales.
        ds = _dataset("x", [
            ColumnSpec("Fecha", "VARCHAR"),  # mal tipado pero matchea por nombre
            ColumnSpec("Valor", "DOUBLE"),
        ])
        assert date_column(ds) == "Fecha"

    def test_raises_when_no_date(self) -> None:
        ds = _dataset("x", [
            ColumnSpec("X", "DOUBLE"),
            ColumnSpec("Y", "DOUBLE"),
        ])
        with pytest.raises(ValueError, match="columna de fecha"):
            date_column(ds)

    def test_find_date_column_returns_none_for_snapshot(self) -> None:
        # Snapshot transversal (sin columna temporal): find_ NO lanza, retorna None.
        ds = _dataset("snap", [
            ColumnSpec("Bucket", "VARCHAR"),
            ColumnSpec("MM_USD", "DOUBLE"),
        ])
        assert find_date_column(ds) is None


@pytest.mark.unit
class TestResolveDatasetId:
    """Fase 1: resuelve dataset_id alucinado al id real más cercano."""

    def _entries(self):
        return [
            _dataset("spread_btp_spc"),
            _dataset("btp_curva"),
            _dataset("clp_monto"),
        ]

    def test_exact_match_no_note(self) -> None:
        rid, note = resolve_dataset_id(self._entries(), "btp_curva")
        assert rid == "btp_curva" and note is None

    def test_case_insensitive(self) -> None:
        rid, note = resolve_dataset_id(self._entries(), "BTP_Curva")
        assert rid == "btp_curva" and note is not None

    def test_hallucinated_suffix_resolves(self) -> None:
        # Caso real visto en los logs del 7B/27B-IQ2.
        rid, note = resolve_dataset_id(self._entries(), "spread_btp_spc_plazo")
        assert rid == "spread_btp_spc" and "no existe" in note

    def test_unrelated_returns_none(self) -> None:
        rid, note = resolve_dataset_id(self._entries(), "xyz_totalmente_otro")
        assert rid is None and note is None


@pytest.mark.unit
class TestResolveColumns:
    def test_case_insensitive_and_fuzzy(self) -> None:
        ds = _dataset("clp", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("CLP", "DOUBLE"),
            ColumnSpec("Monto transado", "BIGINT"),
        ])
        # 'fecha' (case) y 'usdclp' (fuzzy -> CLP).
        resolved, notes = resolve_columns(ds, ["fecha", "usdclp"])
        assert resolved[0] == "Fecha"
        assert resolved[1] == "CLP"
        assert len(notes) == 2

    def test_unresolvable_passes_through(self) -> None:
        ds = _dataset("x", [ColumnSpec("Fecha", "TIMESTAMP"), ColumnSpec("Valor", "DOUBLE")])
        resolved, notes = resolve_columns(ds, ["columna_inventada_zzz"])
        assert resolved == ["columna_inventada_zzz"]  # _validate_columns dará el error


@pytest.mark.unit
class TestBuildFetchSql:
    PARQUET_DIR = Path("/data/parquet")

    def test_basic_select_all(self) -> None:
        ds = _dataset("usdclp_test")
        sql, date_col, cols, _ = build_fetch_sql(ds, parquet_dir=self.PARQUET_DIR)
        assert date_col == "Fecha"
        assert cols == ["Fecha", "Valor"]
        assert 'SELECT "Fecha", "Valor"' in sql
        # Nombre de archivo, sin asumir separador (Windows usa '\', POSIX '/').
        assert "usdclp_test.parquet" in sql
        assert "ORDER BY \"Fecha\" DESC" in sql
        assert "LIMIT 200" in sql

    def test_date_range_quoted(self) -> None:
        ds = _dataset("x")
        sql, _, _, _ = build_fetch_sql(
            ds, parquet_dir=self.PARQUET_DIR,
            fecha_inicio="2024-01-01", fecha_fin="2024-12-31",
        )
        assert "\"Fecha\" >= '2024-01-01'" in sql
        assert "\"Fecha\" <= '2024-12-31'" in sql

    def test_snapshot_without_date_no_order_by(self) -> None:
        # Dataset snapshot (sin fecha): date_col None, sin ORDER BY, lee igual.
        ds = _dataset("snap", [
            ColumnSpec("Bucket", "VARCHAR"),
            ColumnSpec("MM_USD", "DOUBLE"),
        ])
        sql, date_col, cols, _ = build_fetch_sql(ds, parquet_dir=self.PARQUET_DIR)
        assert date_col is None
        assert cols == ["Bucket", "MM_USD"]
        assert "ORDER BY" not in sql
        assert "LIMIT 200" in sql

    def test_snapshot_rejects_date_filter(self) -> None:
        ds = _dataset("snap", [ColumnSpec("Bucket", "VARCHAR"), ColumnSpec("MM_USD", "DOUBLE")])
        with pytest.raises(ValueError, match="no admite filtro temporal"):
            build_fetch_sql(ds, parquet_dir=self.PARQUET_DIR, fecha_inicio="2024-01-01")

    def test_filter_equality_for_enum_column(self) -> None:
        ds = _dataset("btp_test", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Tenor", "VARCHAR", values=["2Y", "10Y"]),
            ColumnSpec("Valor", "DOUBLE"),
        ])
        sql, _, _, _ = build_fetch_sql(
            ds, parquet_dir=self.PARQUET_DIR,
            columns=["Valor"], filters={"Tenor": "10Y"},
        )
        assert "\"Tenor\" = '10Y'" in sql

    def test_filter_in_list(self) -> None:
        ds = _dataset("btp_test", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Tenor", "VARCHAR", values=["2Y", "5Y", "10Y"]),
            ColumnSpec("Valor", "DOUBLE"),
        ])
        sql, _, _, _ = build_fetch_sql(
            ds, parquet_dir=self.PARQUET_DIR,
            filters={"Tenor": ["2Y", "10Y"]},
        )
        assert "\"Tenor\" IN ('2Y', '10Y')" in sql

    def test_limit_capped_at_max(self) -> None:
        ds = _dataset("x")
        sql, _, _, _ = build_fetch_sql(
            ds, parquet_dir=self.PARQUET_DIR, limit=9999,
        )
        assert "LIMIT 500" in sql

    def test_invalid_date_raises(self) -> None:
        ds = _dataset("x")
        with pytest.raises(ValueError, match="fecha_inicio"):
            build_fetch_sql(
                ds, parquet_dir=self.PARQUET_DIR,
                fecha_inicio="ayer",
            )

    def test_invalid_column_raises(self) -> None:
        ds = _dataset("x")
        with pytest.raises(ValueError, match="columnas inválidas"):
            build_fetch_sql(
                ds, parquet_dir=self.PARQUET_DIR,
                columns=["NoExiste"],
            )

    def test_filter_enum_value_rejected(self) -> None:
        ds = _dataset("btp_test", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Tenor", "VARCHAR", values=["2Y", "10Y"]),
            ColumnSpec("Valor", "DOUBLE"),
        ])
        with pytest.raises(ValueError, match="fuera del enum"):
            build_fetch_sql(
                ds, parquet_dir=self.PARQUET_DIR,
                filters={"Tenor": "100Y"},
            )

    def test_filter_unknown_column_rejected(self) -> None:
        ds = _dataset("x")
        with pytest.raises(ValueError, match="no existe"):
            build_fetch_sql(
                ds, parquet_dir=self.PARQUET_DIR,
                filters={"Banco": "BCI"},
            )

    def test_identifier_with_space_quoted(self) -> None:
        ds = _dataset("tib", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Spread TIB-TPM", "DOUBLE"),
        ])
        sql, _, _, _ = build_fetch_sql(
            ds, parquet_dir=self.PARQUET_DIR,
            columns=["Spread TIB-TPM"],
        )
        assert '"Spread TIB-TPM"' in sql

    def test_value_with_apostrophe_escaped(self) -> None:
        ds = _dataset("x", [
            ColumnSpec("Fecha", "TIMESTAMP"),
            ColumnSpec("Banco", "VARCHAR"),
            ColumnSpec("Valor", "DOUBLE"),
        ])
        sql, _, _, _ = build_fetch_sql(
            ds, parquet_dir=self.PARQUET_DIR,
            filters={"Banco": "O'Higgins"},
        )
        # Apostrophe doblada (escape SQL): O''Higgins
        assert "O''Higgins" in sql


# ─────────────────────────────────────────────────────────────────────────────
# discover_query (tool)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDiscoverQuery:
    def _run(self, coro):
        return asyncio.run(coro)

    def _mock_catalog(self, entries):
        return patch(
            "banks_rag.application.agent.tools.discover_query.load_parquet_catalog",
            return_value=entries,
        )

    def test_returns_relevant_dataset(self, datasets: list[ParquetDataset]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        with self._mock_catalog(datasets):
            result = self._run(
                discover_query(state=MagicMock(), query="tipo de cambio dolar usd clp"),
            )
        assert result["n_results"] >= 1
        ids = [r["id"] for r in result["results"]]
        assert "usdclp_test" in ids

    def test_segment_filter(self, datasets: list[ParquetDataset]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        with self._mock_catalog(datasets):
            result = self._run(
                discover_query(state=MagicMock(), query="bonos",
                               segment="renta_fija_chile"),
            )
        assert all(r["segment"] == "renta_fija_chile" for r in result["results"])

    def test_unknown_segment_soft_resolves(self, datasets: list[ParquetDataset]) -> None:
        # Segmento desconocido NO debe fallar (era el error duro de los logs que
        # dejaba al modelo sin datos): busca en todo el catálogo y avisa.
        from banks_rag.application.agent.tools.discover_query import discover_query
        with self._mock_catalog(datasets):
            result = self._run(
                discover_query(state=MagicMock(), query="tipo de cambio",
                               segment="no_existe"),
            )
        assert "error" not in result
        assert result["n_results"] >= 1
        assert "no reconocido" in result.get("segment_note", "")

    def test_segment_alias_resolves(self, datasets: list[ParquetDataset]) -> None:
        # Un alias del dominio ("AFP") se resuelve al segmento canónico. Con el
        # catálogo de prueba (sin fondos_pension) cae al aviso de no-reconocido,
        # así que validamos contra el catálogo real vía search_datasets aparte.
        from banks_rag.domain_knowledge.financial_aliases import resolve_segment
        assert resolve_segment("AFP", {"fondos_pension"}) == "fondos_pension"
        assert resolve_segment("fx", {"mercado_cambiario"}) == "mercado_cambiario"

    def test_top_k_limits_results(self, datasets: list[ParquetDataset]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        with self._mock_catalog(datasets):
            result = self._run(
                discover_query(state=MagicMock(), query="datos", top_k=1),
            )
        assert result["n_results"] <= 1

    def test_result_has_hint(self, datasets: list[ParquetDataset]) -> None:
        from banks_rag.application.agent.tools.discover_query import discover_query
        with self._mock_catalog(datasets):
            result = self._run(
                discover_query(state=MagicMock(), query="cualquier cosa"),
            )
        assert "hint" in result


# ─────────────────────────────────────────────────────────────────────────────
# execute_query (tool)
# ─────────────────────────────────────────────────────────────────────────────

_EXEC_MODULE = "banks_rag.application.agent.tools.execute_query"


@pytest.mark.unit
class TestExecuteQuery:
    @pytest.mark.asyncio
    async def test_unknown_dataset_id(self) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query
        with patch(
            f"{_EXEC_MODULE}.fetch_rows_from_dataset",
            new=AsyncMock(side_effect=ValueError("dataset_id desconocido: 'x'")),
        ):
            result = await execute_query(state=AgentState(), dataset_id="x")
        assert "error" in result
        assert "desconocido" in result["error"]
        assert result["dataset_id"] == "x"

    @pytest.mark.asyncio
    async def test_parquet_file_missing(self) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query
        with patch(
            f"{_EXEC_MODULE}.fetch_rows_from_dataset",
            new=AsyncMock(side_effect=FileNotFoundError("no existe en /x.parquet")),
        ):
            result = await execute_query(state=AgentState(), dataset_id="x")
        assert "error" in result
        assert "no existe" in result["error"]

    @pytest.mark.asyncio
    async def test_happy_path_registers_series(self) -> None:
        from banks_rag.application.agent.tools.execute_query import execute_query
        ds = _dataset("usdclp_test", unit="CLP por USD")
        rows = [
            {"Fecha": "2024-02-01", "Valor": 950.0},
            {"Fecha": "2024-01-01", "Valor": 900.0},
        ]
        state = AgentState()
        with patch(
            f"{_EXEC_MODULE}.fetch_rows_from_dataset",
            new=AsyncMock(return_value=(ds, rows, "Fecha", ["Fecha", "Valor"], [])),
        ):
            result = await execute_query(
                state=state, dataset_id="usdclp_test", columns=["Valor"],
            )
        assert result["dataset_id"] == "usdclp_test"
        assert result["unit"] == "CLP por USD"
        assert result["columns"] == ["Fecha", "Valor"]
        assert result["n_rows"] == 2
        assert result["truncated"] is False
        # Una serie por cada columna no-fecha
        assert "usdclp_test:Valor" in state.series_used
        assert state.series_used["usdclp_test:Valor"]["n_observations"] == 2

    @pytest.mark.asyncio
    async def test_truncated_flag_at_max_rows(self) -> None:
        from banks_rag.application.agent.tools.execute_query import (
            _MAX_ROWS,
            execute_query,
        )
        ds = _dataset("x")
        rows = [{"Fecha": f"2024-01-{i:02d}", "Valor": float(i)}
                for i in range(1, _MAX_ROWS + 1)]
        with patch(
            f"{_EXEC_MODULE}.fetch_rows_from_dataset",
            new=AsyncMock(return_value=(ds, rows, "Fecha", ["Fecha", "Valor"], [])),
        ):
            result = await execute_query(state=AgentState(), dataset_id="x")
        assert result["truncated"] is True
        assert result["truncated_note"] is not None
