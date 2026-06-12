"""Tests de alias del dominio + descubrimiento de datasets sobre el catálogo real.

Regresión directa de los fallos de ``data/logs_intermedios/logsdefronted.jsonl``:
el modelo inventó cifras porque ``discover_query`` no encontraba el dataset.
Estos tests garantizan que el dataset correcto aparece en el top-3.
"""

from __future__ import annotations

import pytest

from banks_rag.domain_knowledge.financial_aliases import (
    dataset_hints,
    expand_query,
    resolve_segment,
    specialist_for,
)
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    load_parquet_catalog,
    search_datasets,
)


@pytest.mark.unit
class TestAliasHelpers:
    def test_expand_query_adds_synonyms(self) -> None:
        out = expand_query("DV01 de fondos de pensiones")
        # Agrega sinónimos de DV01 y de AFP sin perder la query original.
        assert "DV01 de fondos de pensiones" in out
        assert "afp" in out.lower()
        assert "sensibilidad" in out.lower()

    def test_expand_query_noop_without_concepts(self) -> None:
        assert expand_query("hola mundo xyz") == "hola mundo xyz"

    def test_dataset_hints(self) -> None:
        assert "dv01" in dataset_hints("DV01 fondos de pensiones")
        assert any("nr" in h for h in dataset_hints("posicion de no residentes"))

    def test_resolve_segment_alias(self) -> None:
        # Segmentos canónicos del nuevo catálogo (diccionario_parquets.yaml).
        segs = {"afp", "fx", "rf_tasas"}
        assert resolve_segment("AFP", segs) == "afp"
        assert resolve_segment("afp", segs) == "afp"
        assert resolve_segment("desconocido", segs) is None
        assert resolve_segment(None, segs) is None

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("precio del cobre", "fx"),
            ("DV01 fondos de pensiones", "afp"),
            ("posicion de no residentes", "no_residentes"),
            ("curva BTP 10Y", "renta_fija"),
            ("LCR del sistema bancario", "liquidez"),
            ("flujos de fondos mutuos", "fondos_mutuos"),
            ("tipo de cambio dolar", "fx"),
        ],
    )
    def test_specialist_routing(self, query: str, expected: str) -> None:
        assert specialist_for(query) == expected


@pytest.mark.unit
class TestDiscoveryOnRealCatalog:
    """Las consultas que el modelo alucinó deben encontrar su dataset (top-3)."""

    @pytest.fixture(scope="class")
    def catalog(self):
        return load_parquet_catalog()

    def _top_ids(self, catalog, query: str, k: int = 3) -> list[str]:
        return [e.id for e in search_datasets(catalog, query, top_k=k)]

    def test_dv01_pensiones_finds_afp_dataset(self, catalog) -> None:
        assert "dv01_spc_afp" in self._top_ids(catalog, "DV01 de los fondos de pensiones")

    def test_composicion_cartera_afp(self, catalog) -> None:
        ids = self._top_ids(catalog, "composicion de cartera AFP")
        assert "allocation_int_nac" in ids or "allocation" in ids

    def test_afp_chile_extranjero(self, catalog) -> None:
        ids = self._top_ids(catalog, "fondos de AFP en chile y en el extranjero")
        assert "allocation_int_nac" in ids

    def test_no_residentes(self, catalog) -> None:
        ids = self._top_ids(catalog, "posicion de no residentes en instrumentos financieros")
        assert any("nr" in i for i in ids)

    def test_mtm_pensiones(self, catalog) -> None:
        assert "mtm_afp" in self._top_ids(catalog, "MTM de los fondos de pensiones")

    def test_fondos_mutuos(self, catalog) -> None:
        ids = self._top_ids(catalog, "flujos de fondos mutuos")
        assert any("ffmm" in i for i in ids)
