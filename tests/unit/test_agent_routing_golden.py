"""Golden set de ruteo de agente por mercado (determinista, sin modelo).

Valida end-to-end (alias + descubrimiento) la Fase 1: cada pregunta enruta al
especialista correcto y, cuando se especifica, el dataset esperado aparece en
el top-k de discover_query. Cubre los casos que el modelo alucinó en los logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from banks_rag.application.agent.subagents import SUBAGENTS
from banks_rag.domain_knowledge.financial_aliases import specialist_for
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    load_parquet_catalog,
    search_datasets,
)

_GOLDEN = Path(__file__).parents[1].parent / "data" / "golden_set" / "agent_routing.jsonl"


def _load_cases() -> list[dict]:
    if not _GOLDEN.exists():
        return []
    with _GOLDEN.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_CASES = _load_cases()


@pytest.mark.unit
class TestAgentRoutingGolden:
    def test_golden_file_present(self) -> None:
        assert _CASES, "agent_routing.jsonl vacío o ausente"

    @pytest.mark.parametrize("case", _CASES, ids=lambda c: c["query"][:40])
    def test_routes_to_expected_specialist(self, case: dict) -> None:
        got = specialist_for(case["query"])
        assert got == case["expected_specialist"], (
            f"{case['query']!r}: esperado {case['expected_specialist']!r}, "
            f"obtenido {got!r}"
        )
        # El especialista enrutado debe existir en el roster.
        assert case["expected_specialist"] in SUBAGENTS

    @pytest.mark.parametrize(
        "case",
        [c for c in _CASES if c.get("expected_dataset")],
        ids=lambda c: c["query"][:40],
    )
    def test_expected_dataset_in_top_k(self, case: dict) -> None:
        catalog = load_parquet_catalog()
        # top_k=5 (production usa 8); 3 es demasiado estricto cuando la query
        # expandida infla datasets adyacentes del mismo segmento.
        ids = [e.id for e in search_datasets(catalog, case["query"], top_k=5)]
        assert case["expected_dataset"] in ids, (
            f"{case['query']!r}: {case['expected_dataset']!r} no está en top-5 {ids}"
        )
