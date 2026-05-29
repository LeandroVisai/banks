"""Replay de regresión sobre los logs reales del frontend.

Convierte ``data/logs_intermedios/logsdefronted.jsonl`` (transcripción real con
gemma-3-12b, donde el modelo ALUCINABA) en un fixture de regresión offline:
para cada pregunta data-bearing de esos logs, verifica que hoy el ruteo al
especialista y el descubrimiento del dataset son correctos. No invoca el LLM —
valida las capas deterministas (alias + discovery) que antes fallaban.

Si el archivo de logs no existe, el módulo se salta.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from banks_rag.config.paths import ROOT
from banks_rag.domain_knowledge.financial_aliases import specialist_for
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    load_parquet_catalog,
    search_datasets,
)

_LOGS = ROOT / "data" / "logs_intermedios" / "logsdefronted.jsonl"

pytestmark = pytest.mark.skipif(
    not _LOGS.exists(), reason=f"No existe {_LOGS}; nada que reproducir."
)

# Preguntas data-bearing de los logs → expectativa determinista.
# `dataset` (substring que debe aparecer en el top-3 de discover) y/o
# `specialist` (ruteo por mercado). Los follow-ups conversacionales
# ("Consulta 2", "¿de dónde sacaste?") se omiten a propósito.
_EXPECTED = {
    "Tienes el precio del cobre?": {"dataset": "cobre_dxy", "specialist": "fx"},
    "Tienes información del dv01 de los fondos de pensiones?": {
        "dataset": "dv01_spc_afp", "specialist": "afp",
    },
    "cual es la Composición de cartera AFP?": {
        "dataset": "allocation_int_nac", "specialist": "afp",
    },
    "Si, quiero saber cuanta proporcion de los fondos de AFP se encuentran en chile y cuales en el extranjero": {
        "dataset": "allocation_int_nac", "specialist": "afp",
    },
    "cual es la posicion de no residentes con respecto a instrumentos financieros?": {
        "specialist": "no_residentes",
    },
}


@pytest.fixture(scope="module")
def logged_questions() -> set[str]:
    msgs: set[str] = set()
    with _LOGS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("user_message"):
                msgs.add(rec["user_message"])
    return msgs


@pytest.fixture(scope="module")
def catalog():
    return load_parquet_catalog()


@pytest.mark.unit
def test_expected_questions_present_in_logs(logged_questions: set[str]) -> None:
    """Las preguntas curadas existen tal cual en los logs (no se desincronizan)."""
    faltan = [q for q in _EXPECTED if q not in logged_questions]
    assert not faltan, f"Preguntas curadas ausentes en los logs: {faltan}"


@pytest.mark.unit
@pytest.mark.parametrize("question", list(_EXPECTED), ids=lambda q: q[:40])
def test_logged_question_routes_and_discovers(question, catalog) -> None:
    exp = _EXPECTED[question]
    if "specialist" in exp:
        assert specialist_for(question) == exp["specialist"], (
            f"{question!r}: ruteo esperado {exp['specialist']!r}, "
            f"obtenido {specialist_for(question)!r}"
        )
    if "dataset" in exp:
        ids = [e.id for e in search_datasets(catalog, question, top_k=3)]
        assert exp["dataset"] in ids, (
            f"{question!r}: {exp['dataset']!r} no está en top-3 {ids}"
        )
