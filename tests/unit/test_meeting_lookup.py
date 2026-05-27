"""Tests de las tools de reuniones de política monetaria (Fase C)."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import pytest

from banks_rag.application.agent.tools import dispatch
from banks_rag.application.agent.tools.meeting_lookup import (
    _find_tpm_rate,
    _normalize_chunk_text,
    _select_key_chunks,
)
from banks_rag.domain.agent import AgentState

_MODULE = "banks_rag.application.agent.tools.meeting_lookup"


def _chunk(cid: str, pos: int, importance: float, is_decision: bool = False) -> dict:
    return {
        "chunk_id": cid,
        "text": f"Texto del chunk {cid}.",
        "page_start": pos + 1,
        "page_end": pos + 1,
        "position_in_doc": pos,
        "section_type": "DECISION" if is_decision else "CONTENIDO",
        "importance_score": importance,
        "is_policy_decision": is_decision,
    }


# ─────────────────────────────────────────────────────────────────────────────
# _select_key_chunks (función pura)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSelectKeyChunks:
    def test_prioritizes_policy_decision_over_importance(self) -> None:
        chunks = [
            _chunk("c0", 0, 0.5),
            _chunk("c1", 1, 0.9),
            _chunk("c2", 2, 0.3, is_decision=True),  # decisión, baja importancia
            _chunk("c3", 3, 0.8),
        ]
        selected = _select_key_chunks(chunks, max_n=2)
        ids = {c["chunk_id"] for c in selected}
        # La decisión entra pese a su baja importancia; la acompaña la de mayor importancia.
        assert ids == {"c1", "c2"}

    def test_reorders_selection_by_position(self) -> None:
        chunks = [
            _chunk("c0", 5, 0.9),
            _chunk("c1", 1, 0.8),
            _chunk("c2", 3, 0.7),
        ]
        selected = _select_key_chunks(chunks, max_n=3)
        positions = [c["position_in_doc"] for c in selected]
        assert positions == [1, 3, 5]  # orden de lectura, no de ranking

    def test_caps_at_max_n(self) -> None:
        chunks = [_chunk(f"c{i}", i, 0.5) for i in range(10)]
        assert len(_select_key_chunks(chunks, max_n=4)) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Fake repo
# ─────────────────────────────────────────────────────────────────────────────


class _FakeRepo:
    """PostgresRepo de prueba: documentos y chunks en memoria."""

    docs: ClassVar[dict] = {}
    chunks: ClassVar[dict] = {}
    doc_list: ClassVar[list] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_document_by_filename(self, filename: str):
        return self.docs.get(filename)

    def get_chunks_by_document_id(self, document_id: str, **kwargs):
        return self.chunks.get(document_id, [])

    def list_documents(self, **kwargs):
        return list(self.doc_list)


def _make_repo(docs: dict, chunks: dict, doc_list: list | None = None):
    """Construye una subclase de _FakeRepo con datos fijos."""
    return type("RepoFixture", (_FakeRepo,), {
        "docs": docs, "chunks": chunks, "doc_list": doc_list or [],
    })


# ─────────────────────────────────────────────────────────────────────────────
# compare_meetings
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCompareMeetings:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        state = AgentState()
        docs = {
            "comunicado1.pdf": {
                "document_id": "d1", "filename": "comunicado1.pdf",
                "doc_type_category": "COMUNICADO", "document_date": "2024-01-30",
            },
            "comunicado2.pdf": {
                "document_id": "d2", "filename": "comunicado2.pdf",
                "doc_type_category": "COMUNICADO", "document_date": "2024-03-26",
            },
        }
        chunks = {
            "d1": [_chunk("d1c0", 0, 0.9, is_decision=True), _chunk("d1c1", 1, 0.4)],
            "d2": [_chunk("d2c0", 0, 0.8, is_decision=True), _chunk("d2c1", 1, 0.5)],
        }
        repo = _make_repo(docs, chunks)

        with patch(f"{_MODULE}.PostgresRepo", repo):
            result, _ = await dispatch(state, "compare_meetings", {
                "filename_a": "comunicado1.pdf", "filename_b": "comunicado2.pdf",
            })

        assert result["meeting_a"]["date"] == "2024-01-30"
        assert result["meeting_b"]["date"] == "2024-03-26"
        assert len(result["meeting_a"]["chunks"]) == 2
        # Refs globales asignadas en el state compartido (2 + 2 chunks).
        assert len(state.chunks_seen) == 4
        refs = [c["ref"] for c in result["meeting_a"]["chunks"]]
        refs += [c["ref"] for c in result["meeting_b"]["chunks"]]
        assert refs == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_missing_document_returns_error(self) -> None:
        docs = {
            "existe.pdf": {
                "document_id": "d1", "filename": "existe.pdf",
                "doc_type_category": "COMUNICADO", "document_date": "2024-01-30",
            },
        }
        repo = _make_repo(docs, {"d1": [_chunk("d1c0", 0, 0.5)]})

        with patch(f"{_MODULE}.PostgresRepo", repo):
            result, _ = await dispatch(AgentState(), "compare_meetings", {
                "filename_a": "existe.pdf", "filename_b": "fantasma.pdf",
            })

        assert "error" in result
        assert "fantasma.pdf" in result["error"]

    @pytest.mark.asyncio
    async def test_respects_max_chunks_per_doc(self) -> None:
        docs = {
            f"doc{i}.pdf": {
                "document_id": f"d{i}", "filename": f"doc{i}.pdf",
                "doc_type_category": "MINUTA", "document_date": f"2024-0{i}-01",
            }
            for i in (1, 2)
        }
        chunks = {f"d{i}": [_chunk(f"d{i}c{j}", j, 0.5) for j in range(10)] for i in (1, 2)}
        repo = _make_repo(docs, chunks)

        with patch(f"{_MODULE}.PostgresRepo", repo):
            result, _ = await dispatch(AgentState(), "compare_meetings", {
                "filename_a": "doc1.pdf", "filename_b": "doc2.pdf",
                "max_chunks_per_doc": 3,
            })

        assert len(result["meeting_a"]["chunks"]) == 3
        assert len(result["meeting_b"]["chunks"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# get_recent_policy_decisions
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGetRecentPolicyDecisions:
    @pytest.mark.asyncio
    async def test_sorts_by_date_descending(self) -> None:
        state = AgentState()
        doc_list = [
            {"document_id": "d1", "filename": "ene.pdf",
             "doc_type_category": "COMUNICADO", "document_date": "2024-01-30",
             "document_year": 2024},
            {"document_id": "d3", "filename": "may.pdf",
             "doc_type_category": "COMUNICADO", "document_date": "2024-05-29",
             "document_year": 2024},
            {"document_id": "d2", "filename": "mar.pdf",
             "doc_type_category": "COMUNICADO", "document_date": "2024-03-26",
             "document_year": 2024},
        ]
        chunks = {
            d["document_id"]: [_chunk(f"{d['document_id']}c0", 0, 0.8, is_decision=True)]
            for d in doc_list
        }
        repo = _make_repo({}, chunks, doc_list=doc_list)

        with patch(f"{_MODULE}.PostgresRepo", repo):
            result, _ = await dispatch(state, "get_recent_policy_decisions", {"n": 2})

        assert result["n_decisions"] == 2
        # Más reciente primero.
        assert [d["date"] for d in result["decisions"]] == ["2024-05-29", "2024-03-26"]
        assert len(state.chunks_seen) == 2

    @pytest.mark.asyncio
    async def test_empty_corpus(self) -> None:
        repo = _make_repo({}, {}, doc_list=[])
        with patch(f"{_MODULE}.PostgresRepo", repo):
            result, _ = await dispatch(AgentState(), "get_recent_policy_decisions", {})
        assert result["n_decisions"] == 0
        assert "message" in result

    @pytest.mark.asyncio
    async def test_latest_decision_extracts_tpm_level(self) -> None:
        state = AgentState()
        doc_list = [{
            "document_id": "d1", "filename": "comunicado abril 2026.pdf",
            "doc_type_category": "COMUNICADO_RPM", "document_date": "2026-04-28",
            "document_year": 2026,
        }]
        text = (
            "El Consejo del Banco Central de Chile acordó mantener la Tasa de "
            "Política Monetaria en 5,50%. La decisión fue adoptada por unanimidad."
        )
        chunks = {"d1": [{
            "chunk_id": "d1c0", "text": text, "page_start": 1, "page_end": 1,
            "position_in_doc": 0, "section_type": "DECISION",
            "importance_score": 0.8, "is_policy_decision": True,
        }]}
        repo = _make_repo({}, chunks, doc_list=doc_list)

        with patch(f"{_MODULE}.PostgresRepo", repo):
            result, _ = await dispatch(state, "get_recent_policy_decisions", {"n": 1})

        ld = result["latest_decision"]
        assert ld["tpm_level"] == "5,50%"
        assert ld["action"] == "mantener"
        assert ld["date"] == "2026-04-28"
        assert ld["filename"] == "comunicado abril 2026.pdf"
        # La ref apunta a un chunk realmente registrado en el state (citable).
        assert ld["ref"] == 1

    @pytest.mark.asyncio
    async def test_latest_decision_null_when_no_rate(self) -> None:
        state = AgentState()
        doc_list = [{
            "document_id": "d1", "filename": "comunicado.pdf",
            "doc_type_category": "COMUNICADO_RPM", "document_date": "2026-04-28",
            "document_year": 2026,
        }]
        # _chunk() produce texto sin tasa → tpm_level debe ser None (degrada).
        chunks = {"d1": [_chunk("d1c0", 0, 0.8, is_decision=True)]}
        repo = _make_repo({}, chunks, doc_list=doc_list)

        with patch(f"{_MODULE}.PostgresRepo", repo):
            result, _ = await dispatch(state, "get_recent_policy_decisions", {"n": 1})

        assert result["latest_decision"]["tpm_level"] is None

    @pytest.mark.asyncio
    async def test_latest_decision_from_misclassified_opening(self) -> None:
        """Bug real de abril 2026: la tasa vive en el párrafo de apertura mal
        etiquetado como RIESGOS, y la etiqueta DECISION se la lleva un chunk
        [IMAGE] truncado SIN el número. La extracción debe barrer todo el doc."""
        state = AgentState()
        doc_list = [{
            "document_id": "d1", "filename": "comunicado abril 2026.pdf",
            "doc_type_category": "COMUNICADO_RPM", "document_date": "2026-04-28",
            "document_year": 2026,
        }]
        opening = {
            "chunk_id": "d1c0",
            "text": ("Martes 28 de abril de 2026 Reunión de Política Monetaria. "
                     "En su Reunión de Política Monetaria, el Consejo del Banco "
                     "Central de Chile acordó mantener la tasa de interés de "
                     "política monetaria en 4,5%. La decisión fue unánime."),
            "page_start": 1, "page_end": 1, "position_in_doc": 0,
            "section_type": "RIESGOS",  # ← mal clasificado
            "importance_score": 0.667, "is_policy_decision": False,
        }
        decision_img = {
            "chunk_id": "d1c11",
            "text": ("[IMAGE p.1]\nMartes 28 de abril de 2026 ... acordó mantener "
                     "la tasa de interés de política monetari"),  # truncado, sin %
            "page_start": 1, "page_end": 1, "position_in_doc": 11,
            "section_type": "DECISION",
            "importance_score": 0.7, "is_policy_decision": True,
        }
        repo = _make_repo({}, {"d1": [opening, decision_img]}, doc_list=doc_list)

        with patch(f"{_MODULE}.PostgresRepo", repo):
            result, _ = await dispatch(state, "get_recent_policy_decisions", {"n": 1})

        ld = result["latest_decision"]
        assert ld["tpm_level"] == "4,5%"
        assert ld["action"] == "mantener"
        assert ld["date"] == "2026-04-28"
        assert isinstance(ld["ref"], int) and ld["ref"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# _find_tpm_rate / _normalize_chunk_text (funciones puras)
# ─────────────────────────────────────────────────────────────────────────────


def _rate(text: str) -> tuple[str, str | None] | None:
    return _find_tpm_rate(_normalize_chunk_text(text))


@pytest.mark.unit
class TestFindTpmRate:
    def test_mantener(self) -> None:
        assert _rate(
            "El Consejo acordó mantener la Tasa de Política Monetaria en 5,50%."
        ) == ("5,50", "mantener")

    def test_recorte(self) -> None:
        level, action = _rate(
            "El Consejo decidió reducir la Tasa de Política Monetaria en 25 "
            "puntos base, hasta 5,25%."
        )
        assert level == "5,25"
        assert action == "recorte"

    def test_alza_con_sigla_tpm(self) -> None:
        level, action = _rate("El Consejo acordó aumentar la TPM en 25 pb, a 3,00%.")
        assert level == "3,00"
        assert action == "alza"

    def test_none_cuando_porcentaje_no_es_tpm(self) -> None:
        assert _rate("La variación anual del IPC total fue de 2,8% en marzo.") is None

    def test_no_cruza_oraciones(self) -> None:
        # TPM y porcentaje en oraciones distintas → sin match.
        assert _rate(
            "El Consejo se refirió a la Tasa de Política Monetaria. La "
            "inflación subyacente fue 2,8%."
        ) is None

    def test_normaliza_saltos_de_linea_y_prefijo_imagen(self) -> None:
        # El [IMAGE p.1] y los \n del OCR no deben romper el match.
        level, action = _rate(
            "[IMAGE p.1]\nEl Consejo acordó mantener la tasa de interés de\n"
            "política monetaria en 4,5%."
        )
        assert level == "4,5"
        assert action == "mantener"
