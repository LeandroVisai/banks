"""Tests de las tools de reuniones de política monetaria (Fase C)."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import pytest

from banks_rag.application.agent.tools import dispatch
from banks_rag.application.agent.tools.meeting_lookup import _select_key_chunks
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
