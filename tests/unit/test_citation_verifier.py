"""Unit tests para citation_verifier."""

from __future__ import annotations

import pytest

from banks_rag.application.agent.citation_verifier import verify_citations
from banks_rag.domain.agent import AgentState


def _make_state(n_chunks: int) -> AgentState:
    state = AgentState()
    for i in range(n_chunks):
        state.add_chunk({"chunk_id": f"c{i}"})
    return state


@pytest.mark.unit
class TestVerifyCitations:
    def test_no_citations_passes_through(self) -> None:
        state = _make_state(3)
        cleaned, refs = verify_citations("Esta es una respuesta sin citas.", state)
        assert cleaned == "Esta es una respuesta sin citas."
        assert refs == []

    def test_valid_citations_preserved(self) -> None:
        state = _make_state(3)
        cleaned, refs = verify_citations("La TPM bajó [1] según el comunicado [2].", state)
        assert "[1]" in cleaned
        assert "[2]" in cleaned
        assert refs == [1, 2]

    def test_invalid_citation_removed(self) -> None:
        state = _make_state(2)
        cleaned, refs = verify_citations("Algo dijo [99] sobre la inflación.", state)
        assert "[99]" not in cleaned
        assert refs == []

    def test_mixed_valid_and_invalid(self) -> None:
        state = _make_state(2)
        cleaned, refs = verify_citations("Datos [1] y otros [99] datos [2].", state)
        assert "[1]" in cleaned
        assert "[2]" in cleaned
        assert "[99]" not in cleaned
        assert sorted(refs) == [1, 2]

    def test_dedup_in_used_refs(self) -> None:
        state = _make_state(2)
        _, refs = verify_citations("[1] dijo X. Luego [1] otra vez.", state)
        assert refs == [1]

    def test_punctuation_cleanup_after_removal(self) -> None:
        state = _make_state(1)
        # [99] inválido debe removerse junto con el espacio antes de la coma
        cleaned, _ = verify_citations("Hola [99] , mundo.", state)
        assert "  " not in cleaned
        assert " ," not in cleaned

    def test_preserves_order_of_appearance(self) -> None:
        state = _make_state(3)
        _, refs = verify_citations("Tercer [3], luego primero [1], y segundo [2].", state)
        assert refs == [3, 1, 2]
