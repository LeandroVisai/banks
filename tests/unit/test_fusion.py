"""Unit tests para fusion (RRF, MMR, importance_boost)."""

from __future__ import annotations

import numpy as np
import pytest

from banks_rag.application.retrieval import (
    DEFAULT_RRF_K,
    doc_year,
    importance_boost,
    mark_low_confidence,
    mmr_select,
    parse_embedding,
    recency_factor,
    rrf_fuse,
)


def _hit(chunk_id: str, **extra) -> dict:
    base = {"chunk_id": chunk_id, "embedding": [1.0, 0.0, 0.0, 0.0]}
    base.update(extra)
    return base


@pytest.mark.unit
class TestRrfFuse:
    def test_empty_inputs(self) -> None:
        assert rrf_fuse([], []) == []

    def test_single_doc_in_both_lists(self) -> None:
        hit = _hit("a")
        result = rrf_fuse([hit], [hit])
        assert len(result) == 1
        # Score = 1/(k+1) + 1/(k+1) = 2/(k+1)
        expected = 2.0 / (DEFAULT_RRF_K + 1)
        assert result[0]["rrf_score"] == pytest.approx(expected)

    def test_doc_only_in_vector_list(self) -> None:
        result = rrf_fuse([_hit("a")], [])
        assert len(result) == 1
        assert result[0]["_rrf_vector_rank"] == 1

    def test_ranks_combined(self) -> None:
        # 'a' es 1° en vector, 2° en lexical; 'b' es 2° en vector, 1° en lexical
        # Ambos deben tener el mismo score
        vec = [_hit("a"), _hit("b")]
        lex = [_hit("b"), _hit("a")]
        result = rrf_fuse(vec, lex)
        assert len(result) == 2
        assert result[0]["rrf_score"] == result[1]["rrf_score"]

    def test_higher_rank_wins(self) -> None:
        vec = [_hit("a"), _hit("b"), _hit("c")]
        lex = [_hit("a")]
        result = rrf_fuse(vec, lex)
        assert result[0]["chunk_id"] == "a"  # presente en ambos


@pytest.mark.unit
class TestMmrSelect:
    def test_empty_returns_empty(self) -> None:
        q = np.array([1.0, 0.0])
        assert mmr_select([], q, k=5) == []

    def test_first_pick_is_most_relevant(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        # 'a' es más similar a la query que 'b'
        cands = [
            _hit("a", embedding=[1.0, 0.0, 0.0, 0.0]),
            _hit("b", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        result = mmr_select(cands, q, k=2, lambda_param=1.0)
        assert result[0]["chunk_id"] == "a"

    def test_diversifies_when_lambda_low(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        # 'a' y 'a2' son idénticos; 'b' es ortogonal
        cands = [
            _hit("a", embedding=[1.0, 0.0, 0.0, 0.0]),
            _hit("a2", embedding=[1.0, 0.0, 0.0, 0.0]),
            _hit("b", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        result = mmr_select(cands, q, k=2, lambda_param=0.0)  # solo diversidad
        ids = [r["chunk_id"] for r in result]
        # El segundo debería ser 'b' (diverso) en lugar de 'a2' (redundante)
        assert ids[0] in ("a", "a2")
        assert "b" in ids


@pytest.mark.unit
class TestParseEmbedding:
    def test_parses_string(self) -> None:
        result = parse_embedding("[0.1,0.2,0.3]")
        assert result.tolist() == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)

    def test_parses_list(self) -> None:
        result = parse_embedding([0.1, 0.2, 0.3])
        assert result.dtype == np.float32

    def test_parses_array(self) -> None:
        arr = np.array([0.1, 0.2])
        result = parse_embedding(arr)
        assert result.tolist() == pytest.approx([0.1, 0.2])


@pytest.mark.unit
class TestDocYear:
    def test_chunk_date_priority(self) -> None:
        assert doc_year({"chunk_date": "2024-05-15", "document_date": "2020-01-01"}) == 2024

    def test_falls_back_to_document_date(self) -> None:
        assert doc_year({"document_date": "2020-01-01"}) == 2020

    def test_no_date(self) -> None:
        assert doc_year({}) is None


@pytest.mark.unit
class TestRecencyFactor:
    def test_current_year_is_max(self) -> None:
        assert recency_factor(2024, today_year=2024) == 1.0

    def test_exponential_decay(self) -> None:
        # Decay exponencial con vida media de 8 años: a 4 años → 0.5**0.5.
        assert recency_factor(2020, today_year=2024) == pytest.approx(0.7071, abs=0.001)
        # A una vida media exacta (8 años) → 0.5.
        assert recency_factor(2016, today_year=2024) == pytest.approx(0.5, abs=0.001)

    def test_old_docs_decay_near_zero_but_keep_gradient(self) -> None:
        # Documentos muy antiguos tienden a 0 pero conservan gradiente
        # (a diferencia del decay lineal, que colapsaba todo a 0 exacto).
        v1900 = recency_factor(1900, today_year=2024)
        v1950 = recency_factor(1950, today_year=2024)
        assert v1900 < 0.001
        assert v1950 > v1900  # más reciente ⇒ mayor score, no empate en 0

    def test_no_year_neutral(self) -> None:
        assert recency_factor(None) == 0.5


@pytest.mark.unit
class TestImportanceBoost:
    def test_higher_importance_wins_with_same_rrf(self) -> None:
        hits = [
            {"chunk_id": "a", "rrf_score": 0.5, "importance_score": 0.3},
            {"chunk_id": "b", "rrf_score": 0.5, "importance_score": 0.9},
        ]
        result = importance_boost(hits)
        assert result[0]["chunk_id"] == "b"

    def test_final_score_includes_recency(self) -> None:
        hits = [
            {"chunk_id": "a", "rrf_score": 0.5, "importance_score": 0.5,
             "document_date": "2024-01-01"},
        ]
        importance_boost(hits)
        assert "final_score" in hits[0]

    def test_low_confidence_marked(self) -> None:
        results = [{"chunk_id": "a", "final_score": 0.01}]
        mark_low_confidence(results, threshold=0.05)
        assert results[0]["low_confidence"] is True

        results = [{"chunk_id": "a", "final_score": 0.10}]
        mark_low_confidence(results, threshold=0.05)
        assert results[0]["low_confidence"] is False
