"""Tests de CrossEncoderReranker y QueryRouter (Fase 5).

Cubre:
- CrossEncoderReranker: rerank con mock model, ordenamiento, reranker_score,
  top_k, lista vacía, load idempotente, info().
- QueryRouter: detección de SQL, RAG, VISUAL, casos mixtos, fallback RAG.
- hybrid_search: integración del parámetro reranker (mock).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from banks_rag.application.retrieval.query_router import RouteDecision, route_query
from banks_rag.config import override_settings, reset_settings
from banks_rag.config.settings import Settings
from banks_rag.infrastructure.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    _extract_text,
    build_default_reranker,
    reranking_enabled,
    reset_default_reranker,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(text: str, section: str = "") -> dict:
    return {"text": text, "section_type": section, "chunk_id": text[:8]}


def _mock_cross_encoder(scores: list[float]) -> MagicMock:
    """Mock CrossEncoder cuyo predict() devuelve los scores dados."""
    m = MagicMock()
    m.predict.return_value = np.array(scores)
    return m


# ── Tests CrossEncoderReranker ────────────────────────────────────────────────

@pytest.mark.unit
class TestCrossEncoderReranker:
    def _reranker_with_mock(self, scores: list[float]) -> CrossEncoderReranker:
        r = CrossEncoderReranker("jinaai/jina-reranker-v3")
        r._model = _mock_cross_encoder(scores)
        r.loaded = True
        return r

    def test_rerank_sorts_by_score_descending(self) -> None:
        chunks = [_chunk("low relevance"), _chunk("high relevance"), _chunk("medium")]
        r = self._reranker_with_mock([0.1, 0.9, 0.5])
        result = r.rerank("query", chunks)
        assert result[0]["text"] == "high relevance"
        assert result[1]["text"] == "medium"
        assert result[2]["text"] == "low relevance"

    def test_reranker_score_added(self) -> None:
        chunks = [_chunk("a"), _chunk("b")]
        r = self._reranker_with_mock([0.8, 0.2])
        result = r.rerank("q", chunks)
        assert "reranker_score" in result[0]
        assert result[0]["reranker_score"] == pytest.approx(0.8)

    def test_top_k_limits_output(self) -> None:
        chunks = [_chunk(f"chunk{i}") for i in range(5)]
        r = self._reranker_with_mock([0.1, 0.9, 0.5, 0.3, 0.7])
        result = r.rerank("q", chunks, top_k=2)
        assert len(result) == 2
        assert result[0]["reranker_score"] == pytest.approx(0.9)
        assert result[1]["reranker_score"] == pytest.approx(0.7)

    def test_empty_chunks_returns_empty(self) -> None:
        r = self._reranker_with_mock([])
        assert r.rerank("q", []) == []

    def test_original_chunk_not_mutated(self) -> None:
        original = _chunk("text")
        r = self._reranker_with_mock([0.5])
        result = r.rerank("q", [original])
        assert "reranker_score" not in original
        assert "reranker_score" in result[0]

    def test_load_idempotent(self) -> None:
        r = CrossEncoderReranker("jinaai/jina-reranker-v3")
        sentinel = MagicMock()
        r.loaded = True
        r._model = sentinel
        # Llamar load() cuando ya está cargado no debe tocar _model
        r.load()
        assert r._model is sentinel  # el mismo objeto, no se re-creó

    def test_info_returns_expected_keys(self) -> None:
        r = CrossEncoderReranker()
        info = r.info()
        assert "name" in info
        assert "loaded" in info
        assert "batch_size" in info
        assert "max_length" in info

    def test_passes_section_type_in_pairs(self) -> None:
        chunks = [_chunk("texto del chunk", section="DECISION")]
        r = self._reranker_with_mock([0.9])
        r.rerank("consulta", chunks)
        call_args = r._model.predict.call_args[0][0]
        query_text, doc_text = call_args[0]
        assert query_text == "consulta"
        assert "[DECISION]" in doc_text

    def test_extract_text_with_section(self) -> None:
        c = {"text": "hola", "section_type": "DECISION"}
        assert _extract_text(c) == "[DECISION] hola"

    def test_extract_text_without_section(self) -> None:
        c = {"text": "hola"}
        assert _extract_text(c) == "hola"

    def test_load_failure_degrades_not_raises(self) -> None:
        """Si load() falla (modelo ausente / sin trust_remote_code), rerank NO
        propaga: devuelve el orden previo recortado a top_k. Crítico para no
        romper search_documents cuando el modelo de reranker no está disponible."""
        r = CrossEncoderReranker("inexistente/modelo-que-no-carga")
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        with patch.object(r, "load", side_effect=RuntimeError("no model")):
            out = r.rerank("q", chunks, top_k=2)
        assert [c["text"] for c in out] == ["a", "b"]  # orden previo, recortado
        assert r._load_failed is True

    def test_load_failure_not_retried(self) -> None:
        """Tras un fallo de carga, no se reintenta en cada rerank (queda degradado)."""
        r = CrossEncoderReranker("inexistente/modelo")
        load_mock = MagicMock(side_effect=RuntimeError("no model"))
        with patch.object(r, "load", load_mock):
            r.rerank("q", [_chunk("a")])
            r.rerank("q", [_chunk("a")])
        load_mock.assert_called_once()  # solo el primer intento

    def test_predict_failure_degrades(self) -> None:
        """Si predict() falla (OOM, par malformado), se mantiene el orden previo."""
        r = CrossEncoderReranker("jinaai/jina-reranker-v3")
        r._model = MagicMock()
        r._model.predict.side_effect = RuntimeError("OOM")
        r.loaded = True
        out = r.rerank("q", [_chunk("a"), _chunk("b")], top_k=1)
        assert [c["text"] for c in out] == ["a"]


# ── Tests build_default_reranker (singleton + settings/.env) ──────────────────

@pytest.mark.unit
class TestBuildDefaultReranker:
    """El reranker lee de Settings (BANKS_RERANK_*), que carga el .env."""

    def setup_method(self) -> None:
        reset_default_reranker()

    def teardown_method(self) -> None:
        reset_default_reranker()
        reset_settings()

    def test_enabled_by_default(self) -> None:
        override_settings(Settings(rerank_enabled=True))
        assert reranking_enabled() is True
        r = build_default_reranker()
        assert isinstance(r, CrossEncoderReranker)
        # No carga el modelo al construir (lazy en el primer rerank).
        assert r.loaded is False

    def test_disabled_returns_none(self) -> None:
        override_settings(Settings(rerank_enabled=False))
        assert reranking_enabled() is False
        assert build_default_reranker() is None

    def test_singleton_returns_same_instance(self) -> None:
        override_settings(Settings(rerank_enabled=True))
        assert build_default_reranker() is build_default_reranker()

    def test_respects_custom_model(self) -> None:
        override_settings(Settings(rerank_enabled=True, rerank_model="custom/model"))
        assert build_default_reranker().name == "custom/model"


# ── Tests QueryRouter ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestQueryRouter:
    def test_sql_usdclp_explicit(self) -> None:
        r = route_query("¿Cuánto vale el USD/CLP hoy?")
        assert r.sql is True

    def test_sql_btp_curve(self) -> None:
        r = route_query("Muéstrame la curva BTP 5Y y 10Y del último mes")
        assert r.sql is True

    def test_sql_series_historica(self) -> None:
        r = route_query("Quiero los datos históricos de la tasa SPC 3M")
        assert r.sql is True

    def test_sql_precio_cobre(self) -> None:
        r = route_query("¿Cuál es el precio del cobre?")
        assert r.sql is True

    def test_sql_lcr(self) -> None:
        r = route_query("Dame el LCR sistémico de los bancos")
        assert r.sql is True

    def test_rag_comunicado(self) -> None:
        r = route_query("¿Qué dijo el Banco Central en el último comunicado?")
        assert r.rag is True

    def test_rag_decision_politica(self) -> None:
        r = route_query("¿Por qué el Consejo decidió bajar la TPM?")
        assert r.rag is True

    def test_rag_minuta(self) -> None:
        r = route_query("Resume la última minuta del Consejo")
        assert r.rag is True

    def test_visual_grafico(self) -> None:
        r = route_query("Muéstrame el gráfico de inflación")
        assert r.visual is True

    def test_visual_chart(self) -> None:
        r = route_query("¿Hay algún chart con la evolución del dólar?")
        assert r.visual is True

    def test_mixed_sql_and_rag(self) -> None:
        r = route_query("¿Cuál es la TPM actual y qué dijo el comunicado del BCCh?")
        assert r.rag is True  # "comunicado del BCCh" activa RAG
        assert r.sql is True  # "TPM actual" activa SQL

    def test_fallback_to_rag_for_generic_query(self) -> None:
        r = route_query("¿Qué es la política monetaria?")
        assert r.rag is True

    def test_primary_is_max_score(self) -> None:
        r = route_query("¿Cuánto vale el USD/CLP hoy?")
        assert r.primary in ("sql", "rag", "visual")
        assert r.scores[r.primary] == max(r.scores.values())

    def test_scores_between_0_and_1(self) -> None:
        for q in [
            "datos del dólar",
            "gráfico de inflación",
            "comunicado BCCh",
            "¿qué pasó?",
        ]:
            r = route_query(q)
            for val in r.scores.values():
                assert 0.0 <= val <= 1.0

    def test_to_dict_structure(self) -> None:
        r = route_query("curva BTP")
        d = r.to_dict()
        assert "rag" in d and "sql" in d and "visual" in d
        assert "primary" in d and "scores" in d

    def test_route_decision_is_dataclass(self) -> None:
        r = route_query("test")
        assert isinstance(r, RouteDecision)


# ── Test integración reranker en hybrid_search ────────────────────────────────

@pytest.mark.unit
class TestHybridSearchWithReranker:
    """Verifica que hybrid_search pasa los chunks al reranker cuando se provee."""

    def test_reranker_called_when_provided(self) -> None:
        from banks_rag.application.retrieval.hybrid_search import hybrid_search

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [{"text": "reranked", "score": 1.0}]

        mock_embedder = MagicMock()
        mock_embedder.name = "Qwen/Qwen3-Embedding-8B"
        mock_embedder.encode_text.return_value = np.zeros((1, 4096), dtype=np.float32)

        # Mockear toda la capa de BD para evitar conexión real
        with patch(
            "banks_rag.application.retrieval.hybrid_search.PostgresRepo"
        ) as MockRepo, patch(
            "banks_rag.application.retrieval.hybrid_search._recall_and_fuse"
        ) as mock_recall, patch(
            "banks_rag.application.retrieval.hybrid_search.importance_boost"
        ) as mock_boost, patch(
            "banks_rag.application.retrieval.hybrid_search.mark_low_confidence"
        ), patch(
            "banks_rag.application.retrieval.hybrid_search.get_db_embedding_dim",
            return_value=4096,
        ), patch(
            "banks_rag.application.retrieval.hybrid_search.mmr_select",
            side_effect=lambda candidates, *args, **kwargs: candidates,
        ):
            mock_recall.return_value = [{"text": "chunk1", "score": 0.8}]
            mock_boost.return_value = [{"text": "chunk1", "score": 0.8}]
            MockRepo.return_value.__enter__ = MagicMock(return_value=MagicMock())
            MockRepo.return_value.connect.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )

            result = hybrid_search(
                "política monetaria",
                query_embedder=mock_embedder,
                reranker=mock_reranker,
                k=3,
            )

        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        assert call_args[1]["top_k"] == 3 or call_args[0][2] == 3

    def test_no_reranker_uses_slicing(self) -> None:
        from banks_rag.application.retrieval.hybrid_search import hybrid_search

        mock_embedder = MagicMock()
        mock_embedder.name = "Qwen/Qwen3-Embedding-8B"
        mock_embedder.encode_text.return_value = np.zeros((1, 4096), dtype=np.float32)

        chunks = [{"text": f"chunk{i}", "score": float(i)} for i in range(10)]

        with patch(
            "banks_rag.application.retrieval.hybrid_search.PostgresRepo"
        ) as MockRepo, patch(
            "banks_rag.application.retrieval.hybrid_search._recall_and_fuse",
            return_value=chunks,
        ), patch(
            "banks_rag.application.retrieval.hybrid_search.importance_boost",
            return_value=chunks,
        ), patch(
            "banks_rag.application.retrieval.hybrid_search.mark_low_confidence"
        ), patch(
            "banks_rag.application.retrieval.hybrid_search.get_db_embedding_dim",
            return_value=4096,
        ):
            MockRepo.return_value.connect.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            result = hybrid_search(
                "test", query_embedder=mock_embedder, k=3, use_mmr=False
            )

        assert len(result.hits) == 3
