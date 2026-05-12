"""Tests del path multimodal (VL) en vectorize_corpus.

Cubre:
- Dual-embedding: VL model + imagen OK → img*w + txt*(1-w), L2-normalizado.
- Fallback a texto: VL model + imagen falla (None) → solo txt_emb.
- Modelo no-VL: siempre texto, aunque haya image_path.
- image_weight personalizado (vía parámetro y vía env var).
- dual_embed_count en el reporte refleja cuántos chunks usaron dual-embedding.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from banks_rag.application.ingestion.vectorize_corpus import (
    VectorizationReport,
    _embed_image_chunks,
    vectorize_corpus,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

DIM = 4


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _img_vec() -> np.ndarray:
    return _unit(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def _txt_vec() -> np.ndarray:
    return _unit(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def _make_image_chunk(path: str = "/tmp/chart.png") -> dict:
    return {
        "text": "[CHART p.1]\nGráfico 1: TPM histórica",
        "image_path": path,
        "doc_type_category": "COMUNICADO",
        "section_type": "DECISION",
        "economic_variables": {},
        "document_date": "2024-01-01",
        "chunk_date": None,
    }


def _make_text_chunk() -> dict:
    return {
        "text": "El Consejo decidió mantener la TPM en 8,25%.",
        "image_path": None,
        "doc_type_category": "COMUNICADO",
        "section_type": "DECISION",
        "economic_variables": {},
        "document_date": "2024-01-01",
        "chunk_date": None,
    }


class _VLEmbedder:
    """Mock VL embedder: encode_image_safe devuelve img_vec salvo que la ruta sea 'FAIL'."""

    name = "Qwen/Qwen3-VL-Embedding-8B"
    dim = DIM
    max_seq_length = 0
    is_multimodal = True

    def encode_text(self, texts: list[str], batch_size: int = 4) -> np.ndarray:
        return np.stack([_txt_vec() for _ in texts])

    def encode_image_safe(self, image_path: str) -> np.ndarray | None:
        if "FAIL" in image_path or not image_path:
            return None
        return _img_vec()

    def count_tokens(self, text: str) -> int:
        return 0


class _TextEmbedder:
    """Mock embedder de solo texto (is_multimodal=False)."""

    name = "Qwen/Qwen3-Embedding-8B"
    dim = DIM
    max_seq_length = 0
    is_multimodal = False

    def encode_text(self, texts: list[str], batch_size: int = 4) -> np.ndarray:
        return np.stack([_txt_vec() for _ in texts])

    def encode_image_safe(self, image_path: str) -> np.ndarray | None:
        return None

    def count_tokens(self, text: str) -> int:
        return 0


# ── Tests de _embed_image_chunks (unidad) ─────────────────────────────────────

@pytest.mark.unit
class TestEmbedImageChunks:
    def test_vl_dual_embedding_combines_and_normalizes(self) -> None:
        chunks = [_make_image_chunk()]
        embeddings, dual_count = _embed_image_chunks(
            _VLEmbedder(), chunks, [0], image_weight=0.7
        )
        emb = embeddings[0]
        expected_raw = 0.7 * _img_vec() + 0.3 * _txt_vec()
        expected = _unit(expected_raw)
        np.testing.assert_allclose(emb, expected, atol=1e-6)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-6
        assert dual_count == 1

    def test_vl_image_fails_falls_back_to_text(self) -> None:
        chunks = [_make_image_chunk(path="/tmp/FAIL.png")]
        embeddings, dual_count = _embed_image_chunks(
            _VLEmbedder(), chunks, [0], image_weight=0.7
        )
        np.testing.assert_allclose(embeddings[0], _txt_vec(), atol=1e-6)
        assert dual_count == 0

    def test_non_vl_always_uses_text(self) -> None:
        chunks = [_make_image_chunk()]
        embeddings, dual_count = _embed_image_chunks(
            _TextEmbedder(), chunks, [0], image_weight=0.7
        )
        np.testing.assert_allclose(embeddings[0], _txt_vec(), atol=1e-6)
        assert dual_count == 0

    def test_custom_image_weight(self) -> None:
        chunks = [_make_image_chunk()]
        w = 0.5
        embeddings, _ = _embed_image_chunks(
            _VLEmbedder(), chunks, [0], image_weight=w
        )
        expected = _unit(w * _img_vec() + (1 - w) * _txt_vec())
        np.testing.assert_allclose(embeddings[0], expected, atol=1e-6)

    def test_weight_zero_equals_text_only(self) -> None:
        chunks = [_make_image_chunk()]
        embeddings, _ = _embed_image_chunks(
            _VLEmbedder(), chunks, [0], image_weight=0.0
        )
        np.testing.assert_allclose(embeddings[0], _txt_vec(), atol=1e-6)

    def test_weight_one_equals_image_only(self) -> None:
        chunks = [_make_image_chunk()]
        embeddings, _ = _embed_image_chunks(
            _VLEmbedder(), chunks, [0], image_weight=1.0
        )
        np.testing.assert_allclose(embeddings[0], _img_vec(), atol=1e-6)

    def test_multiple_chunks_counts_correctly(self) -> None:
        chunks = [
            _make_image_chunk(),              # OK → dual
            _make_image_chunk("FAIL"),        # falla → fallback
            _make_image_chunk(),              # OK → dual
        ]
        embeddings, dual_count = _embed_image_chunks(
            _VLEmbedder(), chunks, [0, 1, 2], image_weight=0.7
        )
        assert dual_count == 2
        assert len(embeddings) == 3


# ── Tests de vectorize_corpus (integración ligera) ────────────────────────────

@pytest.mark.unit
class TestVectorizeCorporaVL:
    def _run(self, embedder: Any, chunks: list[dict], **kw: Any) -> Any:
        return vectorize_corpus(list(chunks), embedder, **kw)

    def test_report_vl_model_true_and_dual_count(self) -> None:
        chunks = [_make_image_chunk(), _make_text_chunk()]
        result = self._run(_VLEmbedder(), chunks, image_weight=0.7)
        assert result.report is not None
        assert result.report.vl_model is True
        assert result.report.image_chunks == 1
        assert result.report.text_chunks == 1
        assert result.report.dual_embed_count == 1
        assert result.report.image_weight == 0.7

    def test_report_non_vl_dual_count_zero(self) -> None:
        chunks = [_make_image_chunk(), _make_text_chunk()]
        result = self._run(_TextEmbedder(), chunks, image_weight=0.7)
        assert result.report is not None
        assert result.report.vl_model is False
        assert result.report.dual_embed_count == 0

    def test_image_weight_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_VISUAL_IMG_WEIGHT", "0.3")
        chunks = [_make_image_chunk()]
        result = self._run(_VLEmbedder(), chunks)
        assert result.report is not None
        assert result.report.image_weight == pytest.approx(0.3)

    def test_all_chunks_get_embedding(self) -> None:
        chunks = [_make_image_chunk(), _make_text_chunk(), _make_text_chunk()]
        result = self._run(_VLEmbedder(), chunks, image_weight=0.7)
        for c in result.chunks:
            assert "embedding" in c
            assert len(c["embedding"]) == DIM

    def test_report_dict_includes_new_fields(self) -> None:
        chunks = [_make_image_chunk()]
        result = self._run(_VLEmbedder(), chunks, image_weight=0.6)
        d = result.report.to_dict()  # type: ignore[union-attr]
        assert "image_weight" in d
        assert "dual_embed_count" in d
        assert d["image_weight"] == pytest.approx(0.6)
        assert d["dual_embed_count"] == 1
