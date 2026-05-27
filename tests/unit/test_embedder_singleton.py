"""Tests del singleton de proceso de ``build_default_embedder``.

El embedder es caro (modelo de varios GB con carga lazy). Antes se construía una
instancia nueva en cada ``search_documents``, recargando el modelo desde disco
en cada búsqueda — el origen del timeout del Analista de Documentos. Ahora la
factory está memoizada: una sola instancia compartida por proceso.

Estos tests NO cargan ningún modelo: ``build_default_embedder()`` solo construye
el objeto (la carga real ocurre en el primer ``encode``).
"""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.embeddings import (
    SentenceTransformersEmbedder,
    build_default_embedder,
    reset_default_embedder,
)


@pytest.mark.unit
class TestEmbedderSingleton:
    def test_returns_same_instance(self) -> None:
        reset_default_embedder()
        try:
            a = build_default_embedder()
            b = build_default_embedder()
            assert a is b
            assert isinstance(a, SentenceTransformersEmbedder)
        finally:
            reset_default_embedder()

    def test_reset_creates_new_instance(self) -> None:
        reset_default_embedder()
        try:
            a = build_default_embedder()
            reset_default_embedder()
            b = build_default_embedder()
            assert a is not b
        finally:
            reset_default_embedder()

    def test_respects_model_env_on_first_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_default_embedder()
        try:
            monkeypatch.setenv("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
            emb = build_default_embedder()
            # El nombre solicitado se fija en construcción (sin cargar el modelo).
            assert emb._requested_name == "intfloat/multilingual-e5-small"
        finally:
            reset_default_embedder()
