"""Adaptadores de embeddings: Protocol + sentence-transformers + builder de texto."""

from .base import Embedder, is_e5_model, is_qwen_embedding, is_vl_model
from .embed_text_builder import (
    E5_PASSAGE_PREFIX,
    build_embed_text,
    metadata_context_enabled,
)
from .sentence_transformers_embedder import (
    SentenceTransformersEmbedder,
    build_default_embedder,
)

__all__ = [
    "Embedder",
    "is_e5_model",
    "is_qwen_embedding",
    "is_vl_model",
    "E5_PASSAGE_PREFIX",
    "build_embed_text",
    "metadata_context_enabled",
    "SentenceTransformersEmbedder",
    "build_default_embedder",
]
