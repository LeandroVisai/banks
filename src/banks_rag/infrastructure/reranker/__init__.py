"""Adaptadores de reranking cross-encoder.

Expone el Protocol ``Reranker`` y la implementación concreta
``CrossEncoderReranker`` (jina-reranker-v3 u otro modelo compatible).
"""

from .cross_encoder_reranker import (
    CrossEncoderReranker,
    Reranker,
    build_default_reranker,
    reranking_enabled,
    reset_default_reranker,
)

__all__ = [
    "CrossEncoderReranker",
    "Reranker",
    "build_default_reranker",
    "reranking_enabled",
    "reset_default_reranker",
]
