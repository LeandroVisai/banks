"""Adaptadores de reranking cross-encoder.

Expone el Protocol ``Reranker`` y la implementación concreta
``CrossEncoderReranker`` (bge-reranker-v2-m3 u otro modelo compatible).
"""

from .cross_encoder_reranker import CrossEncoderReranker, Reranker

__all__ = ["CrossEncoderReranker", "Reranker"]
