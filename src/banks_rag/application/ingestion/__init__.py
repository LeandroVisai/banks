"""Casos de uso del pipeline de ingesta: extract → enrich → vectorize → persist."""

from .enrich_corpus import (
    EnrichmentReport,
    EnrichmentResult,
    enrich_chunk,
    enrich_corpus,
)
from .extract_corpus import (
    ExtractionReport,
    ExtractionResult,
    extract_corpus,
)
from .persist_corpus import (
    PersistenceResult,
    corpus_stats,
    detect_embedding_dim,
    persist_corpus,
    setup_corpus,
)
from .vectorize_corpus import (
    VectorizationReport,
    VectorizationResult,
    vectorize_corpus,
)

__all__ = [
    "ExtractionReport",
    "ExtractionResult",
    "extract_corpus",
    "EnrichmentReport",
    "EnrichmentResult",
    "enrich_chunk",
    "enrich_corpus",
    "VectorizationReport",
    "VectorizationResult",
    "vectorize_corpus",
    "PersistenceResult",
    "corpus_stats",
    "detect_embedding_dim",
    "persist_corpus",
    "setup_corpus",
]
