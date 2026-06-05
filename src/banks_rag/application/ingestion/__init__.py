"""Casos de uso del pipeline de ingesta: extract → enrich → vectorize → persist."""

from .enrich_corpus import (
    EnrichmentReport,
    EnrichmentResult,
    enrich_chunk,
    enrich_corpus,
)
from .enrich_news import (
    NewsEnrichmentReport,
    NewsEnrichmentResult,
    enrich_news,
    enrich_news_chunk,
)
from .extract_corpus import (
    ExtractionReport,
    ExtractionResult,
    extract_corpus,
)
from .extract_news import (
    NewsExtractionReport,
    NewsExtractionResult,
    extract_news,
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
    "NewsExtractionReport",
    "NewsExtractionResult",
    "extract_news",
    "NewsEnrichmentReport",
    "NewsEnrichmentResult",
    "enrich_news",
    "enrich_news_chunk",
    "VectorizationReport",
    "VectorizationResult",
    "vectorize_corpus",
    "PersistenceResult",
    "corpus_stats",
    "detect_embedding_dim",
    "persist_corpus",
    "setup_corpus",
]
