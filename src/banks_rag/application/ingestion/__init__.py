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

__all__ = [
    "ExtractionReport",
    "ExtractionResult",
    "extract_corpus",
    "EnrichmentReport",
    "EnrichmentResult",
    "enrich_chunk",
    "enrich_corpus",
]
