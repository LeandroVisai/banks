"""Casos de uso del pipeline de ingesta: extract → enrich → vectorize → persist."""

from .extract_corpus import (
    ExtractionReport,
    ExtractionResult,
    extract_corpus,
)

__all__ = ["ExtractionReport", "ExtractionResult", "extract_corpus"]
