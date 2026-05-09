"""Entidades de documentos: Document, Chunk (con kind), EnrichedChunk, VisualAsset."""

from .chunk import Chunk, ChunkKind
from .document import Document
from .enrichment import EnrichedChunk
from .visual_asset import (
    BoundingBox,
    VisualAsset,
    VisualExtractionStats,
    VisualKind,
)

__all__ = [
    "Chunk",
    "ChunkKind",
    "Document",
    "EnrichedChunk",
    "BoundingBox",
    "VisualAsset",
    "VisualExtractionStats",
    "VisualKind",
]
