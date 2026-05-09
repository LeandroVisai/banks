"""Entidades de documentos: Document, Chunk (con kind), EnrichedChunk."""

from .chunk import Chunk, ChunkKind
from .document import Document
from .enrichment import EnrichedChunk

__all__ = ["Chunk", "ChunkKind", "Document", "EnrichedChunk"]
