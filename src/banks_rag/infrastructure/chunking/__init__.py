"""Adaptadores de chunking: jerárquico de texto y por celdas (Excel Monitor PM)."""

from .text_chunker import (
    ChunkingConfig,
    chunk_pages,
    detect_section_title,
    split_paragraphs,
    split_sentences,
)

__all__ = [
    "ChunkingConfig",
    "chunk_pages",
    "split_paragraphs",
    "split_sentences",
    "detect_section_title",
]
