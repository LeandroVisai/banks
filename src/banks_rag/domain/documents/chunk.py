"""Chunk entity — fragmento textual o visual con citación auditable."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChunkKind(str, Enum):
    """Discriminator para el tipo de fragmento.

    - TEXT: texto narrativo (chunks típicos de PDF/Excel).
    - VISUAL: gráfico/figura embebida (con image_path).
    - TABLE: tabla estructurada (futuro — Fase 2/8 con table_extractor).
    """

    TEXT = "TEXT"
    VISUAL = "VISUAL"
    TABLE = "TABLE"


@dataclass
class Chunk:
    """Fragmento de texto crudo (N por documento).

    El campo ``kind`` discrimina entre texto, visual o tabla y permite
    al ``hybrid_search`` filtrar por tipo. ``image_path`` solo aplica a
    ``kind=VISUAL``.

    ``chunk_date`` solo se popula para chunks del Excel Monitor PM
    (cada celda tiene fecha propia). Los PDFs heredan la fecha del documento.
    """

    chunk_id: str
    document_id: str
    text: str
    char_count: int
    token_estimate: int  # heurística: char_count // 4
    page_start: int
    page_end: int
    position_in_doc: int  # 0-based
    section_title_raw: str | None
    chunk_date: str | None = None  # ISO YYYY-MM-DD; solo Monitor PM
    image_path: str | None = None  # solo páginas visuales
    kind: ChunkKind = ChunkKind.TEXT
