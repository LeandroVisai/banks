"""Chunker semántico jerárquico para texto extraído de PDFs.

Algoritmo:
  1. Itera página por página (preserva ``page_start``/``page_end``).
  2. Divide cada página en párrafos (``\\n\\n+``).
  3. Detecta títulos cortos en mayúsculas como sección.
  4. Si un párrafo excede ``max_chunk_chars``, lo divide por oraciones.
  5. Acumula párrafos hasta llegar a ``target_chunk_chars`` y flush.
  6. Aplica overlap entre chunks consecutivos cortando en límite de oración.
  7. Fusiona chunks finales huérfanos < ``min_chunk_chars``.

Funciones puras. La normalización de texto debe haberse aplicado antes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_TARGET_CHUNK_CHARS = 600
DEFAULT_MIN_CHUNK_CHARS = 200
DEFAULT_MAX_CHUNK_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 100

_SECTION_TITLE_RE = re.compile(r"^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{2,60})\s*$")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9])")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9\"])")


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuración del chunker. Todos los valores en chars."""

    target_chunk_chars: int = DEFAULT_TARGET_CHUNK_CHARS
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    return [s.strip() for s in sentences if s.strip()]


def detect_section_title(paragraph: str) -> str | None:
    """Una línea corta en mayúsculas es probable título de sección."""
    if "\n" in paragraph or len(paragraph) > 80:
        return None
    m = _SECTION_TITLE_RE.match(paragraph)
    return m.group(1).strip() if m else None


def _overlap_from_sentence_boundary(tail: str, target_len: int) -> str:
    """Tramo de ``tail`` que empieza en el inicio de la primera oración completa.

    Si no hay límite de oración con longitud razonable, retorna los últimos
    ``target_len`` chars.
    """
    for m in _SENTENCE_BOUNDARY_RE.finditer(tail):
        candidate = tail[m.end():]
        if len(candidate) >= target_len // 2:
            return candidate
    return tail[-target_len:]


def chunk_pages(
    pages: list[str],
    config: ChunkingConfig | None = None,
) -> list[dict]:
    """Convierte una lista de textos de páginas (ya normalizados) en chunks.

    Retorna lista de dicts con keys ``text``, ``page_start``, ``page_end``,
    ``section_title_raw``.
    """
    cfg = config or ChunkingConfig()

    raw: list[dict] = []
    current_section: str | None = None

    for page_num, page_text in enumerate(pages, start=1):
        if not page_text:
            continue

        for para in split_paragraphs(page_text):
            title = detect_section_title(para)
            if title:
                current_section = title
                continue

            if len(para) > cfg.max_chunk_chars:
                for sent in split_sentences(para):
                    raw.append({"text": sent, "page": page_num, "section": current_section})
            else:
                raw.append({"text": para, "page": page_num, "section": current_section})

    chunks: list[dict] = []
    buffer: list[str] = []
    buf_pages: list[int] = []
    buf_section: str | None = None
    buf_len = 0

    def _flush() -> None:
        nonlocal buffer, buf_pages, buf_section, buf_len
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if not text:
            return
        chunks.append({
            "text": text,
            "page_start": min(buf_pages),
            "page_end": max(buf_pages),
            "section_title_raw": buf_section,
        })

    for unit in raw:
        unit_len = len(unit["text"])

        if buf_len + unit_len > cfg.target_chunk_chars and buf_len >= cfg.min_chunk_chars:
            _flush()
            if cfg.overlap_chars > 0 and buffer:
                raw_tail = " ".join(buffer)[-(cfg.overlap_chars * 2):]
                tail = _overlap_from_sentence_boundary(raw_tail, cfg.overlap_chars)
                buffer = [tail]
                buf_len = len(tail)
                buf_pages = [buf_pages[-1]]
            else:
                buffer = []
                buf_pages = []
                buf_len = 0
            buf_section = unit["section"] or buf_section

        buffer.append(unit["text"])
        buf_pages.append(unit["page"])
        buf_len += unit_len + 1
        if buf_section is None:
            buf_section = unit["section"]

    _flush()

    if len(chunks) >= 2 and len(chunks[-1]["text"]) < cfg.min_chunk_chars:
        last = chunks.pop()
        prev = chunks[-1]
        prev["text"] = prev["text"] + " " + last["text"]
        prev["page_end"] = max(prev["page_end"], last["page_end"])

    return chunks
