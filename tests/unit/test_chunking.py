"""Unit tests del chunker jerárquico."""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.chunking import (
    ChunkingConfig,
    chunk_pages,
    detect_section_title,
    split_paragraphs,
    split_sentences,
)


@pytest.mark.unit
class TestSplitParagraphs:
    def test_splits_on_double_newline(self) -> None:
        text = "Párrafo 1.\n\nPárrafo 2.\n\nPárrafo 3."
        assert split_paragraphs(text) == ["Párrafo 1.", "Párrafo 2.", "Párrafo 3."]

    def test_strips_whitespace(self) -> None:
        assert split_paragraphs("  hola  \n\n  mundo  ") == ["hola", "mundo"]

    def test_filters_empty(self) -> None:
        assert split_paragraphs("\n\n\n") == []


@pytest.mark.unit
class TestSplitSentences:
    def test_splits_on_period_uppercase(self) -> None:
        text = "Primera oración. Segunda oración. Tercera."
        result = split_sentences(text)
        assert len(result) == 3
        assert result[0] == "Primera oración."

    def test_question_marks(self) -> None:
        result = split_sentences("¿Pregunta? Respuesta clara.")
        assert len(result) == 2

    def test_does_not_split_on_lowercase(self) -> None:
        # "Sr. Juan" no debería partirse porque después del punto viene minúscula
        result = split_sentences("Llegó el Sr. juan ayer.")
        assert len(result) == 1


@pytest.mark.unit
class TestDetectSectionTitle:
    def test_detects_caps_title(self) -> None:
        assert detect_section_title("DECISIÓN") == "DECISIÓN"

    def test_detects_multi_word_caps(self) -> None:
        assert detect_section_title("DISCUSIÓN Y ACUERDOS") == "DISCUSIÓN Y ACUERDOS"

    def test_rejects_long_text(self) -> None:
        long_text = "X" * 100
        assert detect_section_title(long_text) is None

    def test_rejects_multiline(self) -> None:
        assert detect_section_title("LÍNEA1\nLÍNEA2") is None

    def test_rejects_lowercase(self) -> None:
        assert detect_section_title("decisión") is None


@pytest.mark.unit
class TestChunkPages:
    def test_simple_short_pages(self) -> None:
        pages = ["Texto corto en página 1.", "Texto corto en página 2."]
        chunks = chunk_pages(pages)
        assert len(chunks) >= 1
        assert all("text" in c for c in chunks)
        assert all("page_start" in c and "page_end" in c for c in chunks)

    def test_preserves_page_range(self) -> None:
        pages = ["X" * 700, "Y" * 700, "Z" * 700]
        chunks = chunk_pages(pages, ChunkingConfig(target_chunk_chars=600, max_chunk_chars=1200))
        # Cada chunk debe tener page_start <= page_end y referenciar páginas válidas
        for c in chunks:
            assert 1 <= c["page_start"] <= c["page_end"] <= 3

    def test_section_title_propagated(self) -> None:
        pages = ["DECISIÓN\n\n" + "El Consejo decide " * 30]
        chunks = chunk_pages(pages)
        assert any(c["section_title_raw"] == "DECISIÓN" for c in chunks)

    def test_empty_pages_skipped(self) -> None:
        chunks = chunk_pages(["", "   ", ""])
        assert chunks == []

    def test_long_paragraph_split_by_sentences(self) -> None:
        sentence = "Esta es una oración larga. " * 20  # ~540 chars
        pages = [sentence * 5]  # ~2700 chars en un solo párrafo
        chunks = chunk_pages(pages, ChunkingConfig(max_chunk_chars=600, target_chunk_chars=400))
        assert len(chunks) >= 2
