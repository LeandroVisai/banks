"""Unit tests del normalizador de texto."""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.extractors.text_normalizer import normalize_page_text


@pytest.mark.unit
class TestNormalizePageText:
    def test_empty_input(self) -> None:
        assert normalize_page_text("") == ""

    def test_joins_hyphenated_linebreak(self) -> None:
        assert normalize_page_text("infla-\nción") == "inflación"

    def test_collapses_intra_paragraph_newlines(self) -> None:
        result = normalize_page_text("línea 1\nlínea 2")
        assert "\n" not in result
        assert result == "línea 1 línea 2"

    def test_preserves_paragraph_separator(self) -> None:
        result = normalize_page_text("párrafo 1\n\npárrafo 2")
        assert "\n\n" in result

    def test_collapses_multiple_spaces(self) -> None:
        assert normalize_page_text("hola    mundo") == "hola mundo"

    def test_removes_orphan_page_numbers(self) -> None:
        result = normalize_page_text("Texto previo\n\n42\n\nTexto siguiente")
        assert "42" not in result.split()

    def test_removes_drm_watermark(self) -> None:
        text = "Reporte importante {[{abc123}]}sigue."
        result = normalize_page_text(text)
        assert "{[{" not in result
        assert "}]}" not in result

    def test_strips_control_chars(self) -> None:
        text = "limpio\x00\x01\x02texto"
        result = normalize_page_text(text)
        assert "\x00" not in result
        assert "\x01" not in result

    def test_applies_mojibake_fix(self) -> None:
        # 'política' encoded as latin-1 from utf-8 bytes
        broken = "polÃ\xadtica"
        result = normalize_page_text(broken)
        assert "política" in result.lower()
