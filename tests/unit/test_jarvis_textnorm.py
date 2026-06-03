"""Tests de la normalización Markdown → texto hablable (jarvis_news.textnorm)."""

from __future__ import annotations

import pytest

from jarvis_news.textnorm import to_speakable_text


@pytest.mark.unit
class TestToSpeakableText:
    def test_strips_headings(self) -> None:
        out = to_speakable_text("### Resumen Ejecutivo")
        assert "#" not in out
        assert out.startswith("Resumen Ejecutivo")

    def test_strips_bullets_and_numbered_lists(self) -> None:
        out = to_speakable_text("- primer punto\n1. segundo punto\n2) tercero")
        assert "- " not in out
        assert not out.startswith("1")
        assert "primer punto" in out
        assert "segundo punto" in out
        assert "tercero" in out

    def test_keeps_real_numbers_in_text(self) -> None:
        # Las cifras del contenido NO se tocan (años, porcentajes, montos).
        out = to_speakable_text("- El PIB cayó 1,5% en 2026 a US$ 374 millones")
        assert "1,5%" in out
        assert "2026" in out
        assert "US$ 374 millones" in out

    def test_strips_bold_italic_code(self) -> None:
        out = to_speakable_text("El **Banco Central** bajó la *tasa* a `2.5`")
        assert "*" not in out
        assert "`" not in out
        assert "Banco Central" in out
        assert "tasa" in out

    def test_strips_links_keeping_text(self) -> None:
        out = to_speakable_text("Ver [el informe](https://x.com/a) hoy")
        assert "http" not in out
        assert "el informe" in out

    def test_drops_horizontal_rules(self) -> None:
        out = to_speakable_text("uno\n\n---\n\ndos")
        assert "-" not in out
        assert "uno" in out and "dos" in out

    def test_adds_sentence_pauses_between_blocks(self) -> None:
        # Cada bloque termina con puntuación para una pausa natural al leer.
        out = to_speakable_text("### Título\n\nContenido sin punto")
        assert "Título." in out

    def test_empty_input(self) -> None:
        assert to_speakable_text("") == ""
        assert to_speakable_text("   \n  ") == ""
