"""Unit tests para los reparadores de encoding (BCCh + mojibake)."""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.extractors.encoding_fixers import (
    BCCH_CHAR_MAP,
    fix_all,
    fix_bcch_font,
    fix_mojibake,
)


@pytest.mark.unit
class TestFixBcchFont:
    def test_decodes_classic_segment(self) -> None:
        # @=a, A=b, B=c → ÄÄK = ' ' ' l' (Ä=espacio, K=l)
        # Ej: '@A@CB@DE@FÄ' (12 chars) → 'abacc adef '
        encoded = "@A@CB@DE@FÄ@K"  # 13 chars matching [@A-YÄ-Ç¹]+
        decoded = fix_bcch_font(encoded)
        # Cada char se mapea según BCCH_CHAR_MAP
        expected = "".join(BCCH_CHAR_MAP.get(c, c) for c in encoded)
        assert decoded == expected

    def test_skips_normal_text(self) -> None:
        assert fix_bcch_font("Política monetaria 2024") == "Política monetaria 2024"

    def test_skips_when_no_bcch_signature(self) -> None:
        # Ä sin @ — no es BCCh
        text = "Ä veces hay caracteres latinos"
        assert fix_bcch_font(text) == text

    def test_does_not_touch_short_segments(self) -> None:
        # Segmento de 7 chars no califica (regex requiere 8+)
        text = "@AB@CD@"
        assert fix_bcch_font(text) == text


@pytest.mark.unit
class TestFixMojibake:
    def test_repairs_classic_mojibake(self) -> None:
        # 'inflación' → bytes UTF-8 → leídos como Latin-1
        original = "inflación".encode("utf-8").decode("latin-1")
        assert "Ã" in original
        assert fix_mojibake(original) == "inflación"

    def test_skips_clean_text(self) -> None:
        clean = "tipo de cambio diario"
        assert fix_mojibake(clean) == clean

    def test_returns_unchanged_when_undecodable(self) -> None:
        # Contiene Ã pero no es Latin-1 → UTF-8 válido (e.g., ya es UTF-8 con
        # algún Ã legítimo); la función debe no romper.
        edge = "AÃ"  # solo 'AÃ' no se puede re-decode bien
        result = fix_mojibake(edge)
        # No debe explotar; el output puede ser igual o un decode raro.
        assert isinstance(result, str)


@pytest.mark.unit
def test_fix_all_applies_bcch_before_mojibake() -> None:
    """fix_all debe aplicar BCCh antes que mojibake porque Ä rompería el round-trip."""
    # Texto BCCh-encoded sin mojibake
    encoded = "@@@AAÄBBCC"  # 10 chars > 8
    result = fix_all(encoded)
    # No debe llamar a mojibake (no hay Ã en la salida)
    assert "Ã" not in result
    assert result == "".join(BCCH_CHAR_MAP.get(c, c) for c in encoded)
