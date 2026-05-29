"""Unit tests del grounding numérico (anti-alucinación de cifras)."""

from __future__ import annotations

import pytest

from banks_rag.application.agent.numeric_grounding import (
    extract_financial_numbers,
    extract_numbers,
    has_financial_numbers,
    verify_numbers,
)


@pytest.mark.unit
class TestExtraction:
    def test_detects_percentages_and_decimals(self) -> None:
        nums = extract_financial_numbers("El DV01 fue 0,85% y subió a 0,92%.")
        assert 0.85 in nums and 0.92 in nums

    def test_detects_spanish_decimal_and_thousands(self) -> None:
        nums = extract_financial_numbers("El cobre estaba en 624,9 y el monto 1.069.")
        assert 624.9 in nums
        assert 1069.0 in nums  # >= 1000 cuenta como financiera

    def test_ignores_years(self) -> None:
        nums = extract_financial_numbers("En 2026 hubo 3 reuniones.")
        assert nums == []  # 2026 es año; 3 no es financiera

    def test_has_financial_numbers(self) -> None:
        assert has_financial_numbers("subió 2,3%")
        assert not has_financial_numbers("hubo 3 reuniones en 2024")

    def test_extract_numbers_includes_everything(self) -> None:
        # extract_numbers (para poblar fundados) sí toma años/enteros.
        nums = extract_numbers("3 reuniones en 2024, tasa 5,25")
        assert 3.0 in nums and 2024.0 in nums and 5.25 in nums


@pytest.mark.unit
class TestVerifyNumbers:
    def test_grounded_within_tolerance(self) -> None:
        flagged, ok = verify_numbers("La tasa fue 5,25%.", [5.25])
        assert ok and flagged == []

    def test_rounding_tolerance(self) -> None:
        # 5,2 reportado vs 5,25 fundado: dentro de tolerancia relativa.
        _, ok = verify_numbers("La tasa rondó 5,2%.", [5.25])
        assert ok

    def test_unit_conversion_x100(self) -> None:
        # 6,249 USD/lb proviene de 624,9 USc/lb (÷100): debe quedar fundado.
        _, ok = verify_numbers("El cobre a 6,249 USD/lb.", [624.9])
        assert ok

    def test_ungrounded_flagged(self) -> None:
        flagged, ok = verify_numbers("El DV01 promedió 0,85%.", [])
        assert not ok
        assert 0.85 in flagged

    def test_ungrounded_with_some_grounded(self) -> None:
        flagged, ok = verify_numbers(
            "Subió de 5,25% a 9,99%.", [5.25],
        )
        assert not ok
        assert 9.99 in flagged and 5.25 not in flagged
