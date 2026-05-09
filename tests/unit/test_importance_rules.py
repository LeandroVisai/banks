"""Unit tests para reglas de importance_score."""

from __future__ import annotations

import pytest

from banks_rag.domain_knowledge.importance_rules import (
    IMPORTANCE_WEIGHTS,
    VISUAL_CHUNK_FLOOR,
    calculate_importance,
)


def _var(level: str = "HIGH", indicator: str = "CONTEMPORANEOUS") -> dict:
    """Helper: construye un dict de variable como produce detect_variables."""
    return {"importance": level, "indicator_type": indicator, "mentions": 1, "confidence": 0.5}


@pytest.mark.unit
class TestCalculateImportance:
    def test_empty_chunk_gets_penalty(self) -> None:
        # Sin variables ni datos numéricos
        score = calculate_importance(
            variables={}, numerics=[], section_type="CONTENIDO",
            entities={}, is_fwd=False,
        )
        # Solo aplica no_variables_penalty
        assert score == 0.0  # negativo capped a 0

    def test_critical_var_alone(self) -> None:
        score = calculate_importance(
            variables={"TASA_INTERES": _var("CRITICAL")},
            numerics=[], section_type="CONTENIDO",
            entities={}, is_fwd=False,
        )
        assert score == pytest.approx(IMPORTANCE_WEIGHTS["critical_var"], abs=0.01)

    def test_two_critical_adds_bonus(self) -> None:
        score = calculate_importance(
            variables={
                "TASA_INTERES": _var("CRITICAL"),
                "INFLACION": _var("CRITICAL"),
            },
            numerics=[], section_type="CONTENIDO",
            entities={}, is_fwd=False,
        )
        expected = IMPORTANCE_WEIGHTS["critical_var"] + IMPORTANCE_WEIGHTS["many_critical"]
        assert score == pytest.approx(expected, abs=0.01)

    def test_high_only_no_critical(self) -> None:
        score = calculate_importance(
            variables={"V1": _var("HIGH")},
            numerics=[], section_type="CONTENIDO",
            entities={}, is_fwd=False,
        )
        assert score == pytest.approx(IMPORTANCE_WEIGHTS["high_var_only"], abs=0.01)

    def test_decision_section_boost(self) -> None:
        score_with_decision = calculate_importance(
            variables={"V": _var("CRITICAL")}, numerics=[],
            section_type="DECISION", entities={}, is_fwd=False,
        )
        score_without = calculate_importance(
            variables={"V": _var("CRITICAL")}, numerics=[],
            section_type="CONTENIDO", entities={}, is_fwd=False,
        )
        assert score_with_decision > score_without
        assert score_with_decision - score_without == pytest.approx(
            IMPORTANCE_WEIGHTS["policy_section"], abs=0.01
        )

    def test_numerics_saturate_at_3(self) -> None:
        s_one = calculate_importance(
            variables={"V": _var("CRITICAL")}, numerics=[{"v": "1"}],
            section_type="CONTENIDO", entities={}, is_fwd=False,
        )
        s_three = calculate_importance(
            variables={"V": _var("CRITICAL")},
            numerics=[{"v": str(i)} for i in range(3)],
            section_type="CONTENIDO", entities={}, is_fwd=False,
        )
        s_ten = calculate_importance(
            variables={"V": _var("CRITICAL")},
            numerics=[{"v": str(i)} for i in range(10)],
            section_type="CONTENIDO", entities={}, is_fwd=False,
        )
        # 3 == 10 (saturado); 1 < 3
        assert s_three > s_one
        assert s_three == pytest.approx(s_ten, abs=0.001)

    def test_boilerplate_zeros_score(self) -> None:
        # Boilerplate gana incluso con variable CRITICAL + datos
        score = calculate_importance(
            variables={"V": _var("CRITICAL")},
            numerics=[{"v": "5"}, {"v": "6"}, {"v": "7"}],
            section_type="DECISION",
            entities={"BANCO_CENTRAL_CHILE": 2},
            is_fwd=True,
            text_norm="this email is intended only for the use of the addressee",
        )
        # Boilerplate matchea → 0.0
        assert score == 0.0

    def test_score_capped_at_one(self) -> None:
        score = calculate_importance(
            variables={
                "V1": _var("CRITICAL"),
                "V2": _var("CRITICAL"),
                "V3": _var("CRITICAL"),
                "LIQUIDEZ_MERCADO": _var("HIGH"),
                "MERCADO_NDF": _var("HIGH"),
                "FONDO_PENSION": _var("HIGH"),
            },
            numerics=[{"v": str(i)} for i in range(5)],
            section_type="DECISION",
            entities={"E1": 1, "E2": 1, "E3": 1},
            is_fwd=True,
            text_norm="política monetaria expansiva",
            deviation_flag=True,
        )
        assert score <= 1.0

    def test_visual_chunk_floor_constant(self) -> None:
        # El piso debe ser razonable (entre 0.2 y 0.4)
        assert 0.2 <= VISUAL_CHUNK_FLOOR <= 0.4
