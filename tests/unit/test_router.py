"""Tests del router determinista de especialistas (router-v1).

Reemplaza el loop LLM del orquestador: el ruteo es puro y testeable. Cubre las
preguntas reales de los logs + casos cross-dominio, saludos y default.
"""

from __future__ import annotations

import pytest

from banks_rag.application.agent.router import MAX_SPECIALISTS, select_specialists


def _keys(message, history=None) -> list[str]:
    return [s.key for s in select_specialists(message, history)]


@pytest.mark.unit
class TestSelectSpecialists:
    @pytest.mark.parametrize(
        "message,expected",
        [
            # Mercado (preguntas de los logs).
            ("Dame el precio histórico del cobre", ["fx"]),
            ("¿Tienes el DV01 de los fondos de pensiones?", ["afp"]),
            ("composición de cartera AFP", ["afp"]),
            ("posición de no residentes en instrumentos financieros", ["no_residentes"]),
            ("flujos de los fondos mutuos", ["fondos_mutuos"]),
            ("¿cómo está la curva BTP a 10 años?", ["renta_fija"]),
            ("LCR del sistema bancario", ["liquidez"]),
            # Política monetaria / corpus.
            ("¿Qué decidió el Consejo del BCCh en su última reunión?", ["policy"]),
        ],
    )
    def test_routes_single_domain(self, message, expected) -> None:
        assert _keys(message) == expected

    def test_visual_request_routes_to_document(self) -> None:
        assert "document" in _keys("muéstrame el gráfico de inflación del IPoM")

    def test_cross_domain_includes_multiple(self) -> None:
        # "postura del Consejo" → policy; "curva BTP" → renta_fija.
        keys = _keys("compara la postura del Consejo con la curva BTP")
        assert "renta_fija" in keys and "policy" in keys

    def test_caps_at_max_specialists(self) -> None:
        keys = _keys("dolar, cobre, DV01 AFP, fondos mutuos, curva BTP, LCR y no residentes")
        assert len(keys) <= MAX_SPECIALISTS

    @pytest.mark.parametrize("greeting", ["Hola", "buenos días", "gracias", "¿qué puedes hacer?", "quién eres"])
    def test_greetings_route_to_no_specialists(self, greeting) -> None:
        assert _keys(greeting) == []

    def test_unmatched_substantive_defaults_to_corpus(self) -> None:
        # Pregunta substantiva sin señales claras → corpus general.
        assert _keys("Dame algún antecedente relevante de esta semana") in (
            ["document", "policy"], ["document"], ["policy"],
        ) or set(_keys("xyzzy plugh blarg")) == {"document", "policy"}

    def test_anaphora_uses_history(self) -> None:
        # Follow-up que por sí solo no rutea; el history (DV01 AFP) lo ancla.
        history = [
            {"role": "user", "content": "¿Tienes el DV01 de los fondos de pensiones?"},
            {"role": "assistant", "content": "El DV01 promedió ..."},
        ]
        assert "afp" in _keys("¿de dónde salió ese dato?", history)

    def test_self_contained_question_ignores_prior_topic(self) -> None:
        # Pregunta autocontenida de política: NO debe arrastrar el FX del turno
        # anterior (bug de sobre-ruteo visto en producción).
        history = [
            {"role": "user", "content": "que me puedes decir del mercado cambiario?"},
            {"role": "assistant", "content": "El USD/CLP..."},
        ]
        keys = _keys("¿cuáles fueron los últimos acuerdos de política monetaria?", history)
        assert keys == ["policy"]
        assert "fx" not in keys
