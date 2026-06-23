"""Tests del verificador de afirmaciones del informe + el fix de flujos por ventana.

Sin BD ni LLM real: el verificador determinista opera sobre dicts de facts
sintéticos (mismo shape que ``compute_facts``); el crítico LLM se mockea.
Reproduce el caso que detectó la analista: el mismo "Tipo 1" descrito como
entrada en un bloque y como salida en otro.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from banks_rag.application.reporting.parquet_facts import (
    _flow_window,
    _window_variations,
)
from banks_rag.application.reporting.report_spec import (
    FamilyReportSpec,
    ReportBlock,
)
from banks_rag.application.reporting.specs.ffmm_spec import FFMM_SPEC
from banks_rag.application.reporting.verify import (
    SYNTHESIS_KEY,
    _parse_number,
    build_ledger,
    correct_directions,
    detect_intra_contradiction,
    flag_ungrounded_numbers,
    verify_report,
)
from banks_rag.domain.agent import GenerationResult
from banks_rag.infrastructure.sql.parquet_catalog_loader import ColumnSpec, ParquetDataset


# ── value_kind → is_flow ─────────────────────────────────────────────────────


def _ds(dataset_id: str, *, value_kind: str = "") -> ParquetDataset:
    return ParquetDataset(
        id=dataset_id, file=f"{dataset_id}.parquet", name=dataset_id, description="",
        segment="ffmm", unit="US$ Mill.", date_range=None,
        columns=[ColumnSpec(name="fecha", type="TIMESTAMP")],
        value_kind=value_kind,
    )


def test_value_kind_overrides_keyword_heuristic():
    from banks_rag.application.reporting.parquet_facts import _is_flow

    # 'flujos' en el nombre → keyword diría flow; value_kind=stock lo override.
    assert _is_flow(_ds("flujos_acumulados_x", value_kind="stock")) is False
    # Sin keyword en el nombre pero declarado flow → True.
    assert _is_flow(_ds("var_pos_x", value_kind="flow")) is True
    # Sin value_kind: cae a la heurística de palabras clave.
    assert _is_flow(_ds("flujos_x")) is True
    assert _is_flow(_ds("stock_x")) is False


# ── 1A: flujos por ventana = SUMA, no punto-a-punto ──────────────────────────


def test_flow_window_sums_period_flows():
    series = [("2026-06-01", 100.0), ("2026-06-08", 50.0), ("2026-06-15", -30.0)]
    fw = _flow_window(series)
    assert fw["is_flow_sum"] is True
    assert fw["flujo_periodo"] == pytest.approx(120.0)  # 100+50-30, NO 'último-primero'(-130)
    assert fw["n_observaciones"] == 3


def test_window_variations_flow_vs_stock():
    series = [("2026-06-01", 100.0), ("2026-06-10", 50.0), ("2026-06-20", 40.0)]
    wins = [{"label": "v", "start": "2026-06-01", "end": "2026-06-20"}]
    flow = _window_variations(series, wins, is_flow=True)[0]["variacion"]
    stock = _window_variations(series, wins, is_flow=False)[0]["variacion"]
    assert flow["flujo_periodo"] == pytest.approx(190.0)        # suma
    assert stock["cambio_absoluto"] == pytest.approx(-60.0)     # 40-100 (punto a punto)


# ── Ledger ───────────────────────────────────────────────────────────────────


def _flow_facts_t1_entrada() -> dict:
    """Facts estilo compute_facts: Tipo 1 con flujo del período POSITIVO (entrada)."""
    return {
        "is_flow": True, "shape": "timeseries_categorical", "unit": "US$ Mill.",
        "por_categoria": [
            {"categoria": "Tipo 1", "ultimo_valor": 6.55, "ventanas": [
                {"label": "último mes", "variacion": {
                    "is_flow_sum": True, "flujo_periodo": 297.73, "minimo": -10.0, "maximo": 300.0}},
            ]},
        ],
        "contribuciones": [{"label": "último mes", "drivers": [
            {"categoria": "Tipo 1", "cambio_absoluto": 297.73, "nivel_fin": 297.73, "variacion_pct": None}]}],
    }


def test_build_ledger_directions_and_numbers():
    led = build_ledger(_flow_facts_t1_entrada())
    assert led.is_flow is True
    assert led.direction_by_cat["Tipo 1"] == {1}      # entrada
    assert 297.73 in led.numbers


# ── Corrección determinista de dirección (el caso de la analista) ────────────


def test_correct_directions_fixes_contradiction():
    led = build_ledger(_flow_facts_t1_entrada())
    prose = "El Tipo 1 registró una salida de US$ 297,73 Mill. con rescates en el mes."
    fixed, issues = correct_directions(prose, led, where="flujos_ffmm")
    assert "entrada" in fixed and "salida" not in fixed
    assert any(i.kind == "direction" and i.auto_corrected for i in issues)


def test_correct_directions_keeps_mixed_signs_untouched():
    # Tipo 1 con entrada en una ventana y salida en otra: ambas direcciones válidas.
    facts = {
        "is_flow": True, "shape": "timeseries_categorical",
        "por_categoria": [{"categoria": "Tipo 1", "ultimo_valor": 0.0, "ventanas": [
            {"label": "mes", "variacion": {"is_flow_sum": True, "flujo_periodo": 100.0}},
            {"label": "semana", "variacion": {"is_flow_sum": True, "flujo_periodo": -50.0}},
        ]}],
    }
    led = build_ledger(facts)
    assert led.direction_by_cat["Tipo 1"] == {1, -1}
    prose = "El Tipo 1 muestra salida en la semana."
    fixed, issues = correct_directions(prose, led, where="x")
    assert fixed == prose and not issues  # no toca: los datos justifican ambas


def test_correct_directions_noop_on_non_flow():
    led = build_ledger({"is_flow": False, "shape": "timeseries_single"})
    prose = "Hubo una salida de capitales."
    fixed, issues = correct_directions(prose, led, where="x")
    assert fixed == prose and not issues


# ── Cifras sin sustento + contradicción intra-texto ──────────────────────────


def test_flag_ungrounded_numbers():
    led = build_ledger(_flow_facts_t1_entrada())
    issues = flag_ungrounded_numbers("Tipo 1 con entrada de US$ 999,99 Mill. en 2026.", led, where="x")
    # 999,99 no está en los facts → marcado; 2026 (año) y 'Tipo 1' → ignorados.
    assert len(issues) == 1 and "999" in issues[0].detail


def test_flag_ungrounded_skips_grounded():
    led = build_ledger(_flow_facts_t1_entrada())
    assert flag_ungrounded_numbers("entrada de US$ 297,73 Mill.", led, where="x") == []


def test_detect_intra_contradiction():
    led = build_ledger(_flow_facts_t1_entrada())
    prose = "El Tipo 1 tuvo entrada. Luego el Tipo 1 sufrió una salida."
    issues = detect_intra_contradiction(prose, led, where="x")
    assert any(i.kind == "contradiction" for i in issues)


# ── _parse_number: locales ES y US ───────────────────────────────────────────


@pytest.mark.parametrize("tok,expected", [
    ("297,73", 297.73), ("1.943,09", 1943.09), ("1,943.09", 1943.09),
    ("77.568,29", 77568.29), ("4,62", 4.62), ("-120,41", -120.41), ("285", 285.0),
])
def test_parse_number_locales(tok, expected):
    assert _parse_number(tok) == pytest.approx(expected)


# ── no_text: el bloque mantiene gráfico pero no recibe slot de texto ─────────


def test_no_text_block_has_no_slot():
    from banks_rag.application.reporting.curated_report import _resolve_text_slots

    # Los DOS bloques de flujos comparten source_id (flujos_acum_ffmm): el primero
    # (diferencial t-7/t-30) comenta una vez; el segundo (acumulado) es no_text.
    flujos = [b for b in _resolve_text_slots(FFMM_SPEC) if b.source_id == "flujos_acum_ffmm"]
    assert len(flujos) == 2
    commenting = [b for b in flujos if not b.no_text]
    redundant = [b for b in flujos if b.no_text]
    assert len(commenting) == 1 and len(redundant) == 1
    assert commenting[0].text_slot == "ffmm:flujos_acum_ffmm"   # comenta una vez
    assert redundant[0].text_slot == ""                          # no_text → sin slot


def test_spec_no_text_source_ids():
    # flujos_acum_ffmm también alimenta el bloque comentado (diferencial t-7/t-30),
    # así que NO es redundante: su párrafo debe entrar a la síntesis.
    assert "flujos_acum_ffmm" not in FFMM_SPEC.no_text_source_ids()
    assert FFMM_SPEC.no_text_source_ids() == set()


# ── Integración: verify_report sobre un informe ──────────────────────────────


@dataclass
class _MockLLM:
    responses: list = field(default_factory=list)
    raise_on_call: bool = False

    async def generate(self, messages, *, tools=None, **kwargs):
        if self.raise_on_call:
            raise RuntimeError("servidor caído")
        if self.responses:
            return self.responses.pop(0)
        return GenerationResult(text='{"corrections": [], "contradictions": []}', n_tokens=5)


@dataclass
class _Section:
    dataset_id: str
    paragraph: str
    status: str = "ok"
    facts: dict | None = None
    name: str = "x"
    unit: str = "US$ Mill."
    segment: str = "ffmm"


@dataclass
class _Report:
    sections: list
    overview_md: str = ""


def test_verify_report_autocorrects_direction_no_llm():
    sec = _Section("flujos_ffmm",
                   "El Tipo 1 registró salida de US$ 297,73 Mill.",
                   facts=_flow_facts_t1_entrada())
    report = _Report([sec], overview_md="Tipo 1 con rescates de US$ 297,73 Mill.")
    vr = asyncio.run(verify_report(report, llm=None))
    assert "entrada" in sec.paragraph and "salida" not in sec.paragraph
    assert "entrada" in report.overview_md  # síntesis también corregida
    assert vr.n_direction_fixed >= 2
    assert vr.llm_used is False


def test_verify_report_never_raises_when_llm_fails():
    sec = _Section("flujos_ffmm", "El Tipo 1 tuvo entrada de US$ 297,73 Mill.",
                   facts=_flow_facts_t1_entrada())
    report = _Report([sec], overview_md="Resumen.")
    # llm que revienta en la pasada crítica → no debe tumbar la verificación.
    vr = asyncio.run(verify_report(report, llm=_MockLLM(raise_on_call=True)))
    assert vr is not None  # entregó resultado igual


def test_verify_report_applies_llm_corrections():
    sec = _Section("flujos_ffmm", "El Tipo 1 tuvo entrada de US$ 297,73 Mill.",
                   facts=_flow_facts_t1_entrada())
    report = _Report([sec], overview_md="La cifra global fue de US$ 297,73 Mill.")
    critic = GenerationResult(
        text='{"corrections": [{"find": "global", "replace": "mensual", "reason": "claridad"}], '
             '"contradictions": ["revisar X"]}',
        n_tokens=10)
    vr = asyncio.run(verify_report(report, llm=_MockLLM(responses=[critic])))
    assert "mensual" in report.overview_md
    assert vr.llm_used is True
    assert any(i.where == SYNTHESIS_KEY and "revisar X" in i.detail for i in vr.issues)


def test_verify_report_blocks_llm_numeric_change():
    # El crítico sugiere cambiar una cifra (1.27 -> 2.20): NO se aplica, se marca.
    sec = _Section("flujos_ffmm", "El Tipo 1 tuvo entrada de US$ 297,73 Mill.",
                   facts=_flow_facts_t1_entrada())
    report = _Report([sec], overview_md="DAP incrementó un 1.27% en la semana.")
    critic = GenerationResult(
        text='{"corrections": [{"find": "incrementó un 1.27%", "replace": "incrementó un 2.20%", '
             '"reason": "x"}], "contradictions": []}', n_tokens=10)
    vr = asyncio.run(verify_report(report, llm=_MockLLM(responses=[critic])))
    assert "1.27%" in report.overview_md and "2.20%" not in report.overview_md  # NO aplicado
    assert any("NO aplicado" in i.detail for i in vr.issues)


def test_verify_report_applies_llm_wording_change():
    # Cambio que PRESERVA las cifras (redacción/dirección): sí se aplica.
    sec = _Section("flujos_ffmm", "ok", facts=_flow_facts_t1_entrada())
    report = _Report([sec], overview_md="El Tipo 1 tuvo salida de US$ 297,73 Mill.")
    critic = GenerationResult(
        text='{"corrections": [{"find": "salida de US$ 297,73", "replace": "entrada de US$ 297,73", '
             '"reason": "dir"}], "contradictions": []}', n_tokens=10)
    vr = asyncio.run(verify_report(report, llm=_MockLLM(responses=[critic])))
    assert "entrada de US$ 297,73" in report.overview_md


def test_verify_report_regenerates_flagged_paragraph():
    # Párrafo con cifra inventada (no está en facts) → se regenera y mejora.
    sec = _Section("flujos_ffmm", "El Tipo 1 tuvo entrada de US$ 888,88 Mill.",
                   facts=_flow_facts_t1_entrada())
    report = _Report([sec], overview_md="Resumen.")

    async def _regen(section, issues):
        return "El Tipo 1 tuvo entrada de US$ 297,73 Mill."  # cifra correcta

    vr = asyncio.run(verify_report(report, regenerate=_regen))
    assert "297,73" in sec.paragraph and "888" not in sec.paragraph
    assert vr.n_regenerated == 1


def test_strip_fact_tags_removed_from_prose():
    from banks_rag.application.reporting.parquet_report import _clean_paragraph, _strip_fact_tags

    raw = ("El Tipo 1 tuvo entrada de US$ 297,73 Mill. [ENTRADA: flujo positivo].\n\n"
           "El Tipo 3 con salida [SALIDA: flujo negativo] y Tipo 6 [MENOR ENTRADA: ...].")
    cleaned = _clean_paragraph(raw)
    assert "[ENTRADA" not in cleaned and "[SALIDA" not in cleaned and "MENOR ENTRADA]" not in cleaned
    assert "entrada de US$ 297,73 Mill." in cleaned
    # idempotente y sin doble espacio.
    assert "  " not in _strip_fact_tags("a [ENTRADA: x]  b")


def test_spec_unchanged_other_blocks_keep_text():
    # Garantía: ningún otro bloque ffmm perdió su comentario al introducir no_text.
    spec: FamilyReportSpec = FFMM_SPEC
    no_text = sum(1 for b in spec.blocks if b.no_text)
    assert no_text == 1  # solo flujos_acum_ffmm
    assert isinstance(spec.blocks[0], ReportBlock)
