"""Tests de run_report (modo informe) con LLM mock."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from banks_rag.application.agent import run_report
from banks_rag.application.agent.subagents import SUBAGENTS
from banks_rag.domain.agent import GenerationResult


@dataclass
class _ReportLLM:
    """Mock: análisis sin cifras para los subagentes (tools presentes); informe
    estructurado para la síntesis (tools=None)."""

    n_subagent_calls: int = 0
    n_synth_calls: int = 0
    seen_system: list[str] = field(default_factory=list)

    async def generate(self, messages, *, tools=None, **kwargs):
        if tools is None:
            self.n_synth_calls += 1
            self.seen_system.append(messages[0]["content"])
            return GenerationResult(
                text="# Informe de mercado\n## Resumen ejecutivo\n- Estabilidad general.",
                n_tokens=30,
            )
        self.n_subagent_calls += 1
        # Sin cifras → no dispara el guard anti-alucinación.
        return GenerationResult(text="El segmento mostró estabilidad relativa.", n_tokens=10)

    def count_text_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_tokens(self, messages, tools=None) -> int:
        return 100


@pytest.mark.unit
class TestRunReport:
    @pytest.mark.asyncio
    async def test_full_scope_runs_all_specialists(self) -> None:
        llm = _ReportLLM()
        res = await run_report("condiciones financieras de mayo 2026", llm=llm, scope="full")
        # 8 especialistas + 1 síntesis de informe.
        assert llm.n_subagent_calls == len(SUBAGENTS)
        assert llm.n_synth_calls == 1
        assert "Informe de mercado" in res.response
        assert res.iterations >= 1
        assert isinstance(res.charts, list)

    @pytest.mark.asyncio
    async def test_synthesis_uses_report_prompt(self) -> None:
        llm = _ReportLLM()
        await run_report("spread BTP", llm=llm, scope="full")
        # El system prompt de la síntesis es el de informe (economista jefe).
        assert any("economista jefe" in s for s in llm.seen_system)

    @pytest.mark.asyncio
    async def test_auto_scope_routes_to_subset(self) -> None:
        llm = _ReportLLM()
        await run_report("evolución del spread BTP vs SPC a 10 años", llm=llm, scope="auto")
        # Ruteo determinista: menos especialistas que el roster completo.
        assert 1 <= llm.n_subagent_calls < len(SUBAGENTS)
        assert llm.n_synth_calls == 1
