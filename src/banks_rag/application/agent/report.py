"""``run_report`` — informe de mercado multi-sección (analista senior).

Reutiliza la maquinaria de router-v1 (``run_subagent`` sobre un ``AgentState``
compartido) pero, en vez de una respuesta breve, produce un INFORME estructurado
para la gerencia con ``REPORT_PROMPT``: resumen ejecutivo, secciones por dominio,
riesgos y gráficos referenciados.

Diferencias con ``run_agent``:
  - a los especialistas se les pide explícitamente traer datos y GRAFICAR las
    series clave (``plot_series``), de modo que el informe lleve gráficos;
  - ``scope='full'`` corre el roster completo (informe integral), ``'auto'``
    rutea por la consulta;
  - la síntesis usa ``REPORT_PROMPT`` con un presupuesto de tokens mayor.

Pensado para Qwen3.6-27B en H100, capaz de redactar informes largos coherentes.
"""

from __future__ import annotations

import asyncio
import logging
import time

from banks_rag.domain.agent import AgentResult, AgentState

from .citation_verifier import verify_citations
from .conversation_loop import (
    DEFAULT_SUBAGENT_MAX_ITERATIONS,
    DEFAULT_TOOL_TIMEOUT_S,
    SubAgentResult,
    _format_chunks_seen,
    _with_today,
    run_subagent,
)
from .numeric_grounding import verify_numbers
from .prompts import MAX_ITERATIONS_FALLBACK_MESSAGE, REPORT_PROMPT
from .router import select_specialists
from .subagents import SUBAGENTS, SubAgentSpec

log = logging.getLogger(__name__)

# Más presupuesto que una respuesta de chat: un informe es largo y estructurado.
DEFAULT_REPORT_MAX_TOKENS = 4096
# Instrucción que se añade a la tarea de cada especialista en modo informe.
_REPORT_TASK_HINT = (
    " Trae los datos relevantes y, cuando aporte al informe, GRAFICA con "
    "plot_series las series clave. Entrega un análisis senior (nivel, variación, "
    "lectura), no datos crudos."
)


def _report_specs(topic: str, history: list[dict] | None, scope: str) -> list[SubAgentSpec]:
    """Selecciona los especialistas del informe según ``scope``."""
    if scope == "full":
        return list(SUBAGENTS.values())
    specs = select_specialists(topic, history or [])
    if not specs:  # saludo/ambiguo: para un informe, cae al corpus general.
        specs = [SUBAGENTS["document"], SUBAGENTS["policy"]]
    return specs


def _build_report_user_message(
    topic: str, subs: list[SubAgentResult], charts: list[dict],
) -> str:
    parts = [f"Tema del informe:\n{topic}", "", "Aportes de tus especialistas:"]
    for sub in subs:
        parts.append(f"\n## {sub.display_name}\n{sub.analysis.strip()}")
    if charts:
        parts.append("\nGráficos disponibles (refiérelos como 'gráfico N'):")
        for c in charts:
            parts.append(
                f"- Gráfico {c['id']}: {c.get('title', '')} "
                f"({c.get('dataset_id', '')}, {c.get('chart_type', '')})"
            )
    parts.append(
        "\nRedacta ahora el informe completo para la gerencia siguiendo tu "
        "estructura (resumen ejecutivo, secciones por dominio con aporte, "
        "riesgos). Conserva las citas [N] y refiere los gráficos."
    )
    return "\n".join(parts)


async def run_report(
    topic: str,
    *,
    llm,
    history: list[dict] | None = None,
    scope: str = "auto",
    max_iterations: int = DEFAULT_SUBAGENT_MAX_ITERATIONS,
    max_tool_result_tokens: int = 1500,
    temperature: float | None = None,
    max_tokens: int | None = DEFAULT_REPORT_MAX_TOKENS,
) -> AgentResult:
    """Genera un informe de mercado multi-sección.

    Args:
        topic: tema del informe (p.ej. "condiciones financieras de mayo 2026").
        scope: ``'auto'`` rutea por el tema; ``'full'`` corre los 8 especialistas.
        max_tokens: presupuesto de la síntesis del informe (default 4096).
    """
    state = AgentState()
    t0 = time.perf_counter()

    specs = _report_specs(topic, history, scope)
    log.info("run_report scope=%s → especialistas: %s", scope, [s.key for s in specs])

    sub_max = min(max_iterations, DEFAULT_SUBAGENT_MAX_ITERATIONS)

    async def _run(spec: SubAgentSpec) -> SubAgentResult:
        task = (
            f"Para un informe de mercado del BCCh sobre: {topic}\n\n"
            f"Aporta tu análisis como {spec.display_name}." + _REPORT_TASK_HINT
        )
        return await run_subagent(
            spec, task, llm=llm, state=state,
            max_iterations=sub_max, max_tool_result_tokens=max_tool_result_tokens,
            tool_timeout_s=DEFAULT_TOOL_TIMEOUT_S, temperature=temperature,
            max_tokens=max_tokens,
        )

    subs: list[SubAgentResult] = list(await asyncio.gather(*(_run(s) for s in specs)))
    total_tokens = sum(s.total_tokens for s in subs)

    synth_messages = [
        {"role": "system", "content": _with_today(REPORT_PROMPT)},
        {"role": "user", "content": _build_report_user_message(topic, subs, state.charts)},
    ]
    synth = await llm.generate(
        synth_messages, tools=None, temperature=temperature, max_tokens=max_tokens,
    )
    total_tokens += synth.n_tokens
    report_text = synth.text or MAX_ITERATIONS_FALLBACK_MESSAGE

    cleaned, cited_refs, invalid_refs = verify_citations(report_text, state)
    ungrounded, _ = verify_numbers(cleaned, state.grounded_numbers)
    if invalid_refs:
        log.warning("informe citó refs inválidas %s", invalid_refs)

    return AgentResult(
        response=cleaned,
        iterations=max((s.iterations for s in subs), default=0) + 1,
        tool_trace=list(state.tool_trace),
        chunks_seen=_format_chunks_seen(state),
        series_used=list(state.series_used.values()),
        charts=list(state.charts),
        cited_refs=cited_refs,
        finish_reason=synth.finish_reason,
        total_tokens=total_tokens,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        invalid_refs=invalid_refs,
        ungrounded_numbers=ungrounded,
    )
