"""Loop agentic — orquestador del flujo iterativo LLM ↔ tools.

Pipeline:

    1. El LLM ve system + tools + history + user_message.
    2. Decide: respuesta directa o tool calls.
    3. Si tool_calls: ejecutar en paralelo, inyectar resultados como
       ``role='tool'``, repetir desde 1.
    4. Si no tool_calls: respuesta final, salir del loop.
    5. Hard cap ``max_iterations``: evita loops infinitos.
    6. Verificación de citas: limpia refs inventadas antes de devolver.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from banks_rag.domain.agent import AgentResult, AgentState

from .citation_verifier import verify_citations
from .prompts import MAX_ITERATIONS_FALLBACK_MESSAGE, SYSTEM_PROMPT
from .tools.registry import TOOL_SCHEMAS, dispatch

log = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1500


def _serialize_tool_result(result: Any) -> str:
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(result)


def _summarize(result: Any) -> str:
    """Una línea corta para la traza persistente."""
    if not isinstance(result, dict):
        return str(result)[:120]
    if "error" in result:
        return f"ERROR: {result['error']}"
    if "n_results" in result:
        return f"{result['n_results']} fragmento(s)"
    if "n" in result:
        return f"{result['n']} observación(es)"
    if "n_chunks" in result:
        return f"{result['n_chunks']} chunk(s) del documento"
    if "n_series" in result:
        return f"{result['n_series']} serie(s) en catálogo"
    return "ok"


def _truncate_to_token_budget(text: str, budget: int, count_tokens) -> str:
    """Si el texto supera el budget, lo recorta proporcionalmente por chars."""
    n = count_tokens(text)
    if n <= budget:
        return text
    ratio = budget / max(n, 1)
    cutoff = int(len(text) * ratio * 0.95)
    return text[:cutoff] + "\n\n[...truncated...]"


def _format_chunks_seen(state: AgentState) -> list[dict]:
    """Snapshot de chunks_seen para el AgentResult final."""
    return [
        {
            "ref": state.chunk_id_to_ref[str(c["chunk_id"])],
            "filename": c.get("filename"),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "section": c.get("section_type"),
            "doc_type": c.get("doc_type_category"),
            "date": str(c.get("chunk_date") or c.get("document_date") or ""),
            "importance": round(float(c.get("importance_score") or 0.0), 3),
        }
        for c in state.chunks_seen
    ]


async def run_agent(
    user_message: str,
    history: list[dict],
    *,
    llm,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgentResult:
    """Ejecuta un turno completo del agente.

    Args:
        user_message: pregunta del usuario.
        history: historial previo de la conversación (lista de dicts ``role``+``content``).
        llm: motor LLM que cumple ``LLMEngine`` Protocol.
        max_iterations: hard cap de iteraciones LLM↔tools.
        max_tool_result_tokens: cap por resultado de tool antes de inyectarlo al contexto.
        temperature, max_tokens: overrides para el LLM.
        system_prompt: override del system prompt (default: el de prompts.py).
    """
    state = AgentState()
    t0 = time.perf_counter()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message},
    ]

    iteration = 0
    total_tokens = 0
    final_text = ""
    finish_reason = "stop"
    hit_max = False

    while iteration < max_iterations:
        iteration += 1
        log.info("agent iteration %d/%d", iteration, max_iterations)

        result = await llm.generate(
            messages,
            tools=TOOL_SCHEMAS,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_tokens += result.n_tokens
        finish_reason = result.finish_reason

        if not result.has_tool_calls:
            final_text = result.text
            log.info("agent done at iter %d (no more tool calls)", iteration)
            break

        log.info(
            "agent iter %d: %d tool call(s) %s",
            iteration, len(result.tool_calls), [tc.name for tc in result.tool_calls],
        )

        # Mensaje del assistant con los tool_calls.
        messages.append({
            "role": "assistant",
            "content": result.text or "",
            "tool_calls": [tc.to_message_block() for tc in result.tool_calls],
        })

        # Ejecutar tools en paralelo.
        async def _run_one(tc):
            tool_result, duration_ms = await dispatch(state, tc.name, tc.arguments)
            return tc, tool_result, duration_ms

        executions = await asyncio.gather(*(_run_one(tc) for tc in result.tool_calls))

        for tc, tool_result, duration_ms in executions:
            content_str = _serialize_tool_result(tool_result)
            content_str = _truncate_to_token_budget(
                content_str, max_tool_result_tokens, llm.count_text_tokens,
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": content_str,
            })

            state.add_tool_call_trace(
                iteration=iteration,
                tool=tc.name,
                arguments=tc.arguments,
                result_summary=_summarize(tool_result),
                result_size_chars=len(content_str),
                duration_ms=duration_ms,
            )
    else:
        hit_max = True

    if hit_max:
        log.warning("agent hit max_iterations=%d without finalizing", max_iterations)
        final_text = MAX_ITERATIONS_FALLBACK_MESSAGE
        finish_reason = "max_iterations"

    cleaned_response, cited_refs = verify_citations(final_text, state)

    return AgentResult(
        response=cleaned_response,
        iterations=iteration,
        tool_trace=list(state.tool_trace),
        chunks_seen=_format_chunks_seen(state),
        series_used=list(state.series_used.values()),
        cited_refs=cited_refs,
        finish_reason=finish_reason,
        total_tokens=total_tokens,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
