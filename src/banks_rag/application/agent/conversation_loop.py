"""Loop agentic multi-agente — orquestador ↔ sub-agentes ↔ tools.

Arquitectura (Fase B):

    run_agent  =  ORQUESTADOR
      · ve solo las tools delegate_to_* (una por especialista)
      · descompone la pregunta y delega
      · sintetiza la respuesta final con citas [N]
        │
        ├─ delegate_to_document_analyst ─┐
        ├─ delegate_to_quant_analyst ────┤
        ├─ delegate_to_policy_analyst ───┤  run_subagent  =  ESPECIALISTA
        └─ delegate_to_market_analyst ───┘    · ve solo sus tools de dominio
                                              · ejecuta su propio loop LLM↔tools
                                              · devuelve un análisis al orquestador

Tanto el orquestador como cada sub-agente corren el mismo loop genérico
``_run_tool_loop``; lo que cambia es el system prompt, el set de tools y la
función de dispatch. Todos comparten un único ``AgentState`` para que las
citas ``[N]`` sean globalmente consistentes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from banks_rag.domain.agent import AgentResult, AgentState

from .citation_verifier import verify_citations
from .prompts import MAX_ITERATIONS_FALLBACK_MESSAGE, ORCHESTRATOR_SYSTEM_PROMPT
from .subagents import (
    DELEGATE_SCHEMAS,
    SUBAGENTS,
    SubAgentSpec,
    subagent_for_delegate,
    tool_schemas_for,
)
from .tools.registry import dispatch

log = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_SUBAGENT_MAX_ITERATIONS = 4
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1500
# Las "tool results" del orquestador son análisis completos de sus especialistas:
# merecen más presupuesto que un resultado de tool crudo.
DEFAULT_MAX_DELEGATE_RESULT_TOKENS = 4000
DEFAULT_TOOL_TIMEOUT_S = 30.0
# Una delegación dispara un loop entero del sub-agente (varias llamadas al LLM):
# su techo de tiempo es mucho mayor que el de una tool de dominio.
DEFAULT_DELEGATE_TIMEOUT_S = 240.0

# Firma de un dispatcher de tools: (state, name, arguments) → (resultado, duración_ms).
DispatchFn = Callable[[AgentState, str, dict], Awaitable[tuple[dict, int]]]


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
    if "analyst" in result and "analysis" in result:  # resultado de delegación
        return f"{result['analyst']} — {result.get('iterations', '?')} iteración(es)"
    if "n_results" in result:
        return f"{result['n_results']} fragmento(s)"
    if "n" in result:
        return f"{result['n']} observación(es)"
    if "n_chunks" in result:
        return f"{result['n_chunks']} chunk(s) del documento"
    if "n_series" in result:
        return f"{result['n_series']} serie(s) en catálogo"
    if "n_indicadores" in result:
        return f"{result['n_indicadores']} indicador(es) de mercado"
    if "n_decisions" in result:
        return f"{result['n_decisions']} decisión(es) de política"
    if "meeting_a" in result and "meeting_b" in result:
        return "comparación de 2 reuniones"
    return "ok"


_TEXT_KEYS = ("text", "content", "chunk_text", "snippet", "visual_caption", "analysis")
_TRUNCABLE_LIST_HINT = ("results", "rows", "chunks", "series", "documents", "items", "hits")


def _clip_str(value: Any, max_chars: int) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…"
    return value


def _shrink_texts(obj: Any, text_max: int) -> Any:
    """Recorta recursivamente los campos de texto largos de una estructura."""
    if isinstance(obj, dict):
        return {
            k: (_clip_str(v, text_max) if k in _TEXT_KEYS else _shrink_texts(v, text_max))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_shrink_texts(x, text_max) for x in obj]
    return obj


def _longest_list_key(d: dict) -> str | None:
    """Clave de la lista más larga del dict — candidata a recortar."""
    best, best_len = None, 1
    for k, v in d.items():
        if isinstance(v, list) and len(v) > best_len:
            best, best_len = k, len(v)
    if best is not None:
        return best
    # Si ninguna lista supera 1 elemento, prioriza una clave-hint si existe.
    for k in _TRUNCABLE_LIST_HINT:
        if isinstance(d.get(k), list):
            return k
    return None


def _truncate_tool_result(result: Any, budget: int, count_tokens) -> str:
    """Serializa el resultado de una tool recortándolo al budget de tokens.

    Trunca a nivel de estructura — acorta campos de texto largos y reduce el
    número de elementos de las listas, luego re-serializa — de modo que el
    string entregado al LLM **siempre es JSON válido** (a diferencia de un
    corte ciego del string, que partiría el JSON a la mitad).
    """
    serialized = _serialize_tool_result(result)
    if count_tokens(serialized) <= budget:
        return serialized

    if not isinstance(result, dict):
        # No estructurado: corte por chars como último recurso.
        ratio = budget / max(count_tokens(serialized), 1)
        cutoff = int(len(serialized) * ratio * 0.9)
        return serialized[:cutoff] + " […truncado…]"

    work = dict(result)
    list_key = _longest_list_key(work)
    text_max = 800
    for _ in range(8):
        work = _shrink_texts(work, text_max)
        if list_key and isinstance(work.get(list_key), list) and len(work[list_key]) > 1:
            keep = max(1, len(work[list_key]) // 2)
            work[list_key] = work[list_key][:keep]
        work["truncated"] = True
        candidate = _serialize_tool_result(work)
        if count_tokens(candidate) <= budget:
            return candidate
        text_max = max(120, text_max // 2)

    # Último recurso: conserva solo escalares + nota; serializa garantizado.
    return _serialize_tool_result({
        "truncated": True,
        "truncated_note": "Resultado demasiado grande; se omitió el detalle.",
        **{k: v for k, v in result.items() if not isinstance(v, (list, dict, str))},
    })


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


# ─────────────────────────────────────────────────────────────────────────────
# Loop genérico LLM ↔ tools
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _LoopOutcome:
    """Resultado de una corrida de ``_run_tool_loop``."""

    final_text: str
    finish_reason: str
    total_tokens: int
    iterations: int


async def _run_tool_loop(
    messages: list[dict],
    *,
    llm,
    tool_schemas: list[dict],
    dispatch_fn: DispatchFn,
    state: AgentState,
    agent_label: str,
    max_iterations: int,
    max_tool_result_tokens: int,
    tool_timeout_s: float,
    temperature: float | None,
    max_tokens: int | None,
) -> _LoopOutcome:
    """Ejecuta el loop iterativo LLM↔tools hasta una respuesta final.

    Genérico: lo usan tanto el orquestador (con tools ``delegate_to_*`` y un
    dispatcher de delegación) como cada sub-agente (con sus tools de dominio y
    el dispatcher del registry). ``state`` se comparte entre todos.

    Args:
        messages: conversación inicial (system + user, más history si aplica).
        llm: motor que cumple el Protocol ``LLMEngine``.
        tool_schemas: tools que se le ofrecen al LLM en cada iteración.
        dispatch_fn: ejecuta una tool por nombre y retorna ``(resultado, ms)``.
        state: estado compartido — acumula chunks, series y la traza.
        agent_label: etiqueta del agente para logs y para la traza.
        max_iterations: hard cap de iteraciones.
        max_tool_result_tokens: cap por resultado de tool antes de inyectarlo.
        tool_timeout_s: tiempo límite por tool.
        temperature, max_tokens: overrides del LLM.
    """
    iteration = 0
    total_tokens = 0
    final_text = ""
    finish_reason = "stop"

    while iteration < max_iterations:
        iteration += 1
        # En la última iteración no se ofrecen tools: se fuerza una respuesta
        # final con la evidencia ya reunida.
        is_last = iteration == max_iterations
        tools_arg = None if is_last else tool_schemas

        prompt_tokens = (
            llm.count_tokens(messages, tools_arg)
            if hasattr(llm, "count_tokens") else 0
        )
        log.info(
            "[%s] iteración %d/%d — prompt ≈%d tokens",
            agent_label, iteration, max_iterations, prompt_tokens,
        )

        result = await llm.generate(
            messages,
            tools=tools_arg,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_tokens += result.n_tokens
        finish_reason = result.finish_reason

        if not result.has_tool_calls:
            final_text = result.text
            log.info("[%s] termina en iter %d (sin más tool calls)", agent_label, iteration)
            break

        if is_last:
            # El LLM aún pidió tools pero ya no quedan iteraciones: se usa su
            # texto (o el fallback) en vez de descartar el turno.
            log.warning(
                "[%s] agotó %d iteraciones; se fuerza la respuesta final",
                agent_label, max_iterations,
            )
            final_text = result.text or MAX_ITERATIONS_FALLBACK_MESSAGE
            finish_reason = "max_iterations"
            break

        log.info(
            "[%s] iter %d: %d tool call(s) %s",
            agent_label, iteration, len(result.tool_calls),
            [tc.name for tc in result.tool_calls],
        )

        # Deduplica tool calls idénticas de la misma iteración: el LLM a veces
        # emite la misma llamada varias veces; ejecutarlas todas solo amplifica
        # la carga sin aportar información nueva.
        unique_calls = []
        seen_calls: set[tuple] = set()
        for tc in result.tool_calls:
            key = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
            if key not in seen_calls:
                seen_calls.add(key)
                unique_calls.append(tc)

        messages.append({
            "role": "assistant",
            "content": result.text or "",
            "tool_calls": [tc.to_message_block() for tc in unique_calls],
        })

        # Ejecutar tools en paralelo, cada una con timeout propio. _run_one
        # nunca lanza: captura timeout y excepciones devolviéndolas como dict
        # de error, de modo que una tool defectuosa no aborta el gather.
        async def _run_one(tc):
            t0_tool = time.perf_counter()
            try:
                tool_result, duration_ms = await asyncio.wait_for(
                    dispatch_fn(state, tc.name, tc.arguments),
                    timeout=tool_timeout_s,
                )
                return tc, tool_result, duration_ms
            except TimeoutError:
                duration_ms = int((time.perf_counter() - t0_tool) * 1000)
                log.warning(
                    "[%s] tool %s excedió el timeout de %.0fs",
                    agent_label, tc.name, tool_timeout_s,
                )
                return tc, {
                    "error": f"La herramienta '{tc.name}' excedió el tiempo "
                             f"límite ({tool_timeout_s:.0f}s) y fue cancelada.",
                }, duration_ms
            except Exception as exc:  # noqa: BLE001
                duration_ms = int((time.perf_counter() - t0_tool) * 1000)
                log.exception("[%s] tool %s lanzó una excepción", agent_label, tc.name)
                return tc, {"error": f"Error inesperado en '{tc.name}': {exc}"}, duration_ms

        executions = await asyncio.gather(*(_run_one(tc) for tc in unique_calls))

        for tc, tool_result, duration_ms in executions:
            content_str = _truncate_tool_result(
                tool_result, max_tool_result_tokens, llm.count_text_tokens,
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
                agent=agent_label,
            )

    return _LoopOutcome(
        final_text=final_text,
        finish_reason=finish_reason,
        total_tokens=total_tokens,
        iterations=iteration,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-agente especialista
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SubAgentResult:
    """Resultado de la corrida de un sub-agente especialista."""

    key: str
    display_name: str
    analysis: str
    iterations: int
    finish_reason: str
    total_tokens: int


async def run_subagent(
    spec: SubAgentSpec,
    task: str,
    *,
    llm,
    state: AgentState,
    max_iterations: int = DEFAULT_SUBAGENT_MAX_ITERATIONS,
    max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> SubAgentResult:
    """Ejecuta un especialista sobre una tarea concreta delegada por el orquestador.

    El sub-agente solo ve sus tools de dominio (``spec.tool_names``) y no recibe
    el historial de la conversación: el ``task`` debe ser autocontenido. El
    ``state`` se comparte con el orquestador para que las citas sean globales.
    """
    messages: list[dict] = [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": task},
    ]
    outcome = await _run_tool_loop(
        messages,
        llm=llm,
        tool_schemas=tool_schemas_for(spec),
        dispatch_fn=dispatch,
        state=state,
        agent_label=spec.key,
        max_iterations=max_iterations,
        max_tool_result_tokens=max_tool_result_tokens,
        tool_timeout_s=tool_timeout_s,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return SubAgentResult(
        key=spec.key,
        display_name=spec.display_name,
        analysis=outcome.final_text,
        iterations=outcome.iterations,
        finish_reason=outcome.finish_reason,
        total_tokens=outcome.total_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador
# ─────────────────────────────────────────────────────────────────────────────


async def run_agent(
    user_message: str,
    history: list[dict],
    *,
    llm,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
    tool_timeout_s: float = DEFAULT_DELEGATE_TIMEOUT_S,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str = ORCHESTRATOR_SYSTEM_PROMPT,
) -> AgentResult:
    """Ejecuta un turno completo del agente orquestador.

    El orquestador descompone la pregunta, delega en los especialistas vía las
    tools ``delegate_to_*`` y sintetiza la respuesta final. Los sub-agentes
    corren dentro del dispatch de cada delegación, compartiendo el ``AgentState``.

    Args:
        user_message: pregunta del usuario.
        history: turnos previos de la conversación.
        llm: motor LLM (Protocol ``LLMEngine``).
        max_iterations: hard cap de iteraciones del orquestador.
        max_tool_result_tokens: cap de los resultados de tool **de los
            sub-agentes** (sus datos crudos). El análisis que cada sub-agente
            devuelve al orquestador usa un cap mayor.
        tool_timeout_s: tiempo límite por delegación (un sub-agente entero).
        temperature, max_tokens: overrides del LLM.
        system_prompt: override del prompt del orquestador.
    """
    state = AgentState()
    t0 = time.perf_counter()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message},
    ]

    async def _dispatch_delegate(
        st: AgentState, name: str, arguments: dict,
    ) -> tuple[dict, int]:
        """Dispatcher del orquestador: una tool ``delegate_to_*`` → un sub-agente."""
        spec = subagent_for_delegate(name)
        if spec is None:
            return {
                "error": f"Especialista no disponible: {name!r}.",
                "available": [s.delegate_tool for s in SUBAGENTS.values()],
            }, 0

        task = (arguments or {}).get("task", "")
        if not isinstance(task, str) or not task.strip():
            return {
                "error": (
                    "El argumento 'task' es obligatorio: describe la consulta "
                    "concreta y autocontenida para el especialista."
                ),
            }, 0

        t0_delegate = time.perf_counter()
        sub = await run_subagent(
            spec,
            task.strip(),
            llm=llm,
            state=st,
            max_tool_result_tokens=max_tool_result_tokens,
            tool_timeout_s=DEFAULT_TOOL_TIMEOUT_S,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        duration_ms = int((time.perf_counter() - t0_delegate) * 1000)
        return {
            "analyst": sub.display_name,
            "analysis": sub.analysis,
            "iterations": sub.iterations,
            "finish_reason": sub.finish_reason,
        }, duration_ms

    outcome = await _run_tool_loop(
        messages,
        llm=llm,
        tool_schemas=DELEGATE_SCHEMAS,
        dispatch_fn=_dispatch_delegate,
        state=state,
        agent_label="orquestador",
        max_iterations=max_iterations,
        max_tool_result_tokens=DEFAULT_MAX_DELEGATE_RESULT_TOKENS,
        tool_timeout_s=tool_timeout_s,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    cleaned_response, cited_refs, invalid_refs = verify_citations(outcome.final_text, state)
    if invalid_refs:
        log.warning(
            "el orquestador citó %d ref(s) inválida(s) %s — posible alucinación",
            len(invalid_refs), invalid_refs,
        )

    return AgentResult(
        response=cleaned_response,
        iterations=outcome.iterations,
        tool_trace=list(state.tool_trace),
        chunks_seen=_format_chunks_seen(state),
        series_used=list(state.series_used.values()),
        cited_refs=cited_refs,
        finish_reason=outcome.finish_reason,
        total_tokens=outcome.total_tokens,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        invalid_refs=invalid_refs,
    )
