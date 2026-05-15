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
DEFAULT_TOOL_TIMEOUT_S = 30.0


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


_TEXT_KEYS = ("text", "content", "chunk_text", "snippet", "visual_caption")
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


async def run_agent(
    user_message: str,
    history: list[dict],
    *,
    llm,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
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
        tool_timeout_s: tiempo límite por tool; al excederlo se devuelve un error
            estructurado y el turno continúa (evita que una tool colgada bloquee
            el worker indefinidamente).
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

    while iteration < max_iterations:
        iteration += 1
        # En la última iteración no se ofrecen tools: se fuerza una respuesta
        # final con la evidencia ya reunida, en vez de ejecutar tool calls
        # cuyos resultados ya no podrían sintetizarse.
        is_last = iteration == max_iterations
        tools_arg = None if is_last else TOOL_SCHEMAS

        # Observabilidad de contexto: tamaño aproximado del prompt por iteración.
        prompt_tokens = (
            llm.count_tokens(messages, tools_arg)
            if hasattr(llm, "count_tokens") else 0
        )
        log.info(
            "agent iteration %d/%d — prompt ≈%d tokens",
            iteration, max_iterations, prompt_tokens,
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
            log.info("agent done at iter %d (no more tool calls)", iteration)
            break

        if is_last:
            # El LLM aún pidió tools pero ya no quedan iteraciones: se usa su
            # texto (o el fallback) en vez de descartar el turno.
            log.warning(
                "agent agotó %d iteraciones; se fuerza la respuesta final",
                max_iterations,
            )
            final_text = result.text or MAX_ITERATIONS_FALLBACK_MESSAGE
            finish_reason = "max_iterations"
            break

        log.info(
            "agent iter %d: %d tool call(s) %s",
            iteration, len(result.tool_calls), [tc.name for tc in result.tool_calls],
        )

        # Deduplica tool calls idénticas de la misma iteración: el LLM a veces
        # emite la misma búsqueda varias veces; ejecutarlas todas solo amplifica
        # la carga (N conexiones a Postgres, N cargas del embedder).
        unique_calls = []
        seen_calls: set[tuple] = set()
        for tc in result.tool_calls:
            key = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
            if key not in seen_calls:
                seen_calls.add(key)
                unique_calls.append(tc)

        # Mensaje del assistant con los tool_calls.
        messages.append({
            "role": "assistant",
            "content": result.text or "",
            "tool_calls": [tc.to_message_block() for tc in unique_calls],
        })

        # Ejecutar tools en paralelo, cada una con timeout propio.
        # _run_one nunca lanza: captura timeout y excepciones devolviéndolas
        # como dict de error, de modo que una tool defectuosa no aborta el
        # gather ni descarta los resultados de las demás.
        async def _run_one(tc):
            t0_tool = time.perf_counter()
            try:
                tool_result, duration_ms = await asyncio.wait_for(
                    dispatch(state, tc.name, tc.arguments),
                    timeout=tool_timeout_s,
                )
                return tc, tool_result, duration_ms
            except asyncio.TimeoutError:
                duration_ms = int((time.perf_counter() - t0_tool) * 1000)
                log.warning("tool %s excedió el timeout de %.0fs", tc.name, tool_timeout_s)
                return tc, {
                    "error": f"La herramienta '{tc.name}' excedió el tiempo límite "
                             f"({tool_timeout_s:.0f}s) y fue cancelada.",
                }, duration_ms
            except Exception as exc:  # noqa: BLE001
                duration_ms = int((time.perf_counter() - t0_tool) * 1000)
                log.exception("tool %s lanzó una excepción no controlada", tc.name)
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
            )

    # La última iteración siempre produce final_text (respuesta directa o
    # forzada sin tools), así que el loop nunca termina sin respuesta.
    cleaned_response, cited_refs, invalid_refs = verify_citations(final_text, state)
    if invalid_refs:
        log.warning(
            "agente citó %d ref(s) inválida(s) %s — posible alucinación",
            len(invalid_refs), invalid_refs,
        )

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
        invalid_refs=invalid_refs,
    )
