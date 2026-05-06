"""
El loop agentic — corazón de esta infraestructura.

A diferencia del RAG clásico (chatbot/), aquí el flujo es:

  1. El LLM ve el sistema, las tools disponibles, el historial y la pregunta.
  2. El LLM decide qué tool llamar (o responde directamente).
  3. Si llamó tools:
       a. Las ejecutamos en paralelo
       b. Inyectamos los resultados como mensajes role='tool'
       c. Volvemos al paso 1 con el contexto extendido
  4. Si no llamó tools, esa es la respuesta final.
  5. Hard cap: MAX_AGENT_ITERATIONS para evitar loops infinitos.

`AgentState` mantiene la "memoria" del agente durante una sesión:
  - chunks vistos (con refs globales [1], [2], …)
  - series consultadas
  - traza completa de tool calls (para debugging/auditoría)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import llm, prompts
from .settings import settings
from .tools import TOOL_SCHEMAS, dispatch as dispatch_tool

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Estado del agente durante una sesión de /chat
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    chunks_seen: list[dict] = field(default_factory=list)      # acumulado [1], [2], ...
    chunk_id_to_ref: dict[str, int] = field(default_factory=dict)
    series_used: dict[str, dict] = field(default_factory=dict)
    tool_trace: list[dict] = field(default_factory=list)

    def add_chunk(self, chunk: dict) -> int:
        """Idempotente. Si el chunk ya tiene ref asignada, retorna esa misma."""
        cid = str(chunk.get("chunk_id"))
        if cid in self.chunk_id_to_ref:
            return self.chunk_id_to_ref[cid]
        self.chunks_seen.append(chunk)
        ref = len(self.chunks_seen)
        self.chunk_id_to_ref[cid] = ref
        return ref

    def add_series(self, series_id: str, meta: dict, rows: list[dict]) -> None:
        self.series_used[series_id] = {
            "series_id": series_id,
            "series_name": meta["series_name"],
            "unit": meta["unit"],
            "frequency": meta["frequency"],
            "n_observations": len(rows),
            "first_date": rows[0]["date"].isoformat() if rows else None,
            "last_date": rows[-1]["date"].isoformat() if rows else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Resultado del agente
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    response: str
    iterations: int
    tool_trace: list[dict]
    chunks_seen: list[dict]
    series_used: list[dict]
    cited_refs: list[int]
    finish_reason: str
    total_tokens: int
    latency_ms: int


# ─────────────────────────────────────────────────────────────────────────────
# El loop principal
# ─────────────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(\d+)\]")


async def run(
    user_message: str,
    history: list[dict],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> AgentResult:
    state = AgentState()
    t0 = time.perf_counter()

    messages: list[dict] = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    iteration = 0
    total_tokens = 0
    final_text = ""
    finish_reason = "stop"

    while iteration < settings.max_agent_iterations:
        iteration += 1
        log.info("agent iteration %d/%d", iteration, settings.max_agent_iterations)

        result = await llm.generate(
            messages,
            tools=TOOL_SCHEMAS,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_tokens += result.n_tokens
        finish_reason = result.finish_reason

        # ── Sin tool_calls → respuesta final ──
        if not result.has_tool_calls:
            final_text = result.text
            log.info("agent done at iter %d (no more tool calls)", iteration)
            break

        # ── Con tool_calls → ejecutarlos ──
        log.info("agent iter %d: %d tool call(s) %s",
                 iteration, len(result.tool_calls),
                 [tc.name for tc in result.tool_calls])

        # Insertar el mensaje del assistant con los tool_calls
        messages.append({
            "role": "assistant",
            "content": result.text or "",
            "tool_calls": [tc.to_message_block() for tc in result.tool_calls],
        })

        # Ejecutar tools en paralelo
        async def _run_one(tc):
            tool_result, duration_ms = await dispatch_tool(state, tc.name, tc.arguments)
            return tc, tool_result, duration_ms

        executions = await asyncio.gather(*(_run_one(tc) for tc in result.tool_calls))

        # Truncar resultados muy largos para no explotar el contexto
        for tc, tool_result, duration_ms in executions:
            content_str = _serialize_tool_result(tool_result)
            content_str = _truncate_to_token_budget(content_str, settings.max_tool_result_tokens)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": content_str,
            })

            state.tool_trace.append({
                "iteration": iteration,
                "tool": tc.name,
                "arguments": tc.arguments,
                "result_summary": _summarize(tool_result),
                "result_size_chars": len(content_str),
                "duration_ms": duration_ms,
            })

    else:
        # Salimos del while por max_iterations sin un final_text
        log.warning("agent hit max_iterations=%d without finalizing",
                    settings.max_agent_iterations)
        final_text = (
            "He llegado al límite de iteraciones sin lograr una respuesta concluyente. "
            "Los datos que pude reunir hasta ahora aparecen en la traza, pero no logré "
            "sintetizar una respuesta final. Por favor reformula la pregunta más específica."
        )
        finish_reason = "max_iterations"

    # Verificar citas y limpiar refs inventadas
    cleaned_response, cited_refs = _verify_citations(final_text, state)

    return AgentResult(
        response=cleaned_response,
        iterations=iteration,
        tool_trace=list(state.tool_trace),
        chunks_seen=[
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
        ],
        series_used=list(state.series_used.values()),
        cited_refs=cited_refs,
        finish_reason=finish_reason,
        total_tokens=total_tokens,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_tool_result(result: Any) -> str:
    import json as _json
    try:
        return _json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return str(result)


def _summarize(result: Any) -> str:
    """Una línea corta de qué devolvió la tool, para la traza persistente."""
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


def _truncate_to_token_budget(text: str, budget: int) -> str:
    """Si el texto supera `budget` tokens, lo recortamos por caracteres."""
    n = llm.count_text_tokens(text)
    if n <= budget:
        return text
    ratio = budget / max(n, 1)
    cutoff = int(len(text) * ratio * 0.95)
    return text[:cutoff] + "\n\n[...truncated...]"


def _verify_citations(response: str, state: AgentState) -> tuple[str, list[int]]:
    n_chunks = len(state.chunks_seen)
    used: list[int] = []

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_chunks:
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""

    cleaned = _CITATION_RE.sub(_replace, response)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip(), used
