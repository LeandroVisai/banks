"""POST ``/v1/chat`` — endpoint del agente con tool calling."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from banks_rag.application.agent import (
    PROMPT_VERSION,
    run_agent,
)
from banks_rag.config import get_settings
from banks_rag.interface.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkSeen,
    HistoricalSeriesRef,
    ToolTraceEntry,
)

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Ejecuta un turno del agente y persiste la traza.

    El LLM debe estar cargado (``app.state.deps.llm``). Si no, retorna 503.
    """
    settings = get_settings()
    deps = getattr(request.app.state, "deps", None)
    if deps is None or deps.llm is None:
        raise HTTPException(status_code=503, detail="LLM no inicializado")
    if not getattr(deps.llm, "loaded", False):
        raise HTTPException(status_code=503, detail="LLM cargando — reintenta en breve")

    # Solo se aceptan turnos 'user'/'assistant' del cliente: un mensaje
    # 'system' en el history permitiría sobrescribir el system prompt.
    history = [
        {"role": m.role, "content": m.content}
        for m in body.history
        if m.role != "system"
    ]

    try:
        result = await run_agent(
            body.message,
            history=history,
            llm=deps.llm,
            max_iterations=settings.max_agent_iterations,
            max_tool_result_tokens=settings.max_tool_result_tokens,
            temperature=body.temperature if body.temperature is not None else settings.llm_temperature,
            max_tokens=body.max_tokens if body.max_tokens is not None else settings.llm_max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("run_agent falló en /v1/chat")
        raise HTTPException(
            status_code=502,
            detail=f"El agente no pudo completar el turno: {exc}",
        ) from exc

    return ChatResponse(
        response=result.response,
        iterations=result.iterations,
        finish_reason=result.finish_reason,
        tool_trace=[ToolTraceEntry(**t) for t in result.tool_trace],
        chunks_seen=[ChunkSeen(**c) for c in result.chunks_seen],
        series_used=[HistoricalSeriesRef(**s) for s in result.series_used],
        cited_refs=result.cited_refs,
        total_tokens=result.total_tokens,
        latency_ms=result.latency_ms,
        model=getattr(deps.llm, "name", settings.llm_family),
        prompt_version=PROMPT_VERSION,
    )
