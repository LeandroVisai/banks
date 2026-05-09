"""POST ``/v1/chat`` — endpoint del agente con tool calling."""

from __future__ import annotations

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

    history = [{"role": m.role, "content": m.content} for m in body.history]

    result = await run_agent(
        body.message,
        history=history,
        llm=deps.llm,
        max_iterations=settings.max_agent_iterations,
        max_tool_result_tokens=settings.max_tool_result_tokens,
        temperature=body.temperature if body.temperature is not None else settings.llm_temperature,
        max_tokens=body.max_tokens if body.max_tokens is not None else settings.llm_max_tokens,
    )

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
