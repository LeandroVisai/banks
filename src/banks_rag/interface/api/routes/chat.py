"""POST ``/v1/chat`` — endpoint del agente con tool calling."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from banks_rag.application.agent import (
    PROMPT_VERSION,
    run_agent,
)
from banks_rag.config import get_settings
from banks_rag.infrastructure.observability import log_chat_error, log_chat_turn
from banks_rag.infrastructure.uploads import load_upload
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
    # Se recorta a los últimos `history_max_turns` mensajes (memoria del chatbot)
    # como red de seguridad para no exceder N_CTX en conversaciones largas.
    history = [
        {"role": m.role, "content": m.content}
        for m in body.history
        if m.role != "system"
    ][-settings.history_max_turns:]

    model = getattr(deps.llm, "name", settings.llm_family)
    request_id = getattr(request.state, "request_id", "")

    # Resuelve los archivos adjuntos a un bloque de contexto efímero. Los que ya
    # caducaron o no existen se ignoran en silencio.
    attachments_context = ""
    if body.attachments:
        blocks = []
        for upload_id in body.attachments[:5]:
            rec = load_upload(upload_id)
            if rec:
                label = "Documento adjunto" if rec.get("kind") == "document" else "Datos adjuntos"
                blocks.append(f"--- {label}: {rec.get('name', '')} ---\n{rec.get('text', '')}")
        attachments_context = "\n\n".join(blocks)

    try:
        result = await run_agent(
            body.message,
            history=history,
            llm=deps.llm,
            max_iterations=settings.max_agent_iterations,
            max_tool_result_tokens=settings.max_tool_result_tokens,
            temperature=body.temperature if body.temperature is not None else settings.llm_temperature,
            max_tokens=body.max_tokens if body.max_tokens is not None else settings.llm_max_tokens,
            thinking_mode=body.thinking_mode or settings.thinking_mode,
            attachments_context=attachments_context,
            synthesis_max_tokens=settings.synthesis_max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("run_agent falló en /v1/chat")
        log_chat_error(
            body.message, history, str(exc),
            model=model, prompt_version=PROMPT_VERSION, request_id=request_id,
        )
        raise HTTPException(
            status_code=502,
            detail=f"El agente no pudo completar el turno: {exc}",
        ) from exc

    log_chat_turn(
        body.message, history, result,
        model=model, prompt_version=PROMPT_VERSION, request_id=request_id,
    )

    return ChatResponse(
        response=result.response,
        iterations=result.iterations,
        finish_reason=result.finish_reason,
        tool_trace=[ToolTraceEntry(**t) for t in result.tool_trace],
        chunks_seen=[ChunkSeen(**c) for c in result.chunks_seen],
        series_used=[HistoricalSeriesRef(**s) for s in result.series_used],
        cited_refs=result.cited_refs,
        ungrounded_numbers=result.ungrounded_numbers,
        total_tokens=result.total_tokens,
        latency_ms=result.latency_ms,
        model=model,
        prompt_version=PROMPT_VERSION,
    )
