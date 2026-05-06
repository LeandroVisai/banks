"""
FastAPI app + rutas para el chatbot agentic.

Endpoints (mismo set que chatbot/, mismas semánticas, pero el `/chat`
ejecuta el loop agentic en lugar de un RAG único).
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from . import agent, db, embeddings, llm, prompts
from .schemas import (
    ChatRequest, ChatResponse, ChunkRef, HealthResponse, HistoricalSeriesRef,
    HistoryMessage, SessionHistory, SessionSummary, ToolDescription, ToolTraceEntry,
)
from .settings import settings
from .tools import TOOL_REGISTRY, TOOL_SCHEMAS

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("Starting agentic chatbot — initializing infrastructure")
    await db.init_pool()
    await db.apply_schema_files()

    await asyncio.gather(
        embeddings.warm_up(),
        llm.load_engine(),
    )
    log.info(
        "Startup complete — listening on %s:%d (tools: %s)",
        settings.api_host, settings.api_port, list(TOOL_REGISTRY.keys()),
    )
    try:
        yield
    finally:
        log.info("Shutting down")
        await llm.unload_engine()
        await db.close_pool()


app = FastAPI(
    title="Chatbot Agentic — Banco Central",
    description=(
        "Chatbot con LLM local que decide qué tools llamar (search_documents, "
        "get_historical_series, etc.). Versión agentic para comparar con la "
        "infraestructura RAG clásica del repositorio (chatbot/)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    llm_loaded = llm.is_loaded()
    db_ok = await db.healthcheck()
    if llm_loaded and db_ok:
        status = "ok"
    elif not llm_loaded:
        status = "loading"
    else:
        status = "degraded"
    return HealthResponse(
        status=status,
        llm_loaded=llm_loaded,
        db_ok=db_ok,
        model=settings.chatbot_model_name,
        rag_table_prefix=settings.rag_table_prefix,
        available_tools=list(TOOL_REGISTRY.keys()),
    )


@app.get("/model-info")
async def model_info() -> dict:
    return llm.engine_info() | {
        "prompt_version": prompts.PROMPT_VERSION,
        "max_iterations": settings.max_agent_iterations,
    }


@app.get("/tools", response_model=list[ToolDescription])
async def list_tools() -> list[ToolDescription]:
    return [
        ToolDescription(
            name=s["function"]["name"],
            description=s["function"].get("description", ""),
            parameters=s["function"].get("parameters", {}),
        )
        for s in TOOL_SCHEMAS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Sesiones
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/sessions", response_model=list[SessionSummary])
async def sessions(limit: int = 20) -> list[SessionSummary]:
    rows = await db.list_sessions(limit=limit)
    return [
        SessionSummary(
            session_id=str(r["session_id"]),
            created_at=r["created_at"].isoformat(),
            updated_at=r["updated_at"].isoformat(),
            num_messages=int(r["num_messages"]),
        )
        for r in rows
    ]


@app.get("/chat/{session_id}", response_model=SessionHistory)
async def chat_history(session_id: str) -> SessionHistory:
    if not await db.session_exists(session_id):
        raise HTTPException(404, "Sesión no encontrada")
    messages = await db.get_full_history(session_id)
    return SessionHistory(
        session_id=session_id,
        messages=[
            HistoryMessage(
                role=m["role"],
                content=m["content"],
                tool_trace=m.get("tool_trace"),
                cited_chunks=m.get("cited_chunks"),
                historical_series=m.get("historical_series"),
                iterations=m.get("iterations"),
                token_count=m.get("token_count"),
                latency_ms=m.get("latency_ms"),
                created_at=m["created_at"].isoformat(),
            )
            for m in messages
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# /chat — corre el loop agentic
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not llm.is_loaded():
        raise HTTPException(503, "El modelo LLM aún no está cargado.")

    # 1. Sesión
    if req.session_id and await db.session_exists(req.session_id):
        session_id = req.session_id
    else:
        session_id = await db.create_session()

    # 2. Historial (si existe)
    history = await db.get_recent_history(session_id, max_turns=settings.history_max_turns)

    # 3. Persistir mensaje del usuario antes de generar
    await db.save_message(session_id, "user", req.message)

    # 4. Correr el agente
    try:
        result = await agent.run(
            req.message,
            history=history,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        log.exception("Agent run failed")
        raise HTTPException(500, f"Error en el agente: {e}")

    # 5. Persistir la respuesta + traza completa
    await db.save_message(
        session_id, "assistant", result.response,
        tool_trace=result.tool_trace if settings.tool_trace_enabled else None,
        cited_chunks=result.chunks_seen,
        historical_series=result.series_used,
        iterations=result.iterations,
        token_count=result.total_tokens,
        latency_ms=result.latency_ms,
    )

    log.info(
        "agent done session=%s iterations=%d tools=%d chunks=%d series=%d "
        "tokens=%d latency=%dms finish=%s",
        session_id, result.iterations, len(result.tool_trace),
        len(result.chunks_seen), len(result.series_used),
        result.total_tokens, result.latency_ms, result.finish_reason,
    )

    trace = result.tool_trace if req.include_trace else []

    return ChatResponse(
        session_id=session_id,
        response=result.response,
        iterations=result.iterations,
        finish_reason=result.finish_reason,
        tool_trace=[ToolTraceEntry(**t) for t in trace],
        chunks_seen=[ChunkRef(**c) for c in result.chunks_seen],
        series_used=[HistoricalSeriesRef(**s) for s in result.series_used],
        citations_used=result.cited_refs,
        model=settings.chatbot_model_name,
        prompt_version=prompts.PROMPT_VERSION,
        token_count=result.total_tokens,
        latency_ms=result.latency_ms,
    )
