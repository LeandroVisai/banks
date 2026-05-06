"""
FastAPI app + rutas.

Endpoints:
  GET   /healthz                  liveness — el proceso responde
  GET   /readyz                   readiness — LLM cargado y DB accesible
  GET   /model-info               metadata del modelo cargado
  GET   /historical-series        catálogo de series con cobertura
  GET   /sessions                 lista de sesiones recientes
  GET   /chat/{session_id}        historial de una sesión
  POST  /chat                     envía mensaje, recibe respuesta o SSE stream
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import db, embeddings, llm, prompts, query_analysis, retrieval, sql_context
from data_pipeline import dw_store
from .schemas import (
    ChatRequest, ChatResponse, HealthResponse, HistoricalSeriesCatalogEntry,
    HistoryMessage, SessionHistory, SessionSummary, SourceRef, HistoricalSeriesRef,
)
from .settings import settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("Starting up — initializing infrastructure")
    await db.init_pool()
    await db.apply_schema_files()

    if db.is_enabled():
        await asyncio.gather(embeddings.warm_up(), llm.load_engine())
    else:
        # Sin DB (CHATBOT_SKIP_DB=1): solo LLM, sin embeddings para RAG
        await llm.load_engine()

    db_enabled = db.is_enabled()
    log.info("Startup complete — listening on %s:%d", settings.api_host, settings.api_port)

    try:
        yield
    finally:
        log.info("Shutting down")
        await llm.unload_engine()
        await db.close_pool()


app = FastAPI(
    title="Chatbot RAG — Banco Central",
    description=(
        "Chatbot financiero con LLM local (vLLM en proceso) + retrieval híbrido "
        "sobre PostgreSQL/pgvector + series de tiempo macro directas desde SQL."
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
    )


@app.get("/model-info")
async def model_info() -> dict:
    return llm.engine_info() | {"prompt_version": prompts.PROMPT_VERSION}


# ─────────────────────────────────────────────────────────────────────────────
# Catálogos
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/historical-series", response_model=list[HistoricalSeriesCatalogEntry])
async def historical_series() -> list[HistoricalSeriesCatalogEntry]:
    """Catálogo de series — usa el YAML del catálogo DW (siempre disponible)."""
    rows = dw_store.list_series(catalog_path=settings.catalog_path)
    return [
        HistoricalSeriesCatalogEntry(
            series_id=r["id"],
            series_name=r["name"],
            unit=r.get("unit", ""),
            frequency=r.get("frequency", ""),
            source=r.get("source", ""),
            economic_variable=r.get("variable"),
            num_observations=None,
            first_date=None,
            last_date=None,
        )
        for r in rows
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
                rag_sources=m.get("rag_sources"),
                historical_series=m.get("historical_series"),
                token_count=m.get("token_count"),
                latency_ms=m.get("latency_ms"),
                created_at=m["created_at"].isoformat(),
            )
            for m in messages
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# /chat — el flujo principal
# ─────────────────────────────────────────────────────────────────────────────

async def _build_pipeline(req: ChatRequest):
    """
    Pipeline reutilizable entre el modo streaming y el no-streaming:
      1. Sesión
      2. Análisis de query (intención, variables, fechas)
      3. RAG retrieval (si aplica)
      4. Contexto SQL histórico (si aplica)
      5. Historial de conversación
      6. Construcción de mensajes con presupuesto de tokens

    Retorna (session_id, messages, sources, used_series, analysis).
    """
    if not llm.is_loaded():
        raise HTTPException(503, "El modelo LLM aún no está cargado.")

    # 1. Sesión
    if req.session_id and await db.session_exists(req.session_id):
        session_id = req.session_id
    else:
        session_id = await db.create_session()

    # 2. Análisis de la query
    analysis = query_analysis.analyze(req.message)

    # 3. RAG (en paralelo con el contexto histórico)
    chunks_task = (
        asyncio.create_task(retrieval.retrieve(req.message, k=req.k or settings.rag_top_k))
        if req.use_rag and analysis.needs_rag and db.is_enabled()
        else None
    )

    # 4. Datos históricos (siempre disponible si GET_DATA_PATH está configurado)
    hist_task = (
        asyncio.create_task(sql_context.build_context(analysis))
        if req.use_historical
        else None
    )

    # 5. Historial de conversación
    history_task = asyncio.create_task(
        db.get_recent_history(session_id, max_turns=settings.history_max_turns)
    )

    chunks = await chunks_task if chunks_task else []
    hist_block, used_series = await hist_task if hist_task else ("", [])
    history = await history_task

    sources = retrieval.format_sources(chunks)

    # 6. Construir mensajes (con truncamiento por presupuesto)
    messages = prompts.build_messages(
        user_message=req.message,
        rag_chunks=chunks,
        historical_context=hist_block,
        history=history,
    )

    log.info(
        "chat session=%s intent=%s vars=%s n_chunks=%d hist_series=%d",
        session_id, analysis.intent, analysis.variables, len(chunks), len(used_series),
    )

    return session_id, messages, sources, used_series, analysis


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id, messages, sources, used_series, analysis = await _build_pipeline(req)

    # Persistimos el mensaje del usuario antes de generar (así si la generación
    # falla, queda registrado el input).
    await db.save_message(session_id, "user", req.message)

    # ── Streaming: SSE ──
    if req.stream:
        return StreamingResponse(
            _stream_chat(req, session_id, messages, sources, used_series, analysis),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── No-streaming: una sola respuesta ──
    t0 = time.perf_counter()
    try:
        raw_response, n_tokens = await llm.generate(
            messages,
            temperature=req.temperature,
            top_p=None,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        log.exception("LLM generate failed")
        raise HTTPException(500, f"Error en el LLM: {e}")
    latency_ms = int((time.perf_counter() - t0) * 1000)

    response, used_refs = prompts.verify_citations(raw_response, n_sources=len(sources))

    await db.save_message(
        session_id, "assistant", response,
        rag_sources=sources,
        historical_series=used_series,
        token_count=n_tokens,
        latency_ms=latency_ms,
    )

    return ChatResponse(
        session_id=session_id,
        response=response,
        sources=[SourceRef(**s) for s in sources],
        historical_series=[HistoricalSeriesRef(**s) for s in used_series],
        citations_used=used_refs,
        intent=analysis.intent,
        variables_detected=analysis.variables,
        model=settings.chatbot_model_name,
        prompt_version=prompts.PROMPT_VERSION,
        token_count=n_tokens,
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SSE streaming helper
# ─────────────────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict | str) -> str:
    payload = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else json.dumps({"text": data}, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_chat(
    req: ChatRequest,
    session_id: str,
    messages: list[dict],
    sources: list[dict],
    used_series: list[dict],
    analysis,
) -> AsyncIterator[str]:
    # Evento inicial: metadata (cliente puede pintar fuentes ya)
    yield _sse("meta", {
        "session_id": session_id,
        "intent": analysis.intent,
        "variables_detected": analysis.variables,
        "sources": sources,
        "historical_series": used_series,
        "model": settings.chatbot_model_name,
        "prompt_version": prompts.PROMPT_VERSION,
    })

    t0 = time.perf_counter()
    full_chunks: list[str] = []
    try:
        async for delta in llm.generate_stream(
            messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        ):
            full_chunks.append(delta)
            yield _sse("delta", {"text": delta})
    except Exception as e:
        log.exception("LLM stream failed")
        yield _sse("error", {"message": str(e)})
        return

    latency_ms = int((time.perf_counter() - t0) * 1000)
    raw_response = "".join(full_chunks).strip()
    response, used_refs = prompts.verify_citations(raw_response, n_sources=len(sources))

    # Persistir
    try:
        await db.save_message(
            session_id, "assistant", response,
            rag_sources=sources,
            historical_series=used_series,
            token_count=None,  # vLLM stream no expone el conteo final fácilmente
            latency_ms=latency_ms,
        )
    except Exception:
        log.exception("save_message failed during stream")

    # Evento final
    yield _sse("done", {
        "citations_used": used_refs,
        "latency_ms": latency_ms,
        "response_length": len(response),
    })
