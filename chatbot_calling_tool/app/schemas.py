"""Modelos Pydantic — específicos para el chatbot agentic."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# /chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    include_trace: bool = Field(
        True,
        description="Si False, no devuelve la traza completa de tools en la respuesta (sigue persistiéndose).",
    )


class ToolTraceEntry(BaseModel):
    iteration: int
    tool: str
    arguments: dict[str, Any]
    result_summary: str
    result_size_chars: int
    duration_ms: int


class ChunkRef(BaseModel):
    ref: int
    filename: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    section: Optional[str]
    doc_type: Optional[str]
    date: str
    importance: float


class HistoricalSeriesRef(BaseModel):
    series_id: str
    series_name: str
    unit: str
    frequency: str
    n_observations: int
    first_date: Optional[str]
    last_date: Optional[str]


class ChatResponse(BaseModel):
    session_id: str
    response: str
    iterations: int
    finish_reason: str
    tool_trace: list[ToolTraceEntry]
    chunks_seen: list[ChunkRef]
    series_used: list[HistoricalSeriesRef]
    citations_used: list[int]
    model: str
    prompt_version: str
    token_count: int
    latency_ms: int


# ─────────────────────────────────────────────────────────────────────────────
# Sesiones
# ─────────────────────────────────────────────────────────────────────────────

class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    num_messages: int


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    tool_trace: Optional[list[dict]] = None
    cited_chunks: Optional[list[dict]] = None
    historical_series: Optional[list[dict]] = None
    iterations: Optional[int] = None
    token_count: Optional[int] = None
    latency_ms: Optional[int] = None
    created_at: str


class SessionHistory(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


# ─────────────────────────────────────────────────────────────────────────────
# Health & catálogos
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "loading"]
    llm_loaded: bool
    db_ok: bool
    model: str
    rag_table_prefix: str
    available_tools: list[str]


class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
