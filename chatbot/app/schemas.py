"""Modelos Pydantic para los endpoints HTTP."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# /chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(
        None,
        description="ID de sesión existente. Si es None se crea una nueva.",
    )
    k: Optional[int] = Field(
        None, ge=1, le=20,
        description="Chunks RAG a incluir (default: settings.rag_top_k).",
    )
    use_rag: bool = True
    use_historical: bool = True
    stream: bool = False
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)


class SourceRef(BaseModel):
    ref: int
    filename: str
    doc_type: str
    section: str
    page_start: Optional[int]
    page_end: Optional[int]
    date: str
    importance: float
    text_snippet: str


class HistoricalSeriesRef(BaseModel):
    series_id: str
    series_name: str
    unit: str
    frequency: str
    n_observations: int
    first_date: str
    last_date: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[SourceRef]
    historical_series: list[HistoricalSeriesRef]
    citations_used: list[int]
    intent: str
    variables_detected: list[str]
    model: str
    prompt_version: str
    token_count: int
    latency_ms: int


# ─────────────────────────────────────────────────────────────────────────────
# /sessions y /chat/{session_id}
# ─────────────────────────────────────────────────────────────────────────────

class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    num_messages: int


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    rag_sources: Optional[list[dict]] = None
    historical_series: Optional[list[dict]] = None
    token_count: Optional[int] = None
    latency_ms: Optional[int] = None
    created_at: str


class SessionHistory(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


# ─────────────────────────────────────────────────────────────────────────────
# Health & catalogs
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "loading"]
    llm_loaded: bool
    db_ok: bool
    model: str
    rag_table_prefix: str


class HistoricalSeriesCatalogEntry(BaseModel):
    series_id: str
    series_name: str
    unit: str
    frequency: str
    source: str
    economic_variable: Optional[str]
    num_observations: int
    first_date: Optional[str]
    last_date: Optional[str]
