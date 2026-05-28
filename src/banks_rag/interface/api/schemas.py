"""Modelos Pydantic v2 para request/response del API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "degraded"] = "ok"
    version: str
    llm_loaded: bool
    db_ok: bool
    model: str
    rag_table_prefix: str


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class ChunkSeen(BaseModel):
    ref: int
    filename: str | None
    page_start: int | None
    page_end: int | None
    section: str | None
    doc_type: str | None
    date: str
    importance: float


class HistoricalSeriesRef(BaseModel):
    series_id: str
    series_name: str
    unit: str
    frequency: str = ""   # parquet_catalog no declara frequency a nivel dataset
    n_observations: int
    first_date: str | None
    last_date: str | None


class ToolTraceEntry(BaseModel):
    iteration: int
    tool: str
    arguments: dict
    result_summary: str
    result_size_chars: int
    duration_ms: int
    # Agente que emitió la tool call: "orquestador" o la key de un especialista.
    agent: str = ""


class ChatResponse(BaseModel):
    response: str
    iterations: int
    finish_reason: str
    tool_trace: list[ToolTraceEntry]
    chunks_seen: list[ChunkSeen]
    series_used: list[HistoricalSeriesRef]
    cited_refs: list[int]
    total_tokens: int
    latency_ms: int
    model: str
    prompt_version: str


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────


class SearchFiltersIn(BaseModel):
    """Subset de filtros que el cliente puede pasar explícitamente."""

    model_config = ConfigDict(extra="forbid")

    institutions: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    months: list[int] = Field(default_factory=list)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    max_importance: float = Field(default=1.0, ge=0.0, le=1.0)
    min_section_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    exclude_boilerplate: bool = False
    exclude_doc_types: list[str] = Field(default_factory=list)
    exclude_institutions: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    use_mmr: bool = True
    filters: SearchFiltersIn = Field(default_factory=SearchFiltersIn)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    doc_type_category: str
    document_date: str | None
    page_start: int
    page_end: int
    section_type: str
    importance_score: float
    rrf_score: float
    final_score: float
    economic_variables: dict
    tags: list[str]
    image_path: str | None
    text: str


class SearchResponse(BaseModel):
    query: str
    clean_query: str
    filters: dict
    n_results: int
    results: list[SearchHit]


# ─────────────────────────────────────────────────────────────────────────────
# Catalog (parquet_catalog para alimentar el frontend de gráficos)
# ─────────────────────────────────────────────────────────────────────────────


class DatasetColumn(BaseModel):
    name: str
    type: str
    values: list[str] = Field(default_factory=list)


class DatasetEntry(BaseModel):
    id: str
    file: str
    name: str
    description: str
    segment: str
    unit: str
    date_range: list[str] | None = None
    columns: list[DatasetColumn]


class DatasetListResponse(BaseModel):
    n_entries: int
    entries: list[DatasetEntry]


class QueryResponse(BaseModel):
    dataset_id: str
    name: str
    unit: str
    segment: str
    date_column: str
    columns: list[str]
    rows: list[dict]
    n_rows: int
    last_date_in_data: str | None = None
    truncated: bool
    truncated_note: str | None = None
