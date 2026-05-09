# Arquitectura — banks_rag

> Documento vivo. Se actualiza con cada fase del plan en `~/.claude/plans/necesito-que-entiendas-a-curried-stearns.md`.

## 1 · Visión

```
                          ┌──────────────────────────┐
                          │   Cliente (curl/UI)      │
                          └────────────┬─────────────┘
                                       │ HTTPS
                          ┌────────────▼─────────────┐
                          │ INTERFACE  FastAPI + CLI │
                          │  routes / middleware     │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │ APPLICATION              │
                          │  ingestion · retrieval · │
                          │  agent · evaluation      │
                          └────┬──────────────┬──────┘
                               │              │
                          ┌────▼─────┐   ┌────▼──────┐
                          │  DOMAIN  │   │ INFRA     │
                          │ pure     │◀──│ adapters  │
                          │ entities │   │ implement │
                          │ rules    │   │ Protocols │
                          └──────────┘   └───────────┘
```

## 2 · Capas

### `domain/` — núcleo puro
Entidades, value objects, eventos, Protocols. **Nada de I/O ni librerías externas pesadas**.
- `documents/` — Document, Chunk (kind=TEXT|VISUAL|TABLE), VisualAsset, EnrichedChunk
- `retrieval/` — Query, RetrievalResult, Citation
- `series/` — SeriesDefinition, Observation
- `agent/` — ToolCall, AgentState, Conversation

### `application/` — casos de uso
Orquesta dominio + Protocols. Sin acoplarse a libs concretas.
- `ingestion/` — extract, enrich, vectorize, persist
- `retrieval/` — hybrid_search, reranker, query_router, query_decomposition
- `agent/` — conversation_loop, tools/*, prompts, citation_verifier
- `evaluation/` — ragas_runner, retrieval_metrics, golden_set_loader

### `infrastructure/` — adaptadores
Implementa los Protocols con librerías concretas.
- `persistence/` — postgres_repo (psycopg3), parquet_store, image_store
- `embeddings/` — multimodal_embedder (Qwen3-VL-Embedding-8B)
- `reranker/` — cross_encoder (BAAI/bge-reranker-v2-m3)
- `llm/` — llamacpp_engine, chat_templates (Qwen/Gemma), tool_call_parser
- `extractors/` — pdf, excel, chart_detector, encoding_fixers
- `chunking/` — text_chunker, monitor_pm_chunker
- `sql/` — catalog_loader, catalog_index, safe_executor, parameter_parser
- `observability/` — logging (structlog), metrics (Prometheus), tracing

### `interface/` — entry points
- `cli/` — Typer apps: ingest, search, chat, evaluate
- `api/` — FastAPI con rutas /v1/{chat,search,series,catalog,images}, middleware (request_id, auth, rate_limit)

### `domain_knowledge/` — reglas de negocio del dominio macro
Compartido por `application` y `infrastructure`:
- `taxonomy.py` (variables económicas, secciones canónicas, entidades)
- `importance_rules.py` (pesos del importance_score)
- `chunking_rules.py`, `boilerplate_filters.py`

## 3 · Stack

| Componente | Implementación |
|---|---|
| Embeddings | Qwen3-VL-Embedding-8B (4096-dim, texto + imagen) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| LLM | llama.cpp + Qwen3.6-27B-UD-Q4_K_XL (Gemma 4 26B-A4B-it como segundo backend) |
| Vector DB | PostgreSQL 14 + pgvector (HNSW cosine) |
| Series | Parquets en `data/snapshots/` (DuckDB para queries SQL sobre parquet) |
| API | FastAPI + uvicorn |
| Logging | structlog (JSON) + request_id |
| Métricas | prometheus-client |

## 4 · Reglas de dependencia

- `domain → ∅`
- `domain_knowledge → ∅`
- `application → domain, domain_knowledge`
- `infrastructure → domain, domain_knowledge`
- `interface → application, domain, domain_knowledge`
- **Prohibido**: `domain → infrastructure`, `application → infrastructure`

Verificación con `import-linter` (a configurar en Fase 6).

## 5 · Estado de implementación

| Fase | Estado |
|---|---|
| 0 · Preparación | 🚧 En progreso |
| 1 · Refactor estructural | ⏳ Pendiente |
| 2 · Multimodal embeddings + charts | ⏳ Pendiente |
| 3 · LLM swappable llama.cpp | ⏳ Pendiente |
| 4 · Catálogo SQL agentic | ⏳ Pendiente |
| 5 · Re-ranker + routing | ⏳ Pendiente |
| 6 · Evaluación + golden set | ⏳ Pendiente |
| 7 · Observability + hardening | ⏳ Pendiente |
| 8 · Limpieza + cutover | ⏳ Pendiente |
| 9 · Deploy H100 + validación | ⏳ Pendiente |
