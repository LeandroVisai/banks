# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Sistema RAG + agente multimodal para el Banco Central de Chile. Procesa PDFs financieros (Comunicados BCCh, Minutas del Consejo, Fed Statements, research JPMorgan) y un Excel de Monitor PM en un corpus semántico consultable con citación por página. El agente puede además consultar 23 series financieras del catálogo SQL vía DuckDB offline.

Stack: `sentence-transformers` (Qwen3-VL-Embedding-8B, 4096-dim) + `llama-cpp-python` + PostgreSQL 16 + pgvector (HNSW) + búsqueda híbrida BM25/vector con RRF, MMR y cross-encoder reranking + FastAPI.

**Arquitectura**: Clean Architecture + DDD. Los scripts numerados legacy (`00-05_*.py`, `run.py`, `chatbot/`, `chatbot_calling_tool/`) fueron eliminados. Todo el código productivo vive en `src/banks_rag/`.

## Comandos esenciales

```bash
# Ingesta (extracción → enriquecimiento → embeddings → BD)
banks-ingest run --source Datos_prueba/

# Búsqueda
banks-search "tasa de interés 2022" --k 5
banks-search "commodities riesgos" --k 10 --json

# API
uvicorn banks_rag.interface.api.main:create_app --factory --port 8080

# Evaluación
make eval          # golden set completo → eval_report.md
make eval-ci       # gate CI recall@5

# Tests
PYTHONPATH=src pytest tests/unit/ -q     # 498 tests, <2s
make test                                # suite completa
make lint                                # ruff
```

## Arquitectura — flujo de datos

```
Datos_prueba/Comunicados/*.pdf
Datos_prueba/Minutas/*.pdf
Datos_prueba/Fed/*.pdf
Datos_prueba/Researchs/**/*.pdf
Datos_prueba/Monitor PM/textos_monitor_pm.xlsx
  → banks-ingest run
  → [extract]    chunks.json + documents.json
  → [enrich]     chunks_enriched.json  (taxonomía semántica)
  → [vectorize]  chunks_vectorized.json  (dual-embed: txt+img, Qwen3-VL)
  → [load]       PostgreSQL rag_banco  (HNSW + GIN)
  → hybrid_search  (HNSW + BM25 → RRF → MMR → reranker)
  → agente tool-calling  (Qwen3.6 / Gemma 4 vía llama.cpp)
  → FastAPI  /v1/chat · /v1/search · /metrics
```

## Módulos y responsabilidades

| Paquete | Responsabilidad |
|---|---|
| `domain/` | Dataclasses puras (`Document`, `Chunk`, `EnrichedChunk`, `SearchResult`, `ParsedQuery`, `SearchFilters`) |
| `application/ingestion/` | Extract → enrich → vectorize → load |
| `application/retrieval/` | `hybrid_search`, `query_parser`, `query_router`, `fusion`, `filters` |
| `application/agent/` | Loop de tool-calling, registro de tools, `run_agent` |
| `application/evaluation/` | `retrieval_metrics`, `ragas_runner`, golden set |
| `infrastructure/embeddings/` | `SentenceTransformersEmbedder`, `build_default_embedder` |
| `infrastructure/llm/` | `LlamaCppEngine` (Protocol `LLMEngine`) |
| `infrastructure/reranker/` | `CrossEncoderReranker` (Protocol `Reranker`) |
| `infrastructure/persistence/` | `PostgresRepo`, `recall_queries`, `catalog_loader` |
| `infrastructure/observability/` | logging (structlog), metrics (Prometheus), tracing (OTel) |
| `interface/api/` | FastAPI app, middlewares, routes |
| `interface/cli/` | `banks-ingest`, `banks-search`, `banks-eval` |

## Invariantes críticos

- **`src/banks_rag/domain/` es la fuente de verdad de estructuras**: no duplicar dataclasses en otros módulos.

- **`taxonomy.py` en `infrastructure/extractors/`**: los patrones de variables económicas, secciones, entidades y boilerplate viven ahí. Los pasos de ingesta lo importan en tiempo de ejecución.

- **`doc_type` se hereda del filepath, nunca del contenido del chunk**: `detect_doc_type()` opera sobre la ruta relativa.

- **Embeddings normalizados L2 + `vector_cosine_ops`**: los embeddings se normalizan en `vectorize_corpus()`. El índice HNSW usa `vector_cosine_ops`. Cambiar uno sin el otro rompe la similitud.

- **Dual embedding para chunks visuales**: `combined = IMAGE_WEIGHT * img_emb + (1 - IMAGE_WEIGHT) * txt_emb`, normalizado L2. `IMAGE_WEIGHT` default 0.7, override con `RAG_VISUAL_IMG_WEIGHT`.

- **Modelos en `models/<owner>--<name>/`**: convención offline H100. Si el directorio existe se usa; si no, se descarga desde HuggingFace. Nunca hardcodear rutas absolutas.

- **LlamaCppEngine usa lazy import**: `from llama_cpp import Llama` solo en `_sync_load()`. Permite tests sin el binario instalado.

- **CrossEncoderReranker usa lazy import**: `from sentence_transformers import CrossEncoder` solo en `load()`.

- **Catálogo de datasets en `sql_catalog/parquet_catalog.yaml`**: 107 datasets sobre parquets en `data_pipeline/parquet/`, con esquema completo (columnas, tipos, valores de enum, `date_range`). El LLM elige `dataset_id` + columnas/filtros vía `discover_query`/`execute_query`/analytics; **la SQL la arma siempre la tool** (`_parquet_query.build_fetch_sql`) — el LLM NUNCA escribe SQL.

- **`data_pipeline/` ya NO extrae de SQL**: la extracción en vivo del DW (`dw_store`, `extract.py`), los snapshots y `series_catalog.yaml` fueron eliminados. Los parquets en `data_pipeline/parquet/` son la única fuente; se regeneran fuera del repo y se copian. No quedan tools que consulten el SQL Server (la antigua `historical_series` se eliminó).

- **Logging de turnos del agente**: cada turno de `/v1/chat` se persiste como una línea JSON en `data/chat_logs/chat-YYYY-MM-DD.jsonl` vía `infrastructure/observability/chat_log.py` (best-effort, nunca tumba el request). Guarda pregunta, respuesta y evidencia (tool_trace, chunks_seen, series_used, citas) para contrastar respuestas reales vs. esperadas. Control: `BANKS_CHAT_LOG_ENABLED` / `BANKS_CHAT_LOG_DIR`.

- **Tests unitarios sin BD ni modelos**: todos los tests en `tests/unit/` usan mocks. `PYTHONPATH=src pytest tests/unit/ -q` debe pasar en < 2s sin internet ni GPU.

- **Golden set en `data/golden_set/`**: `retrieval.jsonl` (30 casos), `sql_routing.jsonl` (30 casos), `generation.jsonl` (15 casos). Curado para el dominio BCCh.

- **Gate CI**: `make eval-ci` falla (exit 1) si `recall@5` cae > 5% vs baseline en `eval_baseline.json`.

## Agregar nuevos tipos de documento

1. `infrastructure/extractors/pdf_extractor.py` → `detect_doc_type()`: añadir rama `if` con la carpeta nueva.
2. `infrastructure/extractors/taxonomy.py` → `SECTION_KEYWORDS`: añadir patrones del nuevo tipo.
3. Ejecutar `banks-ingest run` (idempotente).

## Agregar nuevos datasets al catálogo de parquets

1. Dejar el parquet en `data_pipeline/parquet/<nombre>.parquet` con sus columnas reales (no es necesario un schema canónico).
2. Añadir entrada en `sql_catalog/parquet_catalog.yaml` bajo `datasets:` con `id`, `file`, `name`, `description`, `segment`, `unit`, `date_range` y `columns` (cada columna con `name` + `type`; si es categórica, opcionalmente `values:` para enum).
3. Añadir caso a `data/golden_set/sql_routing.jsonl` validando que `discover_query` + el nuevo `dataset_id` respondan la pregunta.

## Ajustar parámetros de búsqueda

| Parámetro | Módulo | Qué controla |
|---|---|---|
| `RECALL_N` | `hybrid_search.py` | Candidatos por rama antes de RRF |
| `RRF_K` | `fusion.py` | Hiperparámetro RRF (60 = estándar) |
| `MMR_LAMBDA` | `fusion.py` | 1.0 = solo relevancia, 0.0 = solo diversidad |
| `IMPORTANCE_BOOST` | `fusion.py` | Peso de importance en el re-rank |
| `IMAGE_WEIGHT` | `vectorize_corpus.py` | Peso imagen en dual embedding (default 0.7) |

## Schema PostgreSQL (referencia rápida)

```sql
documents (document_id PK, doc_type_category, institution,
           document_date TEXT, document_year INT, extraction_warnings JSONB)

chunks    (chunk_id PK, document_id FK,
           text, text_tsv TSVECTOR,              -- BM25
           embedding VECTOR(4096),               -- HNSW cosine
           section_type, section_confidence,
           economic_variables JSONB,             -- GIN
           entities JSONB,                       -- GIN
           importance_score, is_policy_decision, is_forward_looking,
           chunk_date DATE,                      -- solo Monitor PM
           tags TEXT[],                          -- GIN
           kind TEXT,                            -- 'TEXT' | 'VISUAL'
           visual_caption TEXT,
           image_path TEXT)
```

## Variables de entorno

| Variable | Default | Cuándo cambiar |
|---|---|---|
| `PGDATABASE` | `rag_banco` | Usar otra BD |
| `PGUSER` / `PGPASSWORD` | SO / vacío | Servidor con auth |
| `BANKS_LLM_FAMILY` | `mock` | `qwen` o `gemma` en producción |
| `BANKS_LLM_MODEL_PATH` | `` | Ruta al `.gguf` |
| `BANKS_LLM_MAX_TOKENS` | `2048` | Tokens máx por paso (especialistas/iteración) |
| `BANKS_SYNTHESIS_MAX_TOKENS` | `4096` | Tokens máx de la respuesta final (síntesis); subir si se trunca |
| `BANKS_THINKING_MODE` | `adaptive` | `off`/`adaptive`/`on`: thinking de Qwen3. `adaptive` = razona solo en especialistas cuantitativos (multi-paso); no en documentales ni síntesis |
| `RAG_EMBEDDING_MODEL` | `Qwen3-VL-Embedding-8B` | Modelo de embeddings (resuelve a `models/<name>/`) |
| `BANKS_CATALOG_SEMANTIC` | `false` | `true` para activar descubrimiento semántico del catálogo (requiere el embedder cargado) |
| `BANKS_API_KEYS` | `` | CSV de API keys (vacío = sin auth) |
| `BANKS_LOG_JSON` | `false` | `true` en producción |
| `BANKS_TRACING` | `off` | `otlp` para OpenTelemetry |
| `BANKS_RATE_LIMIT_RPM` | `60` | Requests/minuto por API key |
| `RAG_VISUAL_IMG_WEIGHT` | `0.7` | Peso imagen en dual embedding |
| `BANKS_RERANK_ENABLED` | `true` | `false` para apagar el cross-encoder reranker en `search_documents` |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Modelo del reranker (resuelve a `models/<owner>--<name>/`) |
| `BANKS_CHAT_LOG_ENABLED` | `true` | `false` para no persistir turnos del agente |
| `BANKS_CHAT_LOG_DIR` | `data/chat_logs` | Otra ruta para los JSONL de chat |

## Gemelo de desarrollo

Este repo tiene un gemelo en `/Users/leandrovenegas/Desktop/Proyecto_rag/` (sandbox Mac). Mantener paridad entre ambos. `taxonomy.py` en `infrastructure/extractors/` es la fuente de verdad para patrones compartidos.
