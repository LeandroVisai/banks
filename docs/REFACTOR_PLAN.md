# Plan: Reingeniería Profesional — Sistema RAG + Agente Multimodal para Banco Central

> Un plan de transformación de un prototipo investigativo a una arquitectura de producción para uno de los bancos centrales más importantes de Latinoamérica. Construido siguiendo Clean Architecture + DDD, con embeddings multimodales que preservan gráficos, agente con tool-calling autónomo sobre catálogo SQL curado, y modelos LLM intercambiables (Qwen3.6 / Gemma 4) bajo restricciones offline.

> **Branch de trabajo**: `branch/qwenllamacpp` (todo el refactor; `main` permanece intacto hasta Fase 8 cutover).

---

## 0 · Progreso de la implementación (live)

> Sección que se actualiza después de cada commit. El resto del documento es el plan original aprobado.

### Sub-fases completadas

| Sub-fase | Commit | Tests | Verificación de paridad |
|---|---|---|---|
| 0+1A · Skeleton + extract pipeline | `b09443b` | 60 | 10 documentos / 13 357 chunks idénticos al legacy `00_generate_jsons.py` |
| 1B · Enrich pipeline | `43d932c` | +32 (92) | 13 357 enriched chunks idénticos campo-por-campo vs `01_enrich_metadata.py` |
| 1C · Vectorize pipeline | `ee17374` | +22 (114) | `build_embed_text` 0 diferencias / 13 357; `build_cross_references` JSON idéntico |
| 1D · Persist pipeline | `42b7e77` | +41 (155) | schema + 17 índices + DML idénticos a `03_database.py` para dim 384 y 4096 con prefijos `''` y `qwen_` |
| 1E · Hybrid search unificada | `e868d89` | +55 (210) | `parse_query` 8/8 reales idénticas; RRF + MMR + importance_boost: orden y scores idénticos |
| 1F · Agent core (loop + registry + LLM Protocol) | `340cb17` | +27 (237) | Por construcción + 27 tests unitarios con mock LLM |
| 1F.b · Tools concretas (search/lookup/series) | `99733cf` | +12 (249) | 5 tools registradas con schemas alineados; mocks de infra + tests |
| 1G · FastAPI unificado | `b89a2b4` | +16 (265) | TestClient con health/chat/search/auth; reemplaza `chatbot/` + `chatbot_calling_tool/` |
| 2A · VisualAsset domain + chart_detector | `7ca9de5` | +19 (284) | Captions detectadas via regex, surrounding text, BoundingBox |
| 2B+2C · Schema kind/visual_caption + search_visuals + /v1/images | `a00ccd0` | +12 (296) | 6 tools, 9 routes; `kinds` filter en hybrid_search |
| 2D · Wire VisualAsset al extract pipeline | `46dbc9a` | 296 | 32 chunks VISUAL extraídos en datos reales (7 con caption) |

**Total Fase 1+2 parcial: 11 commits, 296/296 tests passing en 0.62s; 32 chunks visuales generados con `kind=VISUAL` + `visual_caption` end-to-end.**

### Sub-fases pendientes

| Fase | Estado | Notas |
|---|---|---|
| 2.b · Multimodal embedder + bbox cropping | ⏳ Pendiente | Qwen3-VL-Embedding-8B en H100 (modelo ~16GB en `models/`) |
| 3 · LLM swappable llama.cpp (Qwen3.6 → Gemma) | ⏳ Pendiente | `LlamaCppEngine` implementando el Protocol existente |
| 4 · SQL catalog agentic | ⏳ Pendiente | 75 queries de `Monitor.py` → `discover_query` + `execute_query` |
| 5 · Re-ranker + query routing | ⏳ Pendiente | bge-reranker-v2-m3 + RAG/SQL/VISUAL classifier |
| 6 · Evaluación reproducible | ⏳ Pendiente | golden set + RAGAS + recall@k + CI gate |
| 7 · Observability + hardening | ⏳ Pendiente | structlog + Prometheus + tracing + rate limit |
| 8 · Limpieza + cutover | ⏳ Pendiente | Eliminar `00–05_*.py`, `chatbot/`, `chatbot_calling_tool/`, `run.py` |
| 9 · Deploy H100 + validación | ⏳ Pendiente | Wheels offline + systemd + 5 queries de validación |

### Arquitectura entregada hasta hoy

```
src/banks_rag/
├── domain/             documents (Document, Chunk+ChunkKind, EnrichedChunk)
│                       retrieval (ParsedQuery, SearchFilters, SearchResult)
│                       agent     (ToolCall, AgentState, AgentResult, GenerationResult)
├── domain_knowledge/   taxonomy + enrichment + importance_rules + cross_references
├── application/        ingestion (extract, enrich, vectorize, persist)
│                       retrieval (parse, filters, fusion, hybrid_search)
│                       agent     (loop, prompts, citation_verifier, 6 tools)
├── infrastructure/     extractors (pdf, excel, chart, encoding, normalizer, doc_metadata)
│                       chunking, embeddings (Protocol + sentence_transformers + builder)
│                       persistence (sql_templates, postgres_repo)
│                       sql (recall_queries), llm (Protocol + tool_call_parser)
├── interface/          cli (banks-ingest, banks-search)
│                       api (FastAPI: /healthz, /readyz, /v1/chat, /v1/search, /v1/images/{id})
└── config/             paths, settings (Pydantic)
```

**CLIs:**
```bash
banks-ingest extract|enrich|vectorize|persist|full
banks-search "tasa interés 2022" 5 --institution BANCO_CENTRAL_CHILE --tags DECISION_POLITICA
python -m banks_rag.interface.api.main         # FastAPI servicio unificado
```

---

## 1 · Contexto y Motivación

**Por qué este cambio.** El repositorio hoy es un prototipo funcional pero estructurado como pipeline de notebooks (scripts numerados `00→05`) con dos chatbots paralelos y duplicados. Para mover esto a producción en un banco central se requiere:

1. **Arquitectura limpia y testeable**: separación dominio / aplicación / infraestructura, dependencias unidireccionales, y un único punto de entrada de servicio.
2. **Multimodalidad real**: actualmente los gráficos de los PDFs se renderizan a PNG pero no se vectorizan ni se devuelven al agente. Hay que cerrar ese ciclo con `Qwen3-VL-Embedding-8B`.
3. **Modelos swappable**: el deploy oficial usará GGUF + llama.cpp (no vLLM como dice el README actual). Hay que soportar `Qwen3.6-27B-UD-Q4_K_XL` y `gemma-4-26B-A4B-it` con la misma interfaz.
4. **Agente SQL autónomo con catálogo curado**: `querys/Monitor.py` contiene **75 queries SQL hand-crafted** para series macro/financieras del DW. El agente debe **discernir cuál usar** desde lenguaje natural, parametrizar, ejecutar y citar — no inventar SQL libre (Text-to-SQL guiado por catálogo).
5. **Eliminar duplicación**: `chatbot/` y `chatbot_calling_tool/` comparten ~70% del código. Unificar en un único servicio modular.
6. **Evaluación reproducible**: golden set + RAGAS + métricas de retrieval (recall@k, MRR) — sin esto, "mejorar el RAG" es opinión, no ingeniería.
7. **Observability + governance**: logging estructurado, tracing por `request_id`, métricas Prometheus, documento de model cards, data lineage.

**Restricciones que no cambian.**
- GPU H100 80GB single-node, sin internet en el server de producción.
- Modelos pre-descargados (vía `enlaces.md`).
- PostgreSQL 14 + pgvector como única BD persistente.
- Python 3.12, CUDA 12.8.
- Datos: PDFs (Comunicados, Minutas, Fed, JPMorgan, Researchs) + Excel Monitor PM + DW SQL Server (a través de snapshots parquet).

---

## 2 · Diagnóstico del Estado Actual

| Área | Estado actual | Problema |
|---|---|---|
| **Estructura** | Scripts numerados en raíz + 2 chatbots paralelos | Sin paquete instalable, sin tests, sin separación de capas |
| **Pipeline** | `00→05` orquestados por `run.py` con `subprocess` | Acoplado al filesystem, difícil de testear unitariamente |
| **Embeddings** | `Qwen3-Embedding-8B` (solo texto, 4096-dim) | Charts de PDFs se extraen a PNG pero **no** se embeben — info perdida |
| **LLM** | vLLM + Qwen3.6-35B-A3B FP8 (asumido) | `enlaces.md` realmente pide GGUF + llama.cpp + Qwen3.6-27B y Gemma 4 26B |
| **Chatbots** | `chatbot/` (RAG clásico) + `chatbot_calling_tool/` (agentic) duplicando settings, llm.py, db.py, embeddings.py, schemas.py | Dos servicios paralelos divergiendo, mantenimiento doble |
| **SQL agente** | `historical_series` tool consulta `dw_store.fetch_series` con catálogo de **17** series | Hay **75 queries** crudas en `Monitor.py` ignoradas — capacidad subutilizada |
| **Series catalog** | `data_pipeline/series_catalog.yaml` con ~80 series, `extract.py` mode `dw`/`mock` | Bien diseñado pero desconectado del agente — el agente solo ve un subset |
| **Búsqueda** | `04_search.py` (~760 líneas) + `04_advanced_search.py` (~580 líneas) duplicadas | Dos implementaciones de hybrid search; sin re-ranker cross-encoder |
| **Evaluación** | Inexistente — solo un check manual ("top-1 debe ser comunicado1.pdf") | No hay forma objetiva de medir si una mejora ayuda o daña |
| **Observability** | Logs por consola, sin tracing ni métricas | Imposible debuggear latencias o regresiones en producción |
| **Tests** | Inexistentes | Refactor = roulette rusa |
| **Docs** | `README.md` desactualizado (menciona vLLM, no Gemma, no multimodal) | Onboarding difícil; el plan debe actualizarlo |

**Inventario rápido:**
- 75 queries `gd.get_data(...)` en `querys/Monitor.py` (2152 líneas)
- 14 parquets ya generados en `data_pipeline/snapshots/`
- `models_cache/` solo tiene `intfloat--multilingual-e5-small` (legacy)
- Pipeline ingesta toca: 4 carpetas en `Datos_prueba/` (Comunicados, Minutas, Fed, Researchs) + `Monitor PM/textos_monitor_pm.xlsx`

---

## 3 · Skill Recomendada (`/find-skills`)

Resultado de `npx skills find rag` y `find agent multimodal`:

| Skill | Installs | Por qué |
|---|---|---|
| **`wshobson/agents@rag-implementation`** | 7.8K | Skill más popular para implementar RAG de extremo a extremo. Cubre chunking, embeddings, retrieval, evaluation. **Recomendada como base.** |
| **`langchain-ai/langchain-skills@langchain-rag`** | 6.2K | Patrones canónicos de LangChain (query routing, decomposition, multi-vector retrieval). Útil aunque no usemos LangChain directamente — los patrones son aplicables. |
| **`jeffallan/claude-skills@rag-architect`** | 2K | Específico para diseño arquitectónico de RAG. Útil para el refactor. |
| **`oimiragieo/agent-studio@text-to-sql`** | 108 | Patterns para Text-to-SQL — relevante para el agente SQL con catálogo. |
| **`latestaiagents/agent-skills@agentic-rag`** | 41 | Patterns específicos para agentes con tool-calling sobre RAG. |

**Combo a instalar (Fase 0):**
```bash
npx skills add wshobson/agents@rag-implementation -g -y
npx skills add jeffallan/claude-skills@rag-architect -g -y
npx skills add oimiragieo/agent-studio@text-to-sql -g -y
```

Los tres ya existen en este entorno como skills locales (`rag`, `architecture-patterns`) que también se aprovecharán: `Skill rag`, `Skill architecture-patterns`, `Skill clean-code`, `Skill refactor`, `Skill modular-code`, `Skill prompt-engineering-patterns`, `Skill serving-llms-vllm`.

---

## 4 · Arquitectura Objetivo

### 4.1 · Stack tecnológico revisado

| Capa | Componente | Notas |
|---|---|---|
| **Embeddings (texto + visual)** | `Qwen3-VL-Embedding-8B` (4096-dim, multimodal) | Reemplaza al text-only Qwen3-Embedding-8B. MTEB multilingual #1. |
| **Re-ranker** | `BAAI/bge-reranker-v2-m3` o `Qwen3-Reranker-4B` | Cross-encoder cabal pasada final sobre top-50 de hybrid search. |
| **LLM (intercambiables)** | `Qwen3.6-27B-UD-Q4_K_XL.gguf` (Unsloth) **o** `gemma-4-26B-A4B-it.gguf` (MoE 4B activos) | Selección por env `BANKS_LLM_MODEL`. |
| **Inference engine** | `llama.cpp` (binarios CUDA 12.x) | Reemplaza vLLM porque los modelos ya están en GGUF y llama.cpp es offline-friendly + sin Python en runtime. |
| **Vector DB** | PostgreSQL 14 + pgvector (HNSW cosine) | Sin cambios. |
| **Storage de imágenes** | Filesystem local (`data/images/`) + path en BD | Simple y portable; opcionalmente migrar a MinIO si se quiere object store. |
| **Series macro** | Parquets en `data_pipeline/snapshots/` (ya existen 14) | Sin cambios; expandir catálogo. |
| **Catálogo SQL** | NUEVO `sql_catalog/` con 75 queries de `Monitor.py` extraídas y curadas | Pieza central de la transformación. |
| **API** | FastAPI + Uvicorn (un único servicio) | Reemplaza los dos servicios actuales. |
| **Logging** | `structlog` JSON + `request_id` | Rastreable en ELK/Loki. |
| **Métricas** | Prometheus client + endpoint `/metrics` | Latencia p50/p95/p99, recall@k, tokens, tool calls. |
| **Tests** | pytest + pytest-asyncio + coverage ≥ 70% | |
| **Eval** | RAGAS + golden set propio (50–100 pares) | Reproducible, regresiones detectables. |

### 4.2 · Estructura de carpetas objetivo

```
banks/
├── README.md                              ← REESCRITO (refleja arch nueva)
├── CLAUDE.md                              ← actualizado: nuevo flujo
├── pyproject.toml                         ← NUEVO (reemplaza requirements.txt sueltos)
├── Makefile                               ← NUEVO (ingest, serve, eval, lint, test)
├── docker-compose.yml                     ← NUEVO (postgres + servicio para dev)
├── .env.example
├── .gitignore
│
├── docs/                                  ← NUEVA: documentación viva
│   ├── ARCHITECTURE.md                    ← Clean Architecture aplicada al RAG
│   ├── DEPLOYMENT_H100.md                 ← guía offline paso a paso
│   ├── EVALUATION.md                      ← cómo correr el golden set + RAGAS
│   ├── DATA_GOVERNANCE.md                 ← lineage, retención, PII
│   ├── SQL_CATALOG.md                     ← cómo extender el catálogo
│   └── model_cards/
│       ├── qwen3-vl-embedding-8b.md
│       ├── qwen3.6-27b.md
│       └── gemma-4-26b-a4b-it.md
│
├── src/banks_rag/                         ← PAQUETE INSTALABLE (reemplaza scripts)
│   │
│   ├── domain/                            ← núcleo puro: entidades + reglas
│   │   ├── documents/
│   │   │   ├── document.py                ← migra de models.py
│   │   │   ├── chunk.py                   ← Chunk con kind = TEXT|VISUAL|TABLE
│   │   │   ├── visual_asset.py            ← NUEVO: gráfico/figura con caption
│   │   │   └── enrichment.py
│   │   ├── retrieval/
│   │   │   ├── query.py                   ← parsed query con filters
│   │   │   ├── result.py                  ← retrieval result
│   │   │   └── citation.py
│   │   ├── series/
│   │   │   └── series_definition.py
│   │   └── agent/
│   │       ├── tool_call.py
│   │       ├── agent_state.py
│   │       └── conversation.py
│   │
│   ├── application/                       ← casos de uso (orquestación, sin I/O directa)
│   │   ├── ingestion/
│   │   │   ├── extract_corpus.py          ← reemplaza 00_generate_jsons.py
│   │   │   ├── enrich_corpus.py           ← reemplaza 01_enrich_metadata.py
│   │   │   ├── vectorize_corpus.py        ← reemplaza 02_vectorize.py (texto + visual)
│   │   │   └── persist_corpus.py          ← reemplaza 03_database.py (load only)
│   │   ├── retrieval/
│   │   │   ├── hybrid_search.py           ← unifica 04_search.py + 04_advanced_search.py
│   │   │   ├── reranker.py                ← NUEVO cross-encoder pass
│   │   │   ├── query_decomposition.py     ← NUEVO: parte queries complejas
│   │   │   └── query_router.py            ← NUEVO: rag-only / sql-only / hybrid
│   │   ├── agent/
│   │   │   ├── conversation_loop.py       ← reemplaza chatbot/{,_calling_tool/}/app/agent.py
│   │   │   ├── tools/
│   │   │   │   ├── registry.py
│   │   │   │   ├── search_documents.py
│   │   │   │   ├── search_visuals.py      ← NUEVO: busca solo en gráficos
│   │   │   │   ├── discover_query.py      ← NUEVO: top-K queries del catálogo
│   │   │   │   ├── execute_query.py       ← NUEVO: ejecuta query parametrizada
│   │   │   │   ├── historical_series.py
│   │   │   │   └── document_lookup.py
│   │   │   ├── prompts.py
│   │   │   └── citation_verifier.py
│   │   └── evaluation/
│   │       ├── ragas_runner.py            ← NUEVO
│   │       ├── retrieval_metrics.py       ← NUEVO recall@k, MRR, nDCG
│   │       └── golden_set_loader.py       ← NUEVO
│   │
│   ├── infrastructure/                    ← adaptadores: I/O y librerías externas
│   │   ├── persistence/
│   │   │   ├── postgres_repo.py           ← reemplaza ambos db.py
│   │   │   ├── parquet_store.py           ← migra de data_pipeline/
│   │   │   └── image_store.py             ← NUEVO: filesystem o MinIO
│   │   ├── embeddings/
│   │   │   ├── multimodal_embedder.py     ← NUEVO: Qwen3-VL-Embedding-8B (texto + img)
│   │   │   └── instruction_builder.py     ← prefijos por modelo
│   │   ├── reranker/
│   │   │   └── cross_encoder.py           ← NUEVO BAAI/bge-reranker-v2-m3
│   │   ├── llm/
│   │   │   ├── base.py                    ← Protocolo LLM (generate, count_tokens)
│   │   │   ├── llamacpp_engine.py         ← NUEVO (reemplaza vllm_engine)
│   │   │   ├── tool_call_parser.py        ← <tool_call>{...}</tool_call>
│   │   │   └── chat_templates.py          ← Qwen vs Gemma templates
│   │   ├── extractors/
│   │   │   ├── pdf_extractor.py           ← pypdf + PyMuPDF
│   │   │   ├── excel_extractor.py
│   │   │   ├── chart_detector.py          ← detección visual avanzada
│   │   │   ├── table_extractor.py         ← NUEVO (camelot/tabula opcional)
│   │   │   └── encoding_fixers.py         ← BCCh + mojibake (migra de 00_)
│   │   ├── chunking/
│   │   │   ├── text_chunker.py            ← jerárquico (párrafo→oración→overlap)
│   │   │   └── monitor_pm_chunker.py      ← celdas Excel
│   │   ├── sql/
│   │   │   ├── catalog_loader.py          ← lee sql_catalog/catalog.yaml
│   │   │   ├── catalog_index.py           ← embeddings de descripciones
│   │   │   ├── safe_executor.py           ← ejecuta con timeout + read-only
│   │   │   └── parameter_parser.py        ← extrae fechas, años de NL
│   │   └── observability/
│   │       ├── tracing.py                 ← request_id, span tree
│   │       ├── metrics.py                 ← Prometheus
│   │       └── logging.py                 ← structlog JSON
│   │
│   ├── interface/                         ← entry points: CLI + API
│   │   ├── cli/
│   │   │   ├── ingest.py                  ← reemplaza run.py
│   │   │   ├── search.py
│   │   │   ├── chat.py                    ← REPL de prueba
│   │   │   └── evaluate.py
│   │   └── api/
│   │       ├── main.py                    ← un solo FastAPI
│   │       ├── routes/
│   │       │   ├── chat.py                ← POST /v1/chat
│   │       │   ├── search.py              ← POST /v1/search
│   │       │   ├── series.py              ← GET  /v1/series
│   │       │   ├── catalog.py             ← GET  /v1/queries (catálogo SQL)
│   │       │   └── health.py              ← /healthz, /readyz, /metrics
│   │       ├── middleware/
│   │       │   ├── request_id.py
│   │       │   ├── auth.py                ← API key (opcional, configurable)
│   │       │   └── rate_limit.py
│   │       └── schemas.py                 ← Pydantic v2
│   │
│   ├── domain_knowledge/                  ← reglas de negocio del dominio macro
│   │   ├── taxonomy.py                    ← migra de raíz (queda intocado en lo esencial)
│   │   ├── importance_rules.py            ← extraído de 01_enrich_metadata.py
│   │   ├── chunking_rules.py
│   │   └── boilerplate_filters.py
│   │
│   └── config/
│       ├── settings.py                    ← unifica chatbot/settings + chatbot_calling_tool/settings
│       └── presets.py                     ← perfiles: dev, prod-h100, eval
│
├── sql_catalog/                           ← NUEVO: el corazón del agente SQL
│   ├── README.md                          ← cómo agregar/modificar entradas
│   ├── catalog.yaml                       ← 75 queries de Monitor.py extraídas, normalizadas, descritas
│   ├── snippets/                          ← templates SQL parametrizables (uno por categoría)
│   │   ├── tasas_clp/
│   │   ├── bonos_clp/
│   │   ├── bonos_uf/
│   │   ├── curva_spc_clp/
│   │   ├── curva_spc_uf/
│   │   ├── ust/
│   │   ├── ois_sofr/
│   │   ├── expectativas_tpm/
│   │   ├── fx/
│   │   ├── commodities/
│   │   ├── forwards_clp/
│   │   ├── tasas_usd_mn/
│   │   ├── liquidez_mx/
│   │   └── microestructura/
│   ├── synthetic_examples.jsonl           ← pares (NL question → query_id) para fine-tune del router
│   └── extraction_audit.md                ← log de qué se extrajo de Monitor.py y qué quedó pendiente
│
├── data/                                  ← inputs y outputs de runtime
│   ├── raw/                               ← reemplaza Datos_prueba/
│   │   ├── pdfs/{comunicados,minutas,fed,researchs}/
│   │   └── excel/
│   ├── images/                            ← charts extraídos (gitignored)
│   ├── snapshots/                         ← parquets DW (gitignored)
│   ├── golden_set/                        ← NUEVO: pares Q/A para evaluación
│   │   ├── retrieval.jsonl                ← (query, expected_chunk_ids)
│   │   ├── generation.jsonl               ← (query, ideal_answer, must_cite)
│   │   └── sql_routing.jsonl              ← (query, expected_query_id)
│   └── logs_intermedios/                  ← reemplaza logs/ (gitignored)
│
├── scripts/                               ← utilidades fuera del runtime
│   ├── extract_monitor_queries.py         ← NUEVO: parsea Monitor.py → catalog.yaml
│   ├── refresh_snapshots.py               ← wrapper de extract.py
│   ├── benchmark_embeddings.py
│   ├── benchmark_llm.py
│   └── seed_golden_set.py
│
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   ├── chunking/
│   │   ├── enrichment/
│   │   └── sql/
│   ├── integration/
│   │   ├── test_pipeline_smoke.py
│   │   ├── test_retrieval.py
│   │   └── test_agent_loop.py
│   └── e2e/
│       └── test_chat_endpoint.py
│
└── deploy/                                ← NUEVO: artefactos de despliegue
    ├── docker/
    │   └── Dockerfile.api
    ├── systemd/
    │   ├── banks-api.service
    │   └── banks-llamacpp.service
    └── nginx/
        └── banks.conf
```

### 4.3 · Carpetas y archivos a ELIMINAR

| Path | Razón |
|---|---|
| `00_generate_jsons.py` | Migrado a `application/ingestion/extract_corpus.py` |
| `01_enrich_metadata.py` | Migrado a `application/ingestion/enrich_corpus.py` |
| `02_vectorize.py` | Migrado a `application/ingestion/vectorize_corpus.py` |
| `03_database.py` | Migrado a `application/ingestion/persist_corpus.py` + `infrastructure/persistence/postgres_repo.py` |
| `04_search.py` + `04_advanced_search.py` | Unificados en `application/retrieval/hybrid_search.py` |
| `05_query.py` | Migrado a `application/agent/prompts.py` (su contenido se subsume) |
| `run.py` | Reemplazado por `interface/cli/ingest.py` con Typer |
| `chatbot/` (todo) | Unificado en `interface/api/` + `application/agent/` |
| `chatbot_calling_tool/` (todo) | Idem |
| `models_cache/models--intfloat--multilingual-e5-small` | Modelo legacy no usado |
| `ADVANCED_SEARCH.md` | Su contenido va a `docs/RETRIEVAL.md` |
| `__pycache__/` (raíz) | Limpieza |

### 4.4 · Diagrama de capas (Clean Architecture)

```
┌──────────────────────────────────────────────────────────────────┐
│ INTERFACE          CLI │ FastAPI                                 │
│ (entrada/salida)       │  routes, middleware, schemas Pydantic   │
└────────────┬─────────────────────────────────────────────────────┘
             │ depende de ↓
┌────────────▼─────────────────────────────────────────────────────┐
│ APPLICATION        casos de uso, orquestación                    │
│  ingestion · retrieval · agent · evaluation                      │
└────────────┬─────────────────────────────────────────────────────┘
             │ depende de ↓ (Protocols)
┌────────────▼─────────────────────────────────────────────────────┐
│ DOMAIN             entidades, value objects, reglas puras        │
│  documents · retrieval · series · agent                          │
└──────────────────────────────────────────────────────────────────┘
             ▲ implementa Protocols
┌────────────┴─────────────────────────────────────────────────────┐
│ INFRASTRUCTURE     adaptadores: postgres, llamacpp, embeddings,  │
│                    extractors, sql, image_store, observability   │
└──────────────────────────────────────────────────────────────────┘
```

Regla clave: **`domain/` y `application/` jamás importan de `infrastructure/`** — solo definen Protocols (typing) que la infra implementa.

---

## 5 · Plan de Ejecución por Fases

> Fases ordenadas para minimizar riesgo: cada fase es entregable, testeable, y deja el sistema funcional. Se hace en una rama `feat/professional-arch` con commits atómicos por fase.

### Fase 0 · Preparación (1 día)

- [ ] Crear branch `feat/professional-arch` desde `main`.
- [ ] Instalar las skills recomendadas (`npx skills add ...`).
- [ ] Crear `pyproject.toml` con `setuptools`/`hatch`, dependencias unificadas, paquete `banks_rag`.
- [ ] Crear `Makefile` con targets: `install`, `lint`, `format`, `typecheck`, `test`, `ingest`, `serve`, `eval`.
- [ ] Configurar `pre-commit` con `ruff` (lint+format), `mypy`, `pytest --collect-only`.
- [ ] Crear esqueleto de `src/banks_rag/` con `__init__.py` por capa.
- [ ] Crear `tests/` con conftest base.
- [ ] Mover `Datos_prueba/` → `data/raw/` (con symlink temporal de compatibilidad).
- [ ] Crear `docs/` y empezar `ARCHITECTURE.md`.

**Criterio de salida:** `make install && make lint && make typecheck && make test` corre verde con 0 tests aún.

---

### Fase 1 · Refactor estructural sin cambios funcionales (3 días)

> Mover código existente a la nueva estructura **preservando comportamiento**. La idea es estabilizar el contenedor antes de cambiar las piezas.

- [ ] **`domain/`**: copiar `models.py` → `domain/documents/document.py` + `chunk.py` + `enrichment.py`. Agregar campo `kind: ChunkKind = TEXT` al `Chunk` (hoy implícito).
- [ ] **`domain_knowledge/taxonomy.py`**: mover `taxonomy.py` (de raíz) sin cambios.
- [ ] **`infrastructure/extractors/`**: trasladar lógica de extracción de `00_generate_jsons.py` (PDF, Excel, encoding fix BCCh, mojibake) a archivos modulares.
- [ ] **`infrastructure/chunking/`**: extraer `chunk_pages` y `extract_cell_chunks`.
- [ ] **`application/ingestion/extract_corpus.py`**: orquesta extractors + chunking.
- [ ] **`application/ingestion/enrich_corpus.py`**: migra `01_enrich_metadata.py`.
- [ ] **`infrastructure/persistence/postgres_repo.py`**: unifica los dos `db.py` actuales + lo de `03_database.py`. Schema y migrations en `infrastructure/persistence/migrations/`.
- [ ] **`application/retrieval/hybrid_search.py`**: unifica `04_search.py` + `04_advanced_search.py` con `AdvancedFilters` como argumento opcional. Misma lógica RRF + MMR + importance boost.
- [ ] **`infrastructure/llm/`**: dejar `llamacpp_engine.py` como stub que aún llama vLLM (compatibilidad temporal).
- [ ] **`application/agent/conversation_loop.py`**: migra `chatbot_calling_tool/app/agent.py`.
- [ ] **`interface/cli/ingest.py`**: con Typer, replica `run.py full` y `step N`.
- [ ] **`interface/api/main.py`**: un solo FastAPI con rutas migradas.

**Criterio de salida:**
- `python -m banks_rag.interface.cli.ingest full` produce los mismos JSONs/registros que `python3 run.py full`.
- `python -m banks_rag.interface.api.main` levanta el endpoint `/v1/chat` con paridad funcional.
- Test smoke: ingestar 2 PDFs + 1 Excel y hacer una búsqueda con resultados idénticos al sistema anterior.
- `chatbot/` y `chatbot_calling_tool/` siguen intactos (se eliminan en Fase 8).

---

### Fase 2 · Embeddings multimodales + preservación de gráficos (4 días)

> Cerrar el ciclo: detectar charts → renderizar PNG → embeber con VL → almacenar embedding y path → exponerlos al agente.

- [ ] **`domain/documents/visual_asset.py`** — nueva entidad:
  ```python
  @dataclass(frozen=True)
  class VisualAsset:
      asset_id: str          # {document_id}_v{NNNN}
      document_id: str
      page: int
      kind: Literal["CHART", "TABLE", "IMAGE"]
      image_path: Path
      caption: str | None    # extraído por OCR del entorno o VL
      surrounding_text: str  # 200 chars antes/después
      embedding: list[float] | None
  ```
- [ ] **`infrastructure/extractors/chart_detector.py`** — mejorar la detección actual (`_page_has_visuals`):
  - PyMuPDF + heurística: imágenes embebidas, formas rellenas (≥8 = chart probable), tablas (líneas paralelas).
  - Recortar bbox del chart, no toda la página.
  - Extraer caption: regex sobre texto cercano (`/Figura \d+|Gráfico \d+|Cuadro \d+/`).
- [ ] **`infrastructure/embeddings/multimodal_embedder.py`** — nuevo embedder:
  - Carga `Qwen3-VL-Embedding-8B` con `transformers`/`sentence-transformers` (`trust_remote_code=True`).
  - Método `encode_text(texts) → np.ndarray` y `encode_image(images) → np.ndarray` (mismo espacio 4096-dim).
  - Configura instrucciones específicas: `"query: ..."` para queries, sin prefijo para passages.
- [ ] **`application/ingestion/vectorize_corpus.py`** — refactor:
  - Procesa chunks `kind=TEXT` con `encode_text`.
  - Procesa visuales con `encode_image` (con fallback a `encode_text(caption + surrounding_text)` si la imagen falla).
  - Output: `chunks_vectorized.json` + `visuals_vectorized.json`.
- [ ] **Schema PostgreSQL** — migración:
  ```sql
  ALTER TABLE chunks ADD COLUMN kind TEXT NOT NULL DEFAULT 'TEXT'
      CHECK (kind IN ('TEXT', 'TABLE', 'VISUAL'));
  ALTER TABLE chunks ADD COLUMN visual_caption TEXT;
  -- VISUAL chunks tienen image_path; TEXT chunks NULL.
  CREATE INDEX idx_chunks_kind ON chunks(kind);
  ```
  No se separa en otra tabla — `chunks` con `kind` discriminator simplifica el retrieval (un solo HNSW sirve para texto + visuales).
- [ ] **`application/retrieval/hybrid_search.py`** — agregar parámetro `kinds: list[ChunkKind] | None`:
  - default: todos.
  - el agente puede pedir solo `VISUAL` cuando la query es visual ("muéstrame la curva swap CLP en marzo 2024").
- [ ] **`application/agent/tools/search_visuals.py`** — nueva tool:
  ```json
  {
    "name": "search_visuals",
    "description": "Busca gráficos/tablas en los PDFs. Útil cuando el usuario pide ver una figura, curva, distribución, etc. Devuelve image_path para citar.",
    "parameters": {"query": "...", "k": 5, "doc_type": "...", "year": ...}
  }
  ```
  Retorna `{ref, image_path, caption, page, document}`. El agente puede entonces usar `<image>{path}</image>` en su respuesta o devolverla en `chunks_seen` con `kind=VISUAL`.
- [ ] **API**: el endpoint `/v1/chat` retorna `visuals: list[{ref, image_url, caption}]` además de `sources`. El cliente puede pedir el binario en `GET /v1/images/{asset_id}`.
- [ ] Test: ingestar 1 PDF con gráficos conocidos (un IPOM) y verificar que `search_visuals("curva swap")` retorna el gráfico esperado.

**Criterio de salida:**
- ≥ 90% de páginas con charts identificadas se embeben.
- Búsqueda visual top-1 acierta en 5/5 queries de test manual.
- Agente puede citar `[VISUAL 1] gráfico p.X de archivo.pdf`.

---

### Fase 3 · LLM swappable (Qwen3.6 / Gemma 4) con llama.cpp (3 días)

- [ ] **`infrastructure/llm/base.py`** — Protocol:
  ```python
  class LLMEngine(Protocol):
      async def generate(self, messages, tools=None, **gen_args) -> GenerationResult: ...
      async def generate_stream(self, ...) -> AsyncIterator[str]: ...
      def count_tokens(self, messages, tools=None) -> int: ...
      def info(self) -> dict: ...
  ```
- [ ] **`infrastructure/llm/llamacpp_engine.py`**:
  - Usa `llama-cpp-python` con `n_gpu_layers=-1` (todo a GPU H100).
  - Lee path del modelo de `settings.llm_model_path`.
  - Soporta `chat_format` por familia: `qwen` o `gemma`.
  - Implementa `generate()` con `model.create_chat_completion(messages, tools=tools, ...)`.
  - Streaming: `stream=True` yields tokens.
  - Tokenización: `model.tokenize(text)` para `count_tokens`.
- [ ] **`infrastructure/llm/chat_templates.py`**:
  - Plantillas explícitas para Qwen (con `<|im_start|>` / `<|im_end|>`) y Gemma (con `<start_of_turn>` / `<end_of_turn>`).
  - Inyección de `tools` en el system prompt: ambos modelos esperan formato distinto — abstraer en una función `render_tools(tools, family)`.
- [ ] **`infrastructure/llm/tool_call_parser.py`**:
  - Detecta `<tool_call>{...}</tool_call>` (Qwen) y formato JSON crudo (Gemma).
  - Fallbacks: ```json blocks, JSON al final del mensaje.
  - Misma firma de output `(list[ToolCall], cleaned_text)`.
- [ ] **`config/settings.py`** — nuevas variables:
  ```python
  llm_family: Literal["qwen", "gemma"] = "qwen"
  llm_model_path: Path  # apunta a .gguf
  llm_n_ctx: int = 16384
  llm_n_gpu_layers: int = -1
  llm_temperature: float = 0.2
  ```
- [ ] **Profile presets** (`config/presets.py`):
  - `prod-h100-qwen` → Qwen3.6-27B-UD-Q4_K_XL.gguf
  - `prod-h100-gemma` → gemma-4-26B-A4B-it-Q4_K_M.gguf
- [ ] Bench: `scripts/benchmark_llm.py` mide tokens/s, TTFT, p99 con prompt fijo en ambos modelos.

**Criterio de salida:**
- `BANKS_LLM_FAMILY=qwen make serve` y `BANKS_LLM_FAMILY=gemma make serve` levantan el mismo endpoint.
- Test e2e: misma pregunta → ambos responden coherentemente, con tool calls válidos.
- Bench documentado en `docs/model_cards/`.

---

### Fase 4 · Catálogo SQL agentic — el corazón del agente (5 días)

> Este es el cambio de mayor valor: convertir `Monitor.py` (75 queries hardcoded) en un catálogo navegable + ejecutable por el agente.

#### 4.1 Extracción del catálogo

- [ ] **`scripts/extract_monitor_queries.py`** — parser AST:
  - Lee `querys/Monitor.py` con `ast.parse`.
  - Encuentra todas las llamadas a `gd.get_data(...)` y la variable a la que se asignan.
  - Extrae el SQL string (incluye triple-quoted).
  - Identifica parámetros implícitos: fechas hardcoded (`'2019-01-01'`), filtros (`Paridad = 'USD'`).
  - Para cada query, infiere:
    - `query_id`: snake_case del nombre de variable (`USD` → `fx_usdclp_diario`).
    - `category`: heurística por columnas/tablas (`mesadineOLTP_.mercado.divisas` → `fx`).
    - `tables_used`, `columns`, `filters`, `aggregations`.
  - Escribe `sql_catalog/extraction_audit.md` con qué fue extraído y qué requiere revisión humana.
  - Escribe `sql_catalog/catalog.yaml` (autogenerado, marcado como `# AUTO-GENERATED — review needed`).

- [ ] **Curado humano** (mitad scripted, mitad manual):
  - Cada entrada debe tener:
    ```yaml
    query_id: fx_usdclp_serie_diaria
    category: fx
    description: "Serie diaria del tipo de cambio USD/CLP (cotización mercado spot)."
    intent_synonyms:
      - "tipo de cambio dólar peso chileno"
      - "USD/CLP histórico"
      - "cotización del dólar"
    parameters:
      - name: date_from
        type: date
        default: "2019-01-01"
        description: "Fecha desde (inclusive)."
      - name: date_to
        type: date
        default: null
        description: "Fecha hasta (inclusive). Null = hoy."
    sql_template: |
      SELECT Fecha, Cotizacion
      FROM mesadineOLTP_.mercado.divisas
      WHERE Paridad = 'USD'
        AND Fecha BETWEEN :date_from AND COALESCE(:date_to, GETDATE())
      ORDER BY Fecha DESC
    output_schema:
      - {name: Fecha, type: date}
      - {name: Cotizacion, type: numeric, unit: "CLP/USD"}
    economic_variables: [TIPO_CAMBIO]
    safety:
      max_rows: 5000
      timeout_seconds: 30
      read_only: true
    ```
  - Las **75 queries** se clasifican en las 14 categorías ya existentes (`tasas_clp`, `bonos_clp`, etc.).

#### 4.2 Indexación semántica del catálogo

- [ ] **`infrastructure/sql/catalog_loader.py`**: carga + valida el YAML.
- [ ] **`infrastructure/sql/catalog_index.py`**:
  - En cold-start, embebe `description + intent_synonyms + " ".join(economic_variables)` con `Qwen3-VL-Embedding-8B`.
  - Almacena en una tabla nueva `sql_catalog_embeddings` (PostgreSQL + pgvector).
  - Método `search(intent: str, k=5) → list[CatalogEntry]`.

#### 4.3 Tools del agente

- [ ] **`application/agent/tools/discover_query.py`**:
  ```json
  {
    "name": "discover_query",
    "description": "Busca en el catálogo de queries SQL las que mejor responden la pregunta del usuario. Devuelve top-K con descripción, parámetros esperados y output schema. NO ejecuta nada — solo descubre opciones.",
    "parameters": {
      "intent": "string — qué dato quieres encontrar",
      "category": "string opcional — fx, bonos_clp, tasas_clp, ...",
      "k": "integer (default 5)"
    }
  }
  ```
- [ ] **`application/agent/tools/execute_query.py`**:
  ```json
  {
    "name": "execute_query",
    "description": "Ejecuta una query del catálogo con parámetros. SOLO acepta query_id existente — no se permite SQL libre. Retorna filas (max 5000) o error si parámetros inválidos.",
    "parameters": {
      "query_id": "string — del catálogo",
      "parameters": "object — clave/valor según el schema de la query"
    }
  }
  ```
  Implementación: usa `safe_executor` con timeout, valida tipos, sanitiza con `sqlalchemy.text(...).bindparams(...)`, **read-only enforcement** (regex que rechaza `INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE` aunque estén commented).

- [ ] **`infrastructure/sql/safe_executor.py`**:
  - Ejecuta solo contra el DW configurado (env `BANKS_DW_DSN`).
  - Si el server del chatbot no tiene acceso al DW (offline), modo `parquet` que ejecuta la query equivalente sobre los snapshots con `duckdb` (que sí entiende SQL completo sobre parquet).

- [ ] **`application/agent/tools/historical_series.py`** — refactor: pasa a delegar en `execute_query` cuando el `series_id` mapea a un `query_id`. Mantiene la API del catálogo de 17 series para compatibilidad.

#### 4.4 Datos sintéticos para evaluación del router

- [ ] **`scripts/seed_sql_router_examples.py`** — genera `sql_catalog/synthetic_examples.jsonl`:
  - Para cada `query_id`, genera 5–10 paráfrasis NL con un LLM local + curado.
  - Usado en `tests/integration/test_sql_router.py` para medir `top-K accuracy`.

**Criterio de salida:**
- 75/75 queries de `Monitor.py` migradas al catálogo (con audit).
- `discover_query("inflación esperada")` retorna `expectativas_tpm`/`expectativas_inflacion` en top-3.
- `execute_query("fx_usdclp_serie_diaria", {date_from: "2024-01-01"})` retorna ≥ 200 filas.
- Test de routing: top-1 accuracy ≥ 75% sobre 200 ejemplos sintéticos.

---

### Fase 5 · Re-ranker y query routing (3 días)

- [ ] **`infrastructure/reranker/cross_encoder.py`**: carga `BAAI/bge-reranker-v2-m3`. Método `rerank(query, documents, top_k=10)`.
- [ ] **`application/retrieval/reranker.py`**: pipeline `hybrid_search(top=50) → cross_encoder_rerank(top=20) → mmr(top=k)`.
- [ ] **`application/retrieval/query_router.py`**:
  - Clasifica intent: `RAG_ONLY` (texto narrativo) | `SQL_ONLY` (datos numéricos) | `HYBRID` (combina) | `VISUAL` (gráfico).
  - Implementación inicial: heurística sobre keywords + estructura (preguntas con cifras, fechas → SQL); fallback a un clasificador basado en embeddings si la heurística falla.
- [ ] **`application/retrieval/query_decomposition.py`**:
  - Para queries complejas multi-parte ("compara TPM y inflación durante el tightening de 2022"), llama al LLM con un sub-prompt para descomponer en 2–3 sub-queries.
  - Cada sub-query corre por su propio camino (RAG/SQL) y los resultados se combinan en el contexto final.

**Criterio de salida:**
- Recall@5 sube ≥ 8% sobre el golden set tras agregar el reranker.
- Latencia p95 del retrieval se mantiene ≤ 500ms (reranker en GPU).

---

### Fase 6 · Evaluación reproducible (3 días)

- [ ] **Golden set inicial** (`data/golden_set/`):
  - `retrieval.jsonl`: 50 (query, expected_chunk_ids[]) — curado a mano.
  - `generation.jsonl`: 30 (query, ideal_answer, must_cite_documents[]).
  - `sql_routing.jsonl`: 50 (query, expected_query_id) — usa el sintético + revisión.
- [ ] **`application/evaluation/retrieval_metrics.py`**: recall@k, MRR, nDCG@k.
- [ ] **`application/evaluation/ragas_runner.py`**: integra RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) — corre offline contra LLM local.
- [ ] **`interface/cli/evaluate.py`**: `make eval` corre todo y emite `eval_report.md` con tablas comparativas vs último baseline.
- [ ] **CI hook**: si recall@5 cae > 5% vs baseline, el commit no merge-ea.

**Criterio de salida:**
- `make eval` corre en < 10 min con LLM local.
- Reporte `eval_report.md` autogenerado con histórico de métricas.

---

### Fase 7 · Observability + Production hardening (2 días)

- [ ] **`infrastructure/observability/logging.py`**: structlog JSON, contextvars para `request_id`, `session_id`, `tool_call_id`.
- [ ] **`infrastructure/observability/metrics.py`**:
  - Counter: `chat_requests_total`, `tool_calls_total{tool=}`.
  - Histogram: `retrieval_latency_seconds`, `llm_generation_latency_seconds`, `tokens_generated`.
  - Gauge: `agent_iterations`, `chunks_in_context`.
- [ ] **`infrastructure/observability/tracing.py`**: OpenTelemetry opcional (off por default; on con `BANKS_TRACING=otlp`).
- [ ] **API middleware**:
  - `request_id.py`: inyecta UUID en headers + logs.
  - `auth.py`: API key (env `BANKS_API_KEYS=key1,key2`); endpoints públicos: `/healthz`, `/metrics`.
  - `rate_limit.py`: por API key, configurable.
- [ ] **`/metrics`** endpoint Prometheus.
- [ ] **`docs/DEPLOYMENT_H100.md`**: guía paso a paso para el server (systemd units, nginx reverse proxy, log rotation, backup de la BD).

**Criterio de salida:**
- Logs en JSON con `request_id` end-to-end.
- Curl a `/metrics` retorna métricas válidas.
- Dashboard de Grafana template incluido en `deploy/grafana/`.

---

### Fase 8 · Limpieza + cutover (2 días)

- [ ] Verificar paridad funcional con tests integration (mismo input → mismo output que el sistema viejo).
- [ ] Eliminar archivos enumerados en §4.3.
- [ ] Eliminar `requirements.txt` redundantes (queda solo `pyproject.toml`).
- [ ] Eliminar `chatbot/` y `chatbot_calling_tool/`.
- [ ] Eliminar `00-05_*.py`, `04_advanced_search.py`, `run.py`, `05_query.py`.
- [ ] Actualizar `.gitignore` para nuevas rutas.
- [ ] **Reescribir `README.md`** desde cero con la nueva arquitectura (ver §6).
- [ ] **Actualizar `CLAUDE.md`** con los nuevos invariantes.
- [ ] Borrar `__pycache__/` de raíz.
- [ ] Squash commits triviales.

**Criterio de salida:**
- `tree -L 3 -I '__pycache__|*.parquet|*.pdf'` muestra una estructura limpia.
- `make test` pasa con coverage ≥ 70%.
- README, CLAUDE.md, docs/ coherentes.

---

### Fase 9 · Deploy y prueba en server H100 (2 días)

- [ ] Empaquetar wheels de todas las deps (con `pip download`).
- [ ] Sincronizar `models/` con: `Qwen3-VL-Embedding-8B/`, `Qwen3.6-27B-UD-Q4_K_XL.gguf`, `gemma-4-26B-A4B-it.gguf`, `bge-reranker-v2-m3/`.
- [ ] Transferir tarball al server (sftp).
- [ ] Instalar offline: `pip install --no-index --find-links wheels/ -e .`
- [ ] Configurar `.env` para H100 (paths absolutos, `BANKS_LLM_FAMILY`, etc.).
- [ ] `python -m banks_rag.interface.cli.ingest full` (ingesta completa).
- [ ] `systemctl start banks-llamacpp` + `systemctl start banks-api`.
- [ ] Validación funcional:
  - `/healthz` y `/readyz` OK.
  - 5 preguntas test (3 narrativas, 2 cuantitativas + 1 visual) responden con citas válidas.
  - Switch `BANKS_LLM_FAMILY=qwen ↔ gemma` y reinicio: ambos modelos respondieron coherentemente.
  - `make eval` con golden set produce reporte válido.

**Criterio de salida:**
- Servicio levantado en H100, accesible internamente.
- Reporte de validación `docs/H100_VALIDATION_2026-XX-XX.md`.

---

## 6 · README — esquema del nuevo

El nuevo `README.md` (Fase 8) debe tener:

1. **Hero**: 3-4 líneas explicando qué es y para quién.
2. **Arquitectura visual**: diagrama de capas + flujo `query → router → (RAG | SQL | VISUAL) → agent → respuesta`.
3. **Quick start (offline)**: `make install && make ingest && make serve`.
4. **Capacidades**:
   - Ingesta multimodal (texto + charts).
   - Retrieval híbrido + reranker.
   - Agente con 7 tools (search_documents, search_visuals, discover_query, execute_query, historical_series, document_lookup, list_documents).
   - LLM swappable (Qwen3.6 / Gemma 4) sobre llama.cpp.
   - Evaluación con RAGAS + golden set.
5. **Configuración**: tabla con env vars críticas.
6. **API**: tabla con endpoints + ejemplos curl.
7. **Despliegue en H100**: link a `docs/DEPLOYMENT_H100.md`.
8. **Extender el sistema**:
   - Agregar variable económica → editar `domain_knowledge/taxonomy.py`.
   - Agregar query SQL → editar `sql_catalog/catalog.yaml`.
   - Agregar tipo de documento → editar `infrastructure/extractors/pdf_extractor.py`.
9. **Tests + eval**: cómo correrlos, qué métricas mirar.
10. **Roadmap**: próximos pasos (multi-tenant, fine-tune del reranker, OCR para tablas escaneadas).

---

## 7 · Archivos Críticos (Cheat Sheet)

| Archivo | Cambio | Cuándo |
|---|---|---|
| `pyproject.toml` | NUEVO | Fase 0 |
| `Makefile` | NUEVO | Fase 0 |
| `src/banks_rag/domain/documents/visual_asset.py` | NUEVO | Fase 2 |
| `src/banks_rag/infrastructure/embeddings/multimodal_embedder.py` | NUEVO | Fase 2 |
| `src/banks_rag/infrastructure/llm/llamacpp_engine.py` | NUEVO | Fase 3 |
| `src/banks_rag/infrastructure/llm/chat_templates.py` | NUEVO | Fase 3 |
| `sql_catalog/catalog.yaml` | NUEVO | Fase 4 |
| `scripts/extract_monitor_queries.py` | NUEVO | Fase 4 |
| `src/banks_rag/application/agent/tools/discover_query.py` | NUEVO | Fase 4 |
| `src/banks_rag/application/agent/tools/execute_query.py` | NUEVO | Fase 4 |
| `src/banks_rag/infrastructure/sql/safe_executor.py` | NUEVO | Fase 4 |
| `src/banks_rag/infrastructure/reranker/cross_encoder.py` | NUEVO | Fase 5 |
| `src/banks_rag/application/retrieval/query_router.py` | NUEVO | Fase 5 |
| `src/banks_rag/application/evaluation/ragas_runner.py` | NUEVO | Fase 6 |
| `data/golden_set/*.jsonl` | NUEVO | Fase 6 |
| `README.md` | REESCRITO | Fase 8 |
| `CLAUDE.md` | ACTUALIZADO | Fase 8 |
| `00_generate_jsons.py` … `05_query.py` | ELIMINADOS | Fase 8 |
| `chatbot/` y `chatbot_calling_tool/` | ELIMINADOS | Fase 8 |

Funciones existentes a **reutilizar tal cual** (con migración de path):
- `taxonomy.normalize_text`, `taxonomy.compile_word_pattern`, todas las constantes (`ECONOMIC_VARIABLES`, `SECTION_KEYWORDS`, etc.).
- `_fix_bcch_font` (encoding fix de Minutas BCCh) en `00_generate_jsons.py:68`.
- `_fix_mojibake` en `00_generate_jsons.py:84`.
- `chunk_pages` y `extract_cell_chunks` (lógica de chunking).
- `IMPORTANCE_WEIGHTS` y `calculate_importance` (`01_enrich_metadata.py`).
- `rrf_fuse`, `mmr_select`, `importance_boost` (`04_search.py`) — son correctas, solo migran de archivo.
- `parquet_store` API completa (`data_pipeline/parquet_store.py`).

---

## 8 · Verificación End-to-End

```bash
# 1. Build
make install

# 2. Calidad
make lint
make typecheck
make test

# 3. Ingesta (con datos de prueba en data/raw/)
make ingest                                    # ejecuta extract → enrich → vectorize → persist
psql rag_banco -c "SELECT COUNT(*), kind FROM chunks GROUP BY kind;"
# → debe mostrar TEXT y VISUAL

# 4. Servicio
BANKS_LLM_FAMILY=qwen make serve &
sleep 60                                       # warmup

# 5. Healthchecks
curl -f http://localhost:8080/healthz
curl -f http://localhost:8080/readyz
curl -f http://localhost:8080/metrics | head

# 6. Pregunta narrativa (RAG)
curl -s -X POST http://localhost:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "¿Por qué el BCCh recortó la TPM en enero 2024?"}' | jq '.response, .sources[0]'

# 7. Pregunta cuantitativa (SQL agentic con catálogo)
curl -s -X POST http://localhost:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Muéstrame la serie del USD/CLP de los últimos 6 meses"}' | jq '.tool_trace[] | {tool, query_id: .arguments.query_id}'
# → debe ver discover_query → execute_query con query_id="fx_usdclp_serie_diaria"

# 8. Pregunta visual
curl -s -X POST http://localhost:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Encuentra el gráfico de la curva swap CLP del último IPOM"}' | jq '.visuals[0]'

# 9. Switch de modelo
killall llama-cpp-server
BANKS_LLM_FAMILY=gemma make serve &
sleep 60
# repetir 6, 7, 8 → respuestas coherentes con el otro modelo

# 10. Evaluación
make eval
cat eval_report.md                             # baseline + delta vs anterior
```

---

## 9 · Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Qwen3-VL-Embedding-8B no embebe imágenes con la calidad esperada | Media | Alto | Bench previo (`scripts/benchmark_embeddings.py`) sobre 20 charts test antes de comprometerse. Fallback: caption + surrounding_text con embedder texto. |
| llama.cpp + Qwen3.6-27B no soporta tool calling tan bien como vLLM | Media | Alto | Test con suite de 50 prompts de tool-calling antes de cutover. Si falla, mantener vLLM como engine secundario en `infrastructure/llm/vllm_engine.py`. |
| 75 queries de `Monitor.py` mal extraídas por el AST parser | Alta | Medio | Audit humano obligatorio (`extraction_audit.md`). Curado en pasadas: primero las 14 categorías más usadas, luego el resto. |
| El reranker añade latencia que rompe SLA | Baja | Medio | Bench p95; si excede, mover a procesamiento async post-stream o solo aplicar cuando `len(candidates) > 20`. |
| El agente con catálogo SQL hace tool-calls erróneos (elige query wrong) | Media | Alto | Golden set de routing con 200 ejemplos; si top-1 < 75%, fine-tune del prompt o bias hacia retrieval clásico. |
| Refactor rompe compatibilidad con datos ingeridos previamente | Alta | Medio | Migration scripts en `infrastructure/persistence/migrations/`. Re-ingesta completa documentada como paso de cutover. |
| Pérdida de la lógica de Monitor.py durante extracción | Media | Alto | Mantener `querys/Monitor.py` original intocado en el repo; el catálogo es derivado. |
| Multi-modelo (Qwen + Gemma) con tool formats distintos rompe parsing | Media | Medio | Tests parametrizados sobre ambas familias; `tool_call_parser` con suites de fixtures por familia. |

---

## 10 · Estimación

| Fase | Días | Acumulado |
|---|---|---|
| 0 · Preparación | 1 | 1 |
| 1 · Refactor estructural | 3 | 4 |
| 2 · Multimodal embeddings + charts | 4 | 8 |
| 3 · LLM swappable llama.cpp | 3 | 11 |
| 4 · Catálogo SQL agentic | 5 | 16 |
| 5 · Re-ranker + routing | 3 | 19 |
| 6 · Evaluación + golden set | 3 | 22 |
| 7 · Observability + hardening | 2 | 24 |
| 8 · Limpieza + cutover | 2 | 26 |
| 9 · Deploy H100 + validación | 2 | 28 |

**Total: ~28 días-persona** para un solo desarrollador. Con dos en paralelo (uno en pipelines, otro en agente/API): ~16 días calendario.

---

## 11 · Decisiones tomadas y próximo paso inmediato

**Decisiones confirmadas con el usuario antes de ejecutar:**

| Decisión | Elección |
|---|---|
| **Inference engine** | `llama.cpp` + GGUF — alineado con `enlaces.md`, offline-friendly, sin Python en runtime. vLLM se descarta. |
| **Skills externas a instalar (Fase 0)** | Combo: `wshobson/agents@rag-implementation` + `oimiragieo/agent-studio@text-to-sql` + `jeffallan/claude-skills@rag-architect` |
| **Prioridad de LLMs** | Qwen3.6-27B-UD-Q4_K_XL **primero** (estable con tool-calling, código orientado a Qwen). Gemma 4 26B-A4B-it se incorpora en una sub-fase 3.b una vez estable Qwen. |

**Implicancia de estas decisiones en el plan:**
- Fase 3 implementa **solo Qwen** sobre `llama-cpp-python`. Una sub-fase 3.b (1 día extra) agrega Gemma con la misma abstracción `LLMEngine`.
- Toda referencia a vLLM en la Fase 1 (`infrastructure/llm/vllm_engine.py` como stub) se elimina; el adapter directo es `llamacpp_engine.py` desde el inicio.
- `pyproject.toml` (Fase 0) incluye `llama-cpp-python[server]` con extras de CUDA 12.x; **no** incluye `vllm`.

**Orden de arranque:**
1. **Fase 0** — preparación: branch, skills, `pyproject.toml`, `Makefile`, esqueleto.
2. **Fase 1** — refactor estructural sin cambios funcionales.
3. **Fase 2** — multimodal embeddings + charts.
4. **Fase 3** — llama.cpp + Qwen3.6-27B.
5. **Fase 3.b** (1 día) — agregar Gemma 4 26B-A4B-it al mismo `LLMEngine`.
6. Luego 4 → 5 → 6 → 7 → 8 → 9 según el plan.

**Total revisado: ~29 días-persona** (28 + 1 día de Gemma).
