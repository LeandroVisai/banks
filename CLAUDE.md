# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Sistema RAG + agente multimodal para el Banco Central de Chile. Procesa PDFs financieros (Comunicados BCCh, Minutas del Consejo, Fed Statements, research JPMorgan) y un Excel de Monitor PM en un corpus semántico consultable con citación por página. El agente puede además consultar 23 series financieras del catálogo SQL vía DuckDB offline.

Stack: `sentence-transformers` (Qwen3-VL-Embedding-8B, 4096-dim) + `llama-cpp-python` + PostgreSQL 16 + pgvector (HNSW) + búsqueda híbrida BM25/vector con RRF, MMR y cross-encoder reranking + FastAPI.

**Arquitectura**: Clean Architecture + DDD. Los scripts numerados legacy (`00-05_*.py`, `run.py`, `chatbot/`, `chatbot_calling_tool/`) fueron eliminados. El core productivo vive en `src/banks_rag/`. El analizador de noticias + voz JARVIS vive **aparte** en `src/jarvis_news/` (paquete aislado por seguridad: procesa datos externos scrapeados; ver [`docs/SETUP_JARVIS.md`](docs/SETUP_JARVIS.md)).

## Comandos esenciales

```bash
# Ingesta (extracción → enriquecimiento → embeddings → BD)
banks-ingest run --source Datos_prueba/

# Ingesta de CONTEXTO ACTUAL (noticias) → base Postgres AISLADA (contexto_actual)
banks-ingest-news full --source data_pipeline/Noticias_scrapping/
banks-ingest-news stats

# Búsqueda
banks-search "tasa de interés 2022" --k 5
banks-search "commodities riesgos" --k 10 --json

# API
uvicorn banks_rag.interface.api.main:create_app --factory --port 8080

# LLM serving (backend openai_compat — producción H100; ver docs/SETUP_LLAMA_SERVER.md)
powershell -ExecutionPolicy Bypass -File deploy/start_llama_server.ps1

# Load test de /v1/chat (comparar backends/quants; baseline en data/bench/)
python scripts/load_test_chat.py --users 2 --out data/bench/u2.json

# Informe analítico JARVIS (markdown + html + audio FLAC ES/EN, paquete aislado)
# Salidas ordenadas en data/news_reports/{markdown,html,audio}/
python scripts/jarvis_news_report.py --audio --top-n 25

# Cargar datos REALES del servidor: cada hoja de parquets_unificados.xlsx → un
# parquet en data_pipeline/parquet/ (DuckDB excel ext; sin pandas). Tras correr,
# ejecutar refresh_catalog_dates.py y revisar con audit_parquets.py.
python scripts/xlsx_to_parquet.py
python scripts/refresh_catalog_dates.py
python scripts/audit_parquets.py            # roles + mismatches catálogo↔parquet

# Informe descriptivo de datasets parquet (SIN tools: Python calcula los hechos
# del parquet real vía parquet_facts + series_analytics, y el LLM solo redacta 1
# párrafo de movimientos relevantes por dataset; síntesis global; salidas en
# data/parquet_reports/{markdown,html}/)
python scripts/parquet_report.py --segment ffmm                 # o "fondos mutuos"
python scripts/parquet_report.py --datasets flujos_ffmm,duracion_ffmm --windows 7d,30d,90d

# Informe CURADO por familia (réplica de un correo real del BCCh; Python puro, sin
# LLM). Cada familia escribe en SU carpeta: data/parquet_reports/curated/<familia>/
python scripts/build_family_report.py                           # lista familias
python scripts/build_family_report.py --family fx               # ffmm | afp | nr | fx | dcv
python scripts/build_family_report.py --all
# Para acotar el período de UN gráfico: date_from/date_to (ISO) en su ReportBlock
# del spec. Recorta el PARQUET antes de la transform, así la "última fecha" del
# gráfico, sus ventanas 7d/30d y su párrafo de texto respetan el recorte.

# Informe curado → correo .eml. El .eml ADJUNTA tu HTML final tal cual (SVG +
# tooltips) y pone en el CUERPO una versión plana (SVG→PNG cid:, sin JS, CSS inline)
# que es lo único que renderiza Outlook. Salidas por familia:
#   data/parquet_reports/eml/<familia>/<nombre>.eml
#   data/parquet_reports/plano/<familia>/<nombre>.plano.html  ← el cuerpo, autocontenido
python scripts/reports_to_eml.py --src data/parquet_reports/curated/fx      # una familia
python scripts/reports_to_eml.py --src data/parquet_reports/curated         # todas (recursivo)

# Evaluación
make eval          # golden set completo → eval_report.md
make eval-ci       # gate CI recall@5

# Tests
PYTHONPATH=src pytest tests/unit/ -q     # ~780 tests, pocos segundos
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
             /v1/news-report · /v1/tts  (paquete aislado jarvis_news, reusa el LLM)

Noticias_scrapping/*.json  (paquete aislado jarvis_news)
  → report.generate_news_report   map-reduce → INFORME analítico (Markdown estructurado)
  → html_report.render_html_report → HTML (estilo plantilla: hero + bloques + referencias)
  → report.narrate_report          → RELATO hablable (prosa, base del audio)
  → audio.synthesize_bilingual     → audio FLAC ES + EN (voz JARVIS)
  Salidas: data/news_reports/{markdown,html,audio}/reporte_<fecha>.*
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
| `jarvis_news/` (aislado, **no** `banks_rag`) | Informe analítico de noticias (`report.py` map-reduce → Markdown estructurado; `html_report.py` MD→HTML; `narrate_report` relato) + voz JARVIS (`tts.py`, `effects.py`, `audio.py`, audio FLAC). Reusa el LLM de la app; router `/v1/news-report` · `/v1/tts` |

## Invariantes críticos

- **`src/banks_rag/domain/` es la fuente de verdad de estructuras**: no duplicar dataclasses en otros módulos.

- **`taxonomy.py` en `infrastructure/extractors/`**: los patrones de variables económicas, secciones, entidades y boilerplate viven ahí. Los pasos de ingesta lo importan en tiempo de ejecución.

- **`doc_type` se hereda del filepath, nunca del contenido del chunk**: `detect_doc_type()` opera sobre la ruta relativa.

- **Embeddings normalizados L2 + `vector_cosine_ops`**: los embeddings se normalizan en `vectorize_corpus()`. El índice HNSW usa `vector_cosine_ops`. Cambiar uno sin el otro rompe la similitud.

- **Dual embedding para chunks visuales**: `combined = IMAGE_WEIGHT * img_emb + (1 - IMAGE_WEIGHT) * txt_emb`, normalizado L2. `IMAGE_WEIGHT` default 0.7, override con `RAG_VISUAL_IMG_WEIGHT`.

- **Modelos en `models/<owner>--<name>/`**: convención offline H100. Si el directorio existe se usa; si no, se descarga desde HuggingFace. Nunca hardcodear rutas absolutas.

- **LlamaCppEngine usa lazy import**: `from llama_cpp import Llama` solo en `_sync_load()`. Permite tests sin el binario instalado.

- **Dos backends LLM detrás del Protocol `LLMEngine`** (`BANKS_LLM_BACKEND`): `inprocess` (llama-cpp-python, UNA instancia, requests serializadas — legacy/fallback) y `openai_compat` (`OpenAICompatEngine` vía httpx contra `llama-server` externo con continuous batching + `--cache-reuse` + `--jinja`; producción H100, ver `docs/SETUP_LLAMA_SERVER.md`). Los helpers de comportamiento (`strip_think`, `detect_family`, `flatten_for_gemma`, `parse_args`) viven en `infrastructure/llm/_common.py` y son COMPARTIDOS: nunca duplicarlos en un engine. `llama_cpp_engine` re-exporta los nombres `_strip_think` etc. por compatibilidad. Errores de red/5xx del servidor → `LLMUnavailableError` → HTTP 503 en `/v1/chat`. El semáforo de chat se dimensiona con `BANKS_LLM_SERVER_SLOTS` cuando el backend es `openai_compat`.

- **CrossEncoderReranker usa lazy import**: `from sentence_transformers import CrossEncoder` solo en `load()`.

- **Catálogo de datasets en `sql_catalog/parquet_catalog.yaml`**: 144 datasets sobre parquets en `data_pipeline/parquet/`, con esquema completo (columnas, tipos, valores de enum, `date_range`) y **`chart_type` canónico** (del diccionario del tablero: `stacked_area`, `grouped_bar`, `market_monitor_table`, …). El LLM elige `dataset_id` + columnas/filtros vía `discover_query`/`execute_query`/analytics; **la SQL la arma siempre la tool** (`_parquet_query.build_fetch_sql`) — el LLM NUNCA escribe SQL. Algunos datasets del catálogo pueden no tener parquet local todavía (llegan en la próxima copia); las tools devuelven error controlado y los tests de integración los saltan.

- **`chart_type` del catálogo manda en los gráficos**: el mapping canónico → familia renderizable (`line`/`area`/`bar`/`grouped_bar`/`stacked_bar`/`point`/`pie`/`table`) vive SOLO en `domain/agent/chart_types.py` (`chart_family`, `vega_mark`) — no duplicarlo. `plot_series` usa el `chart_type` del dataset como default (el LLM solo puede forzar marcas simples); `execute_query`/analytics pasan `chart_hint=dataset.chart_type` a `state.add_series`, e `infer_chart_type` lo respeta cuando la forma del dato lo permite (categórico → bar; familias de barra con muchas observaciones degradan a línea). `/v1/catalog` y `/v1/query/{id}` exponen `chart_type` para que el frontend construya el gráfico consistente con el tablero. Tipos desconocidos degradan a `line` (guard en `tests/unit/test_catalog_chart_types.py`).

- **`data_pipeline/` ya NO extrae de SQL**: la extracción en vivo del DW (`dw_store`, `extract.py`), los snapshots y `series_catalog.yaml` fueron eliminados. Los parquets en `data_pipeline/parquet/` son la única fuente; se regeneran fuera del repo y se copian. No quedan tools que consulten el SQL Server (la antigua `historical_series` se eliminó). **Tras cada copia de parquets nuevos, correr `python scripts/refresh_catalog_dates.py`** para actualizar los `date_range` del catálogo desde los datos reales: si quedan desactualizados, el agente en modo thinking se los toma literal, acota sus queries a la fecha vieja y responde con datos antiguos aunque el parquet tenga filas más recientes.

- **Logging de turnos del agente**: cada turno de `/v1/chat` se persiste como una línea JSON en `data/chat_logs/chat-YYYY-MM-DD.jsonl` vía `infrastructure/observability/chat_log.py` (best-effort, nunca tumba el request). Guarda pregunta, respuesta y evidencia (tool_trace, chunks_seen, series_used, citas) para contrastar respuestas reales vs. esperadas. Control: `BANKS_CHAT_LOG_ENABLED` / `BANKS_CHAT_LOG_DIR`.

- **Corpus de "contexto actual" (noticias) en base Postgres AISLADA**: las noticias scrapeadas (un JSON por día, `noticias_YYYY_MM_DD.json`) se ingestan con `banks-ingest-news` a una base de datos SEPARADA (`BANKS_CONTEXT_DB`, default `contexto_actual`), NO al corpus del banco. Aislamiento por construcción: la tool `search_current_context` solo abre conexión a esa base y `search_documents` solo a `rag_banco` — el agente no puede cruzar corpus por error. Pipeline propio (`application/ingestion/extract_news.py` + `enrich_news.py`) con taxonomía noticiera DISTINTA (`doc_type=NOTICIA`, `is_policy_decision` siempre False, sentimiento, `schema_version="news-1.0"`); reusa el embedder Qwen3-VL y `vectorize_corpus`. La fecha de cada noticia sale del campo `date` interno (`DD/MM/YYYY`, un informe diario trae notas de días previos) y cae al NOMBRE del archivo si viene null. **Mojibake**: el texto del scraper viene doble-codificado (UTF-8 leído como cp1252: `inflaciÃ³n`→`inflación`); se repara con `domain_knowledge/text_repair.fix_mojibake` (usa `ftfy` si está, si no round-trip cp1252→utf-8) **por campo**, no sobre texto ya unido — lo comparten `extract_news` y el path de uploads (`infrastructure/uploads/upload_store`). Ingesta aditiva e idempotente (no purga); la frescura la decide el agente con el filtro por fecha + recency a nivel de día. El router dispara el especialista `coyuntura` por señales ("por qué", "noticias", "qué está pasando"); `document`/`policy` también tienen la tool para corroborar.

- **`jarvis_news` es un paquete aislado, NO depende de `banks_rag.application/domain`**: vive aparte por seguridad (procesa datos externos scrapeados). Solo importa el LLM ya cargado por la app (`app.state.deps.llm`) y `banks_rag.config` para flags del `.env`. No mover lógica de `banks_rag` a `jarvis_news` ni al revés. **Pipeline del informe**: `report.py` redacta un **Markdown estructurado** (un `# H1`, bloques `## N. Tema` con `### Síntesis técnica / ### Implicancias económicas / ### Relación entre noticias`, `---` entre bloques y `## Referencias` en APA) → `html_report.render_html_report` lo convierte (determinista, sin librería markdown externa) al HTML de la plantilla (hero + recuadro-disclaimer + `block-heading`/`sub-heading`/`bullet-list`/`separator`) → `report.narrate_report` deriva un **relato hablable** (prosa, sin formato) que es la base del audio. Las salidas se ordenan en `data/news_reports/{markdown,html,audio}/`. **Audio FLAC**: `audio.encode_audio` comprime el WAV a FLAC con `soundfile` (lazy import; su wheel incluye libsndfile → offline); si falta, cae a `.wav` con warning. **Voz JARVIS — dos motores** (en `tts.py`, lazy import de `pyttsx3`/`piper`): `piper` (default, JARVIS auténtico neural) usa **un modelo por idioma** — EN `jgkawell/jarvis` (en_GB RP) con efecto "sala sutil", ES voz latina `gevy` (es_MX) plana; los perfiles por idioma (espeak, length_scale, fx) viven en `voices.py` → `PIPER_PROFILES`. `sapi` (fallback) usa la voz del SO + DSP numpy (`effects.py`). El texto se normaliza con `textnorm.to_speakable_text` antes de sintetizar (la voz NO lee `#`, `*`, `1.`, links). Ver [`docs/SETUP_JARVIS.md`](docs/SETUP_JARVIS.md).

- **Tests unitarios sin BD ni modelos**: todos los tests en `tests/unit/` usan mocks. `PYTHONPATH=src pytest tests/unit/ -q` debe correr en pocos segundos sin internet ni GPU (incluye `jarvis_news` con mocks de TTS).

- **Golden set en `data/golden_set/`**: `retrieval.jsonl` (30 casos), `sql_routing.jsonl` (30 casos), `generation.jsonl` (15 casos). Curado para el dominio BCCh.

- **Gate CI**: `make eval-ci` falla (exit 1) si `recall@5` cae > 5% vs baseline en `eval_baseline.json`.

## Agregar nuevos tipos de documento

1. `infrastructure/extractors/pdf_extractor.py` → `detect_doc_type()`: añadir rama `if` con la carpeta nueva.
2. `infrastructure/extractors/taxonomy.py` → `SECTION_KEYWORDS`: añadir patrones del nuevo tipo.
3. Ejecutar `banks-ingest run` (idempotente).

## Agregar nuevos datasets al catálogo de parquets

1. Dejar el parquet en `data_pipeline/parquet/<nombre>.parquet` con sus columnas reales (no es necesario un schema canónico).
2. Añadir entrada en `sql_catalog/parquet_catalog.yaml` bajo `datasets:` con `id`, `file`, `chart_type` (el tipo canónico del tablero; si es nuevo, mapearlo en `domain/agent/chart_types.py`), `name`, `description`, `segment`, `unit`, `date_range` y `columns` (cada columna con `name` + `type`; si es categórica, opcionalmente `values:` para enum — NO declarar `values` en columnas payload volátiles tipo `_sparkline_json`/`_title_override`).
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
| `BANKS_LLM_BACKEND` | `inprocess` | `openai_compat` para servir con llama-server (concurrencia real; producción) |
| `BANKS_LLM_BASE_URL` | `http://127.0.0.1:8081` | URL del llama-server (solo `openai_compat`) |
| `BANKS_LLM_REQUEST_TIMEOUT_S` | `300` | Techo por request de generación contra el servidor |
| `BANKS_LLM_SERVER_SLOTS` | `4` | Slots `--parallel` del servidor; dimensiona el semáforo de `/v1/chat` |
| `BANKS_LLM_SERVER_API_KEY` | `` | API key del llama-server (`--api-key`); vacío = sin auth |
| `BANKS_LLM_MODEL_PATH` | `` | Ruta al `.gguf` (solo backend `inprocess`) |
| `BANKS_LLM_MAX_TOKENS` | `2048` | Tokens máx por paso (especialistas/iteración) |
| `BANKS_SYNTHESIS_MAX_TOKENS` | `4096` | Tokens máx de la respuesta final (síntesis); subir si se trunca |
| `BANKS_THINKING_MODE` | `adaptive` | `off`/`adaptive`/`on`: **perfil completo** velocidad↔profundidad (no solo thinking). Define `MODE_PROFILES` en `conversation_loop.py`: nº de especialistas (1/2/3), iteraciones (3/4/5) y sampling según la model card de Qwen3 (thinking→temp0.6/top_p0.95; no-thinking→0.4/0.8). El toggle del frontend lo overridea por consulta. La síntesis nunca razona y usa sampling determinista (0.3/0.8) |
| `RAG_EMBEDDING_MODEL` | `Qwen3-VL-Embedding-8B` | Modelo de embeddings (resuelve a `models/<name>/`) |
| `BANKS_CATALOG_SEMANTIC` | `false` | `true` para activar descubrimiento semántico del catálogo (requiere el embedder cargado) |
| `BANKS_API_KEYS` | `` | CSV de API keys (vacío = sin auth) |
| `BANKS_LOG_JSON` | `false` | `true` en producción |
| `BANKS_TRACING` | `off` | `otlp` para OpenTelemetry |
| `BANKS_RATE_LIMIT_RPM` | `60` | Requests/minuto por API key |
| `RAG_VISUAL_IMG_WEIGHT` | `0.7` | Peso imagen en dual embedding |
| `BANKS_RERANK_ENABLED` | `true` | `false` para apagar el cross-encoder reranker en `search_documents` |
| `BANKS_RERANK_MODEL` | `jinaai/jina-reranker-v3` | Modelo del reranker (resuelve a `models/jinaai--jina-reranker-v3/`) |
| `BANKS_CHAT_LOG_ENABLED` | `true` | `false` para no persistir turnos del agente |
| `BANKS_CHAT_LOG_DIR` | `data/chat_logs` | Otra ruta para los JSONL de chat |
| `BANKS_TTS_ENABLED` | `false` | `true` para habilitar el audio (voz JARVIS) en `jarvis_news` |
| `BANKS_TTS_ENGINE` | `piper` | `piper` (neural, JARVIS auténtico EN+ES — default) o `sapi` (voz del SO + efecto DSP, sin modelos; fallback) |
| `BANKS_TTS_MODEL` | `jgkawell--jarvis/jarvis-medium.onnx` | Modelo `.onnx` voz EN (solo `engine=piper`) |
| `BANKS_TTS_MODEL_ES` | `es_MX-gevy/es_MX-gevy-10196-epoch-high.onnx` | Modelo `.onnx` voz ES (solo `engine=piper`) |
| `BANKS_CONTEXT_DB` | `contexto_actual` | Base Postgres AISLADA del corpus de noticias (contexto actual). Solo `search_current_context` la consulta |
| `BANKS_CONTEXT_WINDOW_DAYS` | `20` | Ventana por defecto (días) que la tool de contexto aplica si el agente no pide fechas (no purga: solo acota la búsqueda) |
| `BANKS_CONTEXT_RECENCY_WEIGHT` | `0.30` | Peso de recencia (a nivel de día) en el ranking de noticias |

## Gemelo de desarrollo

Este repo tiene un gemelo en `/Users/leandrovenegas/Desktop/Proyecto_rag/` (sandbox Mac). Mantener paridad entre ambos. `taxonomy.py` en `infrastructure/extractors/` es la fuente de verdad para patrones compartidos.
