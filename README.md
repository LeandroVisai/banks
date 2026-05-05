# Sistema RAG — Banco Central de Chile

Sistema de inteligencia artificial para análisis de política monetaria y macroeconomía.
Procesa documentos financieros del BCCh, Fed, y JPMorgan en un corpus semántico consultable,
y expone un chatbot con LLM local (sin internet, sin API externa) sobre ese corpus.

---

## Visión general

```
Documentos PDF / Excel
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Pipeline RAG  (00 → 04)                                      │
│  Extracción → Enriquecimiento → Embeddings → PostgreSQL       │
│  Tablas: {prefix}documents, {prefix}chunks + HNSW + GIN       │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Chatbot  (chatbot/)                                          │
│  FastAPI + vLLM en proceso + retrieval híbrido                │
│  POST /chat  →  respuesta + fuentes RAG + series históricas   │
└───────────────────────────────────────────────────────────────┘
```

**Stack completo:**
Python 3.12 · pypdf · sentence-transformers (multilingual-e5-small) ·
PostgreSQL 14 + pgvector (HNSW) · vLLM · FastAPI · psycopg3 · CUDA 12.8 · H100

---

## Estructura del repositorio

```
banks/
├── README.md                       ← este archivo
├── CLAUDE.md                       ← arquitectura técnica detallada (para Claude Code)
├── SETUP_PGVECTOR.md               ← instalación PostgreSQL + pgvector
│
├── ── Pipeline RAG ──
├── run.py                          ← orquestador (ejecuta pasos 0-4)
├── models.py                       ← dataclasses Document, Chunk, EnrichedChunk
├── taxonomy.py                     ← variables económicas, secciones, entidades (fuente de verdad)
├── 00_generate_jsons.py            ← PDF/Excel → chunks.json + documents.json
├── 01_enrich_metadata.py           ← enriquecimiento semántico + importance_score
├── 02_vectorize.py                 ← embeddings L2-normalizados
├── 03_database.py                  ← schema PostgreSQL + HNSW + GIN + carga
├── 04_search.py                    ← búsqueda híbrida vector+BM25+RRF+MMR
├── 04_advanced_search.py           ← búsqueda extendida con filtros
├── 05_query.py                     ← prompt RAG listo para LLM
├── requirements.txt
│
├── ── Chatbot ──
└── chatbot/                        ← API del chatbot (ver chatbot/README.md)
    ├── app/                        ← paquete FastAPI
    ├── schema/                     ← SQL de historial + series históricas
    ├── models/                     ← modelos LLM descargados de HuggingFace
    └── scripts/                    ← utilidades (cargar CSVs de series)
```

**Datos de entrada** (`Datos_prueba/`):

| Carpeta | Tipo | Fuente |
|---|---|---|
| `Comunicados/` | Comunicados de política monetaria | BCCh |
| `Minutas/` | Minutas del Consejo | BCCh |
| `Fed/` | Fed Statements | Reserva Federal |
| `Researchs/` | Research reports | JPMorgan |
| `Monitor PM/` | Monitor de Política Monetaria | BCCh (Excel) |

---

## Parte 1 — Pipeline RAG

Transforma documentos en un corpus vectorial consultable en PostgreSQL.

### Flujo de datos

```
Datos_prueba/**/*.pdf
Datos_prueba/Monitor PM/*.xlsx
     │
     ▼ [00] 00_generate_jsons.py
     │      chunking semántico (target 600 chars, overlap 100)
     │      → logs/documents.json + logs/chunks.json
     │
     ▼ [01] 01_enrich_metadata.py
     │      sección canónica, variables económicas, importance_score
     │      → logs/chunks_enriched.json
     │
     ▼ [02] 02_vectorize.py
     │      multilingual-e5-small, prefijo contextual, normalización L2
     │      → logs/chunks_vectorized.json
     │
     ▼ [03] 03_database.py
     │      schema PostgreSQL, índice HNSW (cosine), índice GIN
     │      → {prefix}documents + {prefix}chunks en rag_banco
     │
     ▼ [04] 04_search.py / 04_advanced_search.py
            vector recall + BM25 recall → RRF → MMR → importance boost
            → top-k chunks con citación por página
```

### Quick start — Pipeline

```bash
# Dependencias
pip install -r requirements.txt

# PostgreSQL con pgvector (ver SETUP_PGVECTOR.md si no está instalado)

# Pipeline completo
python3 run.py full

# Verificar que funciona (top-1 debe ser DECISION, importance ≥ 0.90)
python3 04_search.py "política monetaria 2022" 3
```

### Comandos del pipeline

```bash
python3 run.py full                 # todos los pasos
python3 run.py step 0               # extracción PDF/Excel
python3 run.py step 1               # enriquecimiento
python3 run.py step 2               # vectorización
python3 run.py step 3 setup         # crear schema PostgreSQL
python3 run.py step 3 load          # cargar datos
python3 run.py step 3 reset         # drop + recrear + cargar
python3 run.py step 3 stats         # métricas sin tocar nada

# Búsqueda directa (sin chatbot)
python3 04_search.py "tasa de interés 2022" 5
python3 04_search.py "inflación subyacente" 10 --json
python3 04_search.py "riesgos commodities" 5 --no-mmr
```

### Tabla de prefijos RAG

El pipeline soporta varios conjuntos de tablas en paralelo (un prefijo por modelo de embeddings):

| Prefijo | Tablas | Cuándo usarlo |
|---|---|---|
| `gemma_` | `gemma_documents`, `gemma_chunks` | embeddings con modelo Gemma |
| `qwen_` | `qwen_documents`, `qwen_chunks` | embeddings con modelo Qwen |
| *(vacío)* | `documents`, `chunks` | default sin prefijo |

Al ejecutar los scripts te pregunta cuál usar. Se puede fijar con `RAG_TABLE_PREFIX=qwen_`.

### Taxonomía económica (`taxonomy.py`)

Fuente de verdad para todo el sistema. Define:
- **17 variables** (TASA_INTERES, INFLACION, PIB, TIPO_CAMBIO, COMMODITIES, …) con nivel CRITICAL/HIGH/MEDIUM
- **10 secciones canónicas** de documentos (DECISION, VOTACION, PROYECCION, RIESGOS, …)
- **11 secciones del Monitor PM** (MERCADO_CAMBIARIO, TASAS_TPM, RENTA_FIJA, …)
- **Entidades** (BANCO_CENTRAL_CHILE, FEDERAL_RESERVE, PAIS_USA, …)
- **Importance score**: combinación lineal de señales (variable CRITICAL → +0.35, sección DECISION → +0.25, etc.)

---

## Parte 2 — Chatbot

API conversacional con LLM local cargado en proceso. No requiere internet en tiempo de ejecución.

### Arquitectura del chatbot

```
POST /chat
  │
  ├─ query_analysis → intención + variables económicas + rango temporal
  │
  ├─ retrieval.retrieve()         (en paralelo)
  │    vector recall ║ BM25 recall
  │    → RRF → MMR → importance boost → chunks
  │
  ├─ sql_context.build_context()  (solo si intención cuantitativa)
  │    historical_series + historical_data → tabla ASCII
  │
  ├─ prompts.build_messages()
  │    system + <contexto> + <datos_historicos> + historial + query
  │    + truncamiento por presupuesto de tokens
  │
  ├─ llm.generate() / llm.generate_stream()
  │    vLLM AsyncLLMEngine en GPU (sin HTTP, sin API externa)
  │
  └─ prompts.verify_citations()
       elimina [N] hallucinated, retorna fuentes realmente usadas
```

### Modelos LLM soportados

| Modelo | Carpeta en `chatbot/models/` | GPU | Cuantización |
|---|---|---|---|
| **Qwen3.6-35B-A3B** *(default)* | `Qwen3.6-35B-A3B/` | 1× H100 | FP8 (~35 GB) |
| **Kimi K2 Instruct** | `Kimi-K2-Instruct/` | 8× H100 | FP8 (~1 TB) |

Los modelos se descargan manualmente desde HuggingFace (no se necesita internet en el servidor).

### Quick start — Chatbot

```bash
cd chatbot

# Configuración
cp .env.example .env
# Ajusta: PGPASSWORD, RAG_TABLE_PREFIX (qwen_ o gemma_), CHATBOT_MODEL_NAME

# Instalar dependencias (con internet)
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Levantar (carga el modelo en GPU, tarda 1-3 min)
python -m app.main
```

### Endpoints del chatbot

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Readiness (LLM cargado + DB OK) |
| `GET` | `/model-info` | Metadata del modelo activo |
| `GET` | `/historical-series` | Catálogo de series macro con cobertura |
| `GET` | `/sessions` | Sesiones recientes |
| `GET` | `/chat/{session_id}` | Historial de una sesión |
| `POST` | `/chat` | Enviar mensaje (JSON o SSE stream) |
| `GET` | `/docs` | Swagger UI |

```bash
# Ejemplo
curl -s http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "¿Cuál fue la TPM durante 2022 y por qué subió?"}' | jq

# Con streaming SSE
curl -N http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Explica el panorama de inflación", "stream": true}'
```

### Series históricas (datos cuantitativos directos)

El chatbot lee series de tiempo directamente desde SQL (sin vectorizar) e incluye las tablas en el prompt cuando la query es cuantitativa:

| `series_id` | Variable | Unidad |
|---|---|---|
| `tpm` | TPM del BCCh | % anual |
| `ipc_anual` / `ipc_mensual` | Inflación | var % |
| `usdclp_spot` | Tipo de cambio | CLP/USD |
| `pib_trimestral` / `imacec_mensual` | Actividad | var % a/a |
| `precio_cobre` / `precio_petroleo_wti` | Commodities | USD |
| `fed_funds_rate` | Fed Funds | % anual |
| + 10 series más | … | … |

Para cargar datos históricos:
```bash
# CSV con columnas: date,value[,notes]
python -m scripts.load_series tpm data/tpm.csv
```

---

## Variables de entorno

### Pipeline

| Variable | Default | Descripción |
|---|---|---|
| `PGDATABASE` | `rag_banco` | Base de datos PostgreSQL |
| `PGHOST` / `PGPORT` / `PGUSER` / `PGPASSWORD` | localhost/5432/postgres | Conexión |
| `RAG_TABLE_PREFIX` | *(pregunta interactiva)* | `gemma_`, `qwen_`, o vacío |
| `RAG_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Modelo de embeddings |

### Chatbot (`chatbot/.env`)

| Variable | Default | Descripción |
|---|---|---|
| `CHATBOT_MODEL_NAME` | `Qwen3.6-35B-A3B` | Subcarpeta del modelo en `chatbot/models/` |
| `CHATBOT_QUANTIZATION` | `fp8` | Cuantización (vacío = BF16) |
| `CHATBOT_TENSOR_PARALLEL` | `1` | GPUs (8 para Kimi K2) |
| `CHATBOT_MAX_MODEL_LEN` | `8192` | Contexto máximo (tokens) |
| `CHATBOT_MAX_TOKENS` | `1024` | Tokens máximos de respuesta |
| `PROMPT_TOKEN_BUDGET` | `6500` | Límite de tokens del prompt (recorta chunks si supera) |
| `RAG_TABLE_PREFIX` | (vacío) | Prefijo de tablas RAG del pipeline |
| `API_PORT` | `8080` | Puerto FastAPI |
| `LOG_JSON` | `false` | `true` para logs estructurados (ELK/Loki) |

---

## Documentación adicional

| Archivo | Contenido |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Arquitectura técnica detallada, invariantes, schema PostgreSQL |
| [`SETUP_PGVECTOR.md`](SETUP_PGVECTOR.md) | Instalación de PostgreSQL 14 + pgvector |
| [`chatbot/README.md`](chatbot/README.md) | Documentación completa del chatbot |
| [`ADVANCED_SEARCH.md`](ADVANCED_SEARCH.md) | Uso de `04_advanced_search.py` |
