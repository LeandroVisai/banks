# Manual de Operación — Banks RAG

Guía completa, paso a paso, para activar cada funcionalidad del sistema.
Cubre desde la instalación hasta el agente LLM en producción.

---

## Índice

1. [Instalación del entorno](#1-instalación-del-entorno)
2. [PostgreSQL + pgvector](#2-postgresql--pgvector)
3. [Modelos (embeddings, reranker, LLM)](#3-modelos)
4. [Pipeline de ingesta](#4-pipeline-de-ingesta)
5. [Búsqueda híbrida (sin LLM)](#5-búsqueda-híbrida)
6. [Catálogo SQL agentic](#6-catálogo-sql-agentic)
7. [API REST](#7-api-rest)
8. [Agente LLM completo](#8-agente-llm-completo)
9. [Evaluación y métricas](#9-evaluación-y-métricas)
10. [Observabilidad](#10-observabilidad)
11. [Variables de entorno — referencia completa](#11-variables-de-entorno)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Instalación del entorno

### Requisitos

| Componente | Versión mínima | Notas |
|---|---|---|
| Python | 3.12 | `python3.12 --version` |
| PostgreSQL | 16 | con extensión `pgvector` |
| CUDA (GPU) | 12.1 | solo para LLM + embeddings en GPU |
| RAM | 32 GB | 64 GB recomendado con modelos grandes |
| VRAM | 40 GB | H100 80GB para Qwen3.6-27B completo |

### Crear entorno

```bash
git clone <repo-url> banks && cd banks

# Entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

# Instalación base (API + ingesta + búsqueda)
pip install -e ".[api]"

# Instalación desarrollo (+ tests, linting)
pip install -e ".[dev]"

# Instalación evaluación RAGAS (opcional, requiere internet)
pip install -e ".[eval]"
```

### Verificar instalación

```bash
banks-ingest --help     # debe mostrar subcomandos
banks-search --help
banks-eval --help
python -c "import banks_rag; print('OK', banks_rag.__version__)"
```

---

## 2. PostgreSQL + pgvector

### Instalar pgvector

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-16-pgvector

# macOS (Homebrew)
brew install pgvector
```

### Crear base de datos

```sql
-- Como usuario postgres:
sudo -u postgres psql <<'SQL'
CREATE USER banks_rag WITH PASSWORD 'tu_contraseña';
CREATE DATABASE rag_banco OWNER banks_rag;
\c rag_banco
CREATE EXTENSION IF NOT EXISTS vector;
\q
SQL
```

### Variables de conexión

```bash
export PGHOST=localhost
export PGPORT=5432
export PGUSER=banks_rag
export PGPASSWORD=tu_contraseña
export PGDATABASE=rag_banco
```

### Verificar

```bash
psql -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
# Debe mostrar: vector | 0.7.x
```

---

## 3. Modelos

El sistema usa tres tipos de modelos. Todos deben estar en `models/<owner>--<name>/`
o en la raíz de `models/` para los `.gguf`. Se auto-detectan antes de descargar
de HuggingFace.

### 3.1 Embedder — Qwen3-VL-Embedding-8B (texto + imágenes)

**Dimensión de salida**: 4096  
**Uso**: vectorizar chunks de texto Y chunks de gráficos extraídos de PDFs

```bash
# Descargar desde HuggingFace (requiere internet, ~16GB)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-VL-Embedding-8B',
    local_dir='models/Qwen--Qwen3-VL-Embedding-8B',
    ignore_patterns=['*.bin']   # usa solo safetensors
)
"
```

**Verificar**:
```bash
python -c "
from banks_rag.infrastructure.embeddings import build_default_embedder
emb = build_default_embedder()
vec = emb.encode_text(['prueba'])
print('Dim:', vec.shape)    # debe ser (1, 4096)
print('Is multimodal:', emb.is_multimodal)
"
```

**Alternativa más ligera** (E5-multilingual, 768-dim, sin GPU):
```bash
export BANKS_EMBEDDING_MODEL=intfloat/multilingual-e5-large
```
> Nota: si cambias el modelo después de cargar chunks a la BD, debes re-vectorizar e re-indexar todo.

### 3.2 Reranker — bge-reranker-v2-m3 (opcional)

**Uso**: reordenar los top-N chunks recuperados por relevancia cruzada (query, chunk)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'BAAI/bge-reranker-v2-m3',
    local_dir='models/BAAI--bge-reranker-v2-m3'
)
"
```

**Verificar**:
```bash
python -c "
from banks_rag.infrastructure.reranker import CrossEncoderReranker
r = CrossEncoderReranker()
r.load()
resultado = r.rerank('TPM actual', [
    {'text': 'El Consejo decidió mantener la TPM en 5.5%'},
    {'text': 'El precio del cobre subió un 3%'}
])
print(resultado[0]['reranker_score'])   # score más alto primero
print(r.info())
"
```

### 3.3 LLM — Qwen3.6-27B o Gemma 4 26B (vía llama.cpp)

**Uso**: generación de respuestas del agente, tool-calling

#### Instalar llama-cpp-python con soporte CUDA

```bash
# Con GPU CUDA (recomendado para producción)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade

# Sin GPU (solo CPU, muy lento para modelos grandes)
pip install llama-cpp-python
```

#### Descargar modelo cuantizado

```bash
# Qwen3.6-27B (recomendado, ~16GB)
pip install huggingface_hub
huggingface-cli download \
    bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --local-dir models/

# Para producción H100: modelo más capaz
huggingface-cli download \
    unsloth/Qwen3-30B-A3B-GGUF \
    Qwen3-30B-A3B-UD-Q4_K_XL.gguf \
    --local-dir models/
```

#### Configurar

```bash
export BANKS_LLM_FAMILY=qwen
export BANKS_LLM_MODEL_PATH=models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
export BANKS_LLM_N_CTX=16384
export BANKS_LLM_N_GPU_LAYERS=-1    # -1 = todo en GPU
export BANKS_LLM_TEMPERATURE=0.2
export BANKS_LLM_MAX_TOKENS=2048
```

**Verificar**:
```bash
python -c "
import asyncio
from banks_rag.infrastructure.llm import LlamaCppEngine
from banks_rag.config import get_settings

engine = LlamaCppEngine.from_settings(get_settings())
asyncio.run(engine.load())
result = asyncio.run(engine.generate([
    {'role': 'user', 'content': '¿Qué es la política monetaria? Responde en una frase.'}
]))
print(result.text)
print('Tokens:', result.completion_tokens)
"
```

---

## 4. Pipeline de ingesta

Transforma documentos crudos → vectores en PostgreSQL. Cuatro pasos secuenciales.

### Estructura esperada de documentos fuente

```
Datos_prueba/
├── Comunicados/          → tipo COMUNICADO
├── Minutas/              → tipo MINUTA
├── Fed/                  → tipo FED_STATEMENT
├── Researchs/            → tipo RESEARCH
│   └── JPMorgan/
└── Monitor PM/
    └── textos_monitor_pm.xlsx   → tipo MONITOR_PM
```

### Paso 4.1 — Extracción

Extrae texto de PDFs y Excel, chunkea por párrafos (~600 chars, overlap 100):

```bash
banks-ingest extract \
    --data-root Datos_prueba/ \
    --output-dir data/

# Con extracción de gráficos (chunks VISUAL, requiere PyMuPDF)
banks-ingest extract \
    --data-root Datos_prueba/ \
    --output-dir data/ \
    --extract-images \
    --images-dir data/images/
```

**Salida**: `data/chunks.json`, `data/documents.json`

**Verificar**:
```bash
python -c "
import json
chunks = json.load(open('data/chunks.json'))
print(f'{len(chunks)} chunks extraídos')
kinds = {c.get('kind','TEXT') for c in chunks}
print('Tipos:', kinds)
"
```

### Paso 4.2 — Enriquecimiento

Clasifica secciones, variables económicas, importance_score:

```bash
banks-ingest enrich \
    --input data/chunks.json \
    --output data/chunks_enriched.json
```

**Salida**: `data/chunks_enriched.json` (añade `section_type`, `importance_score`, `economic_variables`, `entities`)

**Verificar**:
```bash
python -c "
import json
chunks = json.load(open('data/chunks_enriched.json'))
decisions = [c for c in chunks if c.get('section_type') == 'DECISION']
print(f'Chunks DECISION: {len(decisions)}')
high_imp = [c for c in chunks if c.get('importance_score', 0) > 0.7]
print(f'Importance > 0.7: {len(high_imp)}')
"
```

### Paso 4.3 — Vectorización (embeddings)

Genera embeddings con Qwen3-VL. Chunks VISUAL usan dual embedding (texto + imagen):

```bash
banks-ingest vectorize \
    --input data/chunks_enriched.json \
    --output data/chunks_vectorized.json \
    --batch-size 32

# Con peso personalizado para imagen (default 0.7)
RAG_VISUAL_IMG_WEIGHT=0.8 banks-ingest vectorize \
    --input data/chunks_enriched.json \
    --output data/chunks_vectorized.json
```

**Salida**: `data/chunks_vectorized.json` (añade campo `embedding` de 4096 floats)

**Este paso tarda ~5-30 min dependiendo del número de chunks y la GPU.**

**Verificar**:
```bash
python -c "
import json
chunks = json.load(open('data/chunks_vectorized.json'))
c = chunks[0]
print('Dim embedding:', len(c['embedding']))   # debe ser 4096
dual = [c for c in chunks if c.get('dual_embedded')]
print(f'Dual-embedded (visual): {len(dual)}')
"
```

### Paso 4.4 — Persistencia en PostgreSQL

Crea el schema (HNSW + GIN), carga chunks y documentos:

```bash
banks-ingest persist \
    --chunks data/chunks_vectorized.json \
    --documents data/documents.json

# Con prefijo de tabla (útil para múltiples entornos)
RAG_TABLE_PREFIX=dev_ banks-ingest persist \
    --chunks data/chunks_vectorized.json \
    --documents data/documents.json
```

**Este paso crea las tablas `documents` y `chunks` con índices HNSW y GIN.**

**Verificar**:
```bash
psql -c "SELECT COUNT(*) FROM chunks;"
psql -c "SELECT COUNT(*) FROM chunks WHERE kind='VISUAL';"
psql -c "SELECT section_type, COUNT(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC LIMIT 10;"
psql -c "SELECT doc_type_category, COUNT(*) FROM documents GROUP BY 1;"
```

### Pipeline completo en un comando

```bash
banks-ingest run --source Datos_prueba/

# Equivalente a los 4 pasos anteriores en secuencia
```

---

## 5. Búsqueda híbrida

Combina búsqueda vectorial (HNSW cosine) + BM25 → RRF → MMR → (reranker opcional).

### Desde CLI

```bash
# Búsqueda básica
banks-search "política monetaria 2024" --k 5

# Con filtros
banks-search "TPM" --k 5 --doc-type COMUNICADO --year 2024

# Con reranker
banks-search "decisión del Consejo" --k 5 --rerank

# Salida JSON (para scripts)
banks-search "inflación" --k 3 --json | jq '.hits[].text'

# Sin diversificación MMR
banks-search "tasa de interés" --k 5 --no-mmr
```

### Desde Python

```python
from banks_rag.application.retrieval import hybrid_search
from banks_rag.infrastructure.embeddings import build_default_embedder
from banks_rag.infrastructure.reranker import CrossEncoderReranker

embedder = build_default_embedder()

# Sin reranker
result = hybrid_search(
    "¿Qué decidió el Consejo en la última reunión?",
    query_embedder=embedder,
    k=5,
)
for hit in result.hits:
    print(f"[{hit['section_type']}] {hit['text'][:120]}")
    print(f"  score={hit['score']:.3f} importance={hit['importance_score']:.2f}")

# Con reranker
reranker = CrossEncoderReranker()
result = hybrid_search(
    "¿Cuál es la TPM actual?",
    query_embedder=embedder,
    reranker=reranker,
    k=5,
)
```

### Query routing automático

El router clasifica si la consulta es RAG, SQL o VISUAL antes de buscar:

```python
from banks_rag.application.retrieval.query_router import route_query

r = route_query("¿Cuánto vale el USD/CLP hoy?")
print(r.sql, r.rag, r.visual)   # True False False
print(r.primary)                 # "sql"
print(r.scores)                  # {'rag': 0.0, 'sql': 1.0, 'visual': 0.0}

r = route_query("¿Qué dijo el BCCh en el último comunicado?")
print(r.rag, r.sql)              # True False
```

---

## 6. Catálogo SQL agentic

Consulta 23 series financieras del Monitor PM sin conexión al SQL Server.
Los datos viven en `data_pipeline/snapshots/*.parquet`.

### Requisito: tener los parquets

Los parquets deben generarse previamente desde el SQL Server (con `data_pipeline/extract.py`)
y copiarse a `data_pipeline/snapshots/`. El sistema los lee con DuckDB sin BD adicional.

```bash
ls data_pipeline/snapshots/*.parquet
# fx.parquet, bonos_clp.parquet, commodities.parquet, etc.
```

### Descubrir queries disponibles

```python
import asyncio
from banks_rag.application.agent.tools.discover_query import discover_query
from unittest.mock import MagicMock

state = MagicMock()
result = asyncio.run(discover_query(state=state, query="precio dólar"))
for r in result['results']:
    print(r['query_id'], '—', r['name'])
    print(' ', r['description'][:80])
```

### Ejecutar una query

```python
import asyncio
from banks_rag.application.agent.tools.execute_query import execute_query
from unittest.mock import MagicMock

state = MagicMock()

# USD/CLP últimos 30 días
result = asyncio.run(execute_query(
    state=state,
    query_id="usdclp_historico",
    fecha_inicio="-30d",
    fecha_fin="hoy",
))
print(f"Filas: {result['n_rows']}")
for row in result['rows'][:5]:
    print(row)
```

### Queries disponibles por segmento

| Segmento | query_id |
|---|---|
| Mercado cambiario | `usdclp_historico`, `forwards_clp_curva`, `microestructura_fx`, `indice_monedas_latam` |
| Renta fija Chile | `curva_btp_clp`, `curva_btu_uf`, `btp_btu_comparado`, `spc_clp_curva`, `spc_uf_curva` |
| Renta fija EEUU | `curva_ust` |
| Tasas monetarias | `spreads_dap_swap_clp`, `pdbc_bolsa`, `tib_mercado` |
| Tasas internacionales | `sofr_plazos`, `prime_usd_plazos`, `curva_ois_sofr`, `spread_onshore_dap_usd`, `tasas_tado` |
| Política monetaria | `expectativas_tpm_mipr` |
| Liquidez bancaria | `lcr_mx_bancos`, `nsfr_mx_bancos`, `ratio_liquidez_obligaciones` |
| Commodities | `precio_cobre` |

---

## 7. API REST

### Iniciar el servidor

```bash
# Desarrollo (con recarga automática)
uvicorn banks_rag.interface.api.main:create_app \
    --factory --port 8080 --reload

# Producción
python -m banks_rag.interface.api.main

# Con variables de entorno
BANKS_API_PORT=8080 \
BANKS_LOG_JSON=true \
BANKS_API_KEYS=mi_key_secreta \
python -m banks_rag.interface.api.main
```

### Endpoints disponibles

| Método | Ruta | Descripción | Auth |
|---|---|---|---|
| GET | `/healthz` | Liveness — siempre 200 | No |
| GET | `/readyz` | Readiness — BD + LLM | No |
| GET | `/metrics` | Métricas Prometheus | No |
| GET | `/docs` | Swagger UI interactivo | No |
| POST | `/v1/search` | Búsqueda híbrida sin LLM | Sí |
| POST | `/v1/chat` | Agente LLM con tool-calling | Sí |
| GET | `/v1/images/{chunk_id}` | Imagen de chunk VISUAL | Sí |

### Ejemplos de uso

```bash
# Liveness
curl http://localhost:8080/healthz

# Búsqueda (sin LLM, solo retrieval)
curl -X POST http://localhost:8080/v1/search \
  -H "X-API-Key: mi_key_secreta" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "política monetaria 2024",
    "k": 5,
    "doc_types": ["COMUNICADO", "MINUTA"],
    "use_reranker": false
  }'

# Chat agentic (requiere LLM cargado)
curl -X POST http://localhost:8080/v1/chat \
  -H "X-API-Key: mi_key_secreta" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Cuál es la TPM actual y qué dijo el BCCh?",
    "session_id": "sesion-123"
  }'

# Ver métricas Prometheus
curl http://localhost:8080/metrics | grep banks_
```

### Con Python (requests)

```python
import requests

BASE = "http://localhost:8080"
HEADERS = {"X-API-Key": "mi_key_secreta", "Content-Type": "application/json"}

# Búsqueda
r = requests.post(f"{BASE}/v1/search", headers=HEADERS, json={
    "query": "decisión TPM Banco Central",
    "k": 3
})
for hit in r.json()["hits"]:
    print(hit["section_type"], hit["text"][:100])

# Chat
r = requests.post(f"{BASE}/v1/chat", headers=HEADERS, json={
    "message": "¿Cuánto vale el dólar hoy?"
})
print(r.json()["response"])
```

---

## 8. Agente LLM completo

El agente ejecuta un loop de tool-calling: decide qué tools usar (búsqueda RAG,
SQL, visual), las llama en orden, y formula la respuesta final con citas.

### Prerequisitos

- LLM cargado (ver §3.3)
- Embedder disponible (ver §3.1)
- PostgreSQL con datos (ver §4)
- Parquets en `data_pipeline/snapshots/` (ver §6)

### Desde Python

```python
import asyncio
from banks_rag.application.agent import run_agent
from banks_rag.infrastructure.embeddings import build_default_embedder
from banks_rag.infrastructure.llm import LlamaCppEngine
from banks_rag.infrastructure.reranker import CrossEncoderReranker
from banks_rag.config import get_settings

settings = get_settings()

# Inicializar componentes
embedder = build_default_embedder()
llm = LlamaCppEngine.from_settings(settings)
reranker = CrossEncoderReranker()   # opcional

asyncio.run(llm.load())

# Conversación simple
async def chat():
    history = []
    response = await run_agent(
        user_message="¿Cuál es la TPM actual y por qué la bajaron?",
        conversation_history=history,
        llm=llm,
        embedder=embedder,
        reranker=reranker,     # omitir si no hay modelo
        max_iterations=6,
    )
    print("Respuesta:", response.text)
    print("Tools usadas:", [tc.name for tc in response.tool_calls_made])
    print("Chunks citados:", len(response.cited_chunks))

asyncio.run(chat())
```

### Tools disponibles del agente

| Tool | Cuándo se activa | Qué hace |
|---|---|---|
| `search_documents` | Preguntas narrativas / contexto BCCh | Híbrida RAG sobre corpus de PDFs |
| `search_visuals` | "muéstrame el gráfico", "chart" | Busca chunks de tipo VISUAL |
| `discover_query` | Datos numéricos, series financieras | Busca en catálogo cuáles queries aplican |
| `execute_query` | Tras discover_query | Ejecuta DuckDB sobre parquets |
| `document_lookup` | Cita explícita de documento | Fetch directo de chunks de un doc |
| `historical_series` | Series temporales con fecha | Wrapper de execute_query con filtros de fecha |
| `add_to_context` | Siempre | Agrega chunks seleccionados al contexto final |

### Configurar límites del agente

```bash
export BANKS_MAX_AGENT_ITERATIONS=6    # máximo de rondas tool-call
export BANKS_MAX_TOOL_RESULT_TOKENS=1500   # tokens máximos por resultado de tool
export BANKS_HISTORY_MAX_TURNS=10      # turnos de historial a mantener
```

---

## 9. Evaluación y métricas

### Evaluación offline (sin BD ni LLM real)

```bash
# Todo el golden set → eval_report.md
banks-eval all

# Solo routing SQL (muy rápido, ~0.1s)
banks-eval routing

# Solo retrieval
banks-eval retrieval --k 5

# Solo generación (heurística offline)
banks-eval generation
```

### Guardar baseline para gate CI

```bash
# Primera vez o tras mejora confirmada:
banks-eval all --update-baseline

# Gate CI (falla si recall@5 cae >5%):
banks-eval all --ci
# Equivalente: make eval-ci
```

### Evaluar con BD real (retrieval significativo)

```python
from banks_rag.application.evaluation.retrieval_metrics import (
    evaluate_retrieval, aggregate, load_retrieval_golden_set
)
from banks_rag.application.retrieval import hybrid_search
from banks_rag.infrastructure.embeddings import build_default_embedder

embedder = build_default_embedder()
cases = load_retrieval_golden_set()
results = []

for case in cases[:10]:    # subset para prueba rápida
    result = hybrid_search(case["query"], query_embedder=embedder, k=5)
    r = evaluate_retrieval(
        result.hits,
        expected_doc_types=case["expected_doc_types"],
        expected_sections=case["expected_sections"],
        query=case["query"],
        k=5,
    )
    results.append(r)
    print(f"  recall={r.recall_at_k:.0%}  mrr={r.mrr_at_k:.2f}  [{case['query'][:50]}]")

agg = aggregate(results)
print(f"\nRecall@5: {agg.recall_at_k:.1%}  MRR: {agg.mrr_at_k:.2f}  nDCG: {agg.ndcg_at_k:.2f}")
```

### Golden sets incluidos

| Archivo | Contenido | Casos |
|---|---|---|
| `data/golden_set/retrieval.jsonl` | Queries + doc_types/sections esperados | 30 |
| `data/golden_set/sql_routing.jsonl` | Queries + query_id esperado | 30 |
| `data/golden_set/generation.jsonl` | Queries + keywords + must_cite | 15 |

---

## 10. Observabilidad

### Logging estructurado

```bash
# JSON en producción (recomendado)
export BANKS_LOG_JSON=true
export BANKS_LOG_LEVEL=INFO

# Texto en desarrollo
export BANKS_LOG_JSON=false
export BANKS_LOG_LEVEL=DEBUG
```

```python
# Inyectar contexto en logs (desde middleware o handlers)
from banks_rag.infrastructure.observability import bind_request_context, get_logger

bind_request_context(request_id="req-abc123", session_id="ses-xyz")
log = get_logger(__name__)
log.info("chunk recuperado", n_chunks=5, query="TPM")
# → {"ts":"...","level":"info","request_id":"req-abc123","n_chunks":5,"query":"TPM"}
```

### Métricas Prometheus

```bash
# El endpoint /metrics se activa automáticamente con la API
curl http://localhost:8080/metrics | grep banks_

# Métricas disponibles:
# banks_chat_requests_total{status}
# banks_tool_calls_total{tool,status}
# banks_retrieval_requests_total{status}
# banks_retrieval_latency_seconds (histogram)
# banks_llm_generation_latency_seconds (histogram)
# banks_tokens_generated (histogram)
# banks_agent_iterations (gauge)
# banks_chunks_in_context (gauge)
```

**Configurar Prometheus** (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: banks_rag
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
```

**Importar dashboard Grafana**:
```bash
curl -X POST http://grafana:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @deploy/grafana/banks_rag_dashboard.json
```

### Tracing OpenTelemetry (opcional)

```bash
# Activar con OTLP exporter (a Jaeger, Tempo, etc.)
export BANKS_TRACING=otlp
export BANKS_OTLP_ENDPOINT=http://localhost:4317

# Off por defecto:
export BANKS_TRACING=off
```

### Rate limiting

```bash
# Por API key: 60 req/min, burst de 10 (default)
export BANKS_RATE_LIMIT_RPM=60
export BANKS_RATE_LIMIT_BURST=10

# Aumentar para clientes intensivos
export BANKS_RATE_LIMIT_RPM=300
```

---

## 11. Variables de entorno

### Archivo `.env` (en raíz del repo, no se commitea)

```bash
# ── PostgreSQL ────────────────────────────────────────────────────────────
PGHOST=localhost
PGPORT=5432
PGUSER=banks_rag
PGPASSWORD=tu_contraseña
PGDATABASE=rag_banco
RAG_TABLE_PREFIX=              # prefijo de tablas (ej. "dev_" para desarrollo)

# ── Embeddings ────────────────────────────────────────────────────────────
# (sin prefijo BANKS_; auto-detectado por build_default_embedder)
# BANKS_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B   # default
RAG_VISUAL_IMG_WEIGHT=0.7     # peso imagen en dual embedding (0.0–1.0)

# ── LLM ──────────────────────────────────────────────────────────────────
BANKS_LLM_FAMILY=qwen          # qwen | gemma | mock
BANKS_LLM_MODEL_PATH=models/Qwen3.6-27B-UD-Q4_K_XL.gguf
BANKS_LLM_N_CTX=16384          # context window en tokens
BANKS_LLM_N_GPU_LAYERS=-1      # -1 = todo en GPU, 0 = solo CPU
BANKS_LLM_TEMPERATURE=0.2
BANKS_LLM_TOP_P=0.9
BANKS_LLM_MAX_TOKENS=2048

# ── Agente ───────────────────────────────────────────────────────────────
BANKS_MAX_AGENT_ITERATIONS=6
BANKS_MAX_TOOL_RESULT_TOKENS=1500
BANKS_HISTORY_MAX_TURNS=10

# ── API ──────────────────────────────────────────────────────────────────
BANKS_API_HOST=0.0.0.0
BANKS_API_PORT=8080
BANKS_API_KEYS=key1,key2       # vacío = sin autenticación
BANKS_LOG_LEVEL=INFO           # DEBUG | INFO | WARNING | ERROR
BANKS_LOG_JSON=false           # true en producción
BANKS_RATE_LIMIT_RPM=60
BANKS_RATE_LIMIT_BURST=10

# ── Tracing ──────────────────────────────────────────────────────────────
BANKS_TRACING=off              # off | otlp
BANKS_OTLP_ENDPOINT=http://localhost:4317
```

---

## 12. Troubleshooting

### "No module named 'banks_rag'" en tests

```bash
# Siempre ejecutar tests con PYTHONPATH apuntando a src/
PYTHONPATH=src pytest tests/unit/ -q
```

### "No current event loop" en tests async

```bash
# Usar asyncio.run() en lugar de get_event_loop().run_until_complete()
# Ver tests/unit/test_sql_catalog.py como ejemplo correcto
```

### El embedder descarga el modelo cada vez

```bash
# Crear la carpeta models/ y poner los modelos en models/<owner>--<name>/
# El sistema los auto-detecta antes de ir a HuggingFace
ls models/
# models/Qwen--Qwen3-VL-Embedding-8B/
# models/BAAI--bge-reranker-v2-m3/
```

### llama-cpp-python no usa GPU

```bash
# Verificar que fue compilado con CUBLAS
python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
# Si da False, recompilar:
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
```

### `/readyz` retorna `degraded`

```bash
# La BD no es accesible o las tablas no existen
psql -c "SELECT COUNT(*) FROM chunks;"
# Si falla, correr la ingesta primero: banks-ingest run --source Datos_prueba/
```

### Rate limit 429 en desarrollo

```bash
# Desactivar rate limit para desarrollo
export BANKS_RATE_LIMIT_RPM=10000
# O no poner API keys (auth + rate limit desactivados)
unset BANKS_API_KEYS
```

### Chunks sin resultados en búsqueda

```bash
# Verificar que la BD tiene datos
psql -c "SELECT COUNT(*), AVG(importance_score) FROM chunks;"

# Verificar que el embedder usa el mismo modelo con el que se vectorizó
psql -c "SELECT embedding_dim FROM chunks LIMIT 1;"
# Debe coincidir con el modelo actual (4096 para Qwen3-VL)
```

### El agente no llama tools / responde directamente

```bash
# Ajustar el system prompt o reducir temperatura
export BANKS_LLM_TEMPERATURE=0.1

# Verificar que el modelo soporta tool-calling (chatml format)
export BANKS_LLM_FAMILY=qwen   # o gemma
```
