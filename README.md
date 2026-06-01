# Banks RAG — Sistema RAG + Agente Multimodal · Banco Central de Chile

Sistema de análisis con IA para documentos financieros del BCCh: Comunicados, Minutas del Consejo, Fed Statements, research JPMorgan y Monitor PM. Combina **retrieval híbrido + cross-encoder reranking** sobre un corpus multimodal (texto + gráficos de PDFs) con un **agente de tool-calling** que discierne autónomamente cuándo buscar documentos, ejecutar queries SQL del catálogo curado, o devolver gráficos.

**Diseñado para ejecutarse en GPU H100 80GB sin internet**, con modelos y dependencias pre-descargadas.

---

## Arquitectura

```
Documentos (PDFs + Excel Monitor PM)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  Pipeline ingesta (banks-ingest)                        │
│  extract → enrich → vectorize (Qwen3-VL-Embedding-8B)  │
│  → persist (PostgreSQL 16 + pgvector HNSW cosine)       │
└──────────────────┬──────────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │ hybrid_search   │ vector + BM25 → RRF → MMR → reranker
          └────────┬────────┘  (bge-reranker-v2-m3, activo por defecto)
                   │
      ┌────────────────────────────────────────────────────────┐
      │ Router determinista (router-v1)                        │
      │  selecciona ≥1 especialista por vocabulario            │
      └──────────┬─────────────────────────────────────────────┘
                 │ despacha en paralelo
      ┌──────────▼──────────────────────────────────────────────┐
      │ 8 Sub-agentes especialistas (LlamaCppEngine)            │
      │                                                         │
      │  Mercado Cambiario (FX)  · No Residentes                │
      │  AFP · Fondos Mutuos (FFMM)                            │
      │  Renta Fija · Liquidez & Balance                       │
      │  Analista Documentos · Analista Política Monetaria     │
      │                                                         │
      │  Tools cuantitativos: discover_query · execute_query   │
      │    compute_variation · compute_spread                  │
      │    compute_composition · compute_aggregate             │
      │    get_series_stats · detect_anomaly                   │
      │                                                         │
      │  Tools documentales: search_documents · search_visuals │
      │    list_documents · get_document_chunks                │
      │    compare_meetings · get_recent_policy_decisions      │
      └────────────┬────────────────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │   FastAPI API   │ /v1/chat · /v1/search · /metrics
          └─────────────────┘
```

### Capas (Clean Architecture)

| Capa | Paquete | Responsabilidad |
|---|---|---|
| Domain | `banks_rag.domain` | Dataclasses puras: `Document`, `Chunk`, `SearchResult`, `ParsedQuery` |
| Application | `banks_rag.application` | Casos de uso: ingesta, retrieval, agente, evaluación |
| Infrastructure | `banks_rag.infrastructure` | Adaptadores: embeddings, LLM, PostgreSQL, reranker, observabilidad |
| Interface | `banks_rag.interface` | FastAPI + CLI (`banks-ingest`, `banks-search`, `banks-eval`) |

---

## Quickstart

### Instalación

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[api]"      # producción
pip install -e ".[dev]"      # desarrollo + tests
```

### Variables de entorno mínimas

```bash
export PGHOST=localhost PGDATABASE=rag_banco PGUSER=<user> PGPASSWORD=<pass>
export BANKS_LLM_FAMILY=mock   # o 'qwen' con BANKS_LLM_MODEL_PATH=<ruta.gguf>
```

**Windows (dev box / RTX 3080):** usar `run_local_cpu.ps1` — fija variables de entorno y levanta la API directamente.

```powershell
.\run_local_cpu.ps1
```

### Ingesta

```bash
banks-ingest run --source Datos_prueba/
```

### Búsqueda

```bash
banks-search "política monetaria 2024" --k 5

# Vía API
uvicorn banks_rag.interface.api.main:create_app --factory --port 8080
curl http://localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "TPM actual", "k": 5}'
```

### Frontend (interface2)

`interface2/` es un dashboard estático (HTML + JS + ApexCharts) que la API monta en `/`. **Mismo origen que el backend** — abrir directamente `http://localhost:8080/`, no servir aparte.

```bash
# Con el paquete instalado (pip install -e ".[api]")
python -m banks_rag.interface.api.main

# Sin instalar (sandbox/dev rápido)
PYTHONPATH=src python3 -m banks_rag.interface.api.main
```

Notas:
- La pill superior derecha pasa por `SIN CONEXIÓN → LLM CARGANDO → OK` mientras `llama-cpp` carga. Los charts del catálogo no esperan al LLM; sólo el panel "Agente IA" sí.
- Los charts usan `/v1/query/{dataset_id}` (DuckDB sobre parquets) y los KPIs son hardcoded en `kpi.js` — no requieren PostgreSQL para renderizar.

### Verificación de sub-agentes

```bash
# Verifica los 8 especialistas contra el LLM del .env (local o H100)
python scripts/verify_subagents.py

# Solo un especialista
python scripts/verify_subagents.py --subagent fx

# Con techo de iteraciones
python scripts/verify_subagents.py --max-iters 6
```

### Evaluación

```bash
make eval          # golden set completo → eval_report.md
make eval-ci       # gate CI (falla si recall@5 cae >5%)
```

---

## Modelos

| Rol | Modelo | Tamaño | Ubicación |
|---|---|---|---|
| Embedding texto + imagen | `Qwen/Qwen3-VL-Embedding-8B` | ~16GB | `models/Qwen--Qwen3-VL-Embedding-8B/` |
| Reranker | `BAAI/bge-reranker-v2-m3` | ~1GB | `models/BAAI--bge-reranker-v2-m3/` |
| LLM principal | `Qwen3.6-27B-UD-Q4_K_XL.gguf` | ~16GB | `models/` |
| LLM alternativo | `gemma-4-26B-A4B-it.gguf` | ~14GB | `models/` |

El pipeline auto-detecta `models/<owner>--<name>/` antes de descargar desde HuggingFace.

---

## Catálogo de parquets (107 datasets)

El agente consulta datos del Monitor PM vía DuckDB sobre parquets offline. Los 107 datasets están todos consultables — los 7 snapshots sin columna de fecha (`posicion_rfl_afp`, `cambiario_afp`, `attribution`, `dcv_composicion_ffmm`, `fixing_*`, `variacion_dcv_afp`) usan `build_fetch_sql` sin filtro temporal; los de serie temporal usan `date_column()` estricta para analytics.

| Segmento | Descripción |
|---|---|
| `mercado_cambiario` | USD/CLP, forwards, microestructura FX, índice monedas LATAM |
| `posiciones_cambiarias` | RFL AFP, posición cambiaria AFP, no residentes |
| `fondos_pension` | AFP composición, FFMM composición, DCV, atribución |
| `renta_fija_chile` | Curvas BTP/BTU, SPC CLP/UF, spreads crédito |
| `instrumentos_bcch` | PDBC, TIB, DAP vs swap CLP |
| `renta_fija_eeuu` | Curva UST, OIS SOFR, SOFR plazos, PRIME USD |
| `liquidez_bancaria` | LCR/NSFR por banco, ratio liquidez/obligaciones |
| `balance_bancario` | Depósitos, colocaciones, fixing, variación DCV |
| `tasas_internacionales` | SOFR, PRIME, OIS, spread onshore DAP USD, TADO |
| `politica_monetaria` | Expectativas TPM (MiPr) |
| `commodities` | Precio cobre |

Parquets en `data_pipeline/parquet/`. Esquemas completos (id, columnas, tipos, enums, `date_range`) en [`sql_catalog/parquet_catalog.yaml`](sql_catalog/parquet_catalog.yaml). La SQL la arma siempre la tool (`_parquet_query.build_fetch_sql`); el LLM solo elige `dataset_id` + filtros — nunca escribe SQL.

---

## Tests

```bash
PYTHONPATH=src pytest tests/unit/ -q    # 722 tests, <2s, sin BD ni modelos
PYTHONPATH=src pytest tests/ -q         # + integración (requiere PostgreSQL)
```

---

## Observabilidad

- **Logs**: JSON estructurado con `request_id`, `session_id`, `tool_call_id` (`BANKS_LOG_JSON=true`).
- **Métricas**: `GET /metrics` (Prometheus). Dashboard en [`deploy/grafana/banks_rag_dashboard.json`](deploy/grafana/banks_rag_dashboard.json).
- **Tracing**: OpenTelemetry OTLP opcional (`BANKS_TRACING=otlp`).
- **Rate limit**: token bucket por API key (`BANKS_RATE_LIMIT_RPM=60`).
- **Auth**: `BANKS_API_KEYS=key1,key2`. `/healthz` y `/metrics` son públicos.

---

## Deploy en H100

Ver [`docs/DEPLOYMENT_H100.md`](docs/DEPLOYMENT_H100.md): systemd, nginx, logrotate, backup, Prometheus, troubleshooting.

Para el setup inicial del runtime en el servidor (variables de entorno, venv, CUDA DLLs, activar el venv) ver [`docs/SETUP_SERVIDOR.txt`](docs/SETUP_SERVIDOR.txt).

---

## Estructura del repo

```
banks/
├── src/banks_rag/
│   ├── domain/          # dataclasses puras
│   ├── application/
│   │   ├── agent/       # router-v1, 8 sub-agentes, tools, prompts
│   │   ├── ingestion/   # extract → enrich → vectorize → load
│   │   ├── retrieval/   # hybrid_search, query_parser, fusion, reranker
│   │   └── evaluation/  # retrieval_metrics, ragas_runner, golden set
│   ├── infrastructure/  # adaptadores (embeddings, llm, postgres, reranker, observability)
│   └── interface/       # FastAPI + CLI
├── tests/
│   ├── unit/            # 722 tests, sin BD ni modelos reales
│   └── integration/     # requieren PostgreSQL + pgvector
├── scripts/
│   └── verify_subagents.py   # verificación e2e de los 8 sub-agentes
├── sql_catalog/         # parquet_catalog.yaml — 107 datasets con esquema
├── data_pipeline/       # parquet/ — 107 datasets crudos del DW (única fuente)
├── data/
│   ├── golden_set/      # retrieval.jsonl, sql_routing.jsonl, generation.jsonl
│   └── chat_logs/       # JSONL diario de turnos del agente (gitignored)
├── deploy/
│   ├── systemd/
│   ├── nginx/
│   └── grafana/         # dashboard JSON
├── docs/
│   ├── REFACTOR_PLAN.md
│   ├── DEPLOYMENT_H100.md
│   └── SETUP_SERVIDOR.txt    # guía de entorno H100 (venv, CUDA, .env)
├── run_local_cpu.ps1    # script Windows dev box (fija env vars + levanta API)
├── pyproject.toml       # única fuente de deps y entry points
└── Makefile
```
