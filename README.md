# Sistema RAG + Agente Multimodal — Banco Central

Sistema de análisis con IA para documentos financieros (Comunicados BCCh, Minutas del Consejo, comunicados de la Fed, reportes JPMorgan, Monitor PM) y series macro/financieras del data warehouse.
Combina **retrieval híbrido + reranking** sobre un corpus multimodal (texto + gráficos extraídos de PDFs) con un **agente de tool-calling** que discierne autónomamente cuándo buscar documentos, ejecutar queries SQL del catálogo curado, o devolver gráficos.

**Diseñado para ejecutarse en GPU H100 80GB sin internet**, con modelos y dependencias pre-descargadas.

> **Estado del repo (mayo 2026)**: en transición desde un prototipo de scripts numerados (`00→05` + dos chatbots paralelos) hacia una arquitectura productiva con Clean Architecture + DDD. El plan completo vive en [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Esta rama (`branch/qwenllamacpp`) ejecuta Fases 0–9 del plan; `main` queda intacto hasta el cutover de Fase 8.

---

## 🎯 Visión objetivo

```
Documentos (PDFs + Excel) + DW SQL Server
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│  Pipeline ingesta (texto + visuales)                         │
│  extract → enrich → vectorize (Qwen3-VL-Embedding-8B)        │
│         → persist (PostgreSQL 14 + pgvector HNSW)            │
└────────────────────┬─────────────────────────────────────────┘
                     │
            ┌────────▼────────┐
            │ Hybrid retrieval│ vector + BM25 → RRF → reranker → MMR
            └────────┬────────┘
                     │
        ┌────────────▼────────────┐
        │ Agente (loop tool-call) │ Qwen3.6-27B / Gemma 4 26B (llama.cpp)
        │  • search_documents     │
        │  • search_visuals       │ ← gráficos embebidos
        │  • discover_query       │ ← top-K del catálogo SQL
        │  • execute_query        │ ← ejecución parametrizada
        │  • historical_series    │
        │  • document_lookup      │
        └────────────┬────────────┘
                     │
              respuesta + citas + visuales
```

### Stack

| Capa | Componente |
|---|---|
| **Embeddings** | `Qwen3-VL-Embedding-8B` (4096-dim, texto + imagen) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| **LLM** | `Qwen3.6-27B-UD-Q4_K_XL.gguf` (default) o `gemma-4-26B-A4B-it.gguf` |
| **Inference** | `llama.cpp` (CUDA 12.x, GGUF, sin Python en runtime) |
| **Vector DB** | PostgreSQL 14 + pgvector (HNSW cosine + GIN para BM25) |
| **Series macro** | Parquets en `data/snapshots/` (DuckDB para SQL sobre parquet) |
| **API** | FastAPI + uvicorn + psycopg3 async |
| **Observability** | structlog (JSON) + Prometheus + request_id tracing |

---

## 📁 Estructura objetivo

```
banks/
├── README.md
├── pyproject.toml                ← paquete instalable banks_rag
├── Makefile                      ← install / lint / test / ingest / serve / eval
├── docs/
│   ├── ARCHITECTURE.md           ← Clean Arch aplicada
│   ├── DEPLOYMENT_H100.md        ← guía offline
│   ├── EVALUATION.md             ← golden set + RAGAS
│   ├── SQL_CATALOG.md
│   └── model_cards/
├── src/banks_rag/
│   ├── domain/                   ← entidades + reglas puras
│   ├── application/              ← casos de uso
│   ├── infrastructure/           ← adaptadores (postgres, llamacpp, embeddings, ...)
│   ├── interface/                ← FastAPI + Typer CLIs
│   ├── domain_knowledge/         ← taxonomy.py, importance_rules
│   └── config/
├── sql_catalog/                  ← 75 queries curadas de querys/Monitor.py
│   ├── catalog.yaml
│   ├── snippets/
│   └── synthetic_examples.jsonl
├── data/
│   ├── raw/                      ← PDFs + Excel
│   ├── images/                   ← gráficos extraídos
│   ├── snapshots/                ← parquets DW (14 ya generados)
│   └── golden_set/               ← (query, expected_*) jsonl
├── scripts/                      ← extract_monitor_queries, benchmarks
├── tests/                        ← unit / integration / e2e
└── deploy/                       ← docker, systemd, nginx
```

Reglas de dependencia: `domain` y `application` **jamás** importan de `infrastructure` (Clean Architecture pura).

---

## 🚀 Quick start (durante el refactor)

```bash
# Setup en máquina de desarrollo
git checkout branch/qwenllamacpp
make install-dev

# Verificar que el esqueleto compila
make lint
make typecheck
make test                          # smoke tests del paquete

# Sistema legacy (sigue funcionando hasta Fase 8)
python3 run.py full                # extract → enrich → vectorize → persist
python3 04_search.py "tasa de interés 2022" 5

# Sistema nuevo (incremental, conforme avancen las fases)
make ingest                        # cuando Fase 1 esté completa
make serve                         # cuando Fase 3 esté completa
make eval                          # cuando Fase 6 esté completa
```

---

## 🗺️ Plan de transformación — fases

| Fase | Entregable | Estado |
|---|---|---|
| 0 · Preparación | branch + skills + pyproject.toml + Makefile + esqueleto | 🚧 En progreso |
| 1 · Refactor estructural | scripts numerados → src/banks_rag (paridad funcional) | ⏳ |
| 2 · Multimodal embeddings | Qwen3-VL para texto + charts; tool `search_visuals` | ⏳ |
| 3 · LLM swappable | llama.cpp + Qwen3.6-27B (luego Gemma 4) | ⏳ |
| 4 · Catálogo SQL agentic | 75 queries de `Monitor.py` → `sql_catalog/`; tools `discover_query` + `execute_query` | ⏳ |
| 5 · Re-ranker + routing | bge-reranker-v2-m3 + query_router (RAG/SQL/VISUAL/HYBRID) | ⏳ |
| 6 · Evaluación | golden set + RAGAS + recall@k; CI gate por regresión | ⏳ |
| 7 · Observability | structlog + Prometheus + tracing + auth + rate limit | ⏳ |
| 8 · Limpieza | eliminar `00–05_*.py`, `chatbot/`, `chatbot_calling_tool/` | ⏳ |
| 9 · Deploy H100 | wheels offline + systemd + validación funcional | ⏳ |

Plan completo: `~/.claude/plans/necesito-que-entiendas-a-curried-stearns.md`.

---

## 📚 Documentación

| Archivo | Contenido |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Capas, reglas de dependencia, Protocols |
| [`CLAUDE.md`](CLAUDE.md) | Instrucciones para Claude Code (legacy + nuevas) |
| [`SETUP_PGVECTOR.md`](SETUP_PGVECTOR.md) | Instalación PostgreSQL 14 + pgvector offline |
| [`enlaces.md`](enlaces.md) | Referencias de descargas de modelos (HF) |
| [`docs/DEPLOYMENT_H100.md`](docs/DEPLOYMENT_H100.md) | Guía paso a paso para H100 (Fase 9) |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Cómo correr el golden set + RAGAS (Fase 6) |
| [`sql_catalog/README.md`](sql_catalog/README.md) | Cómo extender el catálogo SQL (Fase 4) |

---

## 🔧 Configuración (env vars críticas)

```bash
# PostgreSQL
PGHOST=localhost
PGPORT=5432
PGDATABASE=rag_banco
PGUSER=rag_user
PGPASSWORD=...

# RAG
RAG_TABLE_PREFIX=qwen_
RAG_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B
RAG_PURE_TEXT=0

# LLM (Fase 3+)
BANKS_LLM_FAMILY=qwen              # qwen | gemma
BANKS_LLM_MODEL_PATH=/models/Qwen3.6-27B-UD-Q4_K_XL.gguf
BANKS_LLM_N_CTX=16384
BANKS_LLM_N_GPU_LAYERS=-1          # todo a GPU

# API (Fase 7+)
API_HOST=0.0.0.0
API_PORT=8080
BANKS_API_KEYS=key1,key2           # API key auth
LOG_JSON=true
```

---

## 🤝 Cómo contribuir durante el refactor

1. Todo el trabajo del refactor vive en `branch/qwenllamacpp` con commits atómicos por fase. `main` no se toca hasta Fase 8 (cutover).
2. **No tocar `chatbot/`, `chatbot_calling_tool/` ni `00–05_*.py`** hasta Fase 8 — el sistema legacy debe seguir funcionando.
3. Tests obligatorios para cualquier código nuevo en `src/banks_rag/`.
4. `make lint && make typecheck && make test` antes de cada commit.

---

## 🚨 Importante

- `taxonomy.py` (en raíz hoy, en `src/banks_rag/domain_knowledge/taxonomy.py` después de Fase 1) sigue siendo la **única fuente de verdad** para variables económicas, secciones, entidades y patrones.
- `querys/Monitor.py` es el **archivo origen intocable** del que se deriva `sql_catalog/catalog.yaml` (Fase 4). No editarlo directamente — agregar/modificar queries en el catálogo derivado.
