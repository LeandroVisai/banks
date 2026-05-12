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
          └────────┬────────┘  (bge-reranker-v2-m3, opcional)
                   │
      ┌────────────▼────────────┐
      │ Agente (loop tool-call) │ LlamaCppEngine (Qwen3.6 / Gemma 4)
      │  • search_documents     │
      │  • search_visuals       │
      │  • discover_query       │ ← catálogo de 23 series financieras
      │  • execute_query        │ ← DuckDB sobre parquets offline
      │  • add_to_context       │
      └────────────┬────────────┘
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

## Catálogo SQL (23 series financieras)

El agente consulta series del Monitor PM vía DuckDB sobre parquets offline:

| Segmento | Queries |
|---|---|
| Mercado cambiario | `usdclp_historico`, `forwards_clp_curva`, `microestructura_fx`, `indice_monedas_latam` |
| Renta fija Chile | `curva_btp_clp`, `curva_btu_uf`, `btp_btu_comparado`, `spc_clp_curva`, `spc_uf_curva` |
| Renta fija EEUU | `curva_ust` |
| Tasas monetarias | `spreads_dap_swap_clp`, `pdbc_bolsa`, `tib_mercado` |
| Tasas internacionales | `sofr_plazos`, `prime_usd_plazos`, `curva_ois_sofr`, `spread_onshore_dap_usd`, `tasas_tado` |
| Política monetaria | `expectativas_tpm_mipr` |
| Liquidez bancaria | `lcr_mx_bancos`, `nsfr_mx_bancos`, `ratio_liquidez_obligaciones` |
| Commodities | `precio_cobre` |

Parquets en `data_pipeline/snapshots/`. Definiciones en [`sql_catalog/catalog.yaml`](sql_catalog/catalog.yaml).

---

## Tests

```bash
PYTHONPATH=src pytest tests/unit/ -q    # 498 tests, <2s, sin BD ni modelos
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

---

## Estructura del repo

```
banks/
├── src/banks_rag/
│   ├── domain/          # dataclasses puras
│   ├── application/     # casos de uso (ingesta, retrieval, agente, evaluación)
│   ├── infrastructure/  # adaptadores (embeddings, llm, postgres, reranker, observability)
│   └── interface/       # FastAPI + CLI
├── tests/
│   ├── unit/            # 498 tests, sin BD ni modelos reales
│   └── integration/     # requieren PostgreSQL + pgvector
├── sql_catalog/         # catalog.yaml — 23 queries DuckDB
├── data_pipeline/       # snapshots/ (parquets del Monitor PM)
├── data/
│   └── golden_set/      # retrieval.jsonl, sql_routing.jsonl, generation.jsonl
├── deploy/
│   ├── systemd/
│   ├── nginx/
│   └── grafana/         # dashboard JSON
├── docs/
│   ├── REFACTOR_PLAN.md
│   └── DEPLOYMENT_H100.md
├── pyproject.toml       # única fuente de deps y entry points
└── Makefile
```
