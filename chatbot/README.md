# Chatbot RAG — Banco Central

Chatbot financiero **completamente local** (sin internet, sin APIs externas) sobre el corpus pgvector del pipeline padre.
Diseñado para correr en una H100 con Qwen3.6-35B-A3B (default) o un cluster de 8×H100 con Kimi K2.

**Stack:** Python 3.12 · vLLM en proceso · FastAPI (lifespan) · psycopg3 async · PostgreSQL 14 + pgvector · CUDA 12.8

---

## Por qué esta versión

`infraestructura_chatbot/` (la primera iteración) funciona pero acumuló decisiones rápidas:
conexiones síncronas, `print()` para logs, sin presupuesto de tokens, streaming en texto plano,
y citas `[N]` que el LLM puede inventar libremente.

Esta carpeta (`chatbot/`) es la reescritura limpia con las mismas piezas pero mejor arquitectura.

| Aspecto | `infraestructura_chatbot/` | `chatbot/` (este) |
|---|---|---|
| Configuración | `config.py` con `os.getenv` y casts manuales | Pydantic Settings con validación |
| PostgreSQL | psycopg2, conexión nueva por request | psycopg3 + pool async |
| Logging | `print()` | logging stdlib (texto o JSON) |
| Ciclo de vida vLLM | `@on_event("startup")` (deprecated) | `lifespan` async |
| Streaming | texto plano cumulativo | SSE con `event:` + `data:` y deltas reales |
| Presupuesto de tokens | sin protección — puede explotar el contexto | recorta chunks RAG hasta caber |
| Citas `[N]` | el LLM puede inventar refs | post-procesamiento elimina refs inválidas |
| Análisis de query | regex de keywords mezclado con prompt building | módulo `query_analysis` aislado con `Intent` clasificada |
| Estructura | flat — todo en raíz | paquete `app/` con namespaces |

---

## Estructura

```
chatbot/
├── README.md
├── requirements.txt
├── .env.example
├── schema/
│   ├── 001_chat_history.sql            ← chat_sessions / chat_messages
│   └── 002_historical_series.sql       ← historical_series + 17 series predefinidas
├── models/                             ← descargas de HuggingFace
├── scripts/
│   └── load_series.py                  ← cargar CSVs a historical_data
└── app/
    ├── __init__.py
    ├── main.py                         ← entrypoint
    ├── settings.py                     ← Pydantic Settings tipado
    ├── logging_config.py               ← stdlib logging + JSON opcional
    ├── api.py                          ← FastAPI + lifespan + rutas
    ├── schemas.py                      ← Pydantic request/response
    ├── db.py                           ← pool async + queries
    ├── llm.py                          ← vLLM AsyncLLMEngine + tokenización
    ├── embeddings.py                   ← sentence-transformers en threadpool
    ├── retrieval.py                    ← vector + BM25 + RRF + MMR
    ├── query_analysis.py               ← intención + variables + fechas
    ├── sql_context.py                  ← series históricas → texto
    └── prompts.py                      ← templates + token budget + citas
```

---

## Arquitectura del flujo

```
POST /chat
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. query_analysis.analyze(message)                               │
│    → intent: quantitative|qualitative|definition|comparison      │
│    → variables: [TASA_INTERES, INFLACION, ...]                   │
│    → date_from / date_to                                         │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼  (en paralelo)
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ 2a. retrieval.retrieve()     │  │ 2b. sql_context.build()      │
│   embed_query (threadpool)   │  │   (solo si intent cuantitat. │
│   ↓                          │  │    + variables detectadas)   │
│   vector_recall ║ lex_recall │  │   fetch_series_meta          │
│   ↓                          │  │   fetch_series_rows          │
│   RRF → MMR → boost          │  │   format_block               │
│   ↓                          │  │   ↓                          │
│   chunks                     │  │   <datos_historicos>...      │
└──────────────────────────────┘  └──────────────────────────────┘
   │                                  │
   └──────────────┬───────────────────┘
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. prompts.build_messages()                                      │
│    system + <contexto> + <datos_historicos> + history + user     │
│    + truncate_chunks_to_budget(prompt_token_budget)              │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. llm.generate() o llm.generate_stream()                        │
│    vLLM AsyncLLMEngine en proceso (sin HTTP)                     │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. prompts.verify_citations()                                    │
│    elimina [N] con N > len(sources) (anti-hallucination)         │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼
        respuesta + sources + citations_used + intent + métricas
              persistencia en chat_messages
```

---

## Instalación

### 1. Modelo LLM en `models/`

En un equipo con internet, descarga desde HuggingFace y copia los archivos:

- Qwen3.6-35B-A3B → `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/main`
- Kimi K2 → `https://huggingface.co/moonshotai/Kimi-K2-Instruct/tree/main`

```
chatbot/models/
└── Qwen3.6-35B-A3B/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── model-*.safetensors
```

### 2. Dependencias

```bash
# Con internet
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Sin internet (descarga wheels en otro equipo)
pip download -r requirements.txt -d ./wheels/
pip install --no-index --find-links ./wheels/ -r requirements.txt
```

### 3. Variables de entorno

```bash
cp .env.example .env
# Edita .env con tus credenciales y prefijo de tablas RAG
```

Lo mínimo a ajustar:

```
PGPASSWORD=...
RAG_TABLE_PREFIX=qwen_      # o gemma_ — debe coincidir con tu pipeline padre
CHATBOT_MODEL_NAME=Qwen3.6-35B-A3B
```

### 4. Schemas SQL (idempotente — el lifespan los aplica solo)

Si quieres aplicarlos manualmente:

```bash
psql -d rag_banco -f schema/001_chat_history.sql
psql -d rag_banco -f schema/002_historical_series.sql
```

### 5. Cargar series históricas

El catálogo de 17 series ya está creado por el schema. Para popularlas:

```bash
# CSV con columnas: date,value[,notes]
python -m scripts.load_series tpm data/tpm.csv
python -m scripts.load_series ipc_anual data/ipc.csv --replace
```

---

## Uso

```bash
cd chatbot
python -m app.main
```

Al arrancar el `lifespan`:

1. Abre el pool de PostgreSQL
2. Aplica los schemas (idempotente)
3. Carga el embedding model y vLLM **en paralelo** (en GPU)
4. Queda escuchando en `0.0.0.0:8080`

---

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/healthz` | Liveness — el proceso responde |
| `GET` | `/readyz` | Readiness — LLM cargado + DB OK |
| `GET` | `/model-info` | Detalles del modelo + `prompt_version` |
| `GET` | `/historical-series` | Catálogo de series con cobertura de fechas |
| `GET` | `/sessions` | Sesiones recientes |
| `GET` | `/chat/{session_id}` | Historial completo de una sesión |
| `POST` | `/chat` | Mensaje + respuesta (JSON o SSE stream) |
| `GET` | `/docs` | Swagger UI |

### Ejemplos

```bash
# Pregunta no-streaming
curl -s http://localhost:8080/chat -H 'Content-Type: application/json' -d '{
    "message": "¿Cuál fue la decisión de tasa del BCCh en julio 2022 y cómo evolucionó la TPM ese año?"
}' | jq

# La respuesta incluye:
# - response (con citas [N] verificadas)
# - sources (chunks RAG usados)
# - historical_series (series SQL inyectadas — TPM, IPC...)
# - citations_used (qué refs aparecen efectivamente)
# - intent (quantitative en este caso)
# - variables_detected (TASA_INTERES, INFLACION...)
# - token_count + latency_ms

# Streaming SSE
curl -N http://localhost:8080/chat -H 'Content-Type: application/json' -d '{
    "message": "Explica el panorama actual de inflación",
    "stream": true
}'
# event: meta
# data: {"session_id":"...", "intent":"qualitative", "sources":[...], ...}
#
# event: delta
# data: {"text":"En el último "}
#
# event: delta
# data: {"text":"trimestre, "}
# ...
# event: done
# data: {"citations_used":[1,3], "latency_ms": 4231, "response_length": 850}
```

---

## Series históricas predefinidas

| `series_id` | Nombre | Unidad | Frecuencia | Variable taxonomía |
|---|---|---|---|---|
| `tpm` | Tasa de Política Monetaria | % anual | mensual | TASA_INTERES |
| `ipc_anual` / `ipc_mensual` / `ipcx_anual` | IPC (varios cortes) | var % | mensual | INFLACION |
| `usdclp_spot` | Tipo de cambio USD/CLP | CLP | diario | TIPO_CAMBIO |
| `pib_trimestral` / `imacec_mensual` | Actividad económica | var % a/a | trim/mensual | PIB |
| `bcu_5y` / `btp_5y` | Bonos BCCh 5y | % anual | diario | TASAS_LARGO_PLAZO |
| `exp_inflacion_12m` / `_24m` / `exp_tpm_12m` | Expectativas EEE | % anual | mensual | EXPECTATIVAS |
| `precio_cobre` / `precio_petroleo_wti` | Commodities | USD | diario | COMMODITIES |
| `fed_funds_rate` | Fed Funds (objetivo) | % anual | mensual | TASA_INTERES |
| `cds_chile_5y` | CDS Chile 5y | pb | diario | RIESGO_CREDITO |
| `desempleo` | Tasa de desocupación | % | mensual | MERCADO_LABORAL |

Para añadir una serie nueva:

```sql
INSERT INTO historical_series
    (series_id, series_name, unit, frequency, source, economic_variable)
VALUES
    ('imacec_no_minero', 'IMACEC no minero', 'var % a/a', 'mensual', 'BCCh', 'PIB');
```

Luego en `app/sql_context.py`, agrega la entrada en `VARIABLE_TO_SERIES` para que la detección automática la asocie a la variable correspondiente.

---

## Variables de entorno principales

Definidas y validadas en `app/settings.py`. Ver `.env.example` para la lista completa.

| Variable | Default | Notas |
|---|---|---|
| `CHATBOT_MODEL_NAME` | `Qwen3.6-35B-A3B` | Subcarpeta en `models/` |
| `CHATBOT_QUANTIZATION` | `fp8` | Vacío o `none` para BF16 |
| `CHATBOT_TENSOR_PARALLEL` | `1` | `8` para Kimi K2 |
| `CHATBOT_MAX_MODEL_LEN` | `8192` | Tokens totales (prompt + respuesta) |
| `CHATBOT_MAX_TOKENS` | `1024` | Tokens de la respuesta |
| `PROMPT_TOKEN_BUDGET` | `6500` | Recorta chunks RAG si el prompt excede esto |
| `RAG_TABLE_PREFIX` | (vacío) | `qwen_` o `gemma_` según pipeline padre |
| `RAG_TOP_K` | `5` | Chunks finales tras MMR |
| `LOG_JSON` | `false` | `true` → logs estructurados para ELK/Loki |
| `PG_POOL_MIN` / `PG_POOL_MAX` | `2` / `8` | Conexiones del pool async |

---

## Diferencias clave vs `infraestructura_chatbot/`

1. **Token budgeting real** — `prompts._truncate_chunks_to_budget` cuenta tokens del prompt completo (chat template aplicado) y retira chunks de menor importancia hasta caber.
2. **SSE streaming** — `text/event-stream` con `event: meta` (fuentes y metadata antes del primer token), `event: delta` (tokens), y `event: done` (cierre con métricas). El cliente puede pintar las fuentes inmediatamente.
3. **Citation guard** — `prompts.verify_citations` parsea `[N]` en la respuesta y elimina las que el LLM inventó (N > n_sources).
4. **Análisis de intención** — `query_analysis.analyze` clasifica la query en `quantitative|qualitative|definition|comparison` y eso decide si vale la pena llamar a `sql_context`.
5. **Persistencia rica** — `chat_messages` guarda `rag_sources`, `historical_series`, `token_count`, `latency_ms`. Sirve para auditar la calidad del sistema en el tiempo.
6. **Pool async + paralelismo** — vector_recall y lexical_recall corren `asyncio.gather`. El pool psycopg3 evita el handshake TCP en cada request.

---

## Qué falta / mejoras posibles

- Cross-encoder reranker (precisión ↑ latencia ↑) — añadir `app/reranker.py` opcional
- Cache de embeddings para queries repetidas (Redis o in-memory LRU)
- Métricas Prometheus (`/metrics`)
- Auth (API key o JWT) si se expone fuera del intranet
- Carga de series desde APIs (BCCh, INE) en lugar de CSVs estáticos
