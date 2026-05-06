# Chatbot Agentic (tool-calling) — Banco Central

Infraestructura **alternativa** al `chatbot/` clásico. Aquí el LLM **decide qué buscar**: recibe definiciones de herramientas (search_documents, get_historical_series, etc.) y las llama iterativamente hasta tener suficiente evidencia para responder.

> Esta carpeta existe en paralelo a `chatbot/` para poder comparar ambos enfoques con la misma base de datos. Corre en el puerto **8081** (vs 8080 del clásico).

**Stack:** Python 3.12 · vLLM AsyncLLMEngine + tool calling · FastAPI · psycopg3 async · PostgreSQL 14 + pgvector · CUDA 12.8

---

## RAG clásico vs Agentic — la diferencia clave

### `chatbot/` (clásico)
```
query → embed → vector+BM25 recall → RRF → MMR → top-k chunks → LLM → respuesta
```
Pipeline determinístico. El LLM **solo razona** sobre lo que el retrieval ya trajo. Una sola pasada.

### `chatbot_calling_tool/` (este)
```
query → LLM decide → tool_call → ejecutar → resultados → LLM razona → ¿más tools? → ...
```
El LLM tiene agencia. Puede:
- Hacer múltiples búsquedas con queries refinadas
- Filtrar por `doc_type`, `year`, `date_range` que el clásico no expone
- Descubrir series con `list_historical_series` antes de pedir datos específicos
- Leer un documento completo si una búsqueda apuntó a algo interesante
- Responder más rápido sin búsquedas si la pregunta no las requiere

Hard cap de **6 iteraciones** para evitar loops infinitos.

---

## Estructura

```
chatbot_calling_tool/
├── README.md
├── requirements.txt
├── .env.example
├── schema/
│   ├── 001_chat_history.sql            ← agent_sessions / agent_messages (con tool_trace)
│   └── 002_historical_series.sql       ← compartido con chatbot/ clásico
├── models/
├── scripts/
│   └── load_series.py
└── app/
    ├── settings.py                     ← + max_agent_iterations
    ├── logging_config.py
    ├── main.py
    ├── api.py                          ← /chat ahora corre el agente
    ├── schemas.py                      ← + ToolTraceEntry, ChunkRef
    ├── db.py                           ← agent_sessions/messages + filtros (doc_type, year)
    ├── llm.py                          ← apply_chat_template(tools=...) + parser tool_calls
    ├── embeddings.py
    ├── prompts.py                      ← system prompt agentic (sin contexto pre-cargado)
    ├── agent.py                        ← EL LOOP: planifica → tool → results → repeat
    └── tools/
        ├── __init__.py                 ← bootstrap del registry
        ├── registry.py                 ← @register decorator + dispatch
        ├── search_documents.py         ← RAG con filtros
        ├── historical_series.py        ← list_historical_series + get_historical_series
        └── document_lookup.py          ← list_documents + get_document_chunks
```

---

## Herramientas disponibles

| Tool | Para qué sirve |
|---|---|
| `search_documents(query, k, doc_type?, year?, date_from?, date_to?)` | Búsqueda híbrida con filtros — el agente la llama refinando |
| `list_historical_series(variable?)` | Descubrir qué series macro existen (TPM, IPC, USD/CLP, …) |
| `get_historical_series(series_id, date_from?, date_to?, limit?)` | Traer datos numéricos de una serie |
| `list_documents(doc_type?, year?)` | Listar documentos por tipo/año |
| `get_document_chunks(filename)` | Leer un documento completo |

Las definiciones JSON-Schema se inyectan al prompt vía `tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS)`. Qwen3 y Kimi K2 soportan esto nativamente.

---

## El loop del agente

```python
# app/agent.py (resumido)
async def run(user_message, history):
    state = AgentState()       # acumula chunks vistos, series, traza
    messages = [system, *history, user]

    for iteration in range(MAX_ITERATIONS):
        result = await llm.generate(messages, tools=TOOL_SCHEMAS)

        if not result.tool_calls:
            return result.text  # ← respuesta final

        # Insertar tool_calls + ejecutar en paralelo
        messages.append(assistant_with_tool_calls)
        for tc in result.tool_calls:
            tool_result = await dispatch(state, tc.name, tc.arguments)
            messages.append(tool_message(tc.id, tool_result))
            state.tool_trace.append(...)

    # max_iterations sin respuesta final
    return fallback_message
```

**Acumulación de citas globales (`AgentState.add_chunk`):** cada chunk recibe un ref `[N]` único y estable durante toda la sesión. Si el agente busca dos veces y trae chunks overlapping, los refs no se duplican. La respuesta final puede citar `[1]`, `[3]`, `[5]` apuntando a chunks vistos en cualquier iteración.

---

## Ejemplo de traza real

**Pregunta:** *"¿Cuál fue la decisión del BCCh en enero 2024 y cómo se comparaba con la inflación de ese momento?"*

```
ITER 1: <tool_call>{"name": "search_documents",
                    "arguments": {"query": "decisión TPM enero 2024",
                                  "doc_type": "COMUNICADO", "year": 2024}}</tool_call>
        → 5 fragmentos, ref [1]–[5]

ITER 2: <tool_call>{"name": "get_historical_series",
                    "arguments": {"series_id": "ipc_anual",
                                  "date_from": "2024-01-01",
                                  "date_to": "2024-03-31"}}</tool_call>
        → 3 observaciones IPC anual ene/feb/mar 2024

ITER 3: <tool_call>{"name": "get_historical_series",
                    "arguments": {"series_id": "tpm",
                                  "date_from": "2023-12-01",
                                  "date_to": "2024-02-29"}}</tool_call>
        → 3 observaciones TPM (cambia de 11.25 → 8.25 → ...)

ITER 4: (sin tool_calls)
        "El Consejo del BCCh acordó recortar la TPM en 100 pb a 7,25% en su
         reunión de enero 2024 [1], decisión consistente con el descenso
         observado de la inflación anual desde ... [2]. Para ese momento, el
         IPC anual estaba en X% (INE), aún por encima de la meta de 3%, pero
         con tendencia descendente clara..."
```

La traza completa se persiste en `agent_messages.tool_trace` (JSONB) — se puede auditar después.

---

## Configuración necesaria para tool calling

El modelo debe soportar tool calling en su chat template. Probados:

| Modelo | Soporte | Notas |
|---|---|---|
| **Qwen3.6-35B-A3B** | ✅ | Formato `<tool_call>` JSON, tokenizer maneja `tools=` |
| **Kimi K2 Instruct** | ✅ | Function calling nativo (compatible OpenAI) |
| Llama 3.1+ | ✅ | También soporta apply_chat_template(tools=) |

`CHATBOT_MAX_MODEL_LEN=16384` por default (más alto que el clásico) porque las trazas con resultados de tools acumuladas crecen rápido.

---

## Quick start

```bash
cd chatbot_calling_tool
cp .env.example .env
# Editar .env: PGPASSWORD, RAG_TABLE_PREFIX, CHATBOT_MODEL_NAME

# Compartir el modelo con chatbot/ (evita duplicar 35GB)
ln -s ../chatbot/models/Qwen3.6-35B-A3B models/Qwen3.6-35B-A3B

# Dependencias
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Lanzar (puerto 8081)
python -m app.main
```

---

## Endpoints

| Método | Ruta | Notas |
|---|---|---|
| `GET` | `/healthz` / `/readyz` | Liveness / Readiness |
| `GET` | `/model-info` | Detalles + `max_iterations` |
| `GET` | `/tools` | Lista las tools y sus JSON-schemas |
| `GET` | `/sessions` | Sesiones recientes |
| `GET` | `/chat/{session_id}` | Historial (con `tool_trace` por mensaje) |
| `POST` | `/chat` | Pregunta → corre el agente → respuesta + traza |

```bash
curl -s http://localhost:8081/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "¿Cuál fue el acuerdo de tasa en enero 2024 y la inflación de ese mes?",
    "include_trace": true
  }' | jq
```

La respuesta incluye:
- `response` — texto final con citas `[N]` validadas
- `iterations` — cuántos ciclos LLM↔tools se ejecutaron
- `tool_trace[]` — cada tool call con args, resultado resumido, duración
- `chunks_seen[]` — todos los chunks acumulados, con su `ref`
- `series_used[]` — series consultadas
- `citations_used[]` — refs efectivamente citadas en la respuesta

---

## Comparación con `chatbot/` clásico

| Aspecto | `chatbot/` | `chatbot_calling_tool/` |
|---|---|---|
| Pasadas al LLM por query | 1 | 1–6 (default 6 cap) |
| Latencia esperada | Baja | 2–4× más alta (varias generaciones) |
| Costo en tokens | Predecible | Variable (depende de iteraciones) |
| Filtros sobre el corpus | No | Sí (doc_type, year, date_range) |
| Multi-step reasoning | No | Sí |
| Determinismo | Alto | Bajo (el modelo decide) |
| Auditabilidad | Buena (sources fijas) | Excelente (traza completa) |
| Fallos típicos | Chunk relevante no entró al top-k | Modelo malinterpreta una tool, loop sin converger |
| Mejor para | Preguntas simples, latencia crítica | Preguntas complejas, comparativas, multi-período |

**Cuándo cada uno gana:**
- *"¿Qué dijo el comunicado de enero 2024?"* → ambos similares; clásico es más rápido.
- *"Compara las decisiones del BCCh entre 2022 y 2024 y relaciónalas con la trayectoria del IPC"* → agentic gana claramente; el clásico no puede iterar.
- *"¿Hay alguna minuta que mencione expectativas de mercado divergentes con la EEE?"* → agentic puede filtrar por doc_type y refinar; el clásico depende del retrieval inicial.

---

## Limitaciones conocidas

- **Latencia**: cada iteración es una generación completa. Una respuesta agentic típica toma 10–30s vs 3–8s del clásico.
- **Tokens**: el contexto crece con cada tool result. Por eso `MAX_TOOL_RESULT_TOKENS=1500` y truncamos resultados largos.
- **Loop divergente**: si el modelo no converge, el cap de iteraciones evita explosiones pero la respuesta puede ser parcial.
- **Tool parsing**: el parser maneja `<tool_call>...</tool_call>` y JSON crudo. Algunos modelos pueden emitir formatos no soportados — revisar `app/llm.py:_parse_tool_calls` si aparecen casos.
- **Streaming**: no implementado en esta versión. La traza completa solo está disponible al finalizar.
