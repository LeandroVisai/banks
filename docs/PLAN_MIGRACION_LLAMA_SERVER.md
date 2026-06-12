# Plan de migración: llama-cpp-python in-process → llama-server (Windows nativo, 1×H100)

> **Objetivo**: eliminar la serialización de requests entre usuarios (cola en `/v1/chat`) y
> reducir la latencia por consulta del agente, **sin tocar la capa de agente**
> (router-v1, especialistas, guards anti-alucinación) y **sin salir de Windows nativo**.
>
> **Estrategia**: el LLM deja de vivir dentro del proceso FastAPI. Se sirve con el binario
> oficial `llama-server` (releases Windows+CUDA de ggml-org) con **continuous batching**,
> **prefix/KV-cache reuse** y **plantilla Jinja del GGUF** (tool-calling nativo de Qwen).
> La app se conecta vía un nuevo adapter `OpenAICompatEngine` que implementa el Protocol
> `LLMEngine` existente. El backend in-process queda como fallback configurable.

---

## 0. Contexto del problema (por qué)

- `LlamaCppEngine` (src/banks_rag/infrastructure/llm/llama_cpp_engine.py) usa
  `ThreadPoolExecutor(max_workers=1)`: **todas** las llamadas al LLM se serializan.
  El `asyncio.gather` de especialistas en `conversation_loop.run_agent` es paralelismo
  ilusorio: 2 usuarios × 2 especialistas × 4 iteraciones + síntesis = ~18 llamadas en fila.
- Cada iteración del loop re-procesa el prompt completo (prefill) porque no hay
  reutilización de KV-cache entre llamadas.
- `llama-server` resuelve ambos: slots paralelos con continuous batching (decode de varias
  secuencias en el mismo forward pass) y caché de prefijo por slot (`--cache-reuse`).
- Bonus: gramáticas/`json_schema` server-side (tool calls JSON válidos por construcción),
  `/tokenize` real (reemplaza la heurística ×1.15 de `count_tokens`), streaming SSE,
  `/metrics` Prometheus, y el modelo sobrevive a reinicios de la API.

## 0.1 Decisiones de modelo (VRAM H100 = 80 GB)

| Componente | Elección | VRAM aprox |
|---|---|---|
| LLM principal | **Qwen3.6-27B-UD-Q6_K_XL.gguf** (25.6 GB) — sube desde Q4_K_XL | ~26 GB |
| KV cache | `--ctx-size 131072 --parallel 4` (32k/slot), `-ctk q8_0 -ctv q8_0`, flash-attn | ~8–16 GB |
| Embedder | Qwen3-VL-Embedding-8B (in-process, igual que hoy) | ~17 GB |
| Reranker | jina-reranker-v3 (igual que hoy) | ~3 GB |
| Margen/overhead | activaciones, CUDA ctx, fragmentación | ~8 GB |

Total estimado ≈ 60–70 GB → cabe. **Verificar con `nvidia-smi` en Fase 1 y ajustar**:
si aprieta, bajar a `--ctx-size 65536 --parallel 4` (16k/slot, equivalente al n_ctx actual).

- Justificación del Q6: el output del sistema son **cifras para gerentes del BCCh**; Q6_K_XL
  reduce el error de quantización vs Q4 con VRAM de sobra. Q8_K_XL (35.3 GB) es viable pero
  dejarlo para un benchmark posterior (Fase 6).
- Fallback Gemma: `gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf` (21.2 GB) si se quiere paridad de
  calidad; se sirve en un segundo puerto o con `llama-swap` (Fase 7, opcional).

## 0.2 Invariantes que NO se rompen

1. `LLMEngine` es un Protocol (`infrastructure/llm/base.py`) — el adapter nuevo lo implementa.
2. Tests unitarios corren sin BD, sin modelos, sin red real (`PYTHONPATH=src pytest tests/unit/ -q`).
3. `_strip_think` (cierre colgante `</think>` de la plantilla Qwen) y `parse_tool_calls`
   (fallback texto) se reutilizan en el adapter.
4. Familia `gemma`: se conserva `_flatten_for_gemma` + tools-como-texto (extraer a módulo común).
5. `MODE_PROFILES`, sampling por componente, síntesis sin thinking: sin cambios.
6. Modelos en `models/<owner>--<name>/`; nunca rutas absolutas hardcodeadas.
7. Chat log JSONL, métricas y citación: sin cambios.

---

## Fase 0 — Baseline y binarios (sin tocar código)

1. **Baseline de rendimiento** (para comparar al final):
   - `make eval` → guardar `eval_report.md` actual.
   - Crear `scripts/load_test_chat.py`: N usuarios concurrentes (2, 4, 8) → POST `/v1/chat`
     con 6–8 preguntas representativas del golden set; reporta p50/p95 de latencia E2E,
     tokens/s agregados y tasa de error. Correrlo contra el stack actual y guardar JSON en
     `data/bench/baseline_inprocess.json`.
2. **Descargar llama-server**: release oficial Windows CUDA de
   https://github.com/ggml-org/llama.cpp/releases (zip `llama-bXXXX-bin-win-cuda-x64.zip`,
   incluye cudart). Descomprimir en `deploy/llama-server/` (añadir a `.gitignore` los binarios).
3. **Descargar el GGUF Q6**: `unsloth/Qwen3.6-27B-GGUF` → `Qwen3.6-27B-UD-Q6_K_XL.gguf`
   en `models/unsloth--Qwen3.6-27B-GGUF/` (convención del repo).
4. Smoke test manual: `llama-server.exe -m <gguf> -ngl 999 -c 16384 --port 8081` +
   `curl http://localhost:8081/health`.

**Criterio de salida**: baseline guardado; servidor levanta y responde `/health` y un
`/v1/chat/completions` simple.

---

## Fase 1 — llama-server como servicio Windows

### 1.1 Script de arranque `deploy/start_llama_server.ps1`

Parámetros leídos de variables de entorno (mismos nombres que usará la app):

```powershell
# deploy/start_llama_server.ps1
$model   = $env:BANKS_LLAMA_SERVER_MODEL    # ruta al .gguf (default models\unsloth--Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q6_K_XL.gguf)
$port    = $env:BANKS_LLAMA_SERVER_PORT     # default 8081
$slots   = $env:BANKS_LLAMA_SERVER_SLOTS    # default 4
$ctx     = $env:BANKS_LLAMA_SERVER_CTX      # default 131072 (TOTAL; por slot = ctx/slots)

& "$PSScriptRoot\llama-server\llama-server.exe" `
  --model $model `
  --n-gpu-layers 999 `
  --ctx-size $ctx `
  --parallel $slots `
  --cont-batching `
  --cache-reuse 256 `
  --flash-attn on `
  --cache-type-k q8_0 --cache-type-v q8_0 `
  --jinja `
  --host 127.0.0.1 --port $port `
  --metrics `
  --no-warmup:$false
```

Notas técnicas:
- `--jinja` usa la **plantilla embebida del GGUF** → tool-calling nativo de Qwen3.6
  (renderiza `tools`, `tool_calls` y rol `tool` como `<tool_response>`), igual que el
  `chat_format=None` actual. Sin esto el servidor cae a un template genérico y se repite
  el bug histórico de "discover_query en bucle".
- `--cache-reuse 256` habilita reutilización de KV por trozos → cada iteración del loop
  del especialista solo prefillea el delta (tool result nuevo), no todo el historial.
- `--parallel 4`: 4 slots = 4 secuencias simultáneas (2 usuarios × 2 especialistas, o
  4 usuarios en modo rápido). El selector de slot usa longest-prefix-match → los
  especialistas con el mismo system prompt comparten prefijo.
- KV q8_0 requiere flash-attn activo (para V). Si hay artefactos numéricos raros en
  outputs largos, primer rollback: quitar `--cache-type-*` (KV fp16) y bajar `--ctx-size`.
- Solo `127.0.0.1`: el servidor NO se expone; la única puerta sigue siendo FastAPI
  (auth por API keys, rate limit — sin cambios).

### 1.2 Servicio Windows

- Opción A (recomendada): **Task Scheduler** al boot, con reinicio en fallo
  (`schtasks` o XML en `deploy/llama_server_task.xml`).
- Opción B: **NSSM** (`nssm install banks-llama-server ...`) si está permitido instalarlo.
- Documentar en `docs/SETUP_LLAMA_SERVER.md`: instalación, arranque, logs, troubleshooting
  (DLLs CUDA, VRAM, cómo leer `/metrics` y `/slots`).

**Criterio de salida**: servicio levantado al boot; `nvidia-smi` confirma presupuesto VRAM;
2 curls simultáneos a `/v1/chat/completions` se atienden en paralelo (verificar con
`/slots` y tiempos solapados).

---

## Fase 2 — Adapter `OpenAICompatEngine`

### 2.1 Nuevo módulo `src/banks_rag/infrastructure/llm/openai_compat_engine.py`

Implementa el Protocol `LLMEngine` contra cualquier endpoint OpenAI-compatible
(llama-server hoy; TabbyAPI o un fork de vLLM mañana, sin tocar la app):

- `__init__(base_url, *, model_name, family, temperature, top_p, max_tokens, timeout_s, api_key=None)`
  — `family` con el mismo auto-detect por nombre (`_detect_family`, extraído a módulo común).
- `load()` → `GET {base_url}/health` con reintentos/backoff (el servidor puede estar
  cargando el modelo); fija `loaded=True`. `unload()` → no-op (el servidor es externo).
- `generate(messages, *, tools, temperature, top_p, max_tokens)`:
  - **httpx.AsyncClient** (añadir `httpx>=0.27` a pyproject; ya viene transitivo con FastAPI)
    → `POST /v1/chat/completions`. **Sin executor**: concurrencia asyncio real.
  - family `qwen` → pasa `messages` + `tools` + `tool_choice="auto"` tal cual (el servidor
    aplica la plantilla del GGUF). family `gemma` → reutiliza `_flatten_for_gemma` +
    `_format_tools_as_text` y NO manda `tools`.
  - Respuesta: leer `message.content`, `message.tool_calls` (nativos), `finish_reason`,
    `usage.completion_tokens`. Si el servidor separa razonamiento en
    `message.reasoning_content`, ignorarlo para `text`; aplicar igualmente `_strip_think`
    sobre `content` (cubre ambos formatos). Si no hay tool_calls nativos →
    `parse_tool_calls(cleaned)` como hoy.
  - Mapear a `GenerationResult` idéntico al engine actual (mismo contrato, mismos tests
    de comportamiento).
  - Errores: timeout/conexión → excepción tipada (`LLMUnavailableError`) que la route
    convierte en 503 con mensaje claro; NUNCA respuesta vacía silenciosa.
- `count_text_tokens(text)` → `POST /tokenize` (cachear con LRU pequeño los textos
  repetidos, p.ej. system prompts). `count_tokens(messages, tools)` →
  `POST /apply-template` si está disponible (+ `/tokenize` del resultado) y, si no,
  conservar la heurística ×1.15 como fallback.
- `info()` → incluye `base_url`, `backend: "openai_compat"`, y lo que devuelva
  `GET /props` del servidor (nombre real del modelo cargado, n_ctx por slot).

### 2.2 Refactor mínimo compartido

- Extraer de `llama_cpp_engine.py` a `src/banks_rag/infrastructure/llm/_common.py`:
  `_strip_think`, `_detect_family`, `_parse_args`, `_format_tools_as_text`,
  `_flatten_for_gemma`. Ambos engines los importan (cero duplicación; `llama_cpp_engine`
  queda funcionalmente idéntico).

### 2.3 Settings y factory

En `config/settings.py`:

```python
# ── LLM backend ──
llm_backend: Literal["inprocess", "openai_compat"] = "inprocess"  # BANKS_LLM_BACKEND
llm_base_url: str = "http://127.0.0.1:8081"                       # BANKS_LLM_BASE_URL
llm_request_timeout_s: float = 300.0                              # BANKS_LLM_REQUEST_TIMEOUT_S
```

En el lifespan (`interface/api/main.py`): si `llm_backend == "openai_compat"` construir
`OpenAICompatEngine` (no requiere `llm_model_path`; `llm_family` sigue mandando para el
flatten de Gemma), si no, `LlamaCppEngine` como hoy. `mock` sigue igual.
**Default `inprocess`** → la migración es opt-in y el rollback es una variable de entorno.

### 2.4 Tests unitarios (sin red)

- `tests/unit/infrastructure/llm/test_openai_compat_engine.py` con transport falso de httpx
  (`httpx.MockTransport`, sin dependencias nuevas): casos —
  tool_calls nativos; `<tool_call>` en texto; `</think>` colgante; gemma flatten
  (snapshot del payload enviado: sin `tools`, con prefijo de sistema fusionado);
  timeout → `LLMUnavailableError`; `/health` con reintentos; `count_text_tokens` vía
  `/tokenize` y fallback.
- Tests de `_common.py` (mover los existentes de strip/parse/flatten).

**Criterio de salida**: `PYTHONPATH=src pytest tests/unit/ -q` verde y rápido; con
`BANKS_LLM_BACKEND=openai_compat` la API responde `/v1/chat` end-to-end contra el servidor.

---

## Fase 3 — Concurrencia real en la app

1. `chat_concurrency` (semáforo de `/v1/chat`): subir default a **4** cuando
   `llm_backend == "openai_compat"` (alinear con `--parallel`). Documentar la regla:
   `chat_concurrency × max_specialists ≤ slots` idealmente; si se excede, llama-server
   encola internamente (degradación suave, no error).
2. Verificar que **nada más** del camino caliente bloquee el event loop:
   - `dispatch` de tools ya es async; los `compute_*`/DuckDB corren en threads — OK.
   - El reranker y el embedder de query son sync (torch): confirmar que se llaman vía
     `asyncio.to_thread`/executor en `search_documents` (si no, envolverlos — son las
     únicas piezas GPU-sync que quedan en el proceso de la API).
3. `DEFAULT_TOOL_TIMEOUT_S`/`DEFAULT_DELEGATE_TIMEOUT_S`: revisar contra los nuevos tiempos
   reales (con batching, una iteración bajo carga puede tardar algo más que sola).
4. Métricas: histograma Prometheus de latencia por llamada LLM etiquetado por
   `component` (specialist/synthesis) y `backend` — ya existe infraestructura de métricas;
   añadir el label.

**Criterio de salida**: `scripts/load_test_chat.py` con 2 y 4 usuarios concurrentes muestra
solapamiento real (p95 con 2 usuarios ≪ 2× p95 de 1 usuario).

---

## Fase 4 — Streaming SSE de la síntesis (opcional, alto impacto UX)

1. Añadir `stream: bool = False` al request de `/v1/chat` (schema Pydantic).
2. En `run_agent`: especialistas igual que hoy (no se streamean); la **síntesis** acepta
   un modo streaming → el adapter pasa `stream=true` al servidor y re-emite deltas.
3. Route: `StreamingResponse` SSE (`text/event-stream`) con eventos
   `{"type":"delta","text":...}` y un evento final `{"type":"done", ...AgentResult sin el texto...}`
   para que el frontend pinte citas/series al cierre. El chat log JSONL se escribe al final
   con la respuesta completa (sin cambios de formato).
4. `verify_citations`/`verify_numbers` corren sobre el texto completo al terminar el stream;
   si fallan, se emite un evento `correction` final (diseño simple: el guard ya solo
   anota/reemplaza, no re-genera).

**Criterio de salida**: TTFB de la síntesis < 1 s percibido en frontend; respuesta final
idéntica a la no-streamed.

---

## Fase 5 — Speculative decoding (opcional, tras estabilizar)

1. Descargar draft pequeño de la misma familia (p.ej. `Qwen3.6-1B`/`0.6B` GGUF Q4) a `models/`.
2. Añadir al PS1: `--model-draft <draft.gguf> --gpu-layers-draft 999 --draft-max 16 --draft-min 4`.
3. Bench A/B con `load_test_chat.py`: aceptar si decode tokens/s mejora ≥ 1.3× sin degradar
   calidad (la aceptación especulativa es exacta: misma distribución, así que solo medir velocidad).
   VRAM extra ~1–2 GB.

---

## Fase 6 — Validación, quant y gate

1. **Calidad**: `make eval` completo con Q6_K_XL + backend nuevo.
   - `recall@5` no debe moverse (retrieval no se toca) → `make eval-ci` verde.
   - `generation.jsonl` (15 casos): comparar lado a lado vs baseline Q4 (diff manual o
     juez si existe); esperable: igual o mejor fidelidad numérica.
2. **Rendimiento**: `load_test_chat.py` 1/2/4/8 usuarios → `data/bench/llama_server_q6.json`;
   tabla comparativa en `docs/SETUP_LLAMA_SERVER.md`.
3. **Bench opcional Q8_K_XL** (35.3 GB): si VRAM y latencia lo permiten, repetir 1–2.
4. **Rollback documentado**: `BANKS_LLM_BACKEND=inprocess` + apagar el servicio. Nada del
   path viejo se borra en esta migración.
5. Actualizar `CLAUDE.md` (sección comandos + variables de entorno + invariante nuevo:
   "el LLM se sirve fuera de proceso vía llama-server; el adapter OpenAICompatEngine es la
   frontera") y `README.md`.

**Criterios de aceptación globales**
- 2 usuarios simultáneos: p95 E2E < 1.5× el p95 de 1 usuario (hoy ≈ 2×, cola pura).
- Latencia por consulta en modo `adaptive` reducida ≥ 30% (cache-reuse + batching de especialistas).
- `make eval-ci` verde; suite unit verde y rápida; cero regresión en generation set.
- Crash/restart de FastAPI no recarga el modelo (servidor independiente).

---

## Fase 7 — Operación dual Qwen/Gemma (opcional)

- Opción simple: segunda instancia de llama-server (puerto 8082) con
  `gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf` solo cuando se necesite comparar (no residente:
  no cabe junto a Qwen Q6 + embedder con margen cómodo).
- Opción cómoda: **llama-swap** (binario Go, Windows nativo) delante de ambos GGUF con
  swap automático por `model` del request. Útil para A/B desde el mismo `BANKS_LLM_BASE_URL`.

---

## Orden de PRs sugerido (cada uno verde en CI)

1. **PR-1**: Fase 0 (script de load test + bench baseline, sin cambios de src).
2. **PR-2**: Fase 2.2 (`_common.py`, refactor puro + tests movidos).
3. **PR-3**: Fase 2.1/2.3/2.4 (adapter + settings + factory + tests; default `inprocess`).
4. **PR-4**: Fase 1 (deploy/: PS1, task XML, `docs/SETUP_LLAMA_SERVER.md`).
5. **PR-5**: Fase 3 (semáforo, to_thread del reranker si aplica, label de métricas).
6. **PR-6**: Fase 6 (bench + docs + CLAUDE.md). Switch de default a `openai_compat`
   SOLO después de este PR.
7. **PR-7/8** (opcionales): Fase 4 (streaming) y Fase 5 (speculative).

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Plantilla Jinja del servidor difiere sutilmente de la del GGUF vía llama-cpp-python | Test de humo comparando tool-call loop completo (golden `sql_routing.jsonl`) entre backends antes del switch |
| KV q8_0 degrada outputs largos | Flag separado; primer paso de rollback es KV fp16 con ctx menor |
| VRAM justa con Q6 + embedder + 128k ctx total | Medir en Fase 1; degradar a 64k total antes que bajar quant |
| Slots < demanda pico | llama-server encola (no falla); el semáforo de la app da back-pressure visible en métricas |
| Fork del comportamiento entre `inprocess` y `openai_compat` | `_common.py` único + mismos tests de contrato corridos sobre ambos engines |
