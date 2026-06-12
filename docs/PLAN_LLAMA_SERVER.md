# Plan de migración: llama-cpp-python in-process → llama-server (Windows + H100)

**Fecha:** 2026-06-09
**Motivación:** la arquitectura router-v1 lanza 1–3 especialistas con `asyncio.gather`,
pero `LlamaCppEngine` serializa todas las generaciones detrás de una sola instancia
`Llama` + `ThreadPoolExecutor`. llama-server aporta continuous batching + slots
paralelos (paralelismo real en la H100), prompt caching, speculative decoding,
aislamiento de procesos (un crash del LLM no tumba FastAPI) y API OpenAI-compatible
con tool-calling nativo de Qwen3 (`--jinja` usa la plantilla embebida del GGUF —
el mismo comportamiento `chat_format=None` que ya validamos).

**Principio rector:** `LLMEngine` es un Protocol. Se agrega un adapter HTTP nuevo
(`LlamaServerEngine`) detrás del mismo contrato; `conversation_loop`, especialistas,
síntesis y `jarvis_news` no cambian. `LlamaCppEngine` se conserva como fallback.

---

## Fase 0 — Binarios y modelos (sin tocar código)

1. Descargar y descomprimir en `C:\llama\` (o ruta interna equivalente):
   - `https://github.com/ggml-org/llama.cpp/releases/download/b9585/llama-b9585-bin-win-cuda-12.4-x64.zip`
   - `https://github.com/ggml-org/llama.cpp/releases/download/b9585/cudart-llama-bin-win-cuda-12.4-x64.zip`
     (solo si el servidor no tiene CUDA Toolkit; las DLLs van junto a `llama-server.exe`)
   - Nota: variante cuda-12.4 ← recomendada para H100 (sm_90, driver ≥550).
     La cuda-13.3 requiere driver ≥580.
2. Descargar GGUFs a `models/` siguiendo la convención `models/<owner>--<name>/`:
   - `models/unsloth--Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q6_K_XL.gguf` (25.6 GB — principal)
   - `models/unsloth--Qwen3.6-27B-GGUF/mmproj-F16.gguf` (opcional, rama visual)
   - `models/unsloth--gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-MXFP4_MOE.gguf` (16.6 GB)
   - `models/unsloth--gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it.gguf` (462 MB —
     draft MTP para speculative decoding de Gemma)
3. Smoke test manual (PowerShell):

```bat
C:\llama\llama-server.exe ^
  --model models\unsloth--Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q6_K_XL.gguf ^
  --n-gpu-layers 999 ^
  --ctx-size 131072 --parallel 4 ^
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 ^
  --jinja ^
  --host 127.0.0.1 --port 8081 --api-key <key-interna> ^
  --metrics
```

```powershell
# Verificación tool-calling (debe devolver tool_calls estructurado):
curl http://127.0.0.1:8081/v1/chat/completions -H "Authorization: Bearer <key>" `
  -H "Content-Type: application/json" -d '{
    "messages": [{"role":"user","content":"¿Cuál es la TPM vigente?"}],
    "tools": [{"type":"function","function":{"name":"search_documents",
      "description":"Busca en el corpus","parameters":{"type":"object",
      "properties":{"query":{"type":"string"}},"required":["query"]}}}]
  }'
```

Criterio de salida: respuesta con `tool_calls` JSON estructurado, VRAM observada
(`nvidia-smi`) y tokens/s de referencia anotados.

**Notas de dimensionamiento:**
- `--ctx-size` es el TOTAL repartido entre slots: 131072/4 = 32k por especialista
  (el doble del `n_ctx=16384` actual).
- KV q8_0 ≈ mitad de VRAM de KV con pérdida despreciable.
- Presupuesto H100 80 GB ≈ modelo Q6 (25.6) + embedder Qwen3-VL-8B (~16 bf16)
  + reranker jina-v3 (~2) + KV/slots (~25) + margen.

---

## Fase 1 — Adapter `LlamaServerEngine`

**Archivo nuevo:** `src/banks_rag/infrastructure/llm/llama_server_engine.py`

- Implementa el Protocol `LLMEngine` (`load`, `unload`, `generate`, `describe`).
- `generate()` → `POST {base_url}/v1/chat/completions` vía `httpx.AsyncClient`
  (async nativo: desaparece el `ThreadPoolExecutor`).
  - Mapea 1:1 `messages`, `tools`, `temperature`, `top_p`, `max_tokens`.
  - `tool_calls` estructurados del server → `ToolCall` del dominio; conservar
    `parse_tool_calls` (texto `<tool_call>`) como fallback defensivo.
  - Conservar `_strip_think` sobre el contenido (Qwen3 con plantilla nativa puede
    emitir cierre colgante `</think>`; el server además puede poblar
    `reasoning_content` — si viene, descartarlo y usar solo `content`).
- `load()` = health check (`GET /health`, reintentos con backoff); `unload()` = no-op
  (el ciclo de vida del proceso es externo). Timeout de request generoso
  (`BANKS_LLM_SERVER_TIMEOUT`, default 300 s).
- **Config** (`banks_rag/config.py`):
  - `BANKS_LLM_BACKEND` = `inprocess` (default, comportamiento actual) | `server`
  - `BANKS_LLM_SERVER_URL` (default `http://127.0.0.1:8081`)
  - `BANKS_LLM_SERVER_API_KEY`
  - `BANKS_LLM_SERVER_TIMEOUT` (default `300`)
- **Factory:** donde hoy se construye `LlamaCppEngine` (deps de la app), elegir
  según `BANKS_LLM_BACKEND`. Cero cambios fuera de `infrastructure/llm/` + config.
- Sampling: los perfiles de `MODE_PROFILES` y la síntesis (0.3/0.8) ya viajan por
  parámetro en `generate()` — no cambian.

## Fase 2 — Tests unitarios (sin red, sin GPU)

**Archivo nuevo:** `tests/unit/test_llama_server_engine.py`

- Mock del transporte httpx (`respx` o `httpx.MockTransport` — MockTransport no
  agrega dependencia).
- Casos mínimos: tool_calls estructurados → `ToolCall`; fallback `<tool_call>` en
  texto; `_strip_think` / `reasoning_content`; error 5xx / timeout → excepción
  limpia (el loop ya degrada con `MAX_ITERATIONS_FALLBACK_MESSAGE`); health check
  con reintardos.
- Invariante del repo: `PYTHONPATH=src pytest tests/unit/ -q` sigue corriendo en
  segundos sin binarios.

## Fase 3 — Servicio Windows

- Registrar `llama-server.exe` como servicio con **NSSM** (o WinSW):
  restart-on-failure, working dir del repo, stdout/stderr a
  `logs/llama-server.log` con rotación.
- Orden de arranque: llama-server → (health OK) → API FastAPI (también como
  servicio NSSM con dependencia).
- `--host 127.0.0.1` SIEMPRE (solo loopback) + `--api-key`; el perímetro lo sigue
  dando FastAPI con `BANKS_API_KEYS`.
- Documentar en `docs/DEPLOYMENT_H100.md` la sección Windows (hoy asume
  Ubuntu/systemd).

## Fase 4 — Validación (gate antes de adoptar)

1. **Calidad:** `make eval` con `BANKS_LLM_BACKEND=server` y comparar
   `eval_report.md` contra baseline in-process. Gate: `make eval-ci`
   (recall@5 no cae >5%). Aprovechar para A/B de cuantización:
   UD-Q4_K_XL (actual) vs UD-Q6_K_XL.
2. **Paralelismo real:** disparar 4 requests concurrentes a `/v1/chat` en modo
   `on` (3 especialistas) y verificar que la latencia p50 NO escala linealmente
   (hoy sí lo hace). Registrar tokens/s agregados desde `/metrics` del server
   (Prometheus nativo con `--metrics`).
3. **Regresión funcional:** suite completa `make test` + revisar
   `data/chat_logs/` (tool_trace, citas) en 10 preguntas del golden set.

## Fase 5 — Tuning posterior (opcional, en orden de retorno)

1. **Speculative decoding Gemma:** `--model-draft mtp-gemma-4-26B-A4B-it.gguf`
   (draft MTP oficial, 462 MB) → 1.5–2x decode esperado. Para Qwen, evaluar un
   Qwen3.6 pequeño como draft (`--model-draft` + `--draft-max 16`).
2. **Dos servers, dos perfiles:** Gemma MXFP4_MOE (MoE 4B activos, rápido) en
   puerto 8082 para `thinking_mode=off`; Qwen Q6 para `adaptive`/`on`. El factory
   elige base_url según perfil. Solo si el presupuesto de VRAM lo permite
   (~17 GB extra) — medir primero.
3. **Ajustar `--parallel`/ctx** según telemetría real de uso de slots.
4. **A7 de la auditoría (pool psycopg3 async):** con el LLM concurrente, Postgres
   pasa a ser el siguiente cuello de botella — subirlo de prioridad.

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| La plantilla jinja del GGUF difiere del rendering de llama-cpp-python | Fase 4.1 compara con golden set antes de adoptar; `--chat-template-file` permite forzar una plantilla si hiciera falta |
| Timeouts en síntesis largas (4096 tokens) | `BANKS_LLM_SERVER_TIMEOUT=300` + streaming como mejora futura |
| Server caído ↔ API viva | Health check en `load()` + NSSM restart + `BANKS_LLM_BACKEND=inprocess` como rollback de 1 línea en `.env` |
| VRAM insuficiente con embedder+reranker residentes | Medir en Fase 0 con `nvidia-smi`; bajar a UD-Q5_K_XL o reducir `--parallel` |
| Paridad con el gemelo (`Proyecto_rag/` en Mac) | El adapter es portable (httpx); en Mac llama-server se instala vía brew — misma config |

## Resumen de cambios al repo

| Acción | Archivo |
|---|---|
| Nuevo | `src/banks_rag/infrastructure/llm/llama_server_engine.py` |
| Nuevo | `tests/unit/test_llama_server_engine.py` |
| Editar | `src/banks_rag/config.py` (4 env vars nuevas) |
| Editar | factory de deps (selección de backend) |
| Editar | `docs/DEPLOYMENT_H100.md` (sección Windows/NSSM) + `CLAUDE.md` (env vars) |
| Sin cambios | `conversation_loop`, router, tools, `jarvis_news`, ingesta |
