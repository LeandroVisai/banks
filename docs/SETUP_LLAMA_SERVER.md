# SETUP — llama-server (backend `openai_compat`)

El LLM se sirve **fuera del proceso FastAPI** con el binario oficial `llama-server`
de llama.cpp (Windows nativo + CUDA, sin WSL). La app se conecta por HTTP loopback
vía `OpenAICompatEngine`. Beneficios sobre el backend in-process:

| | in-process (`llama-cpp-python`) | `llama-server` |
|---|---|---|
| Requests concurrentes | serializadas (executor de 1 thread) | **continuous batching** (`--parallel`) |
| Especialistas en paralelo | ilusorio (cola en GPU) | real (mismo forward pass) |
| KV-cache entre iteraciones del agente | re-prefill completo | **`--cache-reuse`** (solo el delta) |
| Tool calls JSON | parseo de texto (frágil en Gemma) | plantilla `--jinja` + gramáticas |
| Crash/restart de la API | recarga el modelo (~min) | el servidor sigue arriba |
| Tokenización | heurística ×1.15 | `/tokenize` real |

## 1. Instalación

1. Descargar **DOS zips del mismo release** desde
   https://github.com/ggml-org/llama.cpp/releases y descomprimir AMBOS en
   `deploy\llama-server\` (debe quedar `deploy\llama-server\llama-server.exe`):
   - `llama-bXXXX-bin-win-cuda-12.x-x64.zip` — binarios + `ggml-cuda.dll`.
   - `cudart-llama-bin-win-cuda-12.x-x64.zip` — runtime CUDA (`cudart64_12.dll`,
     `cublas64_12.dll`, `cublasLt64_12.dll`). **NO viene incluido en el primero**:
     sin estas DLLs `ggml-cuda.dll` no carga y el servidor cae a CPU **en
     silencio** (síntoma: `device_info:` sin `CUDA0`, prefill ~40 t/s, decode
     ~5 t/s, timeouts de 300 s en `/v1/chat`).

   Verificar ANTES de arrancar: `llama-server.exe --list-devices` debe listar la
   H100 como `CUDA0`.
2. Descargar el GGUF a `models\` (convención `models\<owner>--<name>\`):
   - Principal: `unsloth/Qwen3.6-27B-GGUF` → **`Qwen3.6-27B-UD-Q6_K_XL.gguf`** (25.6 GB)
     en `models\unsloth--Qwen3.6-27B-GGUF\`.
   - (El Q4_K_XL actual sigue sirviendo para el backend in-process/fallback.)
3. Probar a mano:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\start_llama_server.ps1
# en otra consola:
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8081/props
```

4. Registrar como tarea de arranque (admin, una vez):

```powershell
powershell -ExecutionPolicy Bypass -File deploy\register_llama_server_task.ps1
Start-ScheduledTask -TaskName "banks-llama-server"
```

## 2. Configuración de la app

En el `.env` del repo:

```ini
BANKS_LLM_BACKEND=openai_compat
BANKS_LLM_BASE_URL=http://127.0.0.1:8081
BANKS_LLM_FAMILY=qwen
BANKS_LLM_SERVER_SLOTS=4          # = --parallel del servidor
# BANKS_LLM_MODEL_PATH ya no es necesario con openai_compat
```

**Rollback inmediato**: `BANKS_LLM_BACKEND=inprocess` (+ `BANKS_LLM_MODEL_PATH`
al GGUF) y reiniciar la API. Nada del path legacy se eliminó.

## 3. Variables del servidor (defaults en `start_llama_server.ps1`)

| Env var | Default | Notas |
|---|---|---|
| `BANKS_LLAMA_SERVER_MODEL` | `models\unsloth--Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q6_K_XL.gguf` | ruta al GGUF |
| `BANKS_LLAMA_SERVER_PORT` | `8081` | solo loopback |
| `BANKS_LLAMA_SERVER_SLOTS` | `4` | slots de batching; alinear con `BANKS_LLM_SERVER_SLOTS` |
| `BANKS_LLAMA_SERVER_CTX` | `131072` | contexto TOTAL; por slot = ctx/slots (32k) |
| `BANKS_LLAMA_SERVER_KV` | `q8_0` | cuantización KV; `f16` si hay artefactos |
| `BANKS_LLAMA_SERVER_DRAFT` | vacío | GGUF draft para speculative decoding (Fase 5) |
| `BANKS_LLM_SERVER_API_KEY` | vacío | si se fija, la app debe llevar el mismo valor |

**Flag crítico: `--jinja`.** Usa la plantilla embebida del GGUF → tool-calling
nativo de Qwen3.6 (renderiza `tools`, `tool_calls` y rol `tool` como
`<tool_response>`). Sin él, el servidor cae a un template genérico y reaparece
el bug histórico del agente repitiendo `discover_query` en bucle.

## 4. Presupuesto de VRAM (H100 80 GB)

| Componente | Estimado |
|---|---|
| Qwen3.6-27B Q6_K_XL (pesos) | ~26 GB |
| KV cache (131k total, q8_0, flash-attn) | ~8–16 GB |
| Embedder Qwen3-VL-8B (proceso API) | ~17 GB |
| Reranker jina-v3 (proceso API) | ~3 GB |
| Overhead CUDA/activaciones | ~8 GB |

Verificar con `nvidia-smi` tras el arranque de ambos procesos. Si aprieta,
**primero** bajar `BANKS_LLAMA_SERVER_CTX` a `65536`; el quant se baja al último.

## 5. Verificación de concurrencia

```powershell
# slots activos / métricas
curl http://127.0.0.1:8081/slots
curl http://127.0.0.1:8081/metrics

# load test end-to-end contra la API (con la app corriendo):
python scripts/load_test_chat.py --users 1 --out data/bench/u1.json
python scripts/load_test_chat.py --users 2 --out data/bench/u2.json
python scripts/load_test_chat.py --users 4 --out data/bench/u4.json
```

Criterio: con 2 usuarios, p95 ≪ 2× el p95 de 1 usuario (antes era ≈2×: cola pura).

## 6. Validación de calidad antes de cambiar el default

1. `make eval-ci` (gate recall@5 — no debe moverse: retrieval no se toca).
2. `make eval` y comparar `generation.jsonl` contra el baseline Q4 in-process.
3. Smoke del loop de tools: correr los 30 casos de `sql_routing.jsonl` por `/v1/chat`
   y verificar que `discover_query → execute_query` avanza (no loopea).

## 7. Troubleshooting

- **`/health` 503 sostenido**: el modelo aún carga (Q6 ≈ 30–60 s desde NVMe) o
  no hay VRAM libre — mirar el log de consola del servidor y `nvidia-smi`.
- **Lento (prefill ~40 t/s, decode ~5 t/s) y timeouts de 300 s**: el servidor
  cayó a CPU. Causa típica: falta el runtime CUDA junto al exe (`cudart64_12.dll`,
  `cublas64_12.dll`, `cublasLt64_12.dll`) — viene en el zip SEPARADO
  `cudart-llama-bin-win-cuda-12.x-x64.zip` del mismo release. Diagnóstico:
  `llama-server.exe --list-devices` no muestra `CUDA0` / el banner `device_info:`
  solo lista la CPU. Fix offline: copiar esas 3 DLLs desde
  `.venv\Lib\site-packages\torch\lib\` (PyTorch CUDA las trae) o desde
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`. Verificar al
  relanzar: `device_info:` con `CUDA0: NVIDIA H100` y `offloaded X/X layers to GPU`.
  No mezclar con DLLs de otra versión mayor de CUDA en el PATH.
- **Outputs raros en respuestas largas**: probar `BANKS_LLAMA_SERVER_KV=f16` y/o
  reducir ctx; si persiste, desactivar `--cache-reuse` para descartar.
- **El agente repite tool calls sin avanzar**: confirmar `--jinja` activo y que
  el GGUF trae chat template (aparece en el log de arranque del servidor).
- **503 "Motor de inferencia no disponible" en /v1/chat**: el servidor está
  caído; `Start-ScheduledTask -TaskName "banks-llama-server"`.
