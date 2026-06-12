# start_llama_server.ps1 — arranca llama-server (llama.cpp) para banks_rag.
#
# El servidor expone /v1/chat/completions OpenAI-compatible en loopback.
# La app FastAPI se conecta con BANKS_LLM_BACKEND=openai_compat.
#
# Binario: descargar DOS zips del MISMO release oficial de
#   https://github.com/ggml-org/llama.cpp/releases y descomprimir ambos en
#   deploy\llama-server\:
#   - llama-bXXXX-bin-win-cuda-12.x-x64.zip   (binarios + ggml-cuda.dll)
#   - cudart-llama-bin-win-cuda-12.x-x64.zip  (cudart64_12/cublas64_12/cublasLt64_12;
#     NO viene en el primero — sin estas DLLs el servidor cae a CPU EN SILENCIO)
#   Verificar: llama-server.exe --list-devices debe mostrar la H100 como CUDA0.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File deploy\start_llama_server.ps1
#   # o con overrides:
#   $env:BANKS_LLAMA_SERVER_SLOTS=2; .\deploy\start_llama_server.ps1
#
# Ver docs\SETUP_LLAMA_SERVER.md para el presupuesto de VRAM y troubleshooting.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

# ── Configuración (env vars con defaults) ─────────────────────────────────────
$model = $env:BANKS_LLAMA_SERVER_MODEL
if (-not $model) {
    $model = Join-Path $repoRoot "models\unsloth--Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q6_K_XL.gguf"
}
$port  = if ($env:BANKS_LLAMA_SERVER_PORT)  { $env:BANKS_LLAMA_SERVER_PORT }  else { "8081" }
# Slots de generación paralela (continuous batching). Mantener alineado con
# BANKS_LLM_SERVER_SLOTS de la app (dimensiona el semáforo de /v1/chat).
$slots = if ($env:BANKS_LLAMA_SERVER_SLOTS) { $env:BANKS_LLAMA_SERVER_SLOTS } else { "4" }
# Contexto TOTAL en tokens; por slot = ctx / slots (131072/4 = 32k por slot).
# Si la VRAM aprieta (ver docs), bajar a 65536 (16k por slot, paridad con el
# n_ctx=16384 del backend in-process).
$ctx   = if ($env:BANKS_LLAMA_SERVER_CTX)   { $env:BANKS_LLAMA_SERVER_CTX }   else { "131072" }
# Cuantización del KV cache: q8_0 ahorra ~50% de VRAM de KV con pérdida mínima.
# Requiere flash-attn. Si aparecen artefactos en outputs largos, fijar a f16.
$kvType = if ($env:BANKS_LLAMA_SERVER_KV)   { $env:BANKS_LLAMA_SERVER_KV }    else { "q8_0" }

# Speculative decoding (opcional, Fase 5): ruta a un draft GGUF pequeño de la
# misma familia (p. ej. Qwen3.6-1B Q4). Vacío = desactivado.
$draft = $env:BANKS_LLAMA_SERVER_DRAFT

$exe = Join-Path $PSScriptRoot "llama-server\llama-server.exe"
if (-not (Test-Path $exe))   { throw "No existe $exe — descarga el release Windows CUDA de llama.cpp (ver cabecera)." }
if (-not (Test-Path $model)) { throw "No existe el modelo: $model" }

# Guard anti fallback-a-CPU: si está el backend CUDA pero falta el runtime,
# ggml-cuda.dll no carga y el servidor corre en CPU SIN avisar (prefill ~40 t/s,
# decode ~5 t/s, timeouts de 300 s en /v1/chat). Mejor abortar con instrucción.
$binDir = Join-Path $PSScriptRoot "llama-server"
if (Test-Path (Join-Path $binDir "ggml-cuda.dll")) {
    foreach ($dll in @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")) {
        if (-not (Test-Path (Join-Path $binDir $dll))) {
            throw ("Falta $dll junto a llama-server.exe — sin el runtime CUDA el servidor cae a CPU. " +
                   "Copia cudart64_12/cublas64_12/cublasLt64_12 (zip cudart-* del mismo release, " +
                   "o desde torch\lib del venv). Ver docs\SETUP_LLAMA_SERVER.md §7.")
        }
    }
}

# ── Argumentos ────────────────────────────────────────────────────────────────
$args = @(
    "--model", $model,
    "--n-gpu-layers", "999",          # todo a GPU (H100)
    "--ctx-size", $ctx,
    "--parallel", $slots,             # slots de continuous batching
    "--cont-batching",
    "--cache-reuse", "256",           # prefix/KV reuse: el loop del agente solo prefillea el delta
    "--flash-attn", "on",
    "--cache-type-k", $kvType,
    "--cache-type-v", $kvType,
    "--jinja",                        # plantilla del GGUF → tool-calling nativo Qwen (CRÍTICO)
    "--host", "127.0.0.1",            # solo loopback: la única puerta pública es FastAPI
    "--port", $port,
    "--metrics"                       # /metrics Prometheus
)
if ($env:BANKS_LLM_SERVER_API_KEY) { $args += @("--api-key", $env:BANKS_LLM_SERVER_API_KEY) }
if ($draft) {
    $args += @(
        "--model-draft", $draft,
        "--gpu-layers-draft", "999",
        "--draft-max", "16",
        "--draft-min", "4"
    )
}

Write-Host "llama-server → modelo=$model puerto=$port slots=$slots ctx=$ctx kv=$kvType"
& $exe @args
