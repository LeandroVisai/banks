---
name: setup-windows-h100
description: Guía interactiva de instalación de banks_rag en el servidor Windows H100 (offline). Asume que los archivos ya están en el servidor. Genera el script setup_windows_h100.ps1 adaptado al estado real del servidor.
---

El usuario ya tiene todos los archivos descargados en el servidor Windows H100.
Tu tarea es guiar la instalación paso a paso y generar los scripts PowerShell necesarios.

## Antes de generar cualquier script, pregunta:

1. ¿Dónde está el código fuente en el servidor? (ej: `C:\banks_rag\`, `D:\Proyecto\banks\`)
2. ¿Dónde están los wheels descargados? (ej: misma carpeta, `C:\wheels\`)
3. ¿Dónde están los modelos .gguf? (ej: `C:\models\`, carpeta `models\` dentro del repo)
4. ¿Dónde están los parquets del catálogo SQL? (ej: `data_pipeline\snapshots\` dentro del repo)
5. ¿Qué versión de Python está instalada? (correr: `py --version` o `python --version`)
6. ¿Existe PostgreSQL instalado? ¿Y pgvector?

Con esas respuestas, genera el script `scripts\setup_windows_h100.ps1` personalizado:

---

## Template del script (adaptar rutas según respuestas)

```powershell
# setup_windows_h100.ps1
# Ejecutar como Administrador en el servidor Windows H100 (sin internet)
# Uso: powershell -ExecutionPolicy Bypass -File scripts\setup_windows_h100.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Rutas (ajustar según instalación) ─────────────────────────────────────
$REPO_DIR    = "C:\banks_rag"          # raíz del repo
$WHEELS_DIR  = "C:\banks_rag\wheels"   # carpeta con los .whl descargados
$MODELS_DIR  = "C:\banks_rag\models"   # carpeta con archivos .gguf
$VENV_DIR    = "$REPO_DIR\.venv"
$ENV_FILE    = "C:\banks_rag.env"
$PYTHON      = "py"                    # o "python" según el servidor

function Write-Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Write-OK($msg)        { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn($msg)      { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Write-Fail($msg)      { Write-Host "  FAIL $msg" -ForegroundColor Red; exit 1 }

# ── 1. Verificar Python ────────────────────────────────────────────────────
Write-Step 1 "Verificando Python"
$pyVer = & $PYTHON --version 2>&1
Write-OK $pyVer
if ($pyVer -notmatch "3\.(11|12|13)") {
    Write-Warn "Se recomienda Python 3.12. Continuando de todas formas."
}

# ── 2. Verificar GPU H100 ──────────────────────────────────────────────────
Write-Step 2 "Verificando GPU"
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    Write-OK $gpu
} catch {
    Write-Fail "nvidia-smi no encontrado. Verificar instalación de drivers NVIDIA."
}

# ── 3. Crear entorno virtual ───────────────────────────────────────────────
Write-Step 3 "Creando entorno virtual en $VENV_DIR"
if (Test-Path $VENV_DIR) {
    Write-Warn "Ya existe un venv en $VENV_DIR — se reutilizará."
} else {
    & $PYTHON -m venv $VENV_DIR
    Write-OK "venv creado."
}

$PIP    = "$VENV_DIR\Scripts\pip.exe"
$PYTHON_VENV = "$VENV_DIR\Scripts\python.exe"

# ── 4. Instalar dependencias desde wheels offline ──────────────────────────
Write-Step 4 "Instalando dependencias desde wheels offline"

# Verificar que hay wheels disponibles
$wheelCount = (Get-ChildItem $WHEELS_DIR -Filter "*.whl" -ErrorAction SilentlyContinue).Count
if ($wheelCount -eq 0) {
    Write-Fail "No se encontraron .whl en $WHEELS_DIR. Verificar ruta."
}
Write-OK "$wheelCount wheels encontrados."

# Instalar todo desde wheels (sin internet)
& $PIP install `
    --no-index `
    --find-links $WHEELS_DIR `
    --no-deps `
    -e "$REPO_DIR" 2>&1 | Select-Object -Last 5

# Instalar dependencias transitivas también desde wheels
& $PIP install `
    --no-index `
    --find-links $WHEELS_DIR `
    fastapi uvicorn pydantic pydantic-settings `
    psycopg pgvector sqlalchemy duckdb `
    sentence-transformers transformers torch `
    Pillow numpy pypdf PyMuPDF openpyxl `
    pandas pyarrow pyyaml `
    structlog prometheus-client typer 2>&1 | Select-Object -Last 5

Write-OK "Dependencias instaladas."

# ── 5. Instalar llama-cpp-python (CUDA) ───────────────────────────────────
Write-Step 5 "Instalando llama-cpp-python con soporte CUDA"
$llamaWheel = Get-ChildItem $WHEELS_DIR -Filter "llama_cpp_python*.whl" | Select-Object -First 1
if ($llamaWheel) {
    & $PIP install $llamaWheel.FullName --no-deps
    Write-OK "llama-cpp-python instalado desde: $($llamaWheel.Name)"
} else {
    Write-Warn "No se encontró wheel de llama-cpp-python."
    Write-Warn "Si hay un .tar.gz, compilar manualmente:"
    Write-Warn "  set CMAKE_ARGS=-DLLAMA_CUDA=on"
    Write-Warn "  $PIP install llama_cpp_python-*.tar.gz --no-build-isolation"
}

# ── 6. Verificar instalación de Python ────────────────────────────────────
Write-Step 6 "Verificando imports críticos"
$checks = @(
    "import fastapi; print('fastapi OK')",
    "import llama_cpp; print('llama_cpp OK')",
    "import sentence_transformers; print('sentence_transformers OK')",
    "import torch; print(f'torch OK | CUDA: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')",
    "import pgvector; print('pgvector OK')",
    "import duckdb; print('duckdb OK')"
)
foreach ($check in $checks) {
    $result = & $PYTHON_VENV -c $check 2>&1
    if ($LASTEXITCODE -eq 0) { Write-OK $result }
    else                      { Write-Warn "FALLO: $result" }
}

# ── 7. Verificar modelos .gguf ─────────────────────────────────────────────
Write-Step 7 "Verificando modelos GGUF"
$ggufs = Get-ChildItem $MODELS_DIR -Recurse -Filter "*.gguf" -ErrorAction SilentlyContinue
if ($ggufs.Count -eq 0) {
    Write-Warn "No se encontraron archivos .gguf en $MODELS_DIR"
    Write-Warn "Asegurarse de que BANKS_LLM_MODEL_PATH apunte al archivo correcto."
} else {
    foreach ($g in $ggufs) {
        $sizeMB = [math]::Round($g.Length / 1MB, 0)
        Write-OK "$($g.Name) ($sizeMB MB)"
    }
}

# ── 8. Verificar parquets del catálogo SQL ────────────────────────────────
Write-Step 8 "Verificando parquets del catálogo SQL"
$snapshotsDir = "$REPO_DIR\data_pipeline\snapshots"
if (Test-Path $snapshotsDir) {
    $parquets = Get-ChildItem $snapshotsDir -Filter "*.parquet"
    Write-OK "$($parquets.Count) parquets encontrados en $snapshotsDir"
    $parquets | ForEach-Object {
        $sizeMB = [math]::Round($_.Length / 1MB, 2)
        Write-Host "     $($_.Name) ($sizeMB MB)"
    }
} else {
    Write-Warn "No existe $snapshotsDir — el catálogo SQL no funcionará."
}

# ── 9. Crear archivo de entorno ────────────────────────────────────────────
Write-Step 9 "Configurando variables de entorno"
if (Test-Path $ENV_FILE) {
    Write-Warn "$ENV_FILE ya existe — no se sobreescribe. Verificar manualmente."
} else {
    @"
# banks_rag — Variables de entorno del servidor Windows H100
# Editar antes de arrancar el servicio.

# PostgreSQL
PGHOST=localhost
PGPORT=5432
PGUSER=banks_rag
PGPASSWORD=CAMBIAR_ESTO
PGDATABASE=rag_banco

# LLM (llama.cpp)
BANKS_LLM_FAMILY=qwen
BANKS_LLM_MODEL_PATH=$MODELS_DIR\Qwen3-8B-Q4_K_M.gguf
BANKS_LLM_N_CTX=16384
BANKS_LLM_N_GPU_LAYERS=-1
BANKS_LLM_TEMPERATURE=0.2
BANKS_LLM_MAX_TOKENS=2048

# API
BANKS_API_HOST=0.0.0.0
BANKS_API_PORT=8080
BANKS_API_KEYS=CAMBIAR_ESTO
BANKS_LOG_LEVEL=INFO
BANKS_LOG_JSON=true
BANKS_RATE_LIMIT_RPM=60
BANKS_RATE_LIMIT_BURST=10

# Tracing (opcional)
BANKS_TRACING=off
"@ | Out-File -FilePath $ENV_FILE -Encoding utf8
    Write-OK "Creado $ENV_FILE — EDITAR PGPASSWORD y BANKS_API_KEYS antes de continuar."
}

# ── 10. Script de arranque manual ──────────────────────────────────────────
Write-Step 10 "Generando script de arranque"
$startScript = "$REPO_DIR\scripts\start_api.ps1"
@"
# start_api.ps1 — Arranque manual de banks-api en Windows
# Cargar variables de entorno
Get-Content "$ENV_FILE" | ForEach-Object {
    if (`$_ -match "^([^#\s][^=]+)=(.+)`$") {
        [System.Environment]::SetEnvironmentVariable(`$matches[1].Trim(), `$matches[2].Trim(), "Process")
    }
}
Set-Location "$REPO_DIR"
Write-Host "Arrancando banks-api en http://0.0.0.0:8080 ..."
& "$PYTHON_VENV" -m banks_rag.interface.api.main
"@ | Out-File -FilePath $startScript -Encoding utf8
Write-OK "Script generado: $startScript"
Write-Host "`nPara arrancar: powershell -ExecutionPolicy Bypass -File $startScript" -ForegroundColor Yellow

# ── Resumen ────────────────────────────────────────────────────────────────
Write-Host "`n=== Instalación completada ===" -ForegroundColor Green
Write-Host "Próximos pasos:"
Write-Host "  1. Editar $ENV_FILE (PGPASSWORD, BANKS_API_KEYS, BANKS_LLM_MODEL_PATH)"
Write-Host "  2. Verificar PostgreSQL: psql -U banks_rag -d rag_banco -c '\dt'"
Write-Host "  3. Correr ingesta: $VENV_DIR\Scripts\banks-ingest.exe run --source $REPO_DIR\Datos_prueba\"
Write-Host "  4. Arrancar API: powershell -ExecutionPolicy Bypass -File $startScript"
Write-Host "  5. Validar: curl http://localhost:8080/healthz"
```

---

## Notas para el asistente

- Si el usuario reporta errores de CUDA en llama.cpp, verificar que `CMAKE_ARGS=-DLLAMA_CUDA=on` fue usado al compilar.
- Si torch no detecta CUDA (`torch.cuda.is_available()` = False), verificar que la versión de torch coincide con la versión de CUDA del driver (`nvcc --version`).
- Si pgvector falla, verificar que la extensión está instalada en PostgreSQL: `CREATE EXTENSION IF NOT EXISTS vector;`
- Los paths en el script usan `\` (Windows) — no usar `/` en rutas del servidor.
- El archivo `C:\banks_rag.env` debe tener permisos restringidos: `icacls C:\banks_rag.env /inheritance:r /grant:r "SISTEMA:(R)" /grant:r "Administradores:(F)"`
