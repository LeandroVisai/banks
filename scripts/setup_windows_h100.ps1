# setup_windows_h100.ps1
# Instalacion de banks_rag en el servidor Windows H100 (offline)
# Ejecutar como Administrador en PowerShell:
#   powershell -ExecutionPolicy Bypass -File C:\Users\Pipe\banks\scripts\setup_windows_h100.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Rutas ──────────────────────────────────────────────────────────────────
$REPO_DIR    = "C:\Users\Pipe\banks"
$WHEELS_DIR  = "$REPO_DIR\wheels"
$MODELS_DIR  = "$REPO_DIR\models"
$SNAPSHOTS   = "$REPO_DIR\data_pipeline\snapshots"
$VENV_DIR    = "$REPO_DIR\.venv"
$ENV_FILE    = "$REPO_DIR\banks_rag.env"
$START_SCRIPT = "$REPO_DIR\scripts\start_api.ps1"

function Write-Step($n, $msg) { Write-Host "`n=== [$n] $msg ===" -ForegroundColor Cyan }
function Write-OK($msg)        { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn($msg)      { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Write-Fail($msg)      { Write-Host "  FAIL $msg" -ForegroundColor Red; exit 1 }

Write-Host "`n===== Banks RAG — Setup Windows H100 =====" -ForegroundColor Magenta
Write-Host "Repo:       $REPO_DIR"
Write-Host "Wheels:     $WHEELS_DIR"
Write-Host "Modelos:    $MODELS_DIR"
Write-Host "Snapshots:  $SNAPSHOTS"
Write-Host ""

# ── 1. Verificar Python ────────────────────────────────────────────────────
Write-Step 1 "Verificando Python"
try {
    $pyVer = py --version 2>&1
    Write-OK $pyVer
    if ($pyVer -notmatch "3\.(11|12|13)") {
        Write-Warn "Se recomienda Python 3.12+. Continuando de todas formas."
    }
} catch {
    Write-Fail "Comando 'py' no encontrado. Verificar instalacion de Python."
}

# ── 2. Verificar GPU H100 ──────────────────────────────────────────────────
Write-Step 2 "Verificando GPU NVIDIA"
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    Write-OK $gpu
} catch {
    Write-Fail "nvidia-smi no responde. Verificar drivers NVIDIA."
}

# ── 3. Verificar CUDA visible para Python ─────────────────────────────────
Write-Step 3 "Verificando CUDA desde Python"
$cudaCheck = py -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>&1
if ($LASTEXITCODE -eq 0) { Write-OK $cudaCheck }
else { Write-Warn "torch no instalado aun (se instalara en paso 5). Ignorar este aviso." }

# ── 4. Crear entorno virtual ───────────────────────────────────────────────
Write-Step 4 "Creando entorno virtual"
if (Test-Path $VENV_DIR) {
    Write-Warn "Ya existe .venv en $VENV_DIR — se reutiliza."
} else {
    py -m venv $VENV_DIR
    Write-OK "venv creado en $VENV_DIR"
}

$PIP         = "$VENV_DIR\Scripts\pip.exe"
$PYTHON_VENV = "$VENV_DIR\Scripts\python.exe"

# Upgrade pip desde wheels si existe, sino continuar
& $PIP install --quiet --upgrade --no-index --find-links $WHEELS_DIR pip 2>&1 | Out-Null

# ── 5. Instalar dependencias desde wheels offline ──────────────────────────
Write-Step 5 "Instalando dependencias desde wheels offline"

$wheelCount = (Get-ChildItem $WHEELS_DIR -Filter "*.whl" -ErrorAction SilentlyContinue).Count
$tgzCount   = (Get-ChildItem $WHEELS_DIR -Filter "*.tar.gz" -ErrorAction SilentlyContinue).Count
Write-OK "$wheelCount wheels (.whl) y $tgzCount archives (.tar.gz) encontrados"

if ($wheelCount -eq 0 -and $tgzCount -eq 0) {
    Write-Fail "No se encontraron archivos en $WHEELS_DIR"
}

# Instalar el paquete principal (banks_rag) en modo editable
Write-Host "  Instalando banks_rag (modo editable)..."
& $PIP install `
    --no-index `
    --find-links $WHEELS_DIR `
    -e "$REPO_DIR" 2>&1 | Select-Object -Last 3

Write-OK "banks_rag instalado."

# Instalar dependencias clave explicitamente (por si pip editable no las jalona)
Write-Host "  Instalando dependencias de la API..."
$apiDeps = @(
    "fastapi", "uvicorn", "pydantic", "pydantic-settings", "starlette",
    "psycopg", "pgvector", "sqlalchemy", "duckdb",
    "sentence-transformers", "transformers", "torch",
    "Pillow", "numpy", "pypdf", "PyMuPDF", "openpyxl",
    "pandas", "pyarrow", "pyyaml",
    "structlog", "prometheus-client", "typer"
)
& $PIP install `
    --no-index `
    --find-links $WHEELS_DIR `
    @apiDeps 2>&1 | Select-Object -Last 3

Write-OK "Dependencias de API instaladas."

# ── 6. Instalar llama-cpp-python ───────────────────────────────────────────
Write-Step 6 "Instalando llama-cpp-python"

$llamaWhl = Get-ChildItem $WHEELS_DIR -Filter "llama_cpp_python*.whl" | Select-Object -First 1
$llamaTgz = Get-ChildItem $WHEELS_DIR -Filter "llama_cpp_python*.tar.gz" | Select-Object -First 1

if ($llamaWhl) {
    & $PIP install $llamaWhl.FullName --no-deps
    Write-OK "llama-cpp-python instalado desde wheel: $($llamaWhl.Name)"
} elseif ($llamaTgz) {
    Write-Host "  Compilando desde fuente (requiere Visual Studio Build Tools + CUDA toolkit)..."
    $env:CMAKE_ARGS = "-DLLAMA_CUDA=on"
    $env:FORCE_CMAKE = "1"
    & $PIP install $llamaTgz.FullName --no-build-isolation
    Write-OK "llama-cpp-python compilado e instalado desde: $($llamaTgz.Name)"
} else {
    Write-Warn "No se encontro llama_cpp_python en $WHEELS_DIR"
    Write-Warn "El agente LLM no funcionara hasta instalarlo."
}

# ── 7. Verificar imports criticos ──────────────────────────────────────────
Write-Step 7 "Verificando imports criticos"

$checks = @{
    "fastapi"              = "import fastapi; print('fastapi', fastapi.__version__)"
    "llama_cpp"            = "import llama_cpp; print('llama_cpp OK')"
    "sentence_transformers"= "import sentence_transformers; print('sentence_transformers', sentence_transformers.__version__)"
    "torch+CUDA"           = "import torch; ok=torch.cuda.is_available(); dev=torch.cuda.get_device_name(0) if ok else 'CPU'; print(f'torch {torch.__version__} | CUDA:{ok} | {dev}')"
    "pgvector"             = "import pgvector; print('pgvector OK')"
    "duckdb"               = "import duckdb; print('duckdb', duckdb.__version__)"
    "banks_rag"            = "import banks_rag; print('banks_rag OK:', banks_rag.__file__)"
}

foreach ($name in $checks.Keys) {
    $result = & $PYTHON_VENV -c $checks[$name] 2>&1
    if ($LASTEXITCODE -eq 0) { Write-OK "$name — $result" }
    else                      { Write-Warn "$name FALLO: $result" }
}

# ── 8. Verificar modelos .gguf ─────────────────────────────────────────────
Write-Step 8 "Verificando modelos GGUF"

if (-not (Test-Path $MODELS_DIR)) {
    Write-Warn "No existe $MODELS_DIR — crear la carpeta y copiar los .gguf"
} else {
    $ggufs = Get-ChildItem $MODELS_DIR -Recurse -Filter "*.gguf"
    if ($ggufs.Count -eq 0) {
        Write-Warn "No se encontraron .gguf en $MODELS_DIR"
    } else {
        foreach ($g in $ggufs) {
            $sizeMB = [math]::Round($g.Length / 1MB, 0)
            Write-OK "$($g.Name) ($sizeMB MB)"
        }
    }
}

# ── 9. Verificar parquets del catalogo SQL ────────────────────────────────
Write-Step 9 "Verificando parquets del catalogo SQL"

if (-not (Test-Path $SNAPSHOTS)) {
    Write-Warn "No existe $SNAPSHOTS — el catalogo SQL no funcionara"
} else {
    $parquets = Get-ChildItem $SNAPSHOTS -Filter "*.parquet"
    if ($parquets.Count -eq 0) {
        Write-Warn "No hay .parquet en $SNAPSHOTS"
    } else {
        foreach ($p in $parquets) {
            $sizeMB = [math]::Round($p.Length / 1MB, 2)
            Write-OK "$($p.Name) ($sizeMB MB)"
        }
    }
}

# ── 10. Verificar PostgreSQL + pgvector ───────────────────────────────────
Write-Step 10 "Verificando PostgreSQL"
try {
    $pgVer = psql --version 2>&1
    Write-OK $pgVer

    # Verificar que la BD existe y pgvector esta activo
    $pgCheck = & $PYTHON_VENV -c @"
import psycopg, sys
try:
    conn = psycopg.connect('dbname=rag_banco', autocommit=True)
    cur = conn.execute('SELECT extname FROM pg_extension WHERE extname=$$vector$$')
    row = cur.fetchone()
    print('BD rag_banco: OK | pgvector:', 'instalado' if row else 'FALTA (ejecutar: CREATE EXTENSION vector;)')
except Exception as e:
    print('Error conexion:', e)
    sys.exit(1)
"@ 2>&1
    Write-OK $pgCheck
} catch {
    Write-Warn "psql no disponible en PATH o error de conexion. Verificar PostgreSQL."
}

# ── 11. Crear archivo de entorno ───────────────────────────────────────────
Write-Step 11 "Configurando variables de entorno"

# Detectar el primer .gguf disponible para sugerir como MODEL_PATH
$firstGguf = Get-ChildItem $MODELS_DIR -Recurse -Filter "*.gguf" | Select-Object -First 1
$suggestedModel = if ($firstGguf) { $firstGguf.FullName } else { "$MODELS_DIR\modelo.gguf" }

if (Test-Path $ENV_FILE) {
    Write-Warn "$ENV_FILE ya existe — no se sobreescribe. Verificar manualmente."
} else {
    @"
# banks_rag — Variables de entorno del servidor Windows H100
# Editar PGPASSWORD y BANKS_API_KEYS antes de arrancar.

# PostgreSQL
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=CAMBIAR_ESTO
PGDATABASE=rag_banco

# LLM (llama.cpp con H100)
BANKS_LLM_FAMILY=qwen
BANKS_LLM_MODEL_PATH=$suggestedModel
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

    Write-OK "Creado $ENV_FILE"
    Write-Warn "EDITAR PGPASSWORD y BANKS_API_KEYS antes de continuar."
}

# Restringir permisos del env file
try {
    icacls $ENV_FILE /inheritance:r /grant:r "${env:USERNAME}:(F)" /grant:r "SISTEMA:(R)" | Out-Null
    Write-OK "Permisos de $ENV_FILE restringidos correctamente."
} catch {
    Write-Warn "No se pudieron restringir permisos de $ENV_FILE (ejecutar como Admin)."
}

# ── 12. Generar script de arranque ─────────────────────────────────────────
Write-Step 12 "Generando scripts de operacion"

# start_api.ps1
@"
# start_api.ps1 — Arranca banks-api en el servidor Windows H100
# Uso: powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1

Get-Content "$ENV_FILE" | ForEach-Object {
    if (`$_ -match "^([^#\s][^=]+)=(.+)`$") {
        [System.Environment]::SetEnvironmentVariable(`$matches[1].Trim(), `$matches[2].Trim(), "Process")
    }
}

Set-Location "$REPO_DIR"
Write-Host "Arrancando banks-api en http://0.0.0.0:8080 ..." -ForegroundColor Green
& "$PYTHON_VENV" -m banks_rag.interface.api.main
"@ | Out-File -FilePath $START_SCRIPT -Encoding utf8
Write-OK "Creado: $START_SCRIPT"

# run_ingest.ps1
$ingestScript = "$REPO_DIR\scripts\run_ingest.ps1"
@"
# run_ingest.ps1 — Ejecuta el pipeline de ingesta completo
# Uso: powershell -ExecutionPolicy Bypass -File scripts\run_ingest.ps1

Get-Content "$ENV_FILE" | ForEach-Object {
    if (`$_ -match "^([^#\s][^=]+)=(.+)`$") {
        [System.Environment]::SetEnvironmentVariable(`$matches[1].Trim(), `$matches[2].Trim(), "Process")
    }
}

Set-Location "$REPO_DIR"
Write-Host "Iniciando pipeline de ingesta..." -ForegroundColor Green
& "$REPO_DIR\.venv\Scripts\banks-ingest.exe" run --source "$REPO_DIR\Datos_prueba"
"@ | Out-File -FilePath $ingestScript -Encoding utf8
Write-OK "Creado: $ingestScript"

# check_gpu.ps1
$gpuScript = "$REPO_DIR\scripts\check_gpu.ps1"
@"
# check_gpu.ps1 — Estado de la H100 antes de arrancar
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu --format=csv
"@ | Out-File -FilePath $gpuScript -Encoding utf8
Write-OK "Creado: $gpuScript"

# ── Resumen final ──────────────────────────────────────────────────────────
Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "  Instalacion completada" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Proximos pasos obligatorios:" -ForegroundColor Yellow
Write-Host "  1. Editar $ENV_FILE"
Write-Host "     - Cambiar PGPASSWORD"
Write-Host "     - Cambiar BANKS_API_KEYS"
Write-Host "     - Verificar BANKS_LLM_MODEL_PATH apunta al .gguf correcto"
Write-Host ""
Write-Host "  2. Crear la BD y activar pgvector (si no esta hecho):"
Write-Host "     psql -U postgres -c ""CREATE DATABASE rag_banco;"""
Write-Host "     psql -U postgres -d rag_banco -c ""CREATE EXTENSION IF NOT EXISTS vector;"""
Write-Host ""
Write-Host "  3. Correr ingesta:"
Write-Host "     powershell -ExecutionPolicy Bypass -File $ingestScript"
Write-Host ""
Write-Host "  4. Arrancar la API:"
Write-Host "     powershell -ExecutionPolicy Bypass -File $START_SCRIPT"
Write-Host ""
Write-Host "  5. Validar:"
Write-Host "     curl http://localhost:8080/healthz"
Write-Host ""
