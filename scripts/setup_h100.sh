#!/usr/bin/env bash
# setup_h100.sh — Corre en el SERVIDOR H100 (sin internet).
#
# Asume que el tarball generado por package_for_h100.sh fue descomprimido y
# que este script está en banks_rag_src/scripts/setup_h100.sh.
#
# Uso (como root o con sudo):
#   cd /tmp && tar -xzf banks_rag_<fecha>.tar.gz
#   bash banks_rag_src/scripts/setup_h100.sh
#
# Qué hace:
#   1. Crea usuario y grupo 'banks_rag' si no existen
#   2. Copia el código a /opt/banks_rag
#   3. Crea venv e instala wheels offline
#   4. Compila llama-cpp-python con soporte CUDA
#   5. Configura /etc/banks_rag.env (template)
#   6. Instala y habilita las systemd units
#   7. Verifica que el servicio arranca

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
WHEELS_DIR="$(cd "$SCRIPT_DIR/../../wheels" && pwd)"
INSTALL_DIR="/opt/banks_rag"
SERVICE_USER="banks_rag"
ENV_FILE="/etc/banks_rag.env"
PYTHON="python3.11"

echo "=== Banks RAG — Setup en H100 ==="
echo "Fuente:  $REPO_SRC"
echo "Destino: $INSTALL_DIR"
echo "Wheels:  $WHEELS_DIR"
echo ""

# ── 1. Usuario de sistema ──────────────────────────────────────────────────
echo "1/7 Configurando usuario $SERVICE_USER..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --shell /bin/false --home "$INSTALL_DIR" --create-home "$SERVICE_USER"
    echo "    Usuario creado."
else
    echo "    Usuario ya existe."
fi

# ── 2. Copiar código ───────────────────────────────────────────────────────
echo "2/7 Copiando código a $INSTALL_DIR..."
rsync -a --delete \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "$REPO_SRC/" "$INSTALL_DIR/"

chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/logs"
chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/logs"

# ── 3. Entorno Python ─────────────────────────────────────────────────────
echo "3/7 Creando venv con $PYTHON..."
if [[ ! -f "$INSTALL_DIR/.venv/bin/python" ]]; then
    "$PYTHON" -m venv "$INSTALL_DIR/.venv"
fi
VENV_PIP="$INSTALL_DIR/.venv/bin/pip"

echo "    Instalando wheels offline..."
"$VENV_PIP" install \
    --no-index \
    --find-links "$WHEELS_DIR" \
    --upgrade pip setuptools wheel

"$VENV_PIP" install \
    --no-index \
    --find-links "$WHEELS_DIR" \
    "$INSTALL_DIR[api]"

# ── 4. Compilar llama-cpp-python con CUDA ─────────────────────────────────
echo "4/7 Compilando llama-cpp-python con soporte CUDA..."
# Requiere CUDA toolkit instalado en el sistema (nvcc en PATH)
if command -v nvcc &>/dev/null; then
    LLAMA_WHEEL="$(ls "$WHEELS_DIR"/llama_cpp_python*.whl 2>/dev/null | head -1 || true)"
    if [[ -n "$LLAMA_WHEEL" ]]; then
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" \
        FORCE_CMAKE=1 \
        "$VENV_PIP" install \
            --no-index \
            --find-links "$WHEELS_DIR" \
            --no-build-isolation \
            "$LLAMA_WHEEL"
        echo "    llama-cpp-python compilado con CUDA."
    else
        echo "    ADVERTENCIA: wheel de llama-cpp-python no encontrado en $WHEELS_DIR"
        echo "    Compilando desde PyPI si hay conectividad, o instalar manualmente."
    fi
else
    echo "    ADVERTENCIA: nvcc no encontrado — llama-cpp-python sin soporte GPU."
    echo "    Instalar CUDA toolkit y re-ejecutar este script."
fi

# ── 5. Archivo de entorno ─────────────────────────────────────────────────
echo "5/7 Configurando $ENV_FILE..."
if [[ ! -f "$ENV_FILE" ]]; then
    cat > "$ENV_FILE" << 'ENVEOF'
# Banks RAG — Variables de entorno
# EDITAR antes de iniciar el servicio

# PostgreSQL
PGHOST=localhost
PGPORT=5432
PGUSER=banks_rag
PGPASSWORD=CAMBIAR_ESTO
PGDATABASE=rag_banco
RAG_TABLE_PREFIX=

# LLM (ajustar nombre del .gguf)
BANKS_LLM_FAMILY=qwen
BANKS_LLM_MODEL_PATH=models/Qwen3.6-27B-UD-Q4_K_XL.gguf
BANKS_LLM_N_CTX=16384
BANKS_LLM_N_GPU_LAYERS=-1
BANKS_LLM_TEMPERATURE=0.2
BANKS_LLM_MAX_TOKENS=2048

# API
BANKS_API_HOST=127.0.0.1
BANKS_API_PORT=8080
BANKS_API_KEYS=CAMBIAR_KEY1,CAMBIAR_KEY2
BANKS_LOG_LEVEL=INFO
BANKS_LOG_JSON=true
BANKS_RATE_LIMIT_RPM=60
BANKS_RATE_LIMIT_BURST=10

# Tracing (off por defecto)
BANKS_TRACING=off
ENVEOF
    chmod 600 "$ENV_FILE"
    echo "    Creado. EDITAR $ENV_FILE antes de iniciar el servicio."
else
    echo "    Ya existe, no se sobreescribe."
fi

# ── 6. Systemd units ──────────────────────────────────────────────────────
echo "6/7 Instalando systemd units..."
cp "$INSTALL_DIR/deploy/systemd/banks-api.service"    /etc/systemd/system/
cp "$INSTALL_DIR/deploy/systemd/banks-ingest.service" /etc/systemd/system/

systemctl daemon-reload
systemctl enable banks-api
echo "    banks-api.service instalado y habilitado."

# ── 7. Verificación ───────────────────────────────────────────────────────
echo "7/7 Verificación previa al inicio..."
"$INSTALL_DIR/.venv/bin/python" -c "import banks_rag; print('    banks_rag importado OK:', banks_rag.__version__)"
"$INSTALL_DIR/.venv/bin/banks-eval" routing 2>/dev/null && echo "    SQL routing golden set: OK" || true

echo ""
echo "=== Setup completo ==="
echo ""
echo "Próximos pasos:"
echo "  1. Editar $ENV_FILE (contraseñas, rutas de modelos, API keys)"
echo "  2. Configurar PostgreSQL + pgvector (ver docs/DEPLOYMENT_H100.md §3)"
echo "  3. Cargar documentos:"
echo "     systemctl start banks-ingest"
echo "  4. Iniciar API:"
echo "     systemctl start banks-api"
echo "  5. Validar:"
echo "     curl http://localhost:8080/healthz"
echo "     bash $INSTALL_DIR/scripts/validate_h100.sh"
