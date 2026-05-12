#!/usr/bin/env bash
# package_for_h100.sh — Corre en la WORKSTATION (Mac/Linux con internet).
#
# Prepara un tarball listo para transferir al H100:
#   - wheels de todas las dependencias (para instalar offline)
#   - código fuente del repo (sin modelos ni parquets)
#
# Uso:
#   bash scripts/package_for_h100.sh [--output /ruta/destino]
#
# El tarball resultante (~500MB–2GB dependiendo de las deps) se transfiere con:
#   sftp <user>@h100-server
#   put banks_rag_<fecha>.tar.gz /tmp/
#
# Los modelos (.gguf, carpetas HuggingFace) se transfieren por separado con rsync
# porque son muy grandes para incluir en el tarball:
#   rsync -avz --progress models/ <user>@h100:/opt/banks_rag/models/
#   rsync -avz --progress data_pipeline/snapshots/ <user>@h100:/opt/banks_rag/data_pipeline/snapshots/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-/tmp}"
DATE="$(date +%Y%m%d_%H%M%S)"
TARBALL_NAME="banks_rag_${DATE}.tar.gz"
WORK_DIR="$(mktemp -d)"

trap 'rm -rf "$WORK_DIR"' EXIT

echo "=== Banks RAG — Empaquetado para H100 ==="
echo "Repo:    $REPO_ROOT"
echo "Destino: $OUTPUT_DIR/$TARBALL_NAME"
echo ""

# ── 1. Descargar wheels ────────────────────────────────────────────────────
# Requiere Python 3.11 en la workstation con la misma plataforma que el H100.
# Si la plataforma difiere (Mac → Linux), usar:
#   pip download ... --platform manylinux2014_x86_64 --python-version 311
#   pip download ... --platform linux_aarch64 ...

WHEELS_DIR="$WORK_DIR/wheels"
mkdir -p "$WHEELS_DIR"

echo "1/4 Descargando wheels (puede tardar 3-5 min)..."

# Detectar si estamos en Mac (cross-platform download) o Linux nativo
if [[ "$(uname)" == "Darwin" ]]; then
    echo "    Detectado Mac → descarga para Linux x86_64 (H100)"
    pip download \
        --dest "$WHEELS_DIR" \
        --platform manylinux2014_x86_64 \
        --python-version 311 \
        --only-binary=:all: \
        "$REPO_ROOT[api]" 2>&1 | tail -5

    # llama-cpp-python no tiene wheels prebuilt para todas las plataformas;
    # se descarga el source wheel y se compila en el server.
    pip download \
        --dest "$WHEELS_DIR" \
        --no-deps \
        "llama-cpp-python" 2>&1 | tail -3 || true

    # sentence-transformers y torch (GPU)
    pip download \
        --dest "$WHEELS_DIR" \
        --platform manylinux2014_x86_64 \
        --python-version 311 \
        --only-binary=:all: \
        "torch==2.3.0+cu121" \
        --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5 || true
else
    echo "    Detectado Linux → descarga nativa"
    pip download --dest "$WHEELS_DIR" "$REPO_ROOT[api]"
fi

echo "    Wheels descargados: $(ls "$WHEELS_DIR" | wc -l)"

# ── 2. Copiar código fuente (sin modelos, parquets, ni cache) ──────────────
echo "2/4 Copiando código fuente..."
CODE_DIR="$WORK_DIR/banks_rag_src"
mkdir -p "$CODE_DIR"

rsync -a \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='models/' \
    --exclude='models_cache/' \
    --exclude='data_pipeline/snapshots/' \
    --exclude='data/chunks*.json' \
    --exclude='data/images/' \
    --exclude='data/logs_intermedios/' \
    --exclude='.venv/' \
    --exclude='dist/' \
    --exclude='*.egg-info/' \
    --exclude='htmlcov/' \
    --exclude='.coverage' \
    "$REPO_ROOT/" "$CODE_DIR/"

# ── 3. Agregar script de instalación ──────────────────────────────────────
cp "$REPO_ROOT/scripts/setup_h100.sh" "$CODE_DIR/scripts/"

# ── 4. Empaquetar ─────────────────────────────────────────────────────────
echo "3/4 Creando tarball..."
tar -czf "$OUTPUT_DIR/$TARBALL_NAME" \
    -C "$WORK_DIR" \
    "banks_rag_src" \
    "wheels"

SIZE="$(du -sh "$OUTPUT_DIR/$TARBALL_NAME" | cut -f1)"
echo "4/4 Listo: $OUTPUT_DIR/$TARBALL_NAME ($SIZE)"
echo ""
echo "Próximos pasos:"
echo "  1. Transferir tarball al H100:"
echo "     scp $OUTPUT_DIR/$TARBALL_NAME <user>@h100:/tmp/"
echo ""
echo "  2. Transferir modelos (por separado, son grandes):"
echo "     rsync -avz --progress $REPO_ROOT/models/ <user>@h100:/opt/banks_rag/models/"
echo "     rsync -avz --progress $REPO_ROOT/data_pipeline/snapshots/ <user>@h100:/opt/banks_rag/data_pipeline/snapshots/"
echo ""
echo "  3. En el H100:"
echo "     cd /tmp && tar -xzf $TARBALL_NAME"
echo "     bash banks_rag_src/scripts/setup_h100.sh"
