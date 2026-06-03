#!/usr/bin/env bash
# Genera el BUNDLE de wheels del TTS de jarvis_news para instalar OFFLINE.
# Variante Linux/macOS: descarga wheels para Windows (el servidor es Windows).
# Idealmente corre el .ps1 EN Windows (más fiable para pywin32).
#
#   ./scripts/build_tts_bundle.sh
#   # copia jarvis_news_tts_wheels/ al servidor Windows, y allí:
#   pip install --no-index --find-links jarvis_news_tts_wheels -r src/jarvis_news/requirements-tts.txt
set -euo pipefail

OUT="${1:-jarvis_news_tts_wheels}"
REQ="src/jarvis_news/requirements-tts.txt"

echo "Descargando wheels (win_amd64, py3.12) de $REQ → $OUT ..."
python -m pip download -r "$REQ" \
  --platform win_amd64 --python-version 312 --only-binary=:all: \
  -d "$OUT"

echo "✓ Bundle en $OUT ($(du -sh "$OUT" | cut -f1)). En el servidor (offline):"
echo "  pip install --no-index --find-links $OUT -r $REQ"
