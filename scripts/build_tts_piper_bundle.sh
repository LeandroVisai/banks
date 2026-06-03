#!/usr/bin/env bash
# Genera el BUNDLE de wheels del motor TTS NEURAL "piper" (voz JARVIS auténtica)
# para instalar OFFLINE. Variante Linux/macOS: descarga wheels para Windows
# (el servidor es Windows H100). ~40 MB.
#
#   ./scripts/build_tts_piper_bundle.sh
#   # copia jarvis_news_piper_wheels/ al servidor Windows, y allí:
#   pip install --no-index --find-links jarvis_news_piper_wheels -r src/jarvis_news/requirements-tts-piper.txt
#
# OJO: además hay que copiar los modelos .onnx a models/ (no son wheels):
#   models/jgkawell--jarvis/  (EN)   ·   models/es_MX-gevy/  (ES)
set -euo pipefail

OUT="${1:-jarvis_news_piper_wheels}"
REQ="src/jarvis_news/requirements-tts-piper.txt"

echo "Descargando wheels (win_amd64, py3.12) de $REQ → $OUT ..."
python -m pip download -r "$REQ" \
  --platform win_amd64 --python-version 312 --only-binary=:all: \
  -d "$OUT"

echo "✓ Bundle en $OUT ($(du -sh "$OUT" | cut -f1)). En el servidor (offline):"
echo "  pip install --no-index --find-links $OUT -r $REQ"
echo "Y copia los modelos .onnx a models/ (jgkawell--jarvis, es_MX-gevy)."
