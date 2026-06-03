# Genera el BUNDLE de wheels del motor TTS NEURAL "piper" (voz JARVIS auténtica)
# para instalar OFFLINE en el servidor Windows H100 (sin internet). ~40 MB.
#
# Correr en una máquina Windows CON internet (mismo Python/arquitectura que el
# servidor: py3.12 / win_amd64). Luego copia la carpeta de wheels al servidor.
#
#   .\scripts\build_tts_piper_bundle.ps1
#   # copia .\jarvis_news_piper_wheels\ al servidor, y allí:
#   pip install --no-index --find-links jarvis_news_piper_wheels -r src/jarvis_news/requirements-tts-piper.txt
#
# OJO: además hay que copiar los modelos .onnx a models/ (no son wheels):
#   models/jgkawell--jarvis/     (EN, ~60 MB)
#   models/es_MX-gevy/           (ES, ~60 MB)

param([string]$OutDir = "jarvis_news_piper_wheels")

$req = "src/jarvis_news/requirements-tts-piper.txt"
Write-Host "Descargando wheels de $req → $OutDir ..."
python -m pip download -r $req -d $OutDir
if ($LASTEXITCODE -ne 0) { Write-Error "pip download falló"; exit 1 }

$size = (Get-ChildItem $OutDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("✓ Bundle listo en {0} ({1:N1} MB)" -f $OutDir, $size)
Write-Host "En el servidor (offline):"
Write-Host "  pip install --no-index --find-links $OutDir -r $req"
Write-Host "Y copia los modelos .onnx a models/ (jgkawell--jarvis, es_MX-gevy)."
