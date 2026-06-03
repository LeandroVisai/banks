# Genera el BUNDLE de wheels del TTS de jarvis_news para instalar OFFLINE en el
# servidor (sin internet) — motor "sapi", SIN modelos. ~15 MB.
#
# Correr en una máquina Windows CON internet (idealmente mismo Python/arquitectura
# que el servidor H100). Luego copia la carpeta de wheels al servidor.
#
#   .\scripts\build_tts_bundle.ps1
#   # copia .\jarvis_news_tts_wheels\ al servidor, y allí:
#   pip install --no-index --find-links jarvis_news_tts_wheels -r src/jarvis_news/requirements-tts.txt

param([string]$OutDir = "jarvis_news_tts_wheels")

$req = "src/jarvis_news/requirements-tts.txt"
Write-Host "Descargando wheels de $req → $OutDir ..."
python -m pip download -r $req -d $OutDir
if ($LASTEXITCODE -ne 0) { Write-Error "pip download falló"; exit 1 }

$size = (Get-ChildItem $OutDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("✓ Bundle listo en {0} ({1:N1} MB)" -f $OutDir, $size)
Write-Host "En el servidor (offline):"
Write-Host "  pip install --no-index --find-links $OutDir -r $req"
