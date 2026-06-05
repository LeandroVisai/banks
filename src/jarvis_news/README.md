# jarvis_news

Paquete **aislado** (fuera de `banks_rag`) para el *analizador de noticias* con
voz JARVIS. Vive aparte por seguridad: procesa datos externos web-scrapeados
(`data_pipeline/Noticias_scrapping/*.json`) y no debe acoplarse al core RAG.

Lo único que toma del core es:
- el **LLM ya cargado** por la app (`app.state.deps.llm`) — no carga su propio modelo;
- `banks_rag.config` para leer flags/límites del `.env` (vía `os.getenv` fallback).

## Qué hace

1. **Informe analítico** (`report.py`): lee el JSON de noticias, prioriza por
   alcance × peso de tópico (banco central, economía internacional…), y genera
   con **map-reduce** sobre el LLM (las ~110 noticias no caben en una ventana)
   un informe en **Markdown estructurado** (bloques temáticos con *Síntesis
   técnica / Implicancias / Relación entre noticias* y *Referencias* en APA).
2. **HTML** (`html_report.py`): convierte ese Markdown, de forma determinista, al
   HTML de la plantilla de referencia (hero + recuadro-disclaimer + bloques).
3. **Audio JARVIS** (`report.narrate_report` + `audio.py` + `tts.py` +
   `effects.py`): deriva un **relato hablable** del informe y lo convierte a voz
   con el timbre de JARVIS, en **español e inglés**, codificado en **FLAC**
   (cae a WAV si falta `soundfile`).

## Voz JARVIS sin modelos descargados (default)

`BANKS_TTS_ENGINE=sapi` usa la voz del SO (`pyttsx3` → SAPI en Windows) y le
aplica un **efecto DSP puro numpy** (`effects.py`): band-pass (~280–3600 Hz),
*flanger* y *reverb* corto. Es la receta clásica del fan-audio de JARVIS
(EQ + modulación de tono + flanger + reverb), **sin descargar ningún modelo**
(~15 MB de wheels en total).

| Motor (`BANKS_TTS_ENGINE`) | Modelo | Peso | Idiomas |
|---|---|---|---|
| `sapi` (default) | ninguno (voz del SO + DSP) | ~15 MB wheels | EN + ES (según voces del SO) |
| `piper` (opt-in) | `jgkawell/jarvis` (neural, en_GB) | ~60 MB | solo EN |

## Instalación del TTS (offline, servidor H100 sin internet)

El efecto DSP usa `numpy`, que ya viene con el core. El motor `sapi` solo añade
`pyttsx3` (+ `pywin32`/`comtypes` en Windows). Para instalar sin internet:

```powershell
# En una máquina Windows CON internet (mismo Python/arquitectura):
.\scripts\build_tts_bundle.ps1            # → jarvis_news_tts_wheels\
# copia jarvis_news_tts_wheels\ al servidor, y allí (offline):
pip install --no-index --find-links jarvis_news_tts_wheels -r src/jarvis_news/requirements-tts.txt
```

(En Linux/macOS: `scripts/build_tts_bundle.sh`, descarga wheels `win_amd64`.)

Luego en el `.env` del servidor:

```ini
BANKS_TTS_ENABLED=true
BANKS_TTS_ENGINE=sapi          # default; voz del SO + efecto DSP JARVIS
```

## Uso

### CLI (genera archivos de texto y audio)

```bash
python scripts/jarvis_news_report.py                 # markdown/ + html/ (JSON más reciente)
python scripts/jarvis_news_report.py --audio         # + narración + audio/ FLAC (ES y EN)
python scripts/jarvis_news_report.py --json otro.json --top-n 30 --out data/news_reports
```

Salidas ordenadas en `data/news_reports/{markdown,html,audio}/reporte_<fecha>.*`.
Flags útiles: `--audio-format wav` (en vez de FLAC), `--no-html`.

Requiere `.env` con `BANKS_LLM_*` (igual que la API) y, para `--audio`,
`BANKS_TTS_ENABLED=true`.

### API (montada en la app banks_rag)

```
POST /v1/news-report   → {report (markdown), report_html, source_file, n_used, n_total}
POST /v1/tts           → audio/wav   (body: {text, lang: "en"|"es", translate_to_en})
```

El router se monta en `banks_rag.interface.api.main` y reusa el LLM de la app.

## Estructura

| Archivo | Responsabilidad |
|---|---|
| `report.py` | Carga + priorización + map-reduce → informe Markdown estructurado; `narrate_report` (relato hablable) |
| `html_report.py` | `render_html_report`: Markdown → HTML determinista (estilo plantilla) |
| `audio.py` | `synthesize_wav`, `synthesize_bilingual` (relato ES + EN traducido), `wav_to_flac`/`encode_audio` (FLAC), `translate_to_english` |
| `tts.py` | `PiperTTSEngine` (default) / `SapiTTSEngine` (fallback), singleton, flags |
| `effects.py` | `apply_jarvis_effect` (band-pass + flanger + reverb, numpy puro) |
| `voices.py` | Perfiles por idioma (`PIPER_PROFILES`), `pick_voice_id`, `RATE`, `JarvisFx` |
| `config.py` | Rutas propias (modelos, noticias) + flags TTS desde el `.env` |
| `api.py` / `schemas.py` | Router FastAPI y modelos Pydantic (`report` + `report_html`) |
| `requirements-tts*.txt` | Deps del TTS (offline bundle); `soundfile` para FLAC |

## Tests

```bash
PYTHONPATH=src pytest tests/unit/test_jarvis_effects.py tests/unit/test_jarvis_tts.py \
                      tests/unit/test_jarvis_html_report.py tests/unit/test_jarvis_audio.py \
                      tests/unit/test_news_report.py -q
```

Los tests no requieren `pyttsx3`, `piper` ni modelos: el efecto DSP se valida con
señales sintéticas y el TTS con mocks.
