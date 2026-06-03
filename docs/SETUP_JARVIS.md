# Setup JARVIS — Analizador de noticias + voz JARVIS

Guía de instalación y uso del paquete **aislado** `src/jarvis_news/`: genera el
reporte de prensa del día (texto) y, opcional, un audio con la voz de **JARVIS**
de Iron Man, en **español e inglés**.

> Vive aparte de `banks_rag` **por seguridad**: procesa datos externos
> web-scrapeados (`data_pipeline/Noticias_scrapping/*.json`). Solo toma del core
> el LLM ya cargado y la config del `.env`. Ver [src/jarvis_news/README.md](../src/jarvis_news/README.md).

---

## 1. Qué hace

```
data_pipeline/Noticias_scrapping/*.json   (noticias web-scrapeadas)
  → report.py    prioriza (alcance × peso de tópico) + map-reduce sobre el LLM
  → reporte de TEXTO  (data/news_reports/reporte_<fecha>.md)
  → audio.py     traduce a EN con el LLM + TTS + efecto DSP JARVIS
  → reporte_<fecha>_es.wav  +  reporte_<fecha>_en.wav
```

El LLM es el **mismo Qwen3.6** del agente (se reusa, no se carga otro modelo).

---

## 2. Voz JARVIS sin descargar modelos (motor por defecto)

El motor `sapi` usa la voz del SO (`pyttsx3` → SAPI en Windows) y le aplica un
**efecto DSP puro numpy** (`effects.py`): band-pass (~280–3600 Hz) + *flanger* +
*reverb* corto. Es la receta clásica del fan-audio de JARVIS (EQ + modulación de
tono + flanger + reverb), **sin descargar ningún modelo** (~15 MB de wheels).

| Motor (`BANKS_TTS_ENGINE`) | Modelo | Peso | Idiomas |
|---|---|---|---|
| `sapi` (default) | ninguno (voz del SO + DSP) | ~15 MB wheels | EN + ES |
| `piper` (opt-in) | `jgkawell/jarvis` (neural, en_GB) | ~60 MB | solo EN |

---

## 3. Instalación del TTS (offline, servidor H100 sin internet)

El efecto DSP usa `numpy`, que ya viene con el core. El motor `sapi` solo añade
`pyttsx3` (+ `pywin32`/`comtypes` en Windows).

**Paso 1 — generar el bundle de wheels** (en una máquina Windows CON internet,
mismo Python/arquitectura que el servidor):

```powershell
.\scripts\build_tts_bundle.ps1          # → jarvis_news_tts_wheels\  (~15 MB)
```

(En Linux/macOS para descargar wheels `win_amd64`: `scripts/build_tts_bundle.sh`.)

**Paso 2 — instalar en el servidor** (offline), copiando esa carpeta:

```powershell
pip install --no-index --find-links jarvis_news_tts_wheels -r src/jarvis_news/requirements-tts.txt
```

**Paso 3 — activar en el `.env`:**

```ini
BANKS_TTS_ENABLED=true
BANKS_TTS_ENGINE=sapi          # default; voz del SO + efecto DSP JARVIS
```

### Alternativa: motor neural `piper` (solo EN, ~60 MB)

```ini
BANKS_TTS_ENGINE=piper
# BANKS_TTS_MODEL=jgkawell--jarvis/jarvis-medium.onnx
```

```bash
pip install piper-tts
# descargar jgkawell/jarvis (jarvis-medium.onnx + .onnx.json) a models/jgkawell--jarvis/
```

---

## 4. Uso por CLI (genera los archivos)

Requiere `.env` con `BANKS_LLM_*` (igual que la API). Para `--audio`,
`BANKS_TTS_ENABLED=true`.

```bash
# Solo texto (al JSON más reciente de Noticias_scrapping/):
python scripts/jarvis_news_report.py
python scripts/jarvis_news_report.py --top-n 30 --out data/news_reports
# → data/news_reports/reporte_<archivo>.md

# Texto + AUDIO (voz JARVIS en ES y EN):
python scripts/jarvis_news_report.py --audio
# → además reporte_<archivo>_es.wav y reporte_<archivo>_en.wav

# Otro archivo concreto:
python scripts/jarvis_news_report.py --json otro.json --top-n 30
```

> El LLM es **serializado**: el reporte (map-reduce) compite con el chat por el
> modelo. Conviene correrlo fuera de horario o por tarea programada.

---

## 5. Uso por API (montada en la app banks_rag)

El router se monta en `banks_rag.interface.api.main` y reusa el LLM de la app:

```
POST /v1/news-report   → {report, source_file, n_used, n_total}
POST /v1/tts           → audio/wav   (body: {text, lang: "en"|"es", translate_to_en})
```

```bash
curl -s -XPOST localhost:8080/v1/news-report -H 'content-type: application/json' \
     -d '{"top_n": 25}' | jq -r .report

curl -s -XPOST localhost:8080/v1/tts -H 'content-type: application/json' \
     -d '{"text":"Buenos días, señor.","lang":"es"}' -o jarvis_es.wav
```

---

## 6. Variables de entorno

| Variable | Default | Qué hace |
|---|---|---|
| `BANKS_TTS_ENABLED` | `false` | `true` para habilitar el audio (voz JARVIS) |
| `BANKS_TTS_ENGINE` | `sapi` | `sapi` (voz del SO + DSP, sin modelos) o `piper` (neural) |
| `BANKS_TTS_MODEL` | `jgkawell--jarvis/...` | Modelo `.onnx` (solo si `engine=piper`) |
| `BANKS_LLM_*` | — | El LLM que sintetiza el reporte y traduce a EN |
| `BANKS_SYNTHESIS_MAX_TOKENS` | `4096` | Tope de tokens del reporte / traducción |

---

## 7. Integración al agente (roadmap — pendiente)

Hoy se usa por CLI (`scripts/jarvis_news_report.py`) y por los endpoints
`/v1/news-report` y `/v1/tts`. Lo siguiente, **aún no implementado**:

1. **Tool del agente** `news_report` / `read_aloud`: registrar el reporte y el
   TTS como tools del loop de tool-calling, para que el agente las invoque cuando
   el usuario pida "resúmeme las noticias de hoy" o "léemelo en voz".
2. **Frontend (interface2)**: botón "Reporte de prensa" + reproductor del `.wav`
   (reusar el control TTS de "leer respuesta en voz" ya existente).
3. **Tarea programada**: cron/job nocturno que genere el reporte del día sin
   competir con el chat (el LLM es serializado).

---

## 8. Troubleshooting

| Síntoma | Causa / arreglo |
|---|---|
| `/v1/tts` responde 503 "TTS deshabilitado" | Falta `BANKS_TTS_ENABLED=true` en el `.env` |
| `TTSError: Falta 'pyttsx3'` | Instalar el bundle (sección 3) |
| Audio sin timbre metálico | El efecto DSP se aplica siempre; si suena plano, ajustar `JarvisFx` en `voices.py` |
| Reporte vacío / `mock` | `BANKS_LLM_FAMILY` está en `mock`/vacío: configurar `qwen` + `BANKS_LLM_MODEL_PATH` |
| EN no traduce | El LLM debe estar cargado (`/v1/tts` usa `app.state.deps.llm`) |

---

Ver también: [src/jarvis_news/README.md](../src/jarvis_news/README.md) ·
[docs/SETUP_SERVIDOR.txt](SETUP_SERVIDOR.txt) (sección 7) · `.env.example`.
