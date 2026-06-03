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

## 2. Dos motores de voz

El motor **recomendado es `piper`** (neural): da la voz JARVIS **auténtica**. Hay
un fallback `sapi` (voz del SO + DSP) que no descarga modelos.

| Motor (`BANKS_TTS_ENGINE`) | Voz EN | Voz ES | Modelos | Wheels |
|---|---|---|---|---|
| **`piper`** (recomendado) | JARVIS británico real (`jgkawell/jarvis`, en_GB RP) + reverb "sala sutil" | voz latina neural `gevy` (es_MX), natural y plana | 2 × ~60 MB `.onnx` | ~40 MB |
| `sapi` (default, fallback) | voz del SO + DSP metálico | voz del SO + DSP | ninguno | ~15 MB |

**Voz por idioma con piper** (un modelo por idioma; los perfiles viven en
`voices.py` → `PIPER_PROFILES`):

- **EN**: `jgkawell/jarvis` fonemizado en `en-gb-x-rp` (Received Pronunciation),
  pausado (`length_scale` 1.18) + efecto `apply_jarvis_effect` "sala sutil"
  (reverb corta, sin band-pass agresivo ni flanger).
- **ES**: voz `gevy` (es_MX) tal cual — natural, plana, **sin** efecto.

> El texto se normaliza con `textnorm.to_speakable_text` antes de sintetizar, así
> la voz **no lee** el Markdown (`#`, `*`, `1.`, links).

---

## 3. Instalación del TTS (offline, servidor H100 sin internet)

### Motor `piper` (recomendado — JARVIS auténtico)

**Paso 1 — generar el bundle de wheels** (en una máquina Windows CON internet,
mismo Python/arquitectura que el servidor: py3.12 / win_amd64):

```powershell
.\scripts\build_tts_piper_bundle.ps1     # → jarvis_news_piper_wheels\  (~40 MB)
```

(En Linux/macOS para descargar wheels `win_amd64`: `scripts/build_tts_piper_bundle.sh`.)

**Paso 2 — copiar los modelos `.onnx` a `models/`** (~120 MB). Los modelos
**ya vienen incluidos** en el bundle `jarvis_news_piper_wheels/` (subdirectorios
`jgkawell--jarvis/` y `es_MX-gevy/`); en el servidor solo cópialos a `models/`:

```powershell
Copy-Item jarvis_news_piper_wheels\jgkawell--jarvis models\ -Recurse -Force
Copy-Item jarvis_news_piper_wheels\es_MX-gevy       models\ -Recurse -Force
```

(Origen de los modelos, si necesitas rearmarlos:
EN `https://huggingface.co/jgkawell/jarvis` ·
ES `https://huggingface.co/spaces/HirCoir/Piper-TTS-Spanish`.)

**Paso 3 — instalar en el servidor** (offline), con el bundle (los `.onnx` los
ignora `pip`; solo instala los wheels):

```powershell
pip install --no-index --find-links jarvis_news_piper_wheels -r src/jarvis_news/requirements-tts-piper.txt
```

**Paso 4 — activar en el `.env`:**

```ini
BANKS_TTS_ENABLED=true
BANKS_TTS_ENGINE=piper
# rutas por defecto (override opcional):
# BANKS_TTS_MODEL=jgkawell--jarvis/jarvis-medium.onnx
# BANKS_TTS_MODEL_ES=es_MX-gevy/es_MX-gevy-10196-epoch-high.onnx
```

### Fallback `sapi` (sin modelos, ~15 MB)

```powershell
.\scripts\build_tts_bundle.ps1           # → jarvis_news_tts_wheels\
pip install --no-index --find-links jarvis_news_tts_wheels -r src/jarvis_news/requirements-tts.txt
```

```ini
BANKS_TTS_ENABLED=true
BANKS_TTS_ENGINE=sapi          # voz del SO + efecto DSP JARVIS
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
| `BANKS_TTS_ENGINE` | `sapi` | `piper` (neural, JARVIS auténtico — recomendado) o `sapi` (voz del SO + DSP, sin modelos) |
| `BANKS_TTS_MODEL` | `jgkawell--jarvis/jarvis-medium.onnx` | Modelo `.onnx` de la voz **EN** (solo `engine=piper`) |
| `BANKS_TTS_MODEL_ES` | `es_MX-gevy/es_MX-gevy-10196-epoch-high.onnx` | Modelo `.onnx` de la voz **ES** (solo `engine=piper`) |
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
