"""Adapters de TTS para la voz JARVIS. Producen WAV a partir de texto.

Dos motores, elegidos por ``BANKS_TTS_ENGINE``:

- ``sapi`` (default): voz del SO vía ``pyttsx3`` (SAPI en Windows) + efecto DSP
  ``apply_jarvis_effect`` (band-pass + flanger + reverb). NO descarga modelos
  (~15 MB de wheels). Habla EN y ES según la voz instalada en el SO.
- ``piper``: modelo neural ``jgkawell/jarvis`` (en_GB) en
  ``models/jgkawell--jarvis/``. Solo inglés; el texto debe ir en inglés.

Carga perezosa (``pyttsx3``/``piper`` solo se importan al sintetizar) y
best-effort: si falta la librería o el modelo, lanza ``TTSError`` con un mensaje
accionable en vez de romper el arranque del server.
"""

from __future__ import annotations

import io
import logging
import threading
import wave
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config import MODELS_DIR
from .config import tts_enabled as _settings_tts_enabled
from .config import tts_engine as _settings_tts_engine
from .config import tts_model as _settings_tts_model
from .config import tts_model_es as _settings_tts_model_es

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    """TTS no disponible o falló (sin piper-tts, modelo ausente, síntesis)."""


@runtime_checkable
class TTSEngine(Protocol):
    name: str
    loaded: bool

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        """Devuelve audio WAV (bytes) para ``text`` en el idioma ``lang``."""
        ...


class SapiTTSEngine:
    """TTS con la voz del SO (pyttsx3 → SAPI en Windows) + efecto DSP JARVIS.

    NO descarga modelos: usa las voces instaladas en el SO. Aplica el efecto
    ``apply_jarvis_effect`` (band-pass + flanger + reverb) para el timbre
    metálico. La voz se elige por idioma con ``voices.pick_voice_id``."""

    name = "sapi+jarvis-fx"

    def __init__(self) -> None:
        self.loaded = False

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        if not (text or "").strip():
            raise TTSError("Texto vacío para sintetizar.")
        try:
            import pyttsx3
        except ImportError as exc:
            raise TTSError("Falta 'pyttsx3'. Instálalo: pip install pyttsx3") from exc

        import tempfile
        from pathlib import Path

        from .effects import apply_jarvis_effect
        from .voices import RATE, pick_voice_id

        try:
            engine = pyttsx3.init()
            voice_id = pick_voice_id(engine.getProperty("voices"), lang)
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.setProperty("rate", RATE)
            tmp = Path(tempfile.gettempdir()) / f"jarvis_{lang}_{id(text)}.wav"
            engine.save_to_file(text, str(tmp))
            engine.runAndWait()
            raw = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Falló la síntesis SAPI: {exc}") from exc
        self.loaded = True
        return apply_jarvis_effect(raw)


class PiperTTSEngine:
    """TTS neural con ``piper-tts``. Una voz por idioma (offline, desde ``models/``):
    EN = JARVIS británico (jgkawell/jarvis) + 'sala sutil'; ES = voz latina 'gevy'
    (es_MX) plana. Los modelos se cargan perezosamente, uno por idioma."""

    def __init__(self, models: dict[str, str]) -> None:
        # {"en": ruta_onnx, "es": ruta_onnx}; 'en' es el fallback obligatorio.
        self._models = {k: v for k, v in models.items() if v}
        self._voices: dict[str, Any] = {}
        self.name = self._models.get("en", next(iter(self._models.values()), ""))
        self.loaded = False
        self._lock = threading.Lock()  # serializa carga + espeak_voice + synth

    @staticmethod
    def _resolve(model: str) -> Path:
        p = Path(model)
        return p if p.is_absolute() else MODELS_DIR / model

    def _voice_for(self, lang: str) -> Any:
        """Voz piper para ``lang`` (carga perezosa); cae a 'en' si no hay modelo."""
        key = lang if lang in self._models else "en"
        if key in self._voices:
            return self._voices[key]
        try:
            from piper import PiperVoice  # import perezoso
        except ImportError as exc:
            raise TTSError(
                "Falta 'piper-tts'. Instálalo en el servidor: pip install piper-tts"
            ) from exc
        onnx = self._resolve(self._models[key])
        if not onnx.exists():
            raise TTSError(
                f"Modelo de voz no encontrado: {onnx}. Descarga el .onnx + .onnx.json "
                f"(EN: jgkawell/jarvis · ES: es_MX-gevy) a models/."
            )
        voice = PiperVoice.load(str(onnx))   # carga el .onnx.json vecino
        self._voices[key] = voice
        self.loaded = True
        log.info("Piper TTS cargado [%s]: %s", key, onnx.name)
        return voice

    def load(self) -> None:
        """Precarga la voz inglesa (las demás se cargan al usarse)."""
        self._voice_for("en")

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        """Sintetiza ``text`` con la voz del perfil de ``lang``: EN = JARVIS
        británico + reverb 'sala sutil'; ES = voz latina 'gevy' natural y plana."""
        if not (text or "").strip():
            raise TTSError("Texto vacío para sintetizar.")

        from .effects import apply_jarvis_effect
        from .voices import piper_profile, piper_syn_config

        prof = piper_profile(lang)
        syn = piper_syn_config(lang)
        buf = io.BytesIO()
        try:
            with self._lock:
                voice = self._voice_for(lang)
                if prof.espeak_voice:  # None = usa la fonemización del modelo
                    voice.config.espeak_voice = prof.espeak_voice
                with wave.open(buf, "wb") as wav_file:
                    # piper-tts >= 1.3: synthesize_wav(text, wav_file, syn_config=...).
                    # Fallback a la firma vieja para versiones previas.
                    try:
                        voice.synthesize_wav(text, wav_file, syn_config=syn)
                    except TypeError:  # pragma: no cover - compat versión antigua
                        voice.synthesize_wav(text, wav_file)
        except TTSError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Falló la síntesis de audio: {exc}") from exc

        wav = buf.getvalue()
        if prof.fx is not None:  # EN = reverb 'sala sutil'; ES = plano (None)
            wav = apply_jarvis_effect(wav, prof.fx)
        return wav

    def info(self) -> dict:
        return {"name": self.name, "loaded": self.loaded}


# ── Singleton + flags (leídos de Settings → .env) ────────────────────────────
_DEFAULT_TTS: TTSEngine | None = None
_DEFAULT_TTS_LOCK = threading.Lock()


def tts_enabled() -> bool:
    return _settings_tts_enabled()


def build_default_tts() -> TTSEngine | None:
    """Engine compartido del proceso, o ``None`` si TTS está deshabilitado.
    Elige el motor por BANKS_TTS_ENGINE: 'sapi' (voz del SO + efecto DSP, sin
    modelos) o 'piper' (modelo neural). Carga perezosa."""
    global _DEFAULT_TTS
    if not tts_enabled():
        return None
    if _DEFAULT_TTS is not None:
        return _DEFAULT_TTS
    with _DEFAULT_TTS_LOCK:
        if _DEFAULT_TTS is None:
            engine = (_settings_tts_engine() or "sapi").strip().lower()
            if engine == "piper":
                _DEFAULT_TTS = PiperTTSEngine(
                    {"en": _settings_tts_model(), "es": _settings_tts_model_es()}
                )
            else:
                _DEFAULT_TTS = SapiTTSEngine()
        return _DEFAULT_TTS


def reset_default_tts() -> None:
    global _DEFAULT_TTS
    with _DEFAULT_TTS_LOCK:
        _DEFAULT_TTS = None
