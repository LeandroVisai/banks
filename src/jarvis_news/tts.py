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
    """TTS con ``piper-tts``. Resuelve el modelo desde ``models/`` (offline)."""

    def __init__(self, model: str) -> None:
        self._requested = model
        self._voice: Any = None
        self.name = model
        self.loaded = False

    def _resolve_model(self) -> Path:
        p = Path(self._requested)
        if not p.is_absolute():
            p = MODELS_DIR / self._requested
        return p

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from piper import PiperVoice  # import perezoso
        except ImportError as exc:
            raise TTSError(
                "Falta 'piper-tts'. Instálalo en el servidor: pip install piper-tts"
            ) from exc

        onnx = self._resolve_model()
        if not onnx.exists():
            raise TTSError(
                f"Modelo de voz no encontrado: {onnx}. Descarga jgkawell/jarvis a "
                f"models/jgkawell--jarvis/ (jarvis-medium.onnx + .onnx.json)."
            )
        self._voice = PiperVoice.load(str(onnx))   # carga el .onnx.json vecino
        self.loaded = True
        log.info("Piper TTS cargado: %s", onnx.name)

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        # El idioma lo define el modelo Piper (lang es para compat con la interfaz).
        if not (text or "").strip():
            raise TTSError("Texto vacío para sintetizar.")
        if not self.loaded:
            self.load()
        buf = io.BytesIO()
        try:
            with wave.open(buf, "wb") as wav_file:
                # API de piper-tts: synthesize(text, wave_write). Algunas versiones
                # exponen synthesize_wav; se intenta el fallback.
                if hasattr(self._voice, "synthesize"):
                    self._voice.synthesize(text, wav_file)
                else:  # pragma: no cover - compat versión antigua
                    self._voice.synthesize_wav(text, wav_file)
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Falló la síntesis de audio: {exc}") from exc
        return buf.getvalue()

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
            _DEFAULT_TTS = (
                PiperTTSEngine(_settings_tts_model()) if engine == "piper"
                else SapiTTSEngine()
            )
        return _DEFAULT_TTS


def reset_default_tts() -> None:
    global _DEFAULT_TTS
    with _DEFAULT_TTS_LOCK:
        _DEFAULT_TTS = None
