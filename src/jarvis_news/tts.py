"""Adapter de TTS con Piper (offline, ONNX). Produce WAV a partir de texto.

Voz JARVIS: modelo ``jgkawell/jarvis`` (en_GB) en ``models/jgkawell--jarvis/``.
Piper habla el idioma del modelo; la voz JARVIS es inglés, así que el texto que
se le pasa debe estar en inglés (la traducción la hace el endpoint con el LLM).

Carga perezosa (``piper`` solo se importa al sintetizar) y best-effort: si falta
la librería o el modelo, lanza ``TTSError`` con un mensaje accionable en vez de
romper el arranque del server.
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
from .config import tts_model as _settings_tts_model

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    """TTS no disponible o falló (sin piper-tts, modelo ausente, síntesis)."""


@runtime_checkable
class TTSEngine(Protocol):
    name: str
    loaded: bool

    def synthesize(self, text: str) -> bytes:
        """Devuelve audio WAV (bytes) para ``text``."""
        ...


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

    def synthesize(self, text: str) -> bytes:
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
_DEFAULT_TTS: PiperTTSEngine | None = None
_DEFAULT_TTS_LOCK = threading.Lock()


def tts_enabled() -> bool:
    return _settings_tts_enabled()


def build_default_tts() -> PiperTTSEngine | None:
    """Engine compartido del proceso, o ``None`` si TTS está deshabilitado.
    No carga el modelo aquí (lazy en el primer ``synthesize``)."""
    global _DEFAULT_TTS
    if not tts_enabled():
        return None
    if _DEFAULT_TTS is not None:
        return _DEFAULT_TTS
    with _DEFAULT_TTS_LOCK:
        if _DEFAULT_TTS is None:
            _DEFAULT_TTS = PiperTTSEngine(_settings_tts_model())
        return _DEFAULT_TTS


def reset_default_tts() -> None:
    global _DEFAULT_TTS
    with _DEFAULT_TTS_LOCK:
        _DEFAULT_TTS = None
