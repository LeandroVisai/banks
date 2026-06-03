"""Config local de jarvis_news (aislada del core).

Paths propios para no depender de ``banks_rag.config.paths``. Los flags TTS sí
se leen de ``banks_rag`` Settings (que carga el ``.env`` con prefijo BANKS_),
para que el usuario los configure desde el mismo ``.env`` — único punto de
contacto con el core, de solo lectura.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/jarvis_news/config.py → raíz del repo (3 niveles arriba).
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
NEWS_SCRAPING_DIR = ROOT / "data_pipeline" / "Noticias_scrapping"


def tts_enabled() -> bool:
    """Lee BANKS_TTS_ENABLED del .env vía Settings; fallback a env del proceso."""
    try:
        from banks_rag.config import get_settings
        return get_settings().tts_enabled
    except Exception:  # noqa: BLE001 - sin core, cae a env del proceso
        return os.getenv("BANKS_TTS_ENABLED", "false").strip().lower() in (
            "1", "true", "yes", "on",
        )


def tts_model() -> str:
    try:
        from banks_rag.config import get_settings
        return get_settings().tts_model
    except Exception:  # noqa: BLE001
        return os.getenv("BANKS_TTS_MODEL", "jgkawell--jarvis/jarvis-medium.onnx")


def tts_engine() -> str:
    """'sapi' (voz del SO + efecto DSP, sin modelos) o 'piper' (modelo neural)."""
    try:
        from banks_rag.config import get_settings
        return get_settings().tts_engine
    except Exception:  # noqa: BLE001
        return os.getenv("BANKS_TTS_ENGINE", "sapi")
