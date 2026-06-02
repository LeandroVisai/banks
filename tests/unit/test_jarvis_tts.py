"""Tests del adapter TTS de jarvis_news (sin piper instalado: best-effort)."""

from __future__ import annotations

import pytest

from banks_rag.config import override_settings, reset_settings
from banks_rag.config.settings import Settings
from jarvis_news.tts import (
    PiperTTSEngine,
    TTSError,
    build_default_tts,
    reset_default_tts,
    tts_enabled,
)


@pytest.mark.unit
class TestTtsFlags:
    def setup_method(self) -> None:
        reset_default_tts()

    def teardown_method(self) -> None:
        reset_default_tts()
        reset_settings()

    def test_disabled_by_default(self) -> None:
        override_settings(Settings(tts_enabled=False))
        assert tts_enabled() is False
        assert build_default_tts() is None

    def test_enabled_builds_engine_lazy(self) -> None:
        override_settings(Settings(tts_enabled=True, tts_model="x/y.onnx"))
        engine = build_default_tts()
        assert isinstance(engine, PiperTTSEngine)
        assert engine.loaded is False        # carga perezosa
        assert build_default_tts() is engine  # singleton


@pytest.mark.unit
class TestSynthesizeErrors:
    def test_empty_text_raises(self) -> None:
        with pytest.raises(TTSError, match="vac"):
            PiperTTSEngine("x.onnx").synthesize("   ")

    def test_missing_piper_or_model_degrades_to_error(self) -> None:
        # Sin piper-tts (ni modelo) → TTSError accionable, no excepción cruda.
        with pytest.raises(TTSError):
            PiperTTSEngine("inexistente/modelo.onnx").synthesize("hola")
