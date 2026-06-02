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


@pytest.mark.unit
class TestAudioHelpers:
    """Helpers compartidos por el script CLI y la API (jarvis_news.audio)."""

    def teardown_method(self) -> None:
        reset_default_tts()
        reset_settings()

    @pytest.mark.asyncio
    async def test_translate_uses_llm(self) -> None:
        from banks_rag.domain.agent import GenerationResult
        from jarvis_news.audio import translate_to_english

        class _LLM:
            async def generate(self, messages, **kwargs):
                # El prompt debe llevar el texto a traducir.
                assert "hola mundo" in messages[-1]["content"]
                return GenerationResult(text="hello world", n_tokens=3)

        assert await translate_to_english(_LLM(), "hola mundo") == "hello world"

    def test_synthesize_wav_raises_when_tts_disabled(self) -> None:
        from jarvis_news.audio import synthesize_wav

        override_settings(Settings(tts_enabled=False))
        with pytest.raises(TTSError):
            synthesize_wav("hello")
