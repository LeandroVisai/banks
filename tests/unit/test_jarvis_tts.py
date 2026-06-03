"""Tests del adapter TTS de jarvis_news (sin piper instalado: best-effort)."""

from __future__ import annotations

import pytest

from banks_rag.config import override_settings, reset_settings
from banks_rag.config.settings import Settings
from jarvis_news.tts import (
    PiperTTSEngine,
    SapiTTSEngine,
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

    def test_enabled_builds_piper_by_default(self) -> None:
        # Default BANKS_TTS_ENGINE=piper → voz JARVIS auténtica neural (EN+ES).
        override_settings(Settings(tts_enabled=True))
        engine = build_default_tts()
        assert isinstance(engine, PiperTTSEngine)
        assert build_default_tts() is engine  # singleton
        assert engine.loaded is False  # carga perezosa

    def test_engine_sapi_opt_in(self) -> None:
        # Fallback explícito: voz del SO + DSP (sin modelos).
        override_settings(Settings(tts_enabled=True, tts_engine="sapi"))
        assert isinstance(build_default_tts(), SapiTTSEngine)

    def test_piper_engine_has_en_and_es_models(self) -> None:
        # El motor piper carga DOS voces: EN (JARVIS) y ES (gevy latino).
        override_settings(Settings(
            tts_enabled=True, tts_engine="piper",
            tts_model="jarvis/en.onnx", tts_model_es="gevy/es.onnx",
        ))
        engine = build_default_tts()
        assert isinstance(engine, PiperTTSEngine)
        assert engine._models == {"en": "jarvis/en.onnx", "es": "gevy/es.onnx"}


@pytest.mark.unit
class TestPiperLangProfiles:
    """Perfiles por idioma del motor neural: EN con efecto, ES plano."""

    def test_en_profile_has_room_fx_and_british_phonemes(self) -> None:
        from jarvis_news.voices import piper_profile

        prof = piper_profile("en")
        assert prof.espeak_voice == "en-gb-x-rp"
        assert prof.fx is not None  # 'sala sutil'

    def test_es_profile_is_flat_and_uses_model_phonemes(self) -> None:
        from jarvis_news.voices import piper_profile

        prof = piper_profile("es")
        assert prof.espeak_voice is None  # usa la fonemización del modelo gevy
        assert prof.fx is None  # voz plana, sin efecto (gevy crudo)

    def test_unknown_lang_falls_back_to_en(self) -> None:
        from jarvis_news.voices import piper_profile

        assert piper_profile("xx").fx is piper_profile("en").fx


@pytest.mark.unit
class TestSynthesizeErrors:
    def test_empty_text_raises(self) -> None:
        with pytest.raises(TTSError, match="vac"):
            PiperTTSEngine({"en": "x.onnx"}).synthesize("   ")

    def test_missing_piper_or_model_degrades_to_error(self) -> None:
        # Sin piper-tts (ni modelo) → TTSError accionable, no excepción cruda.
        with pytest.raises(TTSError):
            PiperTTSEngine({"en": "inexistente/modelo.onnx"}).synthesize("hola")


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
