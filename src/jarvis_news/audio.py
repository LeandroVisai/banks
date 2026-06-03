"""Helpers de audio compartidos por la CLI (scripts/) y la API (api.py).

``translate_to_english`` reusa el LLM del agente (la voz JARVIS es en_GB);
``synthesize_wav`` envuelve el engine Piper. Centralizados aquí para no duplicar
la lógica entre el script ejecutable y el endpoint /v1/tts.
"""

from __future__ import annotations

import asyncio

from .tts import TTSError, build_default_tts

_TRANSLATE_SYSTEM = "You are a professional translator. Output only the translation, nothing else."
_TRANSLATE_PROMPT = (
    "Translate the following Spanish text into natural British English suitable "
    "for being read aloud (clear sentences, expand figures). Text:\n\n"
)


async def translate_to_english(llm, text: str, *, max_tokens: int = 4096) -> str:
    """Traduce ``text`` al inglés con el LLM (para la voz JARVIS en_GB)."""
    res = await llm.generate(
        [{"role": "system", "content": _TRANSLATE_SYSTEM},
         {"role": "user", "content": _TRANSLATE_PROMPT + text}],
        tools=None, temperature=0.3, max_tokens=max_tokens,
    )
    return (res.text or "").strip() or text


def synthesize_wav(text: str, lang: str = "en") -> bytes:
    """Texto → WAV con voz JARVIS (motor según BANKS_TTS_ENGINE) en ``lang``
    ('en'|'es'). Lanza ``TTSError`` si el TTS está deshabilitado o falla."""
    engine = build_default_tts()
    if engine is None:
        raise TTSError("TTS deshabilitado (BANKS_TTS_ENABLED=false).")
    return engine.synthesize(text, lang)


async def synthesize_bilingual(report_es: str, llm, *, max_tokens: int = 4096) -> dict[str, bytes]:
    """Genera el audio en ESPAÑOL (texto tal cual) y en INGLÉS (traducido con el
    LLM), ambos con voz JARVIS. Devuelve ``{"es": wav, "en": wav}``."""
    es = await asyncio.to_thread(synthesize_wav, report_es, "es")
    text_en = await translate_to_english(llm, report_es, max_tokens=max_tokens)
    en = await asyncio.to_thread(synthesize_wav, text_en, "en")
    return {"es": es, "en": en}
