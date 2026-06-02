"""Helpers de audio compartidos por la CLI (scripts/) y la API (api.py).

``translate_to_english`` reusa el LLM del agente (la voz JARVIS es en_GB);
``synthesize_wav`` envuelve el engine Piper. Centralizados aquí para no duplicar
la lógica entre el script ejecutable y el endpoint /v1/tts.
"""

from __future__ import annotations

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


def synthesize_wav(text: str) -> bytes:
    """Texto (en inglés) → WAV con la voz JARVIS (Piper). Lanza ``TTSError`` si
    el TTS está deshabilitado o falta el modelo/lib."""
    engine = build_default_tts()
    if engine is None:
        raise TTSError("TTS deshabilitado (BANKS_TTS_ENABLED=false).")
    return engine.synthesize(text)
