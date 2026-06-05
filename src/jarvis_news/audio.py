"""Helpers de audio compartidos por la CLI (scripts/) y la API (api.py).

``translate_to_english`` reusa el LLM del agente (la voz JARVIS es en_GB);
``synthesize_wav`` envuelve el engine de voz; ``wav_to_flac``/``encode_audio``
comprimen a FLAC (más liviano). Centralizados aquí para no duplicar la lógica
entre el script ejecutable y el endpoint /v1/tts.
"""

from __future__ import annotations

import asyncio
import io
import logging

from .textnorm import to_speakable_text
from .tts import TTSError, build_default_tts

log = logging.getLogger(__name__)

_TRANSLATE_SYSTEM = "You are a professional translator. Output only the translation, nothing else."
_TRANSLATE_PROMPT = (
    "Translate the following Spanish narration into natural British English suitable "
    "for being read aloud (clear flowing sentences, expand figures). Keep it as a "
    "spoken narration, not a list. Text:\n\n"
)


class FlacUnavailableError(RuntimeError):
    """No se pudo codificar FLAC (falta ``soundfile``)."""


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
    ('en'|'es'). Normaliza Markdown → texto hablable (no lee ``#``, ``*``, ``1.``)
    antes de sintetizar. Lanza ``TTSError`` si el TTS está deshabilitado o falla."""
    engine = build_default_tts()
    if engine is None:
        raise TTSError("TTS deshabilitado (BANKS_TTS_ENABLED=false).")
    return engine.synthesize(to_speakable_text(text), lang)


def wav_to_flac(wav_bytes: bytes) -> bytes:
    """WAV (bytes) → FLAC (bytes) con ``soundfile`` (lossless, ~40-60% del tamaño).

    Import perezoso: ``soundfile`` es opcional (sus wheels incluyen libsndfile, así
    que funciona offline). Lanza ``FlacUnavailableError`` si no está instalado."""
    try:
        import soundfile as sf  # import perezoso (dependencia opcional del TTS)
    except Exception as exc:  # ImportError o falta de libsndfile
        raise FlacUnavailableError(
            "Falta 'soundfile' (FLAC). Instálalo: pip install soundfile"
        ) from exc
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="FLAC")
    return buf.getvalue()


def encode_audio(wav_bytes: bytes, fmt: str = "flac") -> tuple[bytes, str]:
    """Codifica el WAV al formato pedido. Devuelve ``(bytes, extensión)``.

    ``fmt='flac'`` comprime con FLAC; si ``soundfile`` no está, cae a WAV con un
    warning (no rompe). Cualquier otro valor devuelve el WAV tal cual."""
    if (fmt or "wav").lower() == "flac":
        try:
            return wav_to_flac(wav_bytes), "flac"
        except FlacUnavailableError as exc:
            log.warning("FLAC no disponible (%s); se guarda WAV.", exc)
    return wav_bytes, "wav"


async def synthesize_bilingual(
    narration_es: str, llm, *, fmt: str = "flac", max_tokens: int = 4096,
) -> dict[str, tuple[bytes, str]]:
    """Sintetiza el RELATO en ESPAÑOL (tal cual) y en INGLÉS (traducido con el LLM),
    ambos con voz JARVIS, y los codifica en ``fmt`` (default FLAC).

    ``narration_es`` es el relato hablable (ver ``report.narrate_report``), no el
    Markdown del informe. Devuelve ``{"es": (data, ext), "en": (data, ext)}``."""
    clean_es = to_speakable_text(narration_es)
    wav_es = await asyncio.to_thread(synthesize_wav, clean_es, "es")
    text_en = await translate_to_english(llm, clean_es, max_tokens=max_tokens)
    wav_en = await asyncio.to_thread(synthesize_wav, text_en, "en")
    return {
        "es": encode_audio(wav_es, fmt),
        "en": encode_audio(wav_en, fmt),
    }
