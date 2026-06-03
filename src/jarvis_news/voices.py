"""Selección de la voz base del SO (SAPI) por idioma + parámetros del efecto.

No descarga modelos: usa las voces instaladas en el SO. Las pistas (substrings)
eligen la "mejor" voz disponible por idioma; el efecto JARVIS (effects.py) se
aplica encima para el timbre metálico.
"""

from __future__ import annotations

from dataclasses import dataclass

from .effects import JarvisFx

# Pistas para elegir voz por idioma (substring en name/id, case-insensitive).
# Orden = preferencia. JARVIS es británico → en_GB primero; ES latino para Chile.
VOICE_HINTS: dict[str, tuple[str, ...]] = {
    "en": ("george", "ryan", "en-gb", "english (united kingdom)", "daniel",
           "hazel", "english"),
    "es": ("raul", "pablo", "sabina", "helena", "jorge", "spanish", "español", "es-"),
}

# Velocidad de habla (SAPI rate). Más lento = más "JARVIS" (sobrio).
RATE = 165

# Parámetros del efecto JARVIS para la voz SAPI del SO (timbre metálico fuerte,
# compensa que la voz base no es británica).
JARVIS_FX = JarvisFx()

# Preset SUTIL para la voz neural piper (ya es JARVIS británico): solo una reverb
# corta de sala ("IA hablando en la habitación"), sin band-pass agresivo ni
# flanger que ensucien la voz neural. Elegido a oído ("sala sutil").
JARVIS_FX_PIPER = JarvisFx(
    band_low_hz=60.0,
    band_high_hz=9000.0,
    flanger_mix=0.0,
    reverb_ms=45.0,
    reverb_decay=0.18,
    output_peak=0.97,
)

# ── Motor neural piper (dos voces: JARVIS en_GB para EN, gevy es_MX para ES) ──
@dataclass(frozen=True)
class PiperLangProfile:
    """Perfil por idioma del motor neural piper. Cada idioma usa SU modelo
    (config ``tts_model``/``tts_model_es``).

    - ``espeak_voice``: fuerza la fonemización (None = usa la del modelo).
    - ``length_scale`` > 1 = más lento/pausado.
    - ``fx``: efecto JARVIS opcional (reverb 'sala sutil'); None = voz plana.
    """

    espeak_voice: str | None = None
    length_scale: float = 1.15
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8
    fx: JarvisFx | None = None


# EN = JARVIS británico auténtico (en-gb RP) + "sala sutil" (reverb corta), pausado.
# ES = voz neural latina 'gevy' (es_MX) tal cual: natural, plana, sin efecto.
PIPER_PROFILES: dict[str, PiperLangProfile] = {
    "en": PiperLangProfile(
        espeak_voice="en-gb-x-rp", length_scale=1.18, fx=JARVIS_FX_PIPER,
    ),
    "es": PiperLangProfile(
        espeak_voice=None, length_scale=1.15, fx=None,
    ),
}


def piper_profile(lang: str) -> PiperLangProfile:
    """Perfil piper para ``lang`` ('en'|'es'); cae a 'en' si no está definido."""
    return PIPER_PROFILES.get((lang or "en").lower(), PIPER_PROFILES["en"])


def piper_syn_config(lang: str = "en"):
    """``SynthesisConfig`` para ``lang`` según su perfil, o ``None`` si piper no
    está instalado (entonces ``synthesize_wav`` usa los defaults del modelo)."""
    try:
        from piper.config import SynthesisConfig
    except Exception:  # noqa: BLE001 - piper opcional
        return None
    prof = piper_profile(lang)
    return SynthesisConfig(
        length_scale=prof.length_scale,
        noise_scale=prof.noise_scale,
        noise_w_scale=prof.noise_w_scale,
        normalize_audio=True,
        volume=1.0,
    )


def pick_voice_id(voices, lang: str) -> str | None:
    """Elige el id de la mejor voz del SO para ``lang`` ('en'|'es').

    ``voices`` es la lista de pyttsx3 ``engine.getProperty('voices')`` (cada uno
    con ``.id`` y ``.name``). Devuelve ``None`` si no hay coincidencia (el caller
    usa la voz por defecto del SO)."""
    hints = VOICE_HINTS.get(lang, ())
    catalog = [(getattr(v, "id", ""), (getattr(v, "name", "") or "")) for v in voices]
    for hint in hints:
        for vid, name in catalog:
            haystack = f"{vid} {name}".lower()
            if hint in haystack:
                return vid
    return None
