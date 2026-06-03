"""Selección de la voz base del SO (SAPI) por idioma + parámetros del efecto.

No descarga modelos: usa las voces instaladas en el SO. Las pistas (substrings)
eligen la "mejor" voz disponible por idioma; el efecto JARVIS (effects.py) se
aplica encima para el timbre metálico.
"""

from __future__ import annotations

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

# Parámetros del efecto JARVIS (ver effects.JarvisFx).
JARVIS_FX = JarvisFx()


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
