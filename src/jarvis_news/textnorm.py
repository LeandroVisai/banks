"""Normaliza Markdown → texto hablable para el TTS.

El reporte se genera en Markdown (encabezados ``###``, viñetas ``-``, listas
numeradas ``1.``, ``**negritas**``). Si se pasa crudo al TTS, la voz lee
"almohadilla almohadilla", "asterisco" o "uno punto". ``to_speakable_text``
quita esa sintaxis y deja frases limpias con pausas naturales (puntos), sin
tocar las cifras reales del texto (años, porcentajes, montos).
"""

from __future__ import annotations

import re

# Emojis y símbolos decorativos comunes (rango básico; no exhaustivo).
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF←-⇿⬀-⯿]"
)


def _strip_inline(line: str) -> str:
    """Quita marcas inline (negrita/itálica/código/links) de una línea."""
    # Links [texto](url) → texto
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
    # Imágenes ![alt](url) → (nada)
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
    # Negrita / itálica: **x**, __x__, *x*, _x_ → x
    line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
    line = re.sub(r"(\*|_)(.+?)\1", r"\2", line)
    # Código `x` → x
    line = line.replace("`", "")
    return line


def to_speakable_text(md: str) -> str:
    """Convierte Markdown en texto plano apto para leer en voz alta."""
    if not md:
        return ""

    out: list[str] = []
    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Separadores horizontales (---, ***, ___) → se omiten.
        if re.fullmatch(r"[-*_]{3,}", line):
            continue

        # Encabezados: ### Título → "Título." (el punto fuerza una pausa).
        line = re.sub(r"^#{1,6}\s*", "", line)

        # Viñetas: "- item" / "* item" / "+ item" → "item"
        line = re.sub(r"^[-*+]\s+", "", line)

        # Listas numeradas: "1. item" / "2) item" → "item" (quita el número guía,
        # SIN tocar cifras dentro del texto).
        line = re.sub(r"^\d+[.)]\s+", "", line)

        # Citas: "> texto" → "texto"
        line = re.sub(r"^>\s*", "", line)

        line = _strip_inline(line)
        line = _EMOJI.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue

        # Asegura puntuación final para una pausa natural entre bloques.
        if line[-1] not in ".!?:;,":
            line += "."
        out.append(line)

    return " ".join(out)
