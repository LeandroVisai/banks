"""Normalización de texto extraído de PDFs.

Aplica encoding fixers y limpieza estructural (guiones de fin de línea,
saltos intra-párrafo, watermarks DRM, caracteres de control) preservando
``\\n\\n`` como separador de párrafo.
"""

from __future__ import annotations

import re

from .encoding_fixers import fix_all

_HYPHEN_LINEBREAK_RE = re.compile(r"-\n(\w)")
_INTRA_PARA_NEWLINE_RE = re.compile(r"(?<!\n)\n(?!\n)")
_MULTIPLE_SPACES_RE = re.compile(r"[ \t]+")
_ORPHAN_PAGENUM_RE = re.compile(r"^\s*\d{1,3}\s*$", flags=re.MULTILINE)
_DRM_WATERMARK_RE = re.compile(r"\{\[\{[^}]*\}\]\}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_OCR_CORRUPTION_RE = re.compile(r"@[À-ÿ]@")


def normalize_page_text(text: str) -> str:
    """Limpia el texto extraído de una página de PDF.

    Pasos:
      1. Reparar encodings (BCCh font + mojibake UTF-8/Latin-1).
      2. Unir palabras cortadas por guión al final de línea.
      3. Colapsar saltos intra-párrafo (preserva \\n\\n como separador).
      4. Colapsar espacios/tabs múltiples.
      5. Eliminar líneas que son solo número de página.
      6. Eliminar watermarks DRM (JPMorgan: ``{[{hash}]}``).
      7. Eliminar caracteres de control y patrones de corrupción.
    """
    if not text:
        return ""

    text = fix_all(text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1", text)
    text = _INTRA_PARA_NEWLINE_RE.sub(" ", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    text = _ORPHAN_PAGENUM_RE.sub("", text)
    text = _DRM_WATERMARK_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _OCR_CORRUPTION_RE.sub("@", text)

    return text.strip()
