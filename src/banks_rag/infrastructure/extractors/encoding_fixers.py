"""Reparadores de encodings corruptos comunes en PDFs financieros.

Dos correcciones independientes y composables:

1. ``fix_bcch_font``: decodifica el encoding custom de las Minutas BCCh,
   donde el cuerpo del documento usa una fuente con índices de glifo
   corridos +1 respecto al alfabeto estándar (@→a, A→b, ..., Y→z, Ä→espacio).

2. ``fix_mojibake``: repara texto donde pypdf interpretó bytes UTF-8 como
   Latin-1 (clásico ``ó`` → ``Ã³``). Aplica re-encoding latin-1 → utf-8.

Ambas son funciones puras: ``str → str``, sin estado ni I/O.

**Orden importa**: ``fix_bcch_font`` debe ejecutarse ANTES que ``fix_mojibake``,
porque el carácter ``Ä`` (0xC4) que usa BCCh rompería el round-trip
latin-1↔utf-8 de la segunda función.
"""

from __future__ import annotations

import re

# Mapeo de caracteres de la fuente custom de las Minutas BCCh.
# Ä→espacio, Æ→fi (ligadura), Ç→fl (ligadura), ¹→ó.
BCCH_CHAR_MAP: dict[str, str] = {
    "@": "a",
    "A": "b", "B": "c", "C": "d", "D": "e", "E": "f",
    "F": "g", "G": "h", "H": "i", "I": "j", "J": "k",
    "K": "l", "L": "m", "M": "n", "N": "o", "O": "p",
    "P": "q", "Q": "r", "R": "s", "S": "t", "T": "u",
    "U": "v", "V": "w", "W": "x", "X": "y", "Y": "z",
    "Ä": " ",
    "Æ": "fi",
    "Ç": "fl",
    "¹": "ó",
}

# Segmento BCCh: 8+ chars consecutivos del charset de la fuente custom.
# Ä-Ç cubre Ä(196) Å(197) Æ(198) Ç(199) en Latin-1.
_BCCH_RE = re.compile("[@A-YÄ-Ç¹]{8,}")


def fix_bcch_font(text: str) -> str:
    """Decodifica el font encoding custom de las Minutas BCCh.

    Solo actúa en segmentos que contienen simultáneamente ``Ä`` (espacio)
    y ``@`` ('a') — la firma inequívoca de esta codificación.
    Texto normal y mayúsculas legítimas no activan el decode.
    """
    if "Ä" not in text or "@" not in text:
        return text

    def _decode(m: re.Match) -> str:
        seg = m.group(0)
        if "Ä" not in seg or "@" not in seg:
            return seg
        return "".join(BCCH_CHAR_MAP.get(c, c) for c in seg)

    return _BCCH_RE.sub(_decode, text)


def fix_mojibake(text: str) -> str:
    """Repara texto donde pypdf leyó bytes UTF-8 como Latin-1.

    Algunos PDFs almacenan texto en bytes UTF-8 pero pypdf los interpreta
    byte a byte como Latin-1, convirtiendo ``ó`` (C3 B3) en ``Ã³``.
    Esta función re-encodea como Latin-1 (recupera los bytes) y decodea
    como UTF-8. Solo se aplica si hay ``Ã`` y la operación produce UTF-8 válido.
    """
    if "Ã" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def fix_all(text: str) -> str:
    """Aplica ambos fixers en el orden correcto: BCCh primero, mojibake después."""
    return fix_mojibake(fix_bcch_font(text))
