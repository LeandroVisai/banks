"""Verificación de *grounding* numérico de las respuestas del agente.

Complemento de ``citation_verifier`` (que solo valida refs ``[N]`` de
documentos). Este módulo ataca la falla más grave de los logs: el agente
emitió cifras financieras detalladas (DV01 0,85 %, allocation 45 %/55 %) que
**ninguna herramienta entregó** — pura alucinación ante gerentes del BCCh.

Estrategia: una cifra de la respuesta está "fundada" si coincide (con
tolerancia, y admitiendo conversiones ×100/÷100 de unidad) con algún número
que las herramientas devolvieron en este turno (``AgentState.grounded_numbers``).

Función pura: no muta estado ni hace I/O.
"""

from __future__ import annotations

import re

# Un número con separadores opcionales, signo y sufijo de unidad financiera.
_NUMBER_RE = re.compile(
    r"[-+]?\d[\d.,]*\d|\d"
)
# Sufijos que marcan una cifra como financiera (porcentaje, puntos base).
_PCT_SUFFIX_RE = re.compile(r"\s*(%|pb|bps|bp|puntos? base)\b", re.IGNORECASE)


def _to_float(raw: str) -> float | None:
    """Parsea un número en formato es/en a float.

    Convención chilena/es: coma = decimal, punto = miles. Casos:
    - ambos separadores → el más a la derecha es el decimal (cubre "1.069,5"
      es-style y "1,069.5" en-style).
    - solo coma → decimal ("624,9" → 624.9; "6,249" → 6.249).
    - solo punto → miles si el último grupo tiene 3 dígitos ("1.069" → 1069);
      si tiene 1-2, es decimal en-style que se coló ("5.25" → 5.25).
    """
    s = raw.strip().replace(" ", "")
    if not s:
        return None
    has_c, has_d = "," in s, "." in s
    if has_c and has_d:
        if s.rfind(",") > s.rfind("."):  # coma decimal
            s = s.replace(".", "").replace(",", ".")
        else:  # punto decimal
            s = s.replace(",", "")
    elif has_c:
        intpart, _, frac = s.rpartition(",")
        s = intpart.replace(",", "") + ("." + frac if frac else "")
    elif has_d:
        intpart, _, frac = s.rpartition(".")
        if len(frac) == 3:  # punto de miles
            s = s.replace(".", "")
        else:  # punto decimal (1-2 dígitos)
            s = intpart.replace(".", "") + "." + frac
    try:
        return float(s)
    except ValueError:
        return None


def _looks_like_year(value: float, raw: str) -> bool:
    """Años (1990–2099) sin parte decimal no son cifras financieras."""
    return "." not in raw and "," not in raw and 1990 <= value <= 2099 and value.is_integer()


def extract_numbers(text: str) -> list[float]:
    """Todos los números parseables del texto (para poblar el set de fundados)."""
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text or ""):
        v = _to_float(m.group(0))
        if v is not None:
            out.append(v)
    return out


def extract_financial_numbers(text: str) -> list[float]:
    """Números que parecen una cifra financiera (nivel, %, monto, ratio).

    Filtra cosas que NO son afirmaciones de dato: años, e ítems de lista /
    enteros chicos sin decimal ni unidad (p. ej. "3 especialistas").
    """
    out: list[float] = []
    text = text or ""
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0)
        v = _to_float(raw)
        if v is None:
            continue
        if _looks_like_year(v, raw):
            continue
        has_decimal = ("." in raw) or ("," in raw)
        has_unit = bool(_PCT_SUFFIX_RE.match(text[m.end():]))
        # Umbral 100: cubre niveles financieros típicos (USD/CLP ~900-1000,
        # cobre ~600 USc, montos) que suelen ir sin decimal ni unidad explícita.
        # Excluye conteos/ordinales chicos (<100) y años (filtrados arriba). El
        # objetivo es no dejar pasar un nivel inventado a un gerente; los falsos
        # positivos solo se marcan, no bloquean.
        is_financial_magnitude = abs(v) >= 100
        if has_decimal or has_unit or is_financial_magnitude:
            out.append(v)
    return out


def has_financial_numbers(text: str) -> bool:
    """¿La respuesta contiene al menos una cifra financiera?"""
    return bool(extract_financial_numbers(text))


def _is_grounded(value: float, grounded: list[float], *, rel_tol: float, abs_tol: float) -> bool:
    """¿``value`` coincide con algún número fundado (con tolerancia y ×100/÷100)?"""
    candidates = (value, value * 100.0, value / 100.0)
    for g in grounded:
        for c in candidates:
            if abs(c - g) <= max(abs_tol, rel_tol * max(abs(c), abs(g))):
                return True
    return False


def verify_numbers(
    text: str,
    grounded: list[float],
    *,
    rel_tol: float = 0.02,
    abs_tol: float = 0.01,
) -> tuple[list[float], bool]:
    """Devuelve ``(cifras_no_fundadas, is_grounded)``.

    - ``cifras_no_fundadas``: financieras del texto que no matchean ningún
      número entregado por las herramientas (señal de alucinación).
    - ``is_grounded``: ``True`` si no hay cifras no fundadas.

    Si ``grounded`` está vacío y el texto tiene cifras, todas se reportan como
    no fundadas (no hubo ninguna evidencia numérica en el turno).
    """
    flagged: list[float] = []
    for v in extract_financial_numbers(text):
        if not _is_grounded(v, grounded, rel_tol=rel_tol, abs_tol=abs_tol):
            if v not in flagged:
                flagged.append(v)
    return flagged, not flagged
