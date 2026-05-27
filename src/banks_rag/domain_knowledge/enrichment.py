"""Detectores semánticos sobre texto crudo.

Funciones puras que producen señales para el enriquecimiento de un chunk:

- ``detect_variables``: variables económicas con menciones y confianza.
- ``detect_section``: sección canónica con confianza, considerando hint del
  paso 0 + scoring por patterns + bias posicional como fallback.
- ``detect_entities``: menciones de bancos/países conocidos.
- ``extract_numerics``: valores numéricos (sobre texto original, no normalizado).
- ``extract_temporal``: años y trimestres referenciados.
- ``compute_signal_strength``: dirección + magnitud por variable (Monitor PM).
- ``compute_trend_direction``: tendencia por variable (PDFs).
- ``extract_forward_guidance``: oración forward-looking más relevante (PDFs).
- ``deviation_pattern_match``: marca contraste/desvío atípico (Monitor PM).

Todos los patterns se compilan a importación (módulo-level) para evitar
recompilarlos por chunk.
"""

from __future__ import annotations

import re

from .taxonomy import (
    DECISION_SENTENCE_PATTERN,
    ECONOMIC_VARIABLES,
    FORWARD_LOOKING_PATTERN,
    NUMERIC_PATTERNS,
    build_entity_patterns,
    build_section_patterns,
    build_variable_patterns,
    compile_word_pattern,
    normalize_text,
)

# Patterns compilados a la importación (compartidos por todos los chunks).
VARIABLE_PATTERNS = build_variable_patterns()
SECTION_PATTERNS = build_section_patterns()
ENTITY_PATTERNS = build_entity_patterns()

# Patrones temporales
YEAR_PATTERN = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")
QUARTER_PATTERN = re.compile(r"\b(q[1-4]|[1-4][ºo]?\s*trimestre)\b", re.IGNORECASE)

# Patrones direccionales — usados en signal_strength y trend_direction.
_DIR_UP_PATTERN = compile_word_pattern([
    "subio", "aumento", "alza", "sube", "presion al alza",
    "subida", "incremento", "escaló", "escalo",
])
_DIR_DOWN_PATTERN = compile_word_pattern([
    "bajo", "disminuyo", "baja", "cae", "cayo", "presion a la baja",
    "caida", "retroceso", "recorte",
])
_DIR_STABLE_PATTERN = compile_word_pattern([
    "se mantuvo", "estable", "sin cambios", "sin variacion",
])

# Contraste/desvío — solo Monitor PM.
DEVIATION_PATTERN = compile_word_pattern([
    "contrasta con", "a pesar de", "pese a", "sin embargo",
    "en contraposicion", "revirtiendo", "corrigio", "sorpresa",
    "sorprendio", "inesperado",
])

# Tendencia para PDFs.
_TREND_RISING_PATTERN = compile_word_pattern([
    "aumento", "subio", "se incremento", "alza", "aceleracion", "escalo",
])
_TREND_FALLING_PATTERN = compile_word_pattern([
    "disminuyo", "bajo", "cayo", "desaceleracion", "retrocedio", "recorte",
])
_TREND_STABLE_PATTERN = compile_word_pattern([
    "se mantuvo", "estable", "sin cambios", "sin variacion",
])


def detect_variables(text_norm: str) -> dict:
    """Detecta variables económicas con conteo de menciones y confianza.

    Confidence: 0.5 base + 0.1 por mención adicional, capa 1.0.
    """
    found: dict[str, dict] = {}
    for var_name, pattern in VARIABLE_PATTERNS.items():
        matches = pattern.findall(text_norm)
        if not matches:
            continue
        mentions = len(matches)
        importance = ECONOMIC_VARIABLES[var_name][1]
        indicator_type = ECONOMIC_VARIABLES[var_name][2]
        confidence = min(1.0, 0.5 + (mentions - 1) * 0.1)
        found[var_name] = {
            "importance": importance,
            "indicator_type": indicator_type,
            "mentions": mentions,
            "confidence": round(confidence, 2),
        }
    return found


def detect_section(
    text_norm: str,
    position_in_doc: int,
    total_chunks_in_doc: int,
    section_hint: str | None,
) -> tuple[str, float]:
    """Determina la sección canónica del chunk.

    Estrategia:
      1. Si ``section_hint`` (del chunker) matchea una sección canónica → 0.95.
      2. Scoring por patterns; gana la de mayor score (confianza por densidad).
      3. Bias posicional según posición relativa en el documento.
    """
    # 1. hint directo
    if section_hint:
        hint_norm = normalize_text(section_hint)
        for section in SECTION_PATTERNS:
            if section.lower() in hint_norm:
                return section, 0.95

    # 1.5 frase canónica de decisión: gana sobre el scoring por frecuencia. El
    # párrafo de apertura del Comunicado mezcla la decisión con contexto de
    # riesgos; sin esto, "riesgos"/"incertidumbre" le ganan por conteo a la
    # decisión y el chunk con la TPM queda mal clasificado como RIESGOS.
    if DECISION_SENTENCE_PATTERN.search(text_norm):
        return "DECISION", 0.95

    # 2. scoring por patterns
    scores: dict[str, int] = {}
    for section, pattern in SECTION_PATTERNS.items():
        matches = pattern.findall(text_norm)
        if matches:
            scores[section] = len(matches)

    if scores:
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = round(min(1.0, 0.40 + scores[best] / (total + 1)), 2)
        return best, confidence

    # 3. bias posicional
    if total_chunks_in_doc > 0:
        rel_pos = position_in_doc / max(1, total_chunks_in_doc - 1)
        if rel_pos < 0.05:
            return "ENCABEZADO", 0.40
        if rel_pos < 0.20:
            return "CONTEXTO_EXTERNO", 0.30
        if rel_pos < 0.35:
            return "ANALISIS", 0.28
        if rel_pos > 0.85:
            return "RIESGOS", 0.28
        if rel_pos > 0.70:
            return "PROYECCION", 0.30

    return "CONTENIDO", 0.25


def detect_entities(text_norm: str) -> dict:
    """Cuenta menciones de cada entidad conocida (banco / país)."""
    found: dict[str, int] = {}
    for name, pattern in ENTITY_PATTERNS.items():
        matches = pattern.findall(text_norm)
        if matches:
            found[name] = len(matches)
    return found


def extract_numerics(text: str, max_results: int = 10) -> list[dict]:
    """Extrae valores numéricos. Opera sobre texto **original** para preservar
    formato (e.g. ``5,25%``). Cap configurable en ``max_results``.
    """
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for pattern, unit in NUMERIC_PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(1)
            raw = m.group(0)
            key = (value, unit)
            if key in seen:
                continue
            seen.add(key)
            results.append({"value": value, "unit": unit, "raw": raw})
            if len(results) >= max_results:
                return results
    return results


def extract_temporal(text_norm: str) -> dict:
    """Años y trimestres referenciados. Devuelve ``{years, quarters}`` (omite vacíos)."""
    years = sorted(set(YEAR_PATTERN.findall(text_norm)))
    quarters = sorted({m.group(0) for m in QUARTER_PATTERN.finditer(text_norm)})
    out: dict = {}
    if years:
        out["years"] = years
    if quarters:
        out["quarters"] = quarters
    return out


def compute_signal_strength(
    text_norm: str,
    variables: dict,
    numerics: list,
) -> dict:
    """Dirección e intensidad de señal por variable. Solo para Monitor PM.

    Returns:
        ``{var_name: {direction: UP|DOWN|STABLE, magnitude: float | None}}``.
    """
    up = bool(_DIR_UP_PATTERN.search(text_norm))
    down = bool(_DIR_DOWN_PATTERN.search(text_norm))

    if up and not down:
        direction = "UP"
    elif down and not up:
        direction = "DOWN"
    else:
        direction = "STABLE"

    magnitude: float | None = None
    if numerics:
        try:
            magnitude = float(numerics[0]["value"].replace(",", "."))
        except (ValueError, KeyError):
            pass

    return {var_name: {"direction": direction, "magnitude": magnitude} for var_name in variables}


def compute_trend_direction(text_norm: str, variables: dict) -> dict:
    """Tendencia por variable detectada. Solo para PDFs."""
    rising = bool(_TREND_RISING_PATTERN.search(text_norm))
    falling = bool(_TREND_FALLING_PATTERN.search(text_norm))
    stable = bool(_TREND_STABLE_PATTERN.search(text_norm))

    if rising and falling:
        direction = "MIXED"
    elif rising:
        direction = "RISING"
    elif falling:
        direction = "FALLING"
    elif stable:
        direction = "STABLE"
    else:
        return {}

    return {var_name: direction for var_name in variables}


def extract_forward_guidance(
    text: str,
    text_norm: str,
    variables: dict,
) -> str | None:
    """Oración forward-looking más relevante (PDFs con is_forward_looking=True).

    Busca la primera oración que sea forward-looking y mencione una variable
    HIGH/CRITICAL. Cap en 300 chars.
    """
    if not FORWARD_LOOKING_PATTERN.search(text_norm):
        return None

    high_vars = {k for k, v in variables.items() if v["importance"] in ("CRITICAL", "HIGH")}
    if not high_vars:
        return None

    for sentence in re.split(r"[.!?]\s+", text):
        s_norm = normalize_text(sentence)
        if not FORWARD_LOOKING_PATTERN.search(s_norm):
            continue
        for var_name in high_vars:
            if VARIABLE_PATTERNS[var_name].search(s_norm):
                return sentence[:300]
    return None
