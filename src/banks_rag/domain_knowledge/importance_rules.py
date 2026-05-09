"""Reglas de cálculo del ``importance_score`` (0.0–1.0).

El score combina señales explícitas y calibradas. La calibración objetivo
(validada sobre los PDFs de prueba):

- chunk narrativo sin variables ni datos → ~0.0
- chunk con 1 variable HIGH + datos      → ~0.4
- chunk con variable CRÍTICA + datos     → ~0.6
- chunk DECISION de política con datos   → ~0.9+

Los pesos pueden sumar > 1.0; el resultado se capa a 1.0. Los pesos
se editan **acá** y se aplican en ``calculate_importance``.
"""

from __future__ import annotations

from .taxonomy import BOILERPLATE_PATTERN, MONITOR_PM_SECTIONS

IMPORTANCE_WEIGHTS: dict[str, float] = {
    "critical_var": 0.35,        # tiene al menos una CRITICAL
    "many_critical": 0.10,        # tiene 2+ CRITICAL
    "high_var_only": 0.20,        # HIGH pero sin CRITICAL
    "medium_var_only": 0.08,      # MEDIUM pero sin HIGH/CRITICAL
    "numerics": 0.20,             # saturable en 3
    "policy_section": 0.25,       # DECISION o VOTACION
    "outlook_section": 0.15,      # PROYECCION o RIESGOS
    "analysis_section": 0.08,
    "monitor_pm_section": 0.12,   # contenido curado de mercado
    "forward_looking": 0.08,
    "entities": 0.10,             # banco central / país (sat. en 2)
    "no_variables_penalty": 0.12, # sin variables ni datos → ruido semántico
    "boilerplate_penalty": 0.50,  # texto legal/disclaimer
    # v1.1 — variables de mercado de dinero y flujos
    "liquidity_var": 0.15,
    "ndf_var": 0.12,
    "pension_var": 0.10,
    "deviation_flag": 0.08,
}

# Piso para chunks visuales: garantiza que importance_boost no los entierre
# por su poco texto.
VISUAL_CHUNK_FLOOR = 0.30


def calculate_importance(
    variables: dict,
    numerics: list,
    section_type: str,
    entities: dict,
    is_fwd: bool,
    text_norm: str = "",
    deviation_flag: bool = False,
) -> float:
    """Calcula el ``importance_score`` calibrado de un chunk.

    Args:
        variables: ``{var_name: {importance, indicator_type, mentions, confidence}}``.
        numerics: lista de valores numéricos extraídos.
        section_type: tipo de sección canónica (DECISION, ANALISIS, etc.).
        entities: ``{ENTIDAD: mentions}``.
        is_fwd: True si el chunk contiene lenguaje prospectivo.
        text_norm: texto normalizado (para detectar boilerplate).
        deviation_flag: True si Monitor PM marca comportamiento atípico.

    Returns:
        Score en [0.0, 1.0] redondeado a 3 decimales.
    """
    # Boilerplate legal: penalización máxima absoluta.
    if text_norm and BOILERPLATE_PATTERN.search(text_norm):
        return 0.0

    levels = {d["importance"] for d in variables.values()}
    critical_count = sum(1 for d in variables.values() if d["importance"] == "CRITICAL")

    score = 0.0
    w = IMPORTANCE_WEIGHTS

    # Variables: escalones mutuamente excluyentes según nivel más alto presente.
    if "CRITICAL" in levels:
        score += w["critical_var"]
        if critical_count >= 2:
            score += w["many_critical"]
    elif "HIGH" in levels:
        score += w["high_var_only"]
    elif "MEDIUM" in levels:
        score += w["medium_var_only"]

    # Datos cuantitativos: satura en 3 menciones.
    score += w["numerics"] * min(1.0, len(numerics) / 3)

    # Sección: escalones por tipo.
    if section_type in ("DECISION", "VOTACION"):
        score += w["policy_section"]
    elif section_type in ("PROYECCION", "RIESGOS"):
        score += w["outlook_section"]
    elif section_type == "ANALISIS":
        score += w["analysis_section"]
    elif section_type in MONITOR_PM_SECTIONS:
        score += w["monitor_pm_section"]

    if is_fwd:
        score += w["forward_looking"]

    if entities:
        score += w["entities"] * min(1.0, len(entities) / 2)

    # Penalizar chunks sin contenido económico
    if not variables and not numerics:
        score -= w["no_variables_penalty"]

    # v1.1 — bonos por variables de mercado de dinero y flujos
    if "LIQUIDEZ_MERCADO" in variables or "FLUJO_NO_RESIDENTES" in variables:
        score += w["liquidity_var"]
    if "MERCADO_NDF" in variables:
        score += w["ndf_var"]
    if "FONDO_PENSION" in variables:
        score += w["pension_var"]
    if deviation_flag:
        score += w["deviation_flag"]

    return round(max(0.0, min(1.0, score)), 3)
