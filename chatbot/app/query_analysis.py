"""
Análisis de la query del usuario antes de enviarla al RAG/LLM.

Detecta:
  - Variables económicas mencionadas (TPM, IPC, USD/CLP, …)
  - Rango temporal (años explícitos, "últimos 6 meses", etc.)
  - Intención: factual/cuantitativa, narrativa/cualitativa, definicional

La intención influye en si activamos el contexto histórico SQL y cuántos
chunks RAG pedir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal, Optional


Intent = Literal["quantitative", "qualitative", "definition", "comparison"]


# ─────────────────────────────────────────────────────────────────────────────
# Variables económicas → keywords (alineado con taxonomy.py)
# ─────────────────────────────────────────────────────────────────────────────

_VARIABLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:tpm|tasa\s+pol[ií]tica\s+monetaria|tasa\s+rectora|tasa\s+de\s+inter[eé]s)\b", re.I), "TASA_INTERES"),
    (re.compile(r"\b(?:fed|reserva\s+federal|federal\s+reserve|fed\s+funds?)\b",                      re.I), "TASA_INTERES"),
    (re.compile(r"\b(?:ipc|inflaci[oó]n|inflacionario|precios?\s+al?\s+consumidor)\b",                re.I), "INFLACION"),
    (re.compile(r"\b(?:pib|producto\s+interno\s+bruto|crecimiento\s+econ[oó]mico|imacec)\b",          re.I), "PIB"),
    (re.compile(r"\b(?:tipo\s+de\s+cambio|usd[/\\]?clp|d[oó]lar|peso\s+chileno|paridad)\b",           re.I), "TIPO_CAMBIO"),
    (re.compile(r"\b(?:expectativas?\s+(?:de\s+)?inflaci[oó]n|expectativas?\s+inflacionarias?|eee)\b", re.I), "EXPECTATIVAS_INFLACIONARIAS"),
    (re.compile(r"\b(?:bcu|btp|bonos?\s+(?:del?\s+)?banco\s+central|tasas?\s+larg[ao])\b",            re.I), "TASAS_LARGO_PLAZO"),
    (re.compile(r"\b(?:cobre|petr[oó]leo|wti|brent|commodities?|materias?\s+primas?)\b",              re.I), "COMMODITIES"),
    (re.compile(r"\b(?:desempleo|desocupaci[oó]n|empleo|mercado\s+laboral)\b",                        re.I), "MERCADO_LABORAL"),
    (re.compile(r"\b(?:cds|riesgo\s+pa[ií]s|riesgo\s+soberano|spread\s+soberano)\b",                  re.I), "RIESGO_CREDITO"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Intención
# ─────────────────────────────────────────────────────────────────────────────

_QUANTITATIVE_HINTS = re.compile(
    r"\b(?:cu[aá]nto|cu[aá]l(?:es)?\s+(?:fue|son|es)|valor|nivel|cifra|"
    r"datos?|serie|hist[oó]ric[oa]|n[uú]mero|porcentaje|variaci[oó]n|"
    r"evoluci[oó]n|trayectoria|tendencia|comportamiento|gr[aá]fico)\b",
    re.I,
)
_QUALITATIVE_HINTS = re.compile(
    r"\b(?:por\s+qu[eé]|raz[oó]n|motivo|justifica|explica|interpret|"
    r"diagn[oó]stico|an[aá]lisis|opini[oó]n|considera|argumenta)\b",
    re.I,
)
_DEFINITION_HINTS = re.compile(
    r"\b(?:qu[eé]\s+es|qu[eé]\s+significa|definici[oó]n|defin[ei]r|"
    r"qu[eé]\s+entiend|qu[eé]\s+se\s+entiende)\b",
    re.I,
)
_COMPARISON_HINTS = re.compile(
    r"\b(?:vs\.?|versus|comparad[oa]|frente\s+a|en\s+relaci[oó]n\s+a|"
    r"diferencia(?:s)?\s+entre|m[aá]s\s+(?:alto|bajo|grande|peque[ñn]o))\b",
    re.I,
)


def _classify_intent(query: str) -> Intent:
    if _DEFINITION_HINTS.search(query):
        return "definition"
    if _COMPARISON_HINTS.search(query):
        return "comparison"
    if _QUANTITATIVE_HINTS.search(query):
        return "quantitative"
    if _QUALITATIVE_HINTS.search(query):
        return "qualitative"
    # Default: si hay una variable económica → cuantitativa, si no → cualitativa
    return "qualitative"


# ─────────────────────────────────────────────────────────────────────────────
# Rango temporal
# ─────────────────────────────────────────────────────────────────────────────

_RELATIVE_RANGES = [
    (re.compile(r"\b(?:[uú]ltim[oa]s?\s+)?3\s+meses?\b", re.I),                90),
    (re.compile(r"\b(?:[uú]ltim[oa]s?\s+)?(?:6|seis)\s+meses?\b", re.I),       183),
    (re.compile(r"\b(?:[uú]ltim[oa]s?\s+)?(?:12|doce)\s+meses?\b", re.I),      366),
    (re.compile(r"\b(?:[uú]ltim[oa]s?\s+)?(?:2|dos)\s+a[ñn]os?\b", re.I),      730),
    (re.compile(r"\b(?:[uú]ltim[oa]\s+)?a[ñn]o\b", re.I),                      366),
    (re.compile(r"\b(?:reciente|actual|hoy|ahora|coyuntural)\b", re.I),         183),
]


def _extract_date_range(query: str) -> tuple[Optional[date], Optional[date]]:
    today = date.today()

    # Años explícitos: "en 2022", "del 2021", "entre 2019 y 2023"
    years = sorted({
        int(y) for y in re.findall(r"\b(20\d{2})\b", query)
        if 2000 <= int(y) <= today.year + 1
    })
    if len(years) >= 2:
        return date(years[0], 1, 1), date(years[-1], 12, 31)
    if len(years) == 1:
        y = years[0]
        return date(y, 1, 1), date(y, 12, 31)

    # Expresiones relativas
    for pattern, days in _RELATIVE_RANGES:
        if pattern.search(query):
            return today - timedelta(days=days), today

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class QueryAnalysis:
    raw_query: str
    intent: Intent
    variables: list[str] = field(default_factory=list)
    date_from: Optional[date] = None
    date_to: Optional[date] = None

    @property
    def needs_historical_data(self) -> bool:
        """¿Vale la pena consultar series SQL?"""
        if not self.variables:
            return False
        return self.intent in ("quantitative", "comparison") or self.date_from is not None

    @property
    def needs_rag(self) -> bool:
        """¿Vale la pena hacer retrieval sobre los documentos?"""
        # Casi siempre sí — incluso preguntas cuantitativas se enriquecen con contexto.
        return True


def analyze(query: str) -> QueryAnalysis:
    seen: set[str] = set()
    variables: list[str] = []
    for pattern, var in _VARIABLE_PATTERNS:
        if var in seen:
            continue
        if pattern.search(query):
            seen.add(var)
            variables.append(var)

    date_from, date_to = _extract_date_range(query)

    return QueryAnalysis(
        raw_query=query,
        intent=_classify_intent(query),
        variables=variables,
        date_from=date_from,
        date_to=date_to,
    )
