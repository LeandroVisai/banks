"""QueryRouter — clasifica una consulta en RAG / SQL / VISUAL.

Regla-based puro: no requiere modelo adicional ni embeddings. Funciona offline.

Lógica:
  - VISUAL: señales de gráfico / figura / imagen.
  - SQL:    peticiones de series numéricas, datos actuales, comparativos.
  - RAG:    contexto político, documentos narrativos, decisiones, análisis.
  Las categorías NO son mutuamente excluyentes: una pregunta puede
  activar SQL + RAG al mismo tiempo (datos + contexto).

``RouteDecision`` contiene flags booleanos y scores 0–1 por categoría
para que el agente decida qué tools invocar primero.

Uso típico en el agente:
    route = route_query("¿Cuál es la TPM actual y qué dijo el Consejo?")
    # route.sql   → True  (valor numérico TPM)
    # route.rag   → True  (comunicado / minuta)
    # route.visual → False
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Patrones de señal ─────────────────────────────────────────────────────────

_VISUAL_PATTERNS = re.compile(
    r"\b(gr[aá]fico|chart|figura|imagen|plot|visuali[zs]|picture|"
    r"tabla\s+de\s+datos|cuadro\s+(?:N[°º]?\s*\d+|\d+)|"
    r"ver\s+el\s+gr[aá]fico|muestra?(?:me)?\s+(?:el|la)\s+gr[aá]fico)\b",
    re.IGNORECASE,
)

_SQL_PATTERNS = re.compile(
    r"\b("
    # series y activos financieros concretos
    r"USD|CLP|UF|BTP|BTU|SPC|PDBC|TIB|SOFR|TADO|OIS|UST|LCR|NSFR|"
    r"d[oó]lar|euro|peso|cobre|"
    # acciones que piden datos
    r"cu[aá]nto\s+(vale|est[aá]|es)|"
    r"precio\s+(de|del|actual)|"
    r"tasa\s+(de|del|actual)|"
    r"valor\s+(de|del|actual)|"
    r"(?:hist[oó]rico|serie|series|evoluci[oó]n)\s+(?:de|del?)|"
    r"(?:datos?|cifras?|n[uú]meros?)\s+(?:de|del?|sobre)|"
    r"(?:[uú]ltimo[s]?|reciente[s]?|hoy|ayer|actual)\s+(?:valor|dato|nivel|precio|tasa|tpm)|"
    r"(?:nivel|valor|cifra)\s+(?:de\s+(?:la\s+)?)?tpm|tpm\s+actual|tpm\s+hoy|"
    r"(?:cuanto|cu[aá]nto)\s+(?:vale|est[aá]|es|fue|ha\s+sido)|"
    r"comparar\s+(?:la\s+)?(?:curva|serie|tasa)|"
    r"curva\s+de\s+(?:bonos?|tasas?|rendimientos?|swaps?)|"
    r"spread\s+(?:de|entre|del?)|"
    r"forwards?|"
    r"expectativas?\s+(?:de\s+)?(?:tpm|tasa|inflaci[oó]n)|"
    r"bid.ask|microestructura"
    r")\b",
    re.IGNORECASE,
)

_RAG_PATTERNS = re.compile(
    r"\b("
    # tipos de documentos
    r"comunicado|minuta|reunión\s+del\s+consejo|actas?|"
    r"informe|reporte|research|fed\s+statement|"
    # tipos de información narrativa
    r"decidi[oó]|votaci[oó]n|decisi[oó]n\s+de\s+política|"
    r"razon(?:es?|amiento)|justific|argument|señal(?:ó|ando)?|"
    r"perspectiva|escenario|riesgo[s]?|proyecci[oó]n|"
    r"análisis|contexto|antecedente[s]?|"
    r"(?:qu[eé]|por\s+qu[eé])\s+(?:dijo|señaló|explicó|mencionó)|"
    r"(?:qué|cu[aá]l)\s+(?:fue|es)\s+la\s+(?:postura|visión|posición)|"
    r"banco\s+central|bcch|consejo|"
    r"política\s+monetaria|tpm"  # TPM en contexto narrativo
    r")\b",
    re.IGNORECASE,
)

# Señales fuertes de SQL que elevan el score (series con nombres exactos)
_SQL_STRONG_PATTERNS = re.compile(
    r"\b(USD.?CLP|BTP\s*\d+[Yy]|BTU\s*\d+[Yy]|SPC\s*\d+[Ymy]|"
    r"SOFR\s*\d+[Mmy]|OIS|UST\s*\d+[Yy]|LCR|NSFR|"
    r"forwards?\s+a?\s+\d+\s*(?:d[íi]as?|meses?)|"
    r"precio\s+(?:del\s+)?cobre|tipo\s+de\s+cambio)\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    """Decisión de routing para una consulta.

    Campos:
        rag:    el agente debe buscar en documentos (search_documents).
        sql:    el agente debe consultar series numéricas (discover_query + execute_query).
        visual: el agente debe buscar chunks visuales (search_visuals).
        scores: scores crudos por categoría (0.0–1.0).
        primary: categoría dominante ("rag" | "sql" | "visual").
    """

    rag: bool
    sql: bool
    visual: bool
    scores: dict[str, float]
    primary: str

    def to_dict(self) -> dict:
        return {
            "rag": self.rag,
            "sql": self.sql,
            "visual": self.visual,
            "primary": self.primary,
            "scores": self.scores,
        }


def route_query(query: str) -> RouteDecision:
    """Clasifica la consulta en RAG / SQL / VISUAL.

    Returns:
        ``RouteDecision`` con flags y scores. Al menos una categoría
        siempre es True (RAG como fallback universal).
    """
    visual_score = _count_matches(_VISUAL_PATTERNS, query)
    sql_score = _count_matches(_SQL_PATTERNS, query)
    sql_score += _count_matches(_SQL_STRONG_PATTERNS, query) * 2  # señales fuertes pesan doble
    rag_score = _count_matches(_RAG_PATTERNS, query)

    # Normalizar a 0–1 con saturación
    scores = {
        "rag": min(1.0, rag_score / 3),
        "sql": min(1.0, sql_score / 4),
        "visual": min(1.0, visual_score / 2),
    }

    is_visual = scores["visual"] > 0
    is_sql = scores["sql"] >= 0.25  # ≥1 señal SQL activa
    is_rag = scores["rag"] >= 0.25 or (not is_sql and not is_visual)  # fallback universal

    # Primario = mayor score
    primary = max(scores, key=lambda k: scores[k])
    if scores[primary] == 0:
        primary = "rag"

    return RouteDecision(
        rag=is_rag,
        sql=is_sql,
        visual=is_visual,
        scores={k: round(v, 3) for k, v in scores.items()},
        primary=primary,
    )


def _count_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))
