"""Tool ``discover_query`` — busca queries analíticas relevantes en el catálogo SQL.

El agente la usa para encontrar qué query del catálogo responde mejor a una
pregunta sobre series financieras (tipo de cambio, curvas de bonos, liquidez,
etc.) antes de llamar a ``execute_query``.

Estrategia de scoring: BM25-like por solapamiento de términos entre la
consulta del usuario y los campos {name, description, tags, segment} de
cada entrada del catálogo. No requiere embeddings ni modelo adicional.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from banks_rag.infrastructure.sql.catalog_loader import load_catalog

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState

_STOP_WORDS = frozenset({
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "en", "con", "por", "para", "que", "qué", "y", "o", "a", "al",
    "se", "es", "son", "fue", "ser", "como", "más", "pero", "si",
    "the", "of", "in", "for", "and", "or", "a", "an", "to", "is",
})

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "discover_query",
        "description": (
            "Busca en el catálogo de queries analíticas cuál es la más adecuada "
            "para responder una pregunta sobre datos financieros o macroeconómicos "
            "(tipo de cambio, tasas, bonos, liquidez bancaria, commodities, etc.). "
            "Retorna las top-k entradas con su query_id, descripción y parámetros. "
            "Úsala ANTES de execute_query para descubrir el query_id correcto."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Pregunta o descripción en lenguaje natural de los datos "
                        "que necesitas (ej. 'curva de bonos BTP en pesos', "
                        "'tipo de cambio dólar último mes', 'LCR sistémico bancos')."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de resultados a retornar (default 5, máx 10).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
                "segment": {
                    "type": "string",
                    "description": (
                        "Filtrar por segmento de mercado: mercado_cambiario, "
                        "renta_fija_chile, renta_fija_eeuu, tasas_monetarias_chile, "
                        "tasas_internacionales, liquidez_bancaria, commodities, "
                        "politica_monetaria."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}


def _score(entry_tokens: set[str], query_tokens: set[str]) -> int:
    return len(entry_tokens & query_tokens)


@register("discover_query", _SCHEMA)
async def discover_query(
    state: "AgentState",
    query: str,
    top_k: int = 5,
    segment: str | None = None,
) -> dict[str, Any]:
    entries = load_catalog()

    if segment:
        entries = [e for e in entries if e.segment == segment]
        if not entries:
            return {
                "error": f"Segmento {segment!r} sin entradas en el catálogo.",
                "segments_disponibles": list({e.segment for e in load_catalog()}),
            }

    query_tokens = _tokenize(query)
    scored = []
    for entry in entries:
        entry_text = " ".join([
            entry.name,
            entry.description,
            entry.segment,
            " ".join(entry.tags),
        ])
        score = _score(_tokenize(entry_text), query_tokens)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: min(top_k, 10)]

    return {
        "results": [e.to_discovery_dict() for _, e in top],
        "n_results": len(top),
        "hint": (
            "Usa execute_query con el query_id elegido y los parámetros "
            "fecha_inicio / fecha_fin / limit para obtener los datos."
        ),
    }
