"""Tool ``search_visuals`` — busca solo entre chunks visuales (charts, tablas, figuras).

Wrapper de ``hybrid_search`` con ``kinds=['VISUAL']`` y ``kinds=['TABLE']``
disponibles. Útil para preguntas como:

  "Muéstrame el gráfico de la curva swap CLP del último IPOM"
  "¿Hay una tabla con la votación de la última minuta?"

Devuelve los mismos campos que ``search_documents`` + ``image_url`` apuntando
al endpoint ``/v1/images/{chunk_id}`` para que el cliente recupere el binario.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Literal

from banks_rag.application.retrieval import hybrid_search
from banks_rag.domain.retrieval import SearchFilters
from banks_rag.infrastructure.embeddings import build_default_embedder
from banks_rag.infrastructure.persistence import PostgresRepo

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState


SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_visuals",
        "description": (
            "Busca **solo gráficos, figuras o tablas** en los PDFs (no texto "
            "narrativo). Útil cuando el usuario pide ver una curva, una "
            "distribución, una tabla de datos, o cualquier elemento visual. "
            "Cada resultado incluye `image_url` que el cliente puede pedir "
            "como binario en `GET /v1/images/{chunk_id}`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Descripción semántica del visual. Sé específico: "
                        "'curva swap CLP marzo 2024', 'tabla votación "
                        "minuta abril', 'gráfico de inflación anual'."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": "Cuántos visuales retornar (1–10). Default 5.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
                "visual_kind": {
                    "type": "string",
                    "description": (
                        "Filtrar por tipo de visual. Default ``VISUAL`` "
                        "(charts + tablas + imágenes). Para excluir tablas "
                        "narrativas usa ``CHART`` (en Fase 2.b — por ahora "
                        "todos los visuales se etiquetan como CHART por default)."
                    ),
                    "enum": ["VISUAL", "TABLE"],
                },
                "doc_type": {
                    "type": "string",
                    "enum": [
                        "COMUNICADO_RPM", "MINUTA_RPM", "MINUTA_IPOM", "IPOM",
                        "IEF", "FED_STATEMENT", "REPORTE_RESEARCH", "MONITOR_PM",
                    ],
                },
                "year": {
                    "type": "integer",
                    "minimum": 2000, "maximum": 2026,
                },
            },
            "required": ["query"],
        },
    },
}


def _format_visual(chunk: dict, ref: int) -> dict:
    """Formato output que el agente devolverá al usuario."""
    return {
        "ref": ref,
        "chunk_id": chunk.get("chunk_id"),
        "filename": chunk.get("filename", ""),
        "doc_type": chunk.get("doc_type_category", ""),
        "page": chunk.get("page_start"),
        "kind": chunk.get("kind", "VISUAL"),
        "caption": chunk.get("visual_caption"),
        "date": str(chunk.get("chunk_date") or chunk.get("document_date") or ""),
        "image_url": f"/v1/images/{chunk.get('chunk_id')}",
        "importance": round(float(chunk.get("importance_score") or 0.0), 3),
    }


@register("search_visuals", SCHEMA)
async def search_visuals(
    state: "AgentState",
    query: str,
    k: int = 5,
    visual_kind: Literal["VISUAL", "TABLE"] = "VISUAL",
    doc_type: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    k = max(1, min(int(k), 10))

    filters = SearchFilters(
        kinds=["VISUAL"] if visual_kind == "VISUAL" else ["TABLE"],
        doc_types=[doc_type] if doc_type else [],
        year_from=year, year_to=year,
    )

    repo = PostgresRepo(prefix=os.getenv("RAG_TABLE_PREFIX", ""))
    embedder = build_default_embedder()

    search_result = await asyncio.to_thread(
        hybrid_search,
        query,
        query_embedder=embedder,
        repo=repo,
        extra_filters=filters,
        k=k,
        use_mmr=True,
    )

    if not search_result.hits:
        return {
            "results": [],
            "n_results": 0,
            "message": (
                "No se encontraron visuales para la query. Verifica que el "
                "PDF esté ingerido con extracción visual (PyMuPDF activo)."
            ),
        }

    results: list[dict] = []
    for chunk in search_result.hits:
        ref = state.add_chunk(chunk)
        results.append(_format_visual(chunk, ref))

    return {
        "results": results,
        "n_results": len(results),
        "filters_applied": {
            "kinds": filters.kinds,
            "doc_type": doc_type,
            "year": year,
        },
    }
