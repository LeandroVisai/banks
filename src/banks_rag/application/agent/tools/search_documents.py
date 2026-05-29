"""Tool ``search_documents`` — búsqueda híbrida sobre el corpus.

La tool más usada por el agente. Permite filtros (tipo de documento, año,
rango de fechas) que el RAG clásico no expone.

Internamente delega a ``hybrid_search`` (sync) ejecutado en ``asyncio.to_thread``
para no bloquear el event loop del agente.
"""

from __future__ import annotations

import asyncio
import os
from datetime import date
from typing import TYPE_CHECKING, Any

from banks_rag.application.retrieval import hybrid_search
from banks_rag.domain.retrieval import SearchFilters
from banks_rag.infrastructure.embeddings import build_default_embedder
from banks_rag.infrastructure.persistence import PostgresRepo

from ._doc_types import DOC_TYPE_VALUES, normalize_doc_types
from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState


SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Busca fragmentos relevantes en el corpus del Banco Central "
            "(Comunicados, Minutas, Fed Statements, JPMorgan Research, "
            "Monitor PM). Retorna fragmentos con texto, metadata y un `ref` "
            "numérico que puedes usar para citar como [N] en tu respuesta "
            "final. Llama varias veces con queries refinadas si la primera "
            "búsqueda no devuelve lo que necesitas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Consulta semántica en español. Sé específico: "
                        "incluye nombres de variables (TPM, IPC, USD/CLP), "
                        "períodos (2024, último trimestre), o secciones "
                        "(decisión, votación, riesgos)."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": "Cuántos fragmentos retornar (1–10). Default: 5.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
                "doc_type": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(DOC_TYPE_VALUES)},
                    "description": (
                        "Filtrar por uno o varios tipos de documento (también "
                        "acepta un solo string). Útil cuando la pregunta es "
                        "específica (p.ej. solo Minutas). Valores: "
                        + ", ".join(DOC_TYPE_VALUES)
                        + "."
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": "Filtrar por año del documento (2018–2026).",
                    "minimum": 2000,
                    "maximum": 2026,
                },
                "date_from": {
                    "type": "string",
                    "description": "Fecha desde (ISO YYYY-MM-DD). Útil con date_to.",
                },
                "date_to": {
                    "type": "string",
                    "description": "Fecha hasta (ISO YYYY-MM-DD).",
                },
            },
            "required": ["query"],
        },
    },
}


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _build_filters(
    doc_type: str | list[str] | None,
    year: int | None,
    date_from: str | None,
    date_to: str | None,
) -> SearchFilters:
    """Convierte los argumentos de la tool a un ``SearchFilters``.

    ``doc_type`` admite string o array; se normaliza a los valores canónicos
    de ``doc_type_category`` (ver ``_doc_types.normalize_doc_types``).
    """
    df = _parse_date(date_from)
    dt = _parse_date(date_to)

    filters = SearchFilters()
    doc_types = normalize_doc_types(doc_type)
    if doc_types:
        filters.doc_types = doc_types

    # Si vienen exact dates iguales → exact_date; si rango → year range.
    if df and dt and df == dt:
        filters.exact_date = df.isoformat()
        filters.day = df.day
        filters.month = df.month
        filters.year_from = df.year
        filters.year_to = df.year
    elif df or dt:
        if df:
            filters.year_from = df.year
        if dt:
            filters.year_to = dt.year
    elif year is not None:
        filters.year_from = year
        filters.year_to = year

    return filters


def _format_chunk(chunk: dict, ref: int) -> dict:
    text = (chunk.get("text") or "").strip()
    if len(text) > 600:
        text = text[:597] + "..."
    return {
        "ref": ref,
        "text": text,
        "filename": chunk.get("filename", ""),
        "doc_type": chunk.get("doc_type_category", ""),
        "section": chunk.get("section_type", ""),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "date": str(chunk.get("chunk_date") or chunk.get("document_date") or ""),
        "importance": round(float(chunk.get("importance_score") or 0.0), 3),
    }


@register("search_documents", SCHEMA)
async def search_documents(
    state: "AgentState",
    query: str,
    k: int = 5,
    doc_type: str | list[str] | None = None,
    year: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Ejecuta hybrid_search y registra cada chunk en el ``AgentState``."""
    k = max(1, min(int(k), 10))
    filters = _build_filters(doc_type, year, date_from, date_to)

    repo = PostgresRepo(prefix=os.getenv("RAG_TABLE_PREFIX", ""))
    embedder = build_default_embedder()

    # hybrid_search es sync (psycopg2) — to_thread para no bloquear el event loop.
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
                "No se encontraron fragmentos. Intenta con otros términos, "
                "quita filtros, o usa list_documents para explorar."
            ),
        }

    results: list[dict] = []
    for chunk in search_result.hits:
        ref = state.add_chunk(chunk)
        results.append(_format_chunk(chunk, ref))

    response: dict[str, Any] = {
        "results": results,
        "n_results": len(results),
        "filters_applied": {
            "doc_type": filters.doc_types or None,
            "year": year,
            "date_from": date_from,
            "date_to": date_to,
        },
    }

    # Avisa al agente si el retrieval tuvo que degradarse: los resultados
    # NO respetan plenamente los filtros pedidos y debe advertirlo al usuario.
    if search_result.relaxed_filters:
        response["filters_relaxed"] = search_result.relaxed_filters
        response["warning"] = (
            "Sin resultados con los filtros estrictos; se ignoraron "
            f"{search_result.relaxed_filters}. Advierte al usuario que la "
            "respuesta puede no ceñirse a esos filtros."
        )
    if search_result.fallback_used:
        response["fallback_used"] = True
        response["warning"] = (
            "Sin resultados por relevancia; se devolvieron fragmentos por "
            "importancia general. Advierte al usuario que la respuesta "
            "puede no ser específica a la consulta."
        )

    return response
