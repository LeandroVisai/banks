"""
Tool: search_documents — búsqueda híbrida sobre el corpus del BCCh.

Es la herramienta más usada por el agente. Permite filtros (tipo de documento,
año, rango de fechas) que el RAG clásico (chatbot/) no expone.

Cada llamada:
  1. Embed de la query
  2. Vector recall + lexical recall en paralelo (con filtros)
  3. RRF + importance boost
  4. Devuelve top-k chunks con refs globales del AgentState
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Optional

from .registry import register
from .. import db, embeddings
from ..settings import settings

if TYPE_CHECKING:
    from ..agent import AgentState


SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Busca fragmentos relevantes en el corpus del Banco Central (Comunicados, "
            "Minutas, Fed Statements, JPMorgan Research, Monitor PM). Retorna fragmentos "
            "con texto, metadata y un `ref` numérico que puedes usar para citar como [N] "
            "en tu respuesta final. Llama varias veces con queries refinadas si la "
            "primera búsqueda no devuelve lo que necesitas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Consulta semántica en español. Sé específico: incluye nombres "
                        "de variables (TPM, IPC, USD/CLP), períodos (2024, último "
                        "trimestre), o secciones (decisión, votación, riesgos)."
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
                    "type": "string",
                    "description": (
                        "Filtrar por tipo de documento. Útil cuando la pregunta es "
                        "específica de un tipo (p.ej. solo Minutas)."
                    ),
                    "enum": [
                        "COMUNICADO",
                        "MINUTA",
                        "RESEARCH_JPMORGAN",
                        "FED_STATEMENT",
                        "MONITOR_PM",
                    ],
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


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


@register("search_documents", SCHEMA)
async def search_documents(
    state: "AgentState",
    query: str,
    k: int = 5,
    doc_type: Optional[str] = None,
    year: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict[str, Any]:
    k = max(1, min(int(k), 10))
    df = _parse_date(date_from)
    dt = _parse_date(date_to)

    query_vec = await embeddings.embed_query(query)

    import asyncio
    vec_task = asyncio.create_task(db.vector_recall(
        query_vec.tolist(), settings.rag_recall_n,
        doc_type=doc_type, year=year, date_from=df, date_to=dt,
    ))
    lex_task = asyncio.create_task(db.lexical_recall(
        query, settings.rag_recall_n,
        doc_type=doc_type, year=year, date_from=df, date_to=dt,
    ))
    vec_hits, lex_hits = await asyncio.gather(vec_task, lex_task)

    # RRF
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for rank, row in enumerate(vec_hits, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank)
        docs[cid] = dict(row)
    for rank, row in enumerate(lex_hits, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank)
        if cid not in docs:
            docs[cid] = dict(row)

    # Boost por importance
    fused = sorted(
        docs.values(),
        key=lambda d: scores[d["chunk_id"]] + 0.15 * float(d.get("importance_score") or 0.0),
        reverse=True,
    )
    selected = fused[:k]

    # Asignar refs globales (acumuladas en el AgentState)
    results: list[dict] = []
    for chunk in selected:
        ref = state.add_chunk(chunk)
        date_str = str(chunk.get("chunk_date") or chunk.get("document_date") or "")
        text = (chunk.get("text") or "").strip()
        if len(text) > 600:
            text = text[:597] + "..."
        results.append({
            "ref": ref,
            "text": text,
            "filename": chunk.get("filename", ""),
            "doc_type": chunk.get("doc_type_category", ""),
            "section": chunk.get("section_type", ""),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "date": date_str,
            "importance": round(float(chunk.get("importance_score") or 0.0), 3),
        })

    if not results:
        return {
            "results": [],
            "n_results": 0,
            "message": (
                "No se encontraron fragmentos. Intenta con otros términos, quita "
                "filtros, o usa list_documents para explorar qué hay disponible."
            ),
        }

    return {
        "results": results,
        "n_results": len(results),
        "filters_applied": {
            "doc_type": doc_type,
            "year": year,
            "date_from": date_from,
            "date_to": date_to,
        },
    }
