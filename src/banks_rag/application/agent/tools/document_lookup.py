"""Tools de exploración del corpus: list_documents + get_document_chunks.

Útiles cuando el agente sabe que existe un documento específico y quiere
ver más contexto que el que dio search_documents.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from banks_rag.infrastructure.persistence import PostgresRepo

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# list_documents
# ─────────────────────────────────────────────────────────────────────────────

LIST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": (
            "Lista documentos del corpus filtrando por tipo y/o año. Útil "
            "para explorar qué documentos hay disponibles antes de hacer "
            "una búsqueda específica. Por ejemplo: '¿qué Minutas hay del "
            "2024?' → list_documents(doc_type='MINUTA', year=2024)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_type": {
                    "type": "string",
                    "enum": [
                        "COMUNICADO_RPM", "MINUTA_RPM", "MINUTA_IPOM", "IPOM",
                        "IEF", "FED_STATEMENT", "REPORTE_RESEARCH", "MONITOR_PM",
                    ],
                    "description": "Tipo de documento a listar.",
                },
                "year": {
                    "type": "integer",
                    "description": "Año del documento.",
                    "minimum": 2000,
                    "maximum": 2026,
                },
                "limit": {
                    "type": "integer",
                    "description": "Máximo de documentos (default 30).",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
        },
    },
}


@register("list_documents", LIST_SCHEMA)
async def list_documents(
    state: "AgentState",
    doc_type: str | None = None,
    year: int | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    repo = PostgresRepo(prefix=os.getenv("RAG_TABLE_PREFIX", ""))
    rows = await asyncio.to_thread(
        repo.list_documents,
        doc_type=doc_type, year=year, limit=int(limit),
    )
    return {
        "documents": [
            {
                "document_id": str(r["document_id"]),
                "filename": r["filename"],
                "doc_type": r["doc_type_category"],
                "date": str(r["document_date"]) if r.get("document_date") else None,
                "year": r.get("document_year"),
            }
            for r in rows
        ],
        "n": len(rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# get_document_chunks
# ─────────────────────────────────────────────────────────────────────────────

GET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_document_chunks",
        "description": (
            "Trae todos los fragmentos de un documento específico (en orden "
            "de página). Útil cuando search_documents devolvió un fragmento "
            "interesante y quieres leer más contexto del mismo documento."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Nombre del archivo (ej. 'comunicado_ene2024.pdf'). "
                        "Lo obtienes de search_documents o list_documents."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Máximo de chunks (default 30).",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["filename"],
        },
    },
}


@register("get_document_chunks", GET_SCHEMA)
async def get_document_chunks(
    state: "AgentState",
    filename: str,
    limit: int = 30,
) -> dict[str, Any]:
    repo = PostgresRepo(prefix=os.getenv("RAG_TABLE_PREFIX", ""))

    doc = await asyncio.to_thread(repo.get_document_by_filename, filename)
    if not doc:
        return {"error": f"Documento no encontrado: {filename!r}"}

    chunks = await asyncio.to_thread(
        repo.get_chunks_by_document_id,
        doc["document_id"],
        limit=int(limit),
    )

    # Inyectar metadata del doc en cada chunk para que add_chunk en state
    # pueda formatear filename y doc_type al exportar chunks_seen.
    results: list[dict] = []
    for c in chunks:
        c_with_meta = {
            **c,
            "filename": doc["filename"],
            "doc_type_category": doc.get("doc_type_category"),
            "document_date": doc.get("document_date"),
        }
        ref = state.add_chunk(c_with_meta)
        text = (c.get("text") or "").strip()
        if len(text) > 600:
            text = text[:597] + "..."
        results.append({
            "ref": ref,
            "text": text,
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "section": c.get("section_type"),
            "importance": round(float(c.get("importance_score") or 0.0), 3),
        })

    return {
        "filename": doc["filename"],
        "doc_type": doc.get("doc_type_category"),
        "document_date": str(doc.get("document_date") or ""),
        "n_chunks": len(results),
        "chunks": results,
    }
