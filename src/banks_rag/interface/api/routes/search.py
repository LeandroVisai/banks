"""POST ``/v1/search`` — búsqueda híbrida sin pasar por el agente."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from banks_rag.application.retrieval import hybrid_search
from banks_rag.domain.retrieval import SearchFilters
from banks_rag.interface.api.schemas import (
    SearchHit,
    SearchRequest,
    SearchResponse,
)

router = APIRouter()


def _to_hit(row: dict) -> SearchHit:
    return SearchHit(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        filename=row.get("filename", ""),
        doc_type_category=row.get("doc_type_category", ""),
        document_date=str(row.get("document_date") or "") or None,
        page_start=int(row["page_start"]),
        page_end=int(row["page_end"]),
        section_type=row.get("section_type", "CONTENIDO"),
        importance_score=float(row.get("importance_score", 0)),
        rrf_score=float(row.get("rrf_score", 0)),
        final_score=float(row.get("final_score", 0)),
        economic_variables=row.get("economic_variables", {}) or {},
        tags=row.get("tags") or [],
        image_path=row.get("image_path"),
        text=row.get("text", ""),
    )


@router.post("/v1/search", response_model=SearchResponse, tags=["search"])
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    """Hybrid search (vector + BM25 → RRF → MMR → importance boost)."""
    deps = getattr(request.app.state, "deps", None)
    if deps is None or deps.embedder is None or deps.repo is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")

    extra = SearchFilters(**body.filters.model_dump())

    result = await asyncio.to_thread(
        hybrid_search,
        body.query,
        query_embedder=deps.embedder,
        repo=deps.repo,
        extra_filters=extra,
        k=body.k,
        use_mmr=body.use_mmr,
    )

    return SearchResponse(
        query=result.query,
        clean_query=result.clean_query,
        filters=result.parsed_filters or {},
        n_results=len(result.hits),
        results=[_to_hit(r) for r in result.hits],
    )
