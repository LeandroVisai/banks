"""Endpoint ``/metrics`` — expone métricas Prometheus."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import Response

from banks_rag.infrastructure.observability.metrics import generate_latest, get_content_type

router = APIRouter()


@router.get("/metrics", tags=["observability"], include_in_schema=False)
async def metrics() -> Response:
    """Métricas Prometheus en formato text exposition (scrape endpoint)."""
    return Response(
        content=generate_latest(),
        media_type=get_content_type(),
    )
