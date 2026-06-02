"""POST ``/v1/news-report`` — analizador de noticias (informe diario).

Lee un JSON de ``data_pipeline/Noticias_scrapping/``, prioriza las noticias más
relevantes del día y genera un reporte estructurado vía map-reduce sobre el LLM.
Puede tardar (varias llamadas al LLM serializado); no es un endpoint interactivo.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from banks_rag.application.news import generate_news_report
from banks_rag.config import get_settings
from banks_rag.interface.api.schemas import NewsReportRequest, NewsReportResponse

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/v1/news-report", response_model=NewsReportResponse, tags=["news"])
async def news_report(request: Request, body: NewsReportRequest) -> NewsReportResponse:
    settings = get_settings()
    deps = getattr(request.app.state, "deps", None)
    if deps is None or deps.llm is None:
        raise HTTPException(status_code=503, detail="LLM no inicializado")
    if not getattr(deps.llm, "loaded", False):
        raise HTTPException(status_code=503, detail="LLM cargando — reintenta en breve")

    try:
        result = await generate_news_report(
            deps.llm,
            json_path=body.json_path,
            top_n=body.top_n,
            report_max_tokens=settings.synthesis_max_tokens,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Error generando el reporte de noticias")
        raise HTTPException(status_code=502, detail=f"No se pudo generar el reporte: {exc}") from exc

    return NewsReportResponse(**result)
