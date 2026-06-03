"""Router de jarvis_news: POST /v1/news-report y POST /v1/tts.

La app principal (banks_rag) monta este router. Reusa el LLM cargado por la app
(``app.state.deps.llm``) — no carga su propio modelo. Único contacto con el core:
el LLM (inyectado) y ``get_settings`` para leer límites/flags del .env.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, Response

from .audio import synthesize_wav, translate_to_english
from .report import generate_news_report
from .schemas import NewsReportRequest, NewsReportResponse, TtsRequest
from .tts import TTSError, build_default_tts

router = APIRouter()
log = logging.getLogger(__name__)


def _require_llm(request: Request):
    deps = getattr(request.app.state, "deps", None)
    if deps is None or deps.llm is None:
        raise HTTPException(status_code=503, detail="LLM no inicializado")
    if not getattr(deps.llm, "loaded", False):
        raise HTTPException(status_code=503, detail="LLM cargando — reintenta en breve")
    return deps.llm


def _synthesis_budget() -> int:
    try:
        from banks_rag.config import get_settings
        return get_settings().synthesis_max_tokens
    except Exception:  # noqa: BLE001
        return 4096


@router.post("/v1/news-report", response_model=NewsReportResponse, tags=["jarvis-news"])
async def news_report(request: Request, body: NewsReportRequest) -> NewsReportResponse:
    """Genera el reporte de prensa del día (map-reduce sobre el LLM)."""
    llm = _require_llm(request)
    try:
        result = await generate_news_report(
            llm, json_path=body.json_path, top_n=body.top_n,
            report_max_tokens=_synthesis_budget(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Error generando el reporte de noticias")
        raise HTTPException(status_code=502, detail=f"No se pudo generar el reporte: {exc}") from exc
    return NewsReportResponse(**result)


@router.post("/v1/tts", tags=["jarvis-news"])
async def tts(request: Request, body: TtsRequest) -> Response:
    """Texto → audio WAV con voz JARVIS (SAPI + efecto DSP, o Piper).
    Si lang='en' y translate_to_en, traduce el texto con el LLM antes."""
    if build_default_tts() is None:
        raise HTTPException(status_code=503, detail="TTS deshabilitado (BANKS_TTS_ENABLED=false).")

    text = body.text
    if body.lang == "en" and body.translate_to_en:
        llm = _require_llm(request)
        text = await translate_to_english(llm, text, max_tokens=_synthesis_budget())

    try:
        wav = await asyncio.to_thread(synthesize_wav, text, body.lang)
    except TTSError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Error sintetizando audio")
        raise HTTPException(status_code=502, detail=f"No se pudo generar el audio: {exc}") from exc

    return Response(content=wav, media_type="audio/wav")
