"""Router de jarvis_news: POST /v1/news-report y POST /v1/tts.

La app principal (banks_rag) monta este router. Reusa el LLM cargado por la app
(``app.state.deps.llm``) — no carga su propio modelo. Único contacto con el core:
el LLM (inyectado) y ``get_settings`` para leer límites/flags del .env.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, Response

from .report import generate_news_report
from .schemas import NewsReportRequest, NewsReportResponse, TtsRequest
from .tts import TTSError, build_default_tts

router = APIRouter()
log = logging.getLogger(__name__)

_TRANSLATE_SYSTEM = "You are a professional translator. Output only the translation, nothing else."
_TRANSLATE_PROMPT = (
    "Translate the following Spanish text into natural British English suitable "
    "for being read aloud (clear sentences, expand figures). Text:\n\n"
)


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
    """Texto → audio WAV (voz JARVIS vía Piper). Traduce a inglés si se pide
    (la voz JARVIS es en_GB)."""
    engine = build_default_tts()
    if engine is None:
        raise HTTPException(status_code=503, detail="TTS deshabilitado (BANKS_TTS_ENABLED=false).")

    text = body.text
    if body.translate_to_en:
        llm = _require_llm(request)
        res = await llm.generate(
            [{"role": "system", "content": _TRANSLATE_SYSTEM},
             {"role": "user", "content": _TRANSLATE_PROMPT + text}],
            tools=None, temperature=0.3, max_tokens=_synthesis_budget(),
        )
        text = (res.text or "").strip() or body.text

    try:
        wav = await asyncio.to_thread(engine.synthesize, text)
    except TTSError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Error sintetizando audio")
        raise HTTPException(status_code=502, detail=f"No se pudo generar el audio: {exc}") from exc

    return Response(content=wav, media_type="audio/wav")
