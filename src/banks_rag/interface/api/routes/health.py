"""Health endpoints: ``/healthz`` (liveness) y ``/readyz`` (readiness)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from banks_rag import __version__
from banks_rag.config import get_settings
from banks_rag.interface.api.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz(request: Request) -> HealthResponse:
    """Liveness: el proceso responde. Siempre 200."""
    settings = get_settings()
    deps = getattr(request.app.state, "deps", None)
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_loaded=bool(deps and deps.llm and getattr(deps.llm, "loaded", False)),
        db_ok=True,  # liveness no chequea DB
        model=settings.llm_model_path or settings.llm_family,
        rag_table_prefix=settings.rag_table_prefix,
    )


@router.get("/readyz", response_model=HealthResponse, tags=["health"])
async def readyz(request: Request) -> HealthResponse:
    """Readiness: BD accesible + LLM cargado.

    Devuelve 200 con ``status='loading'`` mientras el LLM no esté listo, y
    ``'degraded'`` si la BD falla. Esto permite que el orquestador (k8s,
    systemd) decida cuándo dirigir tráfico al pod.
    """
    settings = get_settings()
    deps = getattr(request.app.state, "deps", None)

    llm_loaded = bool(deps and deps.llm and getattr(deps.llm, "loaded", False))

    db_ok = False
    if deps and deps.repo is not None:
        try:
            await _ping_db(deps.repo)
            db_ok = True
        except Exception:  # noqa: BLE001
            db_ok = False

    if not db_ok:
        status = "degraded"
    elif not llm_loaded:
        status = "loading"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        version=__version__,
        llm_loaded=llm_loaded,
        db_ok=db_ok,
        model=settings.llm_model_path or settings.llm_family,
        rag_table_prefix=settings.rag_table_prefix,
    )


async def _ping_db(repo) -> None:
    """Ejecuta ``SELECT 1`` en la BD via threadpool."""
    import asyncio

    def _q() -> None:
        with repo.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

    await asyncio.to_thread(_q)
