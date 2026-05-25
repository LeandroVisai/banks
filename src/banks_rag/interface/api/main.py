"""FastAPI app unificada — reemplaza chatbot/ y chatbot_calling_tool/.

Endpoints:
  - GET  ``/healthz``         — liveness
  - GET  ``/readyz``          — readiness (DB + LLM)
  - POST ``/v1/chat``         — agente con tool calling
  - POST ``/v1/search``       — búsqueda híbrida sin LLM

Uso:

    python -m banks_rag.interface.api.main

o con override de host/port:

    BANKS_API_PORT=8081 python -m banks_rag.interface.api.main

Para tests, importa ``create_app`` directamente y usa ``TestClient(create_app())``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from banks_rag import __version__
from banks_rag.config import get_settings
from banks_rag.config.paths import ROOT
from banks_rag.infrastructure.observability import configure_logging, setup_tracing
from banks_rag.interface.api.dependencies import AppState
from banks_rag.interface.api.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from banks_rag.interface.api.routes import catalog, chat, health, images, search
from banks_rag.interface.api.routes import metrics as metrics_route

# Importar el paquete de tools para que el registry quede poblado.
import banks_rag.application.agent.tools  # noqa: F401

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga LLM, embedder y repo al startup; los descarga al shutdown.

    Para tests, ``create_app(deps=AppState())`` permite saltarse el lifespan
    y configurar dependencias mockeadas directamente.
    """
    settings = get_settings()
    deps: AppState = app.state.deps

    configure_logging(json=settings.log_json, level=settings.log_level)
    setup_tracing()
    log.info("API iniciando — version=%s", __version__)

    # PostgresRepo es barato (sin conexión hasta ser usado).
    if deps.repo is None:
        from banks_rag.infrastructure.persistence import PostgresRepo
        deps.repo = PostgresRepo(prefix=settings.rag_table_prefix)

    # Embedder con lazy load.
    if deps.embedder is None:
        from banks_rag.infrastructure.embeddings import build_default_embedder
        deps.embedder = build_default_embedder()

    # LLM: instancia LlamaCppEngine para familias qwen/gemma.
    if deps.llm is None and settings.llm_family not in ("mock", ""):
        try:
            from banks_rag.infrastructure.llm.llama_cpp_engine import LlamaCppEngine
            deps.llm = LlamaCppEngine.from_settings(settings)
            await deps.llm.load()
            log.info("LLM cargado: %s", deps.llm.name)
        except Exception:
            log.exception("Error cargando LLM — /v1/chat retornará 503")

    yield

    log.info("API cerrando")
    if deps.llm is not None and hasattr(deps.llm, "unload"):
        try:
            await deps.llm.unload()
        except Exception:  # noqa: BLE001
            log.exception("Error al unload LLM")


def create_app(*, deps: AppState | None = None) -> FastAPI:
    """Factory de la aplicación.

    Args:
        deps: instancia de ``AppState`` con dependencies pre-cableadas
            (para tests). Si ``None``, el lifespan las inicializa.
    """
    settings = get_settings()
    app = FastAPI(
        title="banks_rag API",
        version=__version__,
        description=(
            "RAG + agente multimodal para banco central. "
            "Endpoints: /v1/chat (agentic), /v1/search (hybrid retrieval)."
        ),
        lifespan=lifespan,
    )

    app.state.deps = deps if deps is not None else AppState()

    # CORS (necesario si el frontend se sirve desde otro origen en dev).
    cors_origins_env = os.getenv("BANKS_CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middlewares (orden: outer → inner, se ejecutan en orden inverso).
    app.add_middleware(RequestIDMiddleware)
    if settings.api_keys_set:
        app.add_middleware(APIKeyMiddleware, api_keys=settings.api_keys_set)
    app.add_middleware(RateLimitMiddleware)

    # Routes.
    app.include_router(health.router)
    app.include_router(metrics_route.router)
    app.include_router(chat.router)
    app.include_router(search.router)
    app.include_router(images.router)
    app.include_router(catalog.router)

    # Static frontend: sirve interface2/ en / (same-origin con la API).
    # Montado al final para que las rutas /v1/* y /healthz tengan prioridad.
    interface2_dir = ROOT / "interface2"
    if interface2_dir.is_dir():
        no_cache = os.getenv("BANKS_DEV", "1") == "1"

        class _NoCacheStaticFiles(StaticFiles):
            async def get_response(self, path, scope):
                resp = await super().get_response(path, scope)
                if no_cache and hasattr(resp, "headers"):
                    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
                    resp.headers["Pragma"] = "no-cache"
                    resp.headers["Expires"] = "0"
                return resp

        app.mount(
            "/",
            _NoCacheStaticFiles(directory=str(interface2_dir), html=True),
            name="interface2",
        )

    return app


def main() -> None:
    """Entrypoint para ``python -m banks_rag.interface.api.main``."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "banks_rag.interface.api.main:create_app",
        host=settings.api_host,
        port=settings.api_port,
        factory=True,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
