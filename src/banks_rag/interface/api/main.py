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
from contextlib import asynccontextmanager

from fastapi import FastAPI

from banks_rag import __version__
from banks_rag.config import get_settings
from banks_rag.interface.api.dependencies import AppState
from banks_rag.interface.api.middleware import (
    APIKeyMiddleware,
    RequestIDMiddleware,
)
from banks_rag.interface.api.routes import chat, health, images, search

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

    log.info("API iniciando — version=%s", __version__)

    # PostgresRepo es barato (sin conexión hasta ser usado).
    if deps.repo is None:
        from banks_rag.infrastructure.persistence import PostgresRepo
        deps.repo = PostgresRepo(prefix=settings.rag_table_prefix)

    # Embedder con lazy load.
    if deps.embedder is None:
        from banks_rag.infrastructure.embeddings import build_default_embedder
        deps.embedder = build_default_embedder()

    # LLM: en Fase 3 será LlamaCppEngine. Por ahora si no hay implementation
    # registrada, queda como None y los endpoints de chat retornan 503.
    if deps.llm is None and settings.llm_family != "mock":
        log.warning(
            "LLM family=%s pero no hay implementación registrada — "
            "el endpoint /v1/chat retornará 503 hasta Fase 3.",
            settings.llm_family,
        )

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

    # Middlewares (orden: outer → inner).
    app.add_middleware(RequestIDMiddleware)
    if settings.api_keys_set:
        app.add_middleware(APIKeyMiddleware, api_keys=settings.api_keys_set)

    # Routes.
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(search.router)
    app.include_router(images.router)

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
