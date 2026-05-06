"""
Entry point — `python -m app.main` o `uvicorn app.main:app`.

Usar siempre `python -m app.main` para que el logging quede configurado
antes de que uvicorn cree sus propios loggers.
"""
from __future__ import annotations

import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

from .logging_config import configure_logging
from .settings import settings


def main() -> None:
    configure_logging()
    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        access_log=False,
        reload=False,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
