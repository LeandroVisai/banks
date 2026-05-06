"""Entry point — `python -m app.main`."""
from __future__ import annotations

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
    )


if __name__ == "__main__":
    main()
