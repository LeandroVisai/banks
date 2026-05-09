"""Middleware que asigna un ``request_id`` único a cada request.

El ID se inyecta en:
  - Header de respuesta ``X-Request-ID``.
  - Atributo ``request.state.request_id`` (accesible desde handlers/routes).
  - Logs (vía contextvar — pendiente cuando se integre structlog en Fase 7).

Si el cliente envía ``X-Request-ID`` en la request, se respeta (idempotencia
para retries). Si no, se genera un UUID4 corto.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("x-request-id") or _new_id()
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
