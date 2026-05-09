"""Auth middleware basado en API keys (opcional).

Si ``settings.api_keys`` está vacío, el middleware deja pasar todo.
Si tiene valores, exige un header ``X-API-Key`` con uno de los keys configurados.

Endpoints públicos (siempre accesibles): ``/healthz``, ``/readyz``, ``/metrics``.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

PUBLIC_PATHS: frozenset[str] = frozenset({
    "/healthz",
    "/readyz",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, api_keys: set[str]) -> None:
        super().__init__(app)
        self._keys = api_keys

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._keys:
            return await call_next(request)
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("x-api-key", "")
        if provided not in self._keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "API key inválida o ausente"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return await call_next(request)
