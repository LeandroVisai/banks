"""Rate limiting por API key (token bucket en memoria).

Configurable con variables de entorno:
  BANKS_RATE_LIMIT_RPM=60   — requests por minuto por key (default 60)
  BANKS_RATE_LIMIT_BURST=10 — burst máximo acumulable (default 10)

Si ``BANKS_API_KEYS`` está vacío (auth desactivada), el rate limit aplica
a todos los requests agrupados por IP (best-effort).

Endpoints públicos (``PUBLIC_PATHS``) no están sujetos a rate limit.

Algoritmo: token bucket. Cada key tiene un bucket de capacidad ``burst``.
Se añaden ``rpm/60`` tokens por segundo. Si el bucket está vacío → 429.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .auth import PUBLIC_PATHS

_DEFAULT_RPM = 60
_DEFAULT_BURST = 10


class _TokenBucket:
    """Bucket individual por key."""

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self._rate = rate_per_sec  # tokens/s
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()

    def consume(self) -> bool:
        """Consume 1 token. Returns False si no hay tokens (rate limited)."""
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limit por API key / IP.

    Args:
        rpm: requests permitidos por minuto por key.
        burst: tokens acumulables máximos (picos cortos).
    """

    def __init__(
        self,
        app,
        *,
        rpm: int | None = None,
        burst: int | None = None,
    ) -> None:
        super().__init__(app)
        self._rpm = rpm or int(os.getenv("BANKS_RATE_LIMIT_RPM", str(_DEFAULT_RPM)))
        self._burst = burst or int(os.getenv("BANKS_RATE_LIMIT_BURST", str(_DEFAULT_BURST)))
        self._rate_per_sec = self._rpm / 60.0
        self._buckets: dict[str, _TokenBucket] = defaultdict(self._make_bucket)
        self._lock = Lock()

    def _make_bucket(self) -> _TokenBucket:
        return _TokenBucket(self._rate_per_sec, self._burst)

    def _key(self, request: Request) -> str:
        api_key = request.headers.get("x-api-key", "")
        if api_key:
            return f"key:{api_key}"
        # Fallback a IP.
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() if forwarded else (
            request.client.host if request.client else "unknown"
        )
        return f"ip:{ip}"

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path in PUBLIC_PATHS or path == "/" or path.startswith("/assets/"):
            return await call_next(request)

        key = self._key(request)
        with self._lock:
            bucket = self._buckets[key]
            allowed = bucket.consume()

        if not allowed:
            retry_after = int(60 / self._rpm) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit excedido",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
