"""OpenTelemetry tracing — opcional, off por defecto.

Activar con variable de entorno ``BANKS_TRACING=otlp``.
Si ``opentelemetry-sdk`` no está instalado, todos los calls son noop.

Uso:
    from banks_rag.infrastructure.observability.tracing import get_tracer, setup_tracing

    # Al inicio de la app:
    setup_tracing()

    # En cualquier punto:
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("hybrid_search") as span:
        span.set_attribute("query.len", len(query))
        result = hybrid_search(...)
        span.set_attribute("hits.count", len(result.hits))
"""

from __future__ import annotations

import os
from typing import Any


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass
    def record_exception(self, exc: Exception) -> None:
        pass
    def set_status(self, *args, **kwargs) -> None:
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class _NoopTracer:
    def start_as_current_span(self, name: str, **kwargs) -> _NoopSpan:
        return _NoopSpan()
    def start_span(self, name: str, **kwargs) -> _NoopSpan:
        return _NoopSpan()


_NOOP_TRACER = _NoopTracer()
_configured = False


def setup_tracing(
    *,
    service_name: str = "banks-rag",
    otlp_endpoint: str | None = None,
) -> bool:
    """Configura el exporter OTLP si BANKS_TRACING=otlp.

    Returns:
        True si el tracing real quedó activo, False si es noop.
    """
    global _configured
    if _configured:
        return _is_real_tracing_active()

    mode = os.getenv("BANKS_TRACING", "off").lower()
    if mode != "otlp":
        _configured = True
        return False

    try:
        from opentelemetry import trace  # type: ignore[import]
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore[import]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]

        endpoint = otlp_endpoint or os.getenv("BANKS_OTLP_ENDPOINT", "http://localhost:4317")
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _configured = True
        return True
    except ImportError:
        _configured = True
        return False


def _is_real_tracing_active() -> bool:
    try:
        from opentelemetry import trace  # type: ignore[import]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
        return isinstance(trace.get_tracer_provider(), TracerProvider)
    except ImportError:
        return False


def get_tracer(name: str) -> Any:
    """Devuelve un tracer real (si OTel disponible) o noop."""
    try:
        from opentelemetry import trace  # type: ignore[import]
        return trace.get_tracer(name)
    except ImportError:
        return _NOOP_TRACER


def reset() -> None:
    """Reinicia el estado (útil en tests)."""
    global _configured
    _configured = False
