"""Métricas Prometheus para el servicio RAG.

Usa ``prometheus_client`` si está instalado; en caso contrario expone stubs
que aceptan las mismas llamadas sin efecto (modo noop). Esto permite importar
el módulo en cualquier entorno sin dependencias extras.

Métricas expuestas:
  Counters:
    banks_chat_requests_total{status}      — requests al endpoint /v1/chat
    banks_tool_calls_total{tool,status}    — invocaciones de tools del agente
    banks_retrieval_requests_total{status} — llamadas a hybrid_search

  Histograms:
    banks_retrieval_latency_seconds        — latencia end-to-end de retrieval
    banks_llm_generation_latency_seconds   — latencia de generación LLM
    banks_tokens_generated                 — tokens por respuesta

  Gauges:
    banks_agent_iterations                 — iteraciones del agente por request
    banks_chunks_in_context                — chunks enviados al LLM como contexto

Uso:
    from banks_rag.infrastructure.observability.metrics import (
        CHAT_REQUESTS, RETRIEVAL_LATENCY, record_retrieval,
    )
    with RETRIEVAL_LATENCY.time():
        result = hybrid_search(...)
    CHAT_REQUESTS.labels(status="ok").inc()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

# ── Stubs noop (usados cuando prometheus_client no está instalado) ─────────────

class _NoopCounter:
    def labels(self, **kwargs) -> "_NoopCounter":
        return self
    def inc(self, amount: float = 1) -> None:
        pass


class _NoopHistogram:
    def labels(self, **kwargs) -> "_NoopHistogram":
        return self
    def observe(self, value: float) -> None:
        pass
    @contextmanager
    def time(self) -> Generator:
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.observe(time.monotonic() - t0)


class _NoopGauge:
    def labels(self, **kwargs) -> "_NoopGauge":
        return self
    def set(self, value: float) -> None:
        pass
    def inc(self, amount: float = 1) -> None:
        pass
    def dec(self, amount: float = 1) -> None:
        pass


# ── Inicialización lazy de métricas Prometheus ────────────────────────────────

def _make_metrics():
    """Crea métricas Prometheus reales o stubs según disponibilidad."""
    try:
        from prometheus_client import Counter, Gauge, Histogram

        chat_requests = Counter(
            "banks_chat_requests_total",
            "Total de requests al endpoint /v1/chat",
            ["status"],
        )
        tool_calls = Counter(
            "banks_tool_calls_total",
            "Invocaciones de tools del agente",
            ["tool", "status"],
        )
        retrieval_requests = Counter(
            "banks_retrieval_requests_total",
            "Llamadas a hybrid_search",
            ["status"],
        )
        retrieval_latency = Histogram(
            "banks_retrieval_latency_seconds",
            "Latencia end-to-end de hybrid_search",
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        )
        llm_latency = Histogram(
            "banks_llm_generation_latency_seconds",
            "Latencia de generación LLM",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )
        tokens_generated = Histogram(
            "banks_tokens_generated",
            "Tokens generados por respuesta LLM",
            buckets=[50, 100, 200, 500, 1000, 2000],
        )
        agent_iterations = Gauge(
            "banks_agent_iterations",
            "Iteraciones del agente por request",
        )
        chunks_in_context = Gauge(
            "banks_chunks_in_context",
            "Chunks enviados al LLM como contexto",
        )
        return (
            chat_requests, tool_calls, retrieval_requests,
            retrieval_latency, llm_latency, tokens_generated,
            agent_iterations, chunks_in_context,
            True,  # prometheus available
        )
    except ImportError:
        return (
            _NoopCounter(), _NoopCounter(), _NoopCounter(),
            _NoopHistogram(), _NoopHistogram(), _NoopHistogram(),
            _NoopGauge(), _NoopGauge(),
            False,
        )


(
    CHAT_REQUESTS,
    TOOL_CALLS,
    RETRIEVAL_REQUESTS,
    RETRIEVAL_LATENCY,
    LLM_LATENCY,
    TOKENS_GENERATED,
    AGENT_ITERATIONS,
    CHUNKS_IN_CONTEXT,
    _PROMETHEUS_AVAILABLE,
) = _make_metrics()


def prometheus_available() -> bool:
    """Devuelve True si prometheus_client está instalado."""
    return _PROMETHEUS_AVAILABLE


# ── Helpers de alto nivel ─────────────────────────────────────────────────────

def record_retrieval(latency_s: float, *, n_chunks: int, status: str = "ok") -> None:
    """Registra una llamada a hybrid_search."""
    RETRIEVAL_REQUESTS.labels(status=status).inc()
    RETRIEVAL_LATENCY.observe(latency_s)
    CHUNKS_IN_CONTEXT.set(n_chunks)


def record_llm_generation(latency_s: float, *, tokens: int, status: str = "ok") -> None:
    """Registra una generación LLM."""
    LLM_LATENCY.observe(latency_s)
    if tokens > 0:
        TOKENS_GENERATED.observe(tokens)


def record_tool_call(tool: str, *, status: str = "ok") -> None:
    """Registra una invocación de tool del agente."""
    TOOL_CALLS.labels(tool=tool, status=status).inc()


def generate_latest() -> bytes:
    """Serializa todas las métricas en formato Prometheus text exposition."""
    try:
        from prometheus_client import generate_latest as _gen
        return _gen()
    except ImportError:
        return b"# prometheus_client not installed\n"


def get_content_type() -> str:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST
        return CONTENT_TYPE_LATEST
    except ImportError:
        return "text/plain; version=0.0.4"
