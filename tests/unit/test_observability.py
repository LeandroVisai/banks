"""Tests unitarios de Fase 7 — Observability + Production hardening.

Cubre:
- logging: bind_request_context, get_context, configure_logging (stdlib fallback).
- metrics: counters/histograms noop, record_retrieval, record_tool_call.
- tracing: setup_tracing noop, get_tracer noop span.
- rate_limit: _TokenBucket, RateLimitMiddleware.
- metrics endpoint: responde 200 con content-type apropiado.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from banks_rag.infrastructure.observability.logging import (
    bind_request_context,
    get_context,
    get_logger,
    reset,
)
from banks_rag.infrastructure.observability.metrics import (
    _NoopCounter,
    _NoopGauge,
    _NoopHistogram,
    generate_latest,
    get_content_type,
    prometheus_available,
    record_llm_generation,
    record_retrieval,
    record_tool_call,
)
from banks_rag.infrastructure.observability.tracing import (
    _NoopTracer,
    get_tracer,
    reset as reset_tracing,
    setup_tracing,
)
from banks_rag.interface.api.middleware.rate_limit import (
    RateLimitMiddleware,
    _TokenBucket,
)


# ── Tests logging ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestLogging:
    def setup_method(self):
        reset()

    def teardown_method(self):
        reset()

    def test_bind_and_get_context(self) -> None:
        bind_request_context(request_id="req-1", session_id="ses-1")
        ctx = get_context()
        assert ctx["request_id"] == "req-1"
        assert ctx["session_id"] == "ses-1"

    def test_empty_context_returns_empty_dict(self) -> None:
        ctx = get_context()
        assert ctx == {}

    def test_tool_call_id_in_context(self) -> None:
        bind_request_context(tool_call_id="tool-xyz")
        ctx = get_context()
        assert ctx["tool_call_id"] == "tool-xyz"

    def test_partial_bind_only_sets_provided(self) -> None:
        bind_request_context(request_id="abc")
        ctx = get_context()
        assert "request_id" in ctx
        assert "session_id" not in ctx

    def test_configure_logging_stdlib_fallback(self) -> None:
        from banks_rag.infrastructure.observability.logging import configure_logging
        # No structlog instalado en test env normalmente, pero configure_logging
        # no debe lanzar excepción en ningún caso.
        configure_logging(json=False, level="WARNING")

    def test_configure_logging_json(self) -> None:
        from banks_rag.infrastructure.observability.logging import configure_logging
        configure_logging(json=True, level="INFO")

    def test_configure_logging_idempotent(self) -> None:
        from banks_rag.infrastructure.observability.logging import configure_logging
        configure_logging(json=False, level="INFO")
        configure_logging(json=True, level="DEBUG")  # segunda llamada es noop

    def test_get_logger_returns_something(self) -> None:
        logger = get_logger("test.module")
        assert logger is not None

    def test_reset_clears_context(self) -> None:
        bind_request_context(request_id="x", session_id="y")
        reset()
        ctx = get_context()
        assert ctx == {}


# ── Tests metrics (noop) ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestMetricsNoop:
    def test_noop_counter_labels_inc(self) -> None:
        c = _NoopCounter()
        c.labels(status="ok").inc()  # no debe lanzar

    def test_noop_histogram_observe(self) -> None:
        h = _NoopHistogram()
        h.observe(0.5)  # no debe lanzar

    def test_noop_histogram_time_context_manager(self) -> None:
        h = _NoopHistogram()
        with h.time():
            time.sleep(0.001)

    def test_noop_gauge_set_inc_dec(self) -> None:
        g = _NoopGauge()
        g.set(5.0)
        g.inc(2.0)
        g.dec(1.0)

    def test_prometheus_available_is_bool(self) -> None:
        assert isinstance(prometheus_available(), bool)

    def test_generate_latest_returns_bytes(self) -> None:
        data = generate_latest()
        assert isinstance(data, bytes)

    def test_get_content_type_returns_string(self) -> None:
        ct = get_content_type()
        assert "text" in ct or "prometheus" in ct


@pytest.mark.unit
class TestMetricsHelpers:
    def test_record_retrieval_no_exception(self) -> None:
        record_retrieval(0.25, n_chunks=5, status="ok")

    def test_record_retrieval_error_status(self) -> None:
        record_retrieval(0.5, n_chunks=0, status="error")

    def test_record_llm_generation_no_exception(self) -> None:
        record_llm_generation(1.5, tokens=200, status="ok")

    def test_record_llm_generation_zero_tokens_no_observe(self) -> None:
        record_llm_generation(1.0, tokens=0)  # tokens=0 no observa histograma

    def test_record_tool_call_no_exception(self) -> None:
        record_tool_call("search_documents", status="ok")
        record_tool_call("execute_query", status="error")


# ── Tests tracing ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTracing:
    def setup_method(self):
        reset_tracing()

    def teardown_method(self):
        reset_tracing()

    def test_setup_tracing_off_returns_false(self) -> None:
        with patch.dict("os.environ", {"BANKS_TRACING": "off"}):
            result = setup_tracing()
        assert result is False

    def test_setup_tracing_idempotent(self) -> None:
        setup_tracing()
        setup_tracing()  # segunda llamada no debe lanzar

    def test_get_tracer_returns_tracer(self) -> None:
        tracer = get_tracer("test.module")
        assert tracer is not None

    def test_noop_tracer_span_context_manager(self) -> None:
        tracer = _NoopTracer()
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("key", "value")
            span.record_exception(ValueError("test"))

    def test_noop_span_set_status(self) -> None:
        tracer = _NoopTracer()
        span = tracer.start_span("span")
        span.set_status("ok")  # no debe lanzar


# ── Tests _TokenBucket ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTokenBucket:
    def test_fresh_bucket_allows_burst(self) -> None:
        bucket = _TokenBucket(rate_per_sec=1.0, burst=5)
        for _ in range(5):
            assert bucket.consume() is True

    def test_exhausted_bucket_blocks(self) -> None:
        bucket = _TokenBucket(rate_per_sec=0.1, burst=2)
        bucket.consume()
        bucket.consume()
        assert bucket.consume() is False

    def test_bucket_refills_over_time(self) -> None:
        bucket = _TokenBucket(rate_per_sec=100.0, burst=1)
        bucket.consume()  # vacía
        time.sleep(0.02)  # espera 20ms → ~2 tokens a 100/s
        assert bucket.consume() is True

    def test_burst_caps_token_accumulation(self) -> None:
        bucket = _TokenBucket(rate_per_sec=1000.0, burst=3)
        # Incluso con tasa alta, burst limita los tokens acumulables
        time.sleep(0.01)
        for _ in range(3):
            assert bucket.consume() is True
        assert bucket.consume() is False  # sin más tokens


# ── Tests RateLimitMiddleware ─────────────────────────────────────────────────

@pytest.mark.unit
class TestRateLimitMiddleware:
    def _make_request(self, path: str = "/v1/chat", api_key: str = "") -> MagicMock:
        req = MagicMock()
        req.url.path = path
        req.headers = {"x-api-key": api_key} if api_key else {}
        req.client = MagicMock()
        req.client.host = "127.0.0.1"
        return req

    def test_public_path_bypasses_limit(self) -> None:
        mw = RateLimitMiddleware(MagicMock(), rpm=1, burst=1)
        req = self._make_request(path="/healthz")
        # Simula que el bucket está vacío
        mw._buckets["ip:127.0.0.1"]  # fuerza la creación
        import asyncio
        call_next = AsyncMock(return_value=MagicMock())
        asyncio.run(mw.dispatch(req, call_next))
        call_next.assert_called_once()

    def test_rate_limited_returns_429(self) -> None:
        mw = RateLimitMiddleware(MagicMock(), rpm=60, burst=1)
        req = self._make_request(path="/v1/chat", api_key="testkey")
        import asyncio
        from starlette.responses import JSONResponse
        call_next = AsyncMock(return_value=MagicMock())

        # Primera: consume el único token del burst
        asyncio.run(mw.dispatch(req, call_next))
        # Segunda: sin tokens → 429
        call_next2 = AsyncMock()
        result = asyncio.run(mw.dispatch(req, call_next2))
        assert result.status_code == 429

    def test_api_key_used_as_bucket_key(self) -> None:
        mw = RateLimitMiddleware(MagicMock(), rpm=60, burst=10)
        req = self._make_request(api_key="mykey")
        key = mw._key(req)
        assert key == "key:mykey"

    def test_ip_fallback_when_no_api_key(self) -> None:
        mw = RateLimitMiddleware(MagicMock(), rpm=60, burst=10)
        req = self._make_request(api_key="")
        key = mw._key(req)
        assert key.startswith("ip:")


# ── Tests /metrics endpoint ───────────────────────────────────────────────────

@pytest.mark.unit
class TestMetricsEndpoint:
    def test_metrics_route_returns_200(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from banks_rag.interface.api.routes.metrics import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_route_content_type(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from banks_rag.interface.api.routes.metrics import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/metrics")
        ct = resp.headers.get("content-type", "")
        assert "text" in ct or "prometheus" in ct
