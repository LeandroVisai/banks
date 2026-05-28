"""Observabilidad: logging estructurado, métricas Prometheus, tracing OTel."""

from .chat_log import log_chat_error, log_chat_turn
from .logging import bind_request_context, configure_logging, get_context, get_logger
from .metrics import (
    AGENT_ITERATIONS,
    CHAT_REQUESTS,
    CHUNKS_IN_CONTEXT,
    LLM_LATENCY,
    RETRIEVAL_LATENCY,
    RETRIEVAL_REQUESTS,
    TOKENS_GENERATED,
    TOOL_CALLS,
    generate_latest,
    get_content_type,
    prometheus_available,
    record_llm_generation,
    record_retrieval,
    record_tool_call,
)
from .tracing import get_tracer, setup_tracing

__all__ = [
    # logging
    "configure_logging",
    "bind_request_context",
    "get_context",
    "get_logger",
    # chat log
    "log_chat_turn",
    "log_chat_error",
    # metrics
    "CHAT_REQUESTS",
    "TOOL_CALLS",
    "RETRIEVAL_REQUESTS",
    "RETRIEVAL_LATENCY",
    "LLM_LATENCY",
    "TOKENS_GENERATED",
    "AGENT_ITERATIONS",
    "CHUNKS_IN_CONTEXT",
    "prometheus_available",
    "generate_latest",
    "get_content_type",
    "record_retrieval",
    "record_llm_generation",
    "record_tool_call",
    # tracing
    "setup_tracing",
    "get_tracer",
]
