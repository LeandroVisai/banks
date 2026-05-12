"""Logging estructurado con contextvars para trazabilidad end-to-end.

Usa ``structlog`` si está instalado; cae a stdlib logging si no.
Los contextvars permiten propagar request_id, session_id y tool_call_id
sin pasar parámetros explícitos a través del call stack.

Uso típico:
    from banks_rag.infrastructure.observability.logging import (
        configure_logging, bind_request_context, get_logger,
    )

    # Al inicio de la app:
    configure_logging(json=True, level="INFO")

    # En el middleware de request:
    bind_request_context(request_id="abc123", session_id="ses456")

    # En cualquier módulo:
    log = get_logger(__name__)
    log.info("chunk recuperado", n_chunks=5, query_len=42)
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")
_tool_call_id_var: ContextVar[str] = ContextVar("tool_call_id", default="")

_structlog_configured = False


def bind_request_context(
    *,
    request_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
) -> None:
    """Establece los contextvars para el request actual."""
    if request_id:
        _request_id_var.set(request_id)
    if session_id:
        _session_id_var.set(session_id)
    if tool_call_id:
        _tool_call_id_var.set(tool_call_id)


def get_context() -> dict[str, str]:
    """Devuelve el contexto actual como dict (para incluir en logs)."""
    ctx: dict[str, str] = {}
    if rid := _request_id_var.get():
        ctx["request_id"] = rid
    if sid := _session_id_var.get():
        ctx["session_id"] = sid
    if tid := _tool_call_id_var.get():
        ctx["tool_call_id"] = tid
    return ctx


def configure_logging(*, json: bool = False, level: str = "INFO") -> None:
    """Configura el sistema de logging una sola vez.

    Args:
        json: si True, emite JSON; si False, texto legible.
        level: nivel de logging (DEBUG, INFO, WARNING, ERROR).
    """
    global _structlog_configured
    if _structlog_configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    try:
        import structlog

        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            _inject_banks_context,
        ]

        if json:
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

        structlog.configure(
            processors=shared_processors + [renderer],
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Redirigir stdlib logging a structlog.
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=numeric_level,
        )
        for handler in logging.root.handlers:
            handler.setLevel(numeric_level)

    except ImportError:
        # Fallback a stdlib logging.
        fmt = (
            '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
            if json
            else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logging.basicConfig(
            level=numeric_level,
            format=fmt,
            stream=sys.stdout,
        )

    _structlog_configured = True


def _inject_banks_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Processor de structlog que inyecta los contextvars del servicio."""
    ctx = get_context()
    event_dict.update(ctx)
    return event_dict


def get_logger(name: str) -> Any:
    """Devuelve un logger estructurado (structlog) o stdlib según disponibilidad."""
    try:
        import structlog
        return structlog.get_logger(name)
    except ImportError:
        return logging.getLogger(name)


def reset() -> None:
    """Reinicia el estado (útil en tests)."""
    global _structlog_configured
    _structlog_configured = False
    _request_id_var.set("")
    _session_id_var.set("")
    _tool_call_id_var.set("")
