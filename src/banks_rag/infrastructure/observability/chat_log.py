"""Persistencia de los turnos del agente a JSONL diario.

Cada turno de ``/v1/chat`` se anexa como una línea JSON a
``data/chat_logs/chat-YYYY-MM-DD.jsonl`` (configurable). El registro guarda la
pregunta, la respuesta y toda la evidencia que reunió el agente (tool_trace,
chunks_seen, series_used, citas) para poder **contrastar después lo que el bot
respondió con lo que debería haber respondido** y mejorar sus respuestas.

Diseño:
  - Best-effort: nunca propaga excepciones — un fallo al escribir el log no
    debe tumbar el request. Loguea un warning y sigue.
  - Activable/desactivable con ``BANKS_CHAT_LOG_ENABLED`` (default on) y
    redirigible con ``BANKS_CHAT_LOG_DIR``.
  - Un ``threading.Lock`` serializa los appends dentro del proceso para que
    líneas grandes (con tool_trace/chunks) no se entrelacen.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from banks_rag.config import get_settings

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentResult

log = logging.getLogger(__name__)

_write_lock = threading.Lock()

# Versión del esquema del registro. Súbela si cambias campos para que el
# análisis offline pueda distinguir formatos.
SCHEMA_VERSION = 1


def _now() -> datetime:
    return datetime.now(UTC)


def _log_path(log_dir: Path, when: datetime) -> Path:
    return log_dir / f"chat-{when.date().isoformat()}.jsonl"


def _history_summary(history: list[dict]) -> list[dict]:
    """Compacta el history a ``[{role, content}]`` (descarta otros campos)."""
    out = []
    for m in history or []:
        role = m.get("role")
        if role == "system":  # nunca debería llegar, pero por las dudas
            continue
        out.append({"role": role, "content": m.get("content", "")})
    return out


def build_turn_record(
    user_message: str,
    history: list[dict],
    result: AgentResult,
    *,
    model: str,
    prompt_version: str,
    request_id: str = "",
    when: datetime | None = None,
) -> dict[str, Any]:
    """Construye el registro JSON-serializable de un turno exitoso."""
    when = when or _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": when.isoformat(),
        "request_id": request_id,
        "status": "ok",
        "model": model,
        "prompt_version": prompt_version,
        "user_message": user_message,
        "history": _history_summary(history),
        "response": result.response,
        "finish_reason": result.finish_reason,
        "iterations": result.iterations,
        "latency_ms": result.latency_ms,
        "total_tokens": result.total_tokens,
        "cited_refs": result.cited_refs,
        "invalid_refs": result.invalid_refs,
        "ungrounded_numbers": getattr(result, "ungrounded_numbers", []),
        "tool_trace": result.tool_trace,
        "chunks_seen": result.chunks_seen,
        "series_used": result.series_used,
    }


def build_error_record(
    user_message: str,
    history: list[dict],
    error: str,
    *,
    model: str,
    prompt_version: str,
    request_id: str = "",
    when: datetime | None = None,
) -> dict[str, Any]:
    """Construye el registro de un turno que falló antes de producir respuesta."""
    when = when or _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": when.isoformat(),
        "request_id": request_id,
        "status": "error",
        "model": model,
        "prompt_version": prompt_version,
        "user_message": user_message,
        "history": _history_summary(history),
        "error": error,
    }


def append_record(record: dict[str, Any]) -> Path | None:
    """Anexa un registro al JSONL del día. Best-effort: nunca lanza.

    Retorna el path escrito, o ``None`` si el logging está desactivado o falló.
    """
    settings = get_settings()
    if not settings.chat_log_enabled:
        return None

    try:
        when = _now()
        log_dir = settings.chat_log_dir_resolved
        log_dir.mkdir(parents=True, exist_ok=True)
        path = _log_path(log_dir, when)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with _write_lock, path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return path
    except Exception:
        log.warning("no se pudo persistir el log de chat", exc_info=True)
        return None


def log_chat_turn(
    user_message: str,
    history: list[dict],
    result: AgentResult,
    *,
    model: str,
    prompt_version: str,
    request_id: str = "",
) -> Path | None:
    """Persiste un turno exitoso del agente. Best-effort."""
    record = build_turn_record(
        user_message, history, result,
        model=model, prompt_version=prompt_version, request_id=request_id,
    )
    return append_record(record)


def log_chat_error(
    user_message: str,
    history: list[dict],
    error: str,
    *,
    model: str,
    prompt_version: str,
    request_id: str = "",
) -> Path | None:
    """Persiste un turno que falló. Best-effort."""
    record = build_error_record(
        user_message, history, error,
        model=model, prompt_version=prompt_version, request_id=request_id,
    )
    return append_record(record)
