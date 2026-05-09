"""Parser de tool calls desde el output del LLM.

Soporta varios formatos:

  1. ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
     Formato preferido (Qwen3, Hermes-style).
  2. ```json ...``` fenced code block — algunos modelos lo emiten así.
  3. JSON crudo al final del mensaje — fallback más laxo.

El parser maneja también ``arguments`` como string-JSON anidado (algunos
modelos serializan dos veces).
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from banks_rag.domain.agent import ToolCall

log = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```")


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Extrae ``ToolCall``s del texto generado.

    Returns:
        ``(calls, cleaned_text)``. ``cleaned_text`` es la respuesta sin los
        markers ``<tool_call>`` (lo que va al usuario).
    """
    calls: list[ToolCall] = []

    # 1. Formato preferido: <tool_call>...</tool_call>
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            log.warning("tool_call con JSON inválido: %s", e)
            continue

        name = payload.get("name")
        args = payload.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            calls.append(ToolCall(
                id=_new_call_id(),
                name=name,
                arguments=args if isinstance(args, dict) else {},
            ))

    cleaned = _TOOL_CALL_RE.sub("", text).strip()

    # 2. Fallback: si no encontró ningún <tool_call>, intentar JSON crudo o fenced.
    if not calls:
        m = _FENCED_JSON_RE.search(text)
        candidate = m.group(1) if m else (
            text.strip() if text.strip().startswith("{") else None
        )
        if candidate:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                return calls, cleaned

            if (
                isinstance(payload, dict)
                and "name" in payload
                and "arguments" in payload
            ):
                calls.append(ToolCall(
                    id=_new_call_id(),
                    name=payload["name"],
                    arguments=payload.get("arguments") or {},
                ))
                cleaned = ""

    return calls, cleaned
