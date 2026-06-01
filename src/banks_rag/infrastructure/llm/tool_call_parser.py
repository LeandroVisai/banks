"""Parser de tool calls desde el output del LLM.

Soporta varios formatos:

  1. ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
     Formato JSON (modo mock y modelos que serializan JSON).
  2. ``<tool_call><function=NAME><parameter=P>valor</parameter></function></tool_call>``
     Formato XML/Hermes — el que emite **Qwen3 con su plantilla nativa del GGUF**.
  3. ```json ...``` fenced code block — algunos modelos lo emiten así.
  4. JSON crudo al final del mensaje — fallback más laxo.

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

# Formato XML/Hermes de Qwen3 (plantilla nativa del GGUF):
#   <function=discover_query>
#     <parameter=query>spread BTP vs SPC a 10 años</parameter>
#     <parameter=segment>renta_fija_chile</parameter>
#   </function>
# El nombre va en la apertura; cada arg es un <parameter=...>. Lenient: tolera
# que falten los tags de cierre (generación truncada por max_tokens).
_XML_FUNCTION_OPEN_RE = re.compile(r"<function=([^>\s]+)\s*>")
_XML_FUNCTION_END_RE = re.compile(r"</function>|<function=|</tool_call>")
_XML_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)(?:</parameter>|$)", re.DOTALL,
)
# Limpieza: bloques completos y tags sueltos del protocolo de tool calls.
_CLEAN_BLOCK_RE = re.compile(
    r"<tool_call>.*?</tool_call>|<function=[^>]*>.*?</function>", re.DOTALL,
)
_CLEAN_TAG_RE = re.compile(
    r"</?tool_call>|</?function[^>]*>|</?parameter[^>]*>",
)


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _coerce_value(raw: str):
    """Convierte el valor textual de un <parameter> a su tipo nativo si es
    JSON-parseable (números, bool, listas, objetos); si no, lo deja como str."""
    s = raw.strip()
    if not s:
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def _parse_xml_tool_calls(text: str) -> list[ToolCall]:
    """Extrae tool calls en formato XML/Hermes (``<function=...><parameter=...>``)."""
    calls: list[ToolCall] = []
    for fm in _XML_FUNCTION_OPEN_RE.finditer(text):
        name = fm.group(1).strip()
        if not name:
            continue
        rest = text[fm.end():]
        end = _XML_FUNCTION_END_RE.search(rest)
        block = rest[: end.start()] if end else rest
        args = {
            pm.group(1).strip(): _coerce_value(pm.group(2))
            for pm in _XML_PARAM_RE.finditer(block)
        }
        calls.append(ToolCall(id=_new_call_id(), name=name, arguments=args))
    return calls


def _is_registered_tool(name: str) -> bool:
    """True si ``name`` es una tool registrada. Import perezoso para evitar
    un ciclo de imports con el paquete de tools."""
    try:
        from banks_rag.application.agent.tools.registry import TOOL_REGISTRY
        return name in TOOL_REGISTRY
    except Exception:  # noqa: BLE001
        return False


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

    # 2. Formato XML/Hermes de Qwen3 (<function=...><parameter=...>).
    if not calls:
        calls = _parse_xml_tool_calls(text)

    # Limpieza: quita los bloques completos y luego cualquier tag suelto que
    # haya quedado (p. ej. una apertura sin cierre por truncado).
    cleaned = _CLEAN_BLOCK_RE.sub("", text)
    cleaned = _CLEAN_TAG_RE.sub("", cleaned).strip()

    # 3. Fallback: si no encontró ningún <tool_call>, intentar JSON crudo o fenced.
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

            # El fallback solo acepta el candidato si ``name`` es una tool
            # realmente registrada: así una respuesta final que casualmente
            # empiece con `{` o contenga un bloque ```json no se interpreta
            # por error como tool call (lo que vaciaría la respuesta).
            if (
                isinstance(payload, dict)
                and "name" in payload
                and "arguments" in payload
                and _is_registered_tool(payload["name"])
            ):
                calls.append(ToolCall(
                    id=_new_call_id(),
                    name=payload["name"],
                    arguments=payload.get("arguments") or {},
                ))
                cleaned = ""

    return calls, cleaned
