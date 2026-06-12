"""Helpers compartidos entre los engines LLM (in-process y OpenAI-compatible).

Extraídos de ``llama_cpp_engine.py`` para que ``OpenAICompatEngine`` (llama-server
u otro endpoint OpenAI-compatible) reutilice EXACTAMENTE la misma lógica de:

  - limpieza de razonamiento Qwen3 (``strip_think``),
  - detección de familia por nombre (``detect_family``),
  - parseo tolerante de argumentos de tool calls (``parse_args``),
  - aplanado de conversación para Gemma (``flatten_for_gemma`` +
    ``format_tools_as_text``).

Cero duplicación = cero divergencia de comportamiento entre backends.
``llama_cpp_engine`` re-exporta los nombres con guion bajo (``_strip_think``,
``_detect_family``, ...) por compatibilidad con imports/tests existentes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(raw_text: str) -> str:
    """Quita el razonamiento (thinking) del texto entregado al usuario.

    Dos casos:
      1. Bloque cerrado ``<think>...</think>`` — formato clásico.
      2. **Cierre colgante** ``...razonamiento... </think> respuesta`` SIN apertura:
         es lo que emite Qwen3 con su plantilla nativa del GGUF, porque el
         template ya inyecta ``<think>`` al final del prompt, así que el modelo
         arranca DENTRO del bloque y solo emite el ``</think>`` de cierre. Sin
         este caso, todo el chain-of-thought (a menudo en inglés) se filtraba a
         la respuesta final.
    """
    text = _THINK_RE.sub("", raw_text)
    if "<think>" not in text and "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def detect_family(model_path: str) -> str:
    """Infiere la familia del modelo desde el nombre del GGUF (solo basename).

    Devuelve ``"gemma"`` o ``"qwen"`` (este último es también el default para
    cualquier modelo ChatML-compatible). La familia decide CÓMO se construye el
    prompt:

    - ``qwen``  → plantilla de chat embebida en el GGUF / ``--jinja`` del
      servidor (tool-calling nativo: renderiza ``tools``, ``tool_calls`` y los
      resultados rol ``tool`` como ``<tool_response>``).
    - ``gemma`` → aplanado manual a la alternancia user/model
      (``flatten_for_gemma``).
    """
    p = Path(model_path).name.lower()
    if "gemma" in p:
        return "gemma"
    return "qwen"


def parse_args(raw: str | dict) -> dict:
    """Parsea ``arguments`` de un tool call, tolerando string-JSON anidado."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def format_tools_as_text(tools: list[dict]) -> str:
    """Renderiza los schemas de tools como texto para modelos sin tool calling nativo.

    Gemma no recibe bien los schemas vía la API; se inyectan en el prompt como
    texto, con el formato <tool_call> exacto que ``parse_tool_calls`` reconoce.
    """
    lines = [
        "\n\n## Herramientas disponibles",
        "",
        "Para usar una herramienta, responde EXACTAMENTE en este formato "
        "(JSON dentro de etiquetas <tool_call>):",
        "",
        '<tool_call>',
        '{"name": "nombre_exacto", "arguments": {"arg": "valor"}}',
        '</tool_call>',
        "",
        "Herramientas (usa el nombre EXACTO):",
    ]
    for schema in tools:
        fn = schema["function"]
        props = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        arg_parts = []
        for arg_name, _arg_spec in props.items():
            mark = "" if arg_name in required else " (opcional)"
            arg_parts.append(f'{arg_name}{mark}')
        args_str = ", ".join(arg_parts) if arg_parts else "sin argumentos"
        lines.append(f'- `{fn["name"]}` — args: {args_str}. {fn["description"]}')
    lines.append("")
    lines.append(
        "NUNCA respondas una pregunta sustantiva sin llamar primero a una "
        "herramienta. Cuando ya tengas la información, responde en texto SIN "
        "etiquetas <tool_call>."
    )
    return "\n".join(lines)


def flatten_for_gemma(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Aplana mensajes estilo OpenAI a la alternancia user/model de Gemma.

    Gemma no soporta los roles ``system`` ni ``tool``, ni ``assistant`` con
    ``tool_calls`` estructurados. Esta función:
      - fusiona los ``system`` (más las definiciones de tools como texto) en
        el primer mensaje ``user``;
      - renderiza los ``tool_calls`` del assistant como texto <tool_call>;
      - convierte los resultados de tools (rol ``tool``) en texto marcado,
        anexado al turno ``user`` para mantener la alternancia.
    """
    system_parts = [
        m["content"] for m in messages
        if m.get("role") == "system" and m.get("content")
    ]
    prefix = "\n\n".join(system_parts)
    if tools:
        prefix += format_tools_as_text(tools)

    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            content = m.get("content", "") or ""
            if prefix:
                content = prefix + "\n\n---\n\n" + content
                prefix = ""
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            content = m.get("content", "") or ""
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                content += (
                    f'\n<tool_call>\n{{"name": "{fn.get("name")}", '
                    f'"arguments": {args}}}\n</tool_call>'
                )
            out.append({"role": "assistant", "content": content})
        elif role == "tool":
            tool_msg = f'[RESULTADO DE {m.get("name", "herramienta")}]\n{m.get("content", "")}'
            if out and out[-1]["role"] == "user":
                out[-1]["content"] += "\n\n" + tool_msg
            else:
                out.append({"role": "user", "content": tool_msg})

    if prefix:  # no había ningún user; mete el contexto como primer turno
        out.insert(0, {"role": "user", "content": prefix})
    return out
