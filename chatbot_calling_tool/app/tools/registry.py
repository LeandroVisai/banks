"""
Registry pattern para tools.

Cada tool se declara con:
  @register(name="...", schema={... openai-compatible ...})
  async def my_tool(state: AgentState, **arguments) -> dict: ...

`AgentState` permite que la tool acumule citas globalmente (ver agent.py).
`dispatch` ejecuta la tool por nombre y captura excepciones para devolver
errores estructurados al modelo (que puede así corregir su llamada).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import AgentState

log = logging.getLogger(__name__)


# Registro global { tool_name → callable }
TOOL_REGISTRY: dict[str, Callable[..., Awaitable[dict]]] = {}

# Lista de schemas en formato OpenAI-compatible para apply_chat_template
TOOL_SCHEMAS: list[dict] = []


def register(name: str, schema: dict):
    """
    Decorador para registrar una tool.

    `schema` debe ser el dict OpenAI:
        {
          "type": "function",
          "function": {"name": ..., "description": ..., "parameters": {...}}
        }
    """
    if "function" not in schema or schema["function"].get("name") != name:
        raise ValueError(f"Schema inconsistente para tool {name!r}")

    def decorator(fn: Callable[..., Awaitable[dict]]):
        TOOL_REGISTRY[name] = fn
        TOOL_SCHEMAS.append(schema)
        log.debug("tool registered: %s", name)
        return fn

    return decorator


async def dispatch(
    state: "AgentState",
    name: str,
    arguments: dict[str, Any],
) -> tuple[dict, int]:
    """
    Ejecuta la tool y retorna (resultado, duración_ms).
    Si la tool falla, devuelve un dict de error que el LLM puede leer y corregir.
    """
    if name not in TOOL_REGISTRY:
        return {"error": f"Tool no disponible: {name!r}",
                "available_tools": list(TOOL_REGISTRY.keys())}, 0

    fn = TOOL_REGISTRY[name]
    t0 = time.perf_counter()
    try:
        result = await fn(state=state, **(arguments or {}))
    except TypeError as e:
        # Argumentos inválidos — le decimos al modelo qué firma esperaba
        return {
            "error": f"Argumentos inválidos para {name!r}: {e}",
            "expected_signature": _signature_hint(name),
        }, int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        log.exception("Tool %s failed", name)
        return {"error": f"Error ejecutando {name!r}: {e}"}, int((time.perf_counter() - t0) * 1000)

    return result, int((time.perf_counter() - t0) * 1000)


def _signature_hint(name: str) -> Optional[dict]:
    for schema in TOOL_SCHEMAS:
        if schema.get("function", {}).get("name") == name:
            return schema["function"].get("parameters")
    return None
