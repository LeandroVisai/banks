"""Registry pattern para tools del agente.

Cada tool se declara con:

    @register("my_tool", schema={...openai-compatible...})
    async def my_tool(state: AgentState, **arguments) -> dict:
        ...

``AgentState`` permite que la tool acumule citas globalmente (refs ``[1]``,
``[2]``...). ``dispatch`` ejecuta la tool por nombre y captura excepciones
para devolver errores estructurados al modelo (que puede así corregir su
llamada en la próxima iteración).
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from banks_rag.domain.agent import AgentState

log = logging.getLogger(__name__)


# Registro global { tool_name → callable }
TOOL_REGISTRY: dict[str, Callable[..., Awaitable[dict]]] = {}

# Alias de parámetros que los modelos (sobre todo cuantizaciones agresivas)
# confunden: emiten `keywords`/`search_term` en vez de `query`, `table` en vez
# de `dataset_id`, etc. Se mapean al nombre canónico ANTES de invocar la tool,
# de modo que un nombre de argumento equivocado no rompa la llamada (lo que
# obligaba al modelo a reintentar y agotaba iteraciones). Solo aplica si el
# canónico es un parámetro real de la tool y el alias no lo es.
_ARG_ALIASES: dict[str, str] = {
    "keywords": "query", "keyword": "query", "search_term": "query",
    "search": "query", "q": "query", "text": "query", "term": "query",
    "terms": "query", "consulta": "query", "pregunta": "query",
    "question": "query",
    "dataset": "dataset_id", "id": "dataset_id", "table": "dataset_id",
    "table_id": "dataset_id", "series_id": "dataset_id", "dataset_name": "dataset_id",
    "datasetid": "dataset_id", "dataset_ids": "dataset_id",
    "col": "column", "columna": "column", "field": "column", "columns_name": "column",
}


def _accepted_params(fn: Callable) -> tuple[set[str], bool]:
    """Nombres de parámetros que acepta la tool (sin ``state``) y si tiene **kwargs."""
    names: set[str] = set()
    has_var_kw = False
    for p in inspect.signature(fn).parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_kw = True
        elif p.name != "state":
            names.add(p.name)
    return names, has_var_kw


def _normalize_arguments(
    fn: Callable, arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Remapea alias y descarta kwargs no aceptados, para tolerar modelos que
    inventan nombres de argumentos. Retorna ``(args_normalizados, notas)``.

    Si la tool acepta ``**kwargs`` no se toca nada (la tool decide qué hacer).
    """
    if not arguments:
        return arguments, []
    accepted, has_var_kw = _accepted_params(fn)
    if has_var_kw:
        return arguments, []
    arguments = dict(arguments)
    notes: list[str] = []
    # Caso especial: el modelo empaqueta "dataset_id.columna" en un único arg
    # `series`/`serie` (visto en compute_variation/get_series_stats). Si la tool
    # espera `dataset_id` + `column` y no se entregaron, se separa por el punto.
    if {"dataset_id", "column"} <= accepted:
        for skey in ("series", "serie"):
            sval = arguments.get(skey)
            if (
                isinstance(sval, str) and "." in sval
                and "dataset_id" not in arguments and "column" not in arguments
            ):
                ds, _, col = sval.partition(".")
                if ds and col:
                    arguments.pop(skey)
                    arguments["dataset_id"] = ds
                    arguments["column"] = col
                    notes.append(f"{skey!r}→'dataset_id'+'column'")
                break
    out: dict[str, Any] = {}
    for key, val in arguments.items():
        if key in accepted:
            out[key] = val
            continue
        canon = _ARG_ALIASES.get(key)
        if canon and canon in accepted and canon not in arguments and canon not in out:
            # Campos tipo `query` esperan string; algunos modelos mandan lista.
            if canon == "query" and isinstance(val, list):
                val = " ".join(str(v) for v in val)
            out[canon] = val
            notes.append(f"{key!r}→{canon!r}")
        else:
            notes.append(f"ignorado:{key!r}")
    return out, notes

# Lista de schemas en formato OpenAI-compatible para apply_chat_template.
TOOL_SCHEMAS: list[dict] = []


def register(name: str, schema: dict):
    """Decorador para registrar una tool.

    ``schema`` debe ser el dict OpenAI:

        {
          "type": "function",
          "function": {"name": ..., "description": ..., "parameters": {...}}
        }

    El nombre del decorador y el del schema deben coincidir (validación al
    importar el módulo).
    """
    if "function" not in schema or schema["function"].get("name") != name:
        raise ValueError(f"Schema inconsistente para tool {name!r}")

    def decorator(fn: Callable[..., Awaitable[dict]]):
        TOOL_REGISTRY[name] = fn
        TOOL_SCHEMAS.append(schema)
        log.debug("tool registered: %s", name)
        return fn

    return decorator


def signature_hint(name: str) -> dict | None:
    """Retorna los ``parameters`` JSON-schema de una tool para hint de error."""
    for schema in TOOL_SCHEMAS:
        if schema.get("function", {}).get("name") == name:
            return schema["function"].get("parameters")
    return None


async def dispatch(
    state: AgentState,
    name: str,
    arguments: dict[str, Any],
) -> tuple[dict, int]:
    """Ejecuta la tool y retorna ``(resultado, duración_ms)``.

    Si la tool falla, devuelve un dict de error con el mensaje + signature
    hint, que el LLM puede leer y usar para corregir su próxima llamada.
    """
    if name not in TOOL_REGISTRY:
        return {
            "error": f"Tool no disponible: {name!r}",
            "available_tools": list(TOOL_REGISTRY.keys()),
        }, 0

    fn = TOOL_REGISTRY[name]
    norm_args, notes = _normalize_arguments(fn, arguments or {})
    if notes:
        log.info("dispatch %s: argumentos normalizados (%s)", name, ", ".join(notes))
    t0 = time.perf_counter()
    try:
        result = await fn(state=state, **norm_args)
    except TypeError as e:
        return {
            "error": f"Argumentos inválidos para {name!r}: {e}",
            "expected_signature": signature_hint(name),
        }, int((time.perf_counter() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        log.exception("Tool %s failed", name)
        return {
            "error": f"Error ejecutando {name!r}: {e}",
        }, int((time.perf_counter() - t0) * 1000)

    return result, int((time.perf_counter() - t0) * 1000)


def reset_registry() -> None:
    """Limpia el registro — usado en tests para aislar estados.

    No afecta a tools registrados via ``@register`` después de llamar esto;
    los módulos de tools deben re-importarse.
    """
    TOOL_REGISTRY.clear()
    TOOL_SCHEMAS.clear()
