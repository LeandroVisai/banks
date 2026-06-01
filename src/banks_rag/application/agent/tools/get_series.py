"""Tool ``get_series`` — atajo de UN paso: descubre el dataset (si no se entrega)
y trae sus filas en una sola llamada.

Colapsa ``discover_query`` + ``execute_query`` para reducir el encadenamiento:
los modelos a veces se quedan llamando ``discover_query`` sin avanzar a
``execute_query``. ``get_series`` hace ambos pasos internamente (reusando esas
tools, incluida la resolución difusa de ids/columnas) y devuelve directamente
las filas, más qué dataset eligió y qué alternativas había (para que el modelo
corrija si la elección no fue la correcta).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .discover_query import discover_query
from .execute_query import execute_query
from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState


_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_series",
        "description": (
            "Trae los datos de una serie del catálogo de parquets en UN solo "
            "paso: descubre el dataset y devuelve sus filas. Es la forma "
            "PREFERIDA de obtener datos — usa esto en vez de encadenar "
            "discover_query + execute_query. Da `query` (descripción en lenguaje "
            "natural, p.ej. 'spread BTP vs SPC a 10 años') y, si lo conoces, "
            "`dataset_id`. Devuelve las filas, el dataset elegido y las "
            "alternativas por si la elección no es la correcta."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Descripción en lenguaje natural de la serie que "
                        "necesitas. Requerida si no entregas dataset_id."
                    ),
                },
                "dataset_id": {
                    "type": "string",
                    "description": "ID del dataset si ya lo conoces (omite la búsqueda).",
                },
                "segment": {
                    "type": "string",
                    "description": "Segmento del catálogo para acotar la búsqueda (opcional).",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columnas a traer (opcional; por defecto todas).",
                },
                "fecha_inicio": {"type": "string", "description": "Fecha desde (ISO YYYY-MM-DD)."},
                "fecha_fin": {"type": "string", "description": "Fecha hasta (ISO YYYY-MM-DD)."},
                "limit": {"type": "integer", "description": "Máximo de filas.", "minimum": 1},
            },
            "required": [],
        },
    },
}


@register("get_series", _SCHEMA)
async def get_series(
    state: "AgentState",
    query: str | None = None,
    dataset_id: str | None = None,
    segment: str | None = None,
    columns: list[str] | None = None,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    chosen = (dataset_id or "").strip() or None
    discovery: dict[str, Any] | None = None

    if chosen is None:
        if not (query or "").strip():
            return {
                "error": (
                    "Debes entregar `query` (descripción de la serie) o "
                    "`dataset_id`."
                ),
            }
        disc = await discover_query(state, query=query, segment=segment, top_k=5)
        results = disc.get("results") or []
        if not results:
            return {
                "error": "No se encontró un dataset para esa descripción.",
                "hint": disc.get("message"),
            }
        chosen = results[0]["id"]
        discovery = {
            "elegido": chosen,
            "alternativas": [r["id"] for r in results[1:]],
            "nota": (
                "Se eligió el dataset de mayor relevancia. Si no es el correcto, "
                "vuelve a llamar get_series con `dataset_id` de las alternativas."
            ),
        }

    result = await execute_query(
        state,
        dataset_id=chosen,
        columns=columns,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        limit=limit,
    )
    if discovery and isinstance(result, dict):
        result["discovery"] = discovery
    return result
