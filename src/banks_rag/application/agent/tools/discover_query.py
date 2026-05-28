"""Tool ``discover_query`` — busca datasets del parquet_catalog para responder
una pregunta sobre series financieras o macroeconómicas.

El agente la usa para encontrar qué *dataset* del catálogo tiene los datos
que necesita (USD/CLP, curva BTP, LCR, precio del cobre, etc.) antes de
llamar a ``execute_query`` o a alguna de las analytics tools.

Estrategia de scoring: BM25-like delegado en
``parquet_catalog_loader.search_datasets`` (solapamiento de tokens entre la
consulta del usuario y los campos ``id+name+description+segment+unit`` de
cada dataset)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    load_parquet_catalog,
    search_datasets,
)

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState


_SCHEMA = {
    "type": "function",
    "function": {
        "name": "discover_query",
        "description": (
            "Descubre qué datasets del catálogo de parquets responden a una "
            "pregunta sobre datos financieros o macroeconómicos (tipo de "
            "cambio, tasas, bonos, liquidez bancaria, commodities, balance "
            "del sistema, etc.). Retorna los top-k datasets con su id, "
            "esquema de columnas, unidad y rango de fechas disponibles. "
            "Úsala ANTES de execute_query (para traer las filas) o de las "
            "analytics tools (compute_variation, compute_spread, "
            "get_series_stats, detect_anomaly), todas las cuales reciben el "
            "`id` del dataset que esta tool devuelve."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Pregunta o descripción en lenguaje natural de los "
                        "datos que necesitas (ej. 'curva de bonos BTP en "
                        "pesos', 'tipo de cambio dólar último mes', "
                        "'activos del banco BCI en pesos')."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de datasets a retornar (default 5, máx 20).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "segment": {
                    "type": "string",
                    "description": (
                        "Filtrar por segmento del catálogo (ej. "
                        "'mercado_cambiario', 'balance_bancario', "
                        "'renta_fija_chile', 'commodities', "
                        "'liquidez_bancaria', 'politica_monetaria')."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


@register("discover_query", _SCHEMA)
async def discover_query(
    state: AgentState,
    query: str,
    top_k: int = 5,
    segment: str | None = None,
) -> dict[str, Any]:
    entries = load_parquet_catalog()

    if segment:
        in_segment = [e for e in entries if e.segment == segment]
        if not in_segment:
            return {
                "error": f"Segmento {segment!r} sin datasets en el catálogo.",
                "segments_disponibles": sorted({e.segment for e in entries}),
            }
        entries = in_segment

    top_k = max(1, min(int(top_k), 20))
    top = search_datasets(entries, query, top_k=top_k)

    return {
        "results": [e.to_dict() for e in top],
        "n_results": len(top),
        "hint": (
            "Cada resultado tiene `id` (úsalo como `dataset_id`), `columns` "
            "(con tipos y valores de enum cuando aplica) y `date_range` "
            "(primera y última fecha del parquet). Para traer filas crudas "
            "usa execute_query(dataset_id, columns?, fecha_inicio?, "
            "fecha_fin?, filters?, limit?). Para análisis usa "
            "compute_variation / compute_spread / get_series_stats / "
            "detect_anomaly con (dataset_id, column)."
        ),
    }
