"""Tool ``list_parquets`` — descubre datasets analíticos por keyword.

El agente la usa para encontrar qué parquet contiene los datos que necesita
antes de llamar a ``query_parquet`` con SQL libre. Devuelve el schema completo
(columnas, tipos, valores categóricos, rango de fechas) para que el agente
pueda escribir la query correcta sin adivinar nombres de columnas.
"""

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
        "name": "list_parquets",
        "description": (
            "Busca en el catálogo de datasets analíticos (parquets) cuál contiene "
            "los datos que necesitas. Retorna el schema completo (columnas, tipos, "
            "valores posibles de filtros categóricos, rango de fechas) para que "
            "puedas escribir el SQL correcto en query_parquet. "
            "Úsala ANTES de query_parquet para descubrir el parquet_id y el schema. "
            "Cubre ~100 datasets: balance bancario, BTP/BTU/SPC/OIS, flujos "
            "cambiarios, posiciones de AFP/CSV/NR, PDBC, LCR/NSFR, spreads, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Descripción en lenguaje natural de los datos que necesitas. "
                        "Ej: 'posición de AFP en swaps SPC', 'flujo cambiario por sector', "
                        "'curva BTP plazos', 'LCR sistémico bancos', 'spread DAP SOFR'."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de datasets a retornar (default 5, máx 15).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 15,
                },
                "segment": {
                    "type": "string",
                    "description": (
                        "Filtrar por segmento: balance_bancario, renta_fija_chile, "
                        "mercado_cambiario, posiciones_cambiarias, spreads_credito, "
                        "liquidez_bancaria, fondos_pension, posiciones_rfl, "
                        "instrumentos_bcch, expectativas, otros."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


@register("list_parquets", _SCHEMA)
async def list_parquets(
    state: "AgentState",
    query: str,
    top_k: int = 5,
    segment: str | None = None,
) -> dict[str, Any]:
    entries = load_parquet_catalog()

    if segment:
        available_segments = sorted({e.segment for e in entries})
        filtered = [e for e in entries if e.segment == segment]
        if not filtered:
            return {
                "error": f"Segmento {segment!r} no existe en el catálogo.",
                "segments_disponibles": available_segments,
            }
        entries = filtered

    results = search_datasets(entries, query, top_k=min(top_k, 15))

    return {
        "results": [e.to_dict() for e in results],
        "n_results": len(results),
        "hint": (
            "Elige el parquet_id del dataset más relevante y usa query_parquet "
            "con SQL DuckDB sobre read_parquet('{parquet_path}'). "
            "El path exacto del archivo te lo provee query_parquet automáticamente "
            "si usas el placeholder {parquet_path} en tu SQL."
        ),
    }
