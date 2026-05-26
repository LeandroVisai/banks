"""Tool ``execute_query`` — ejecuta una query del catálogo SQL sobre parquets.

Usa DuckDB para consultar los parquets del DW. El agente llama esta tool
después de ``discover_query`` cuando ya sabe el ``query_id`` correcto.

Limita la salida a ``max_rows`` para no saturar el contexto del LLM.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from banks_rag.config.paths import ROOT
from banks_rag.infrastructure.sql.catalog_loader import (
    get_entry,
    load_catalog,
    render_sql,
)
from banks_rag.infrastructure.sql.duckdb_runner import MAX_ROWS as _MAX_ROWS
from banks_rag.infrastructure.sql.duckdb_runner import last_date_in_rows as _last_date
from banks_rag.infrastructure.sql.duckdb_runner import run_duckdb as _run_duckdb

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState

_SNAPSHOTS_DIR = ROOT / "data_pipeline" / "snapshots"
# _MAX_ROWS: techo duro de filas, compartido con duckdb_runner; la tool
# advierte al modelo si el resultado se truncó.

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_query",
        "description": (
            "Ejecuta una query analítica del catálogo SQL sobre los parquets "
            "del Data Warehouse y retorna los datos en formato tabular. "
            "Úsala tras discover_query cuando ya tienes el query_id. "
            "Los datos incluyen series financieras: tipo de cambio, tasas, "
            "bonos, liquidez bancaria, commodities, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_id": {
                    "type": "string",
                    "description": (
                        "Identificador de la query en el catálogo "
                        "(obtenido con discover_query)."
                    ),
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": "Fecha de inicio en formato ISO YYYY-MM-DD.",
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "Fecha de fin en formato ISO YYYY-MM-DD (default: hoy).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Máximo de filas a retornar (default del catálogo, máx 500).",
                    "minimum": 1,
                    "maximum": 500,
                },
            },
            "required": ["query_id"],
        },
    },
}


@register("execute_query", _SCHEMA)
async def execute_query(
    state: AgentState,
    query_id: str,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    entries = load_catalog()
    entry = get_entry(entries, query_id)
    if entry is None:
        available = [e.query_id for e in entries]
        return {
            "error": f"query_id desconocido: {query_id!r}.",
            "available_query_ids": available,
        }

    # Construir params respetando lo que pasó el agente
    params: dict[str, Any] = {}
    if fecha_inicio:
        params["fecha_inicio"] = fecha_inicio
    if fecha_fin:
        params["fecha_fin"] = fecha_fin
    if limit is not None:
        params["limit"] = min(int(limit), _MAX_ROWS)

    sql = render_sql(entry, _SNAPSHOTS_DIR, params)

    try:
        rows = await asyncio.to_thread(_run_duckdb, sql)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Error ejecutando {query_id!r}: {exc}",
            "query_id": query_id,
        }

    truncated = len(rows) >= _MAX_ROWS
    last_date = _last_date(rows)
    return {
        "query_id": query_id,
        "name": entry.name,
        "unit": entry.unit,
        "frequency": entry.frequency,
        "segment": entry.segment,
        "columns": entry.columns,
        "rows": rows,
        "n_rows": len(rows),
        "last_date_in_data": last_date,
        "data_currency_warning": (
            f"El último dato disponible es del {last_date}. "
            "NO asumas que este dato es de hoy; los datos pueden tener rezago."
            if last_date else None
        ),
        "truncated": truncated,
        "truncated_note": (
            f"Resultados truncados a {_MAX_ROWS} filas. "
            "Usa fecha_inicio/fecha_fin más acotados para obtener menos filas."
            if truncated else None
        ),
    }


# _run_duckdb se importa de infrastructure.sql.duckdb_runner (helper compartido
# con las analytics tools). Se re-exporta con nombre privado por compatibilidad.
