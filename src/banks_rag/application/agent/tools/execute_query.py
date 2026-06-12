"""Tool ``execute_query`` — fetcher parametrizado sobre parquet_catalog.

NO es SQL libre escrita por el LLM. El LLM pasa parámetros estructurados
(``dataset_id``, columnas opcionales, rango de fecha, filtros por igualdad/IN
contra columnas enum) y la tool construye la SQL safe internamente vía
``_parquet_query.build_fetch_sql``: identificadores comillados, valores
escapados, ``LIMIT`` acotado por ``MAX_ROWS``.

Después de ejecutar registra cada serie consultada en ``state.add_series``
para que el panel de "series utilizadas" del frontend muestre el dato.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from banks_rag.infrastructure.sql.duckdb_runner import MAX_ROWS as _MAX_ROWS
from banks_rag.infrastructure.sql.duckdb_runner import last_date_in_rows as _last_date

from ._parquet_query import DEFAULT_LIMIT, fetch_rows_from_dataset
from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState


_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_query",
        "description": (
            "Trae filas de un dataset del catálogo de parquets (tipo de "
            "cambio, tasas, bonos, liquidez, commodities, balance bancario, "
            "etc.). Úsala tras discover_query cuando ya tienes el "
            "`dataset_id`. La SQL la arma la tool — tú solo eliges qué "
            "columnas y filtros. Para análisis (variación, spread, "
            "estadística, anomalía) usa las analytics tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": (
                        "ID del dataset (obtenido con discover_query)."
                    ),
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Columnas a traer (además de la fecha, que siempre "
                        "se incluye). Si omites, trae todas las del dataset. "
                        "Usa los nombres EXACTOS del esquema (case-sensitive)."
                    ),
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": "Fecha desde (ISO YYYY-MM-DD).",
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "Fecha hasta (ISO YYYY-MM-DD).",
                },
                "filters": {
                    "type": "object",
                    "description": (
                        "Filtros de igualdad por columna: "
                        "`{\"Banco\": \"BCI\"}` o `{\"Banco\": [\"BCI\", "
                        "\"Chile\"]}`. Las columnas deben existir en el "
                        "esquema; si la columna declara `values` (enum), el "
                        "valor pasado debe estar en esa lista."
                    ),
                    "additionalProperties": True,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Máximo de filas (default {DEFAULT_LIMIT}, máx {_MAX_ROWS}).",
                    "minimum": 1,
                    "maximum": _MAX_ROWS,
                },
            },
            "required": ["dataset_id"],
        },
    },
}


def _register_series(
    state: AgentState,
    dataset_name: str,
    dataset_unit: str,
    dataset_id: str,
    date_col: str,
    select_cols: list[str],
    rows: list[dict],
    chart_hint: str | None = None,
) -> None:
    """Registra una entrada en ``state.series_used`` por cada columna no-fecha
    devuelta. Mirror del patrón usado en analytics.py. ``chart_hint`` es el
    ``chart_type`` canónico del catálogo para el dataset: el frontend grafica
    la serie de forma consistente con el tablero (área apilada, barras, ...)."""
    if not rows:
        return
    for col in select_cols:
        if col == date_col:
            continue
        # date + value por fila → la serie queda graficable en la respuesta.
        col_rows = [{"date": r.get(date_col), "value": r.get(col)} for r in rows]
        state.add_series(
            f"{dataset_id}:{col}",
            {
                "series_name": f"{dataset_name} — {col}",
                "unit": dataset_unit,
            },
            col_rows,
            chart_hint=chart_hint,
        )


@register("execute_query", _SCHEMA)
async def execute_query(
    state: AgentState,
    dataset_id: str,
    columns: list[str] | None = None,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    try:
        dataset, rows, date_col, select_cols = await fetch_rows_from_dataset(
            dataset_id,
            columns=columns,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            filters=filters,
            limit=limit,
        )
    except ValueError as exc:
        return {"error": str(exc), "dataset_id": dataset_id}
    except FileNotFoundError as exc:
        return {"error": str(exc), "dataset_id": dataset_id}
    except Exception as exc:
        return {
            "error": f"Error ejecutando dataset {dataset_id!r}: {exc}",
            "dataset_id": dataset_id,
        }

    _register_series(
        state, dataset.name, dataset.unit, dataset.id, date_col, select_cols, rows,
        chart_hint=dataset.chart_type,
    )

    last_date = _last_date(rows)
    truncated = len(rows) >= _MAX_ROWS
    return {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "unit": dataset.unit,
        "segment": dataset.segment,
        "date_column": date_col,
        "columns": select_cols,
        "rows": rows,
        "n_rows": len(rows),
        "last_date_in_data": last_date,
        "data_currency_warning": (
            f"El último dato disponible es del {last_date}. "
            "NO asumas que este dato es de hoy; los parquets tienen rezago."
            if last_date else None
        ),
        "truncated": truncated,
        "truncated_note": (
            f"Resultados truncados a {_MAX_ROWS} filas. "
            "Usa fecha_inicio/fecha_fin o filters más acotados."
            if truncated else None
        ),
    }
