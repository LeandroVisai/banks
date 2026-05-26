"""Tool ``query_parquet`` — ejecuta DuckDB SQL libre sobre un parquet del catálogo.

El agente la usa después de ``list_parquets`` cuando ya conoce el parquet_id
y el schema. Escribe SQL DuckDB usando ``read_parquet('{parquet_path}')`` como
tabla virtual; el placeholder ``{parquet_path}`` se reemplaza automáticamente
con la ruta absoluta al archivo.

Seguridad:
  - El parquet_id se valida contra el catálogo (no permite path traversal).
  - El SQL se ejecuta en modo lectura (DuckDB en memoria).
  - Resultados truncados a MAX_ROWS filas.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from banks_rag.infrastructure.sql.duckdb_runner import MAX_ROWS as _MAX_ROWS
from banks_rag.infrastructure.sql.duckdb_runner import last_date_in_rows as _last_date
from banks_rag.infrastructure.sql.duckdb_runner import run_duckdb as _run_duckdb
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    get_dataset,
    get_parquet_dir,
    load_parquet_catalog,
)

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_parquet",
        "description": (
            "Ejecuta SQL DuckDB libre sobre un parquet del catálogo analítico. "
            "Usa 'read_parquet(\\'{parquet_path}\\')' como nombre de tabla en tu SQL; "
            "el placeholder {parquet_path} se reemplaza automáticamente con la ruta real. "
            "Úsala tras list_parquets cuando ya conoces el parquet_id y el schema. "
            "Ejemplo: SELECT Fecha, Banco, SUM(Monto) FROM read_parquet('{parquet_path}') "
            "WHERE Tipo_banco = 'Bancos Sistemicos' GROUP BY 1, 2 ORDER BY 1 DESC LIMIT 20"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "parquet_id": {
                    "type": "string",
                    "description": (
                        "ID del parquet en el catálogo (obtenido con list_parquets). "
                        "Ej: 'act_mn', 'btp_curva', 'flujo_cambiario', 'lcr'."
                    ),
                },
                "sql": {
                    "type": "string",
                    "description": (
                        "Query DuckDB válida. Usa read_parquet('{parquet_path}') como tabla. "
                        "Respeta los nombres exactos de columnas del schema (case-sensitive). "
                        "Columnas con espacios o acentos deben ir entre comillas dobles: "
                        "\"RT Exigible\", \"Monto transado\", \"1 a 90 dias\". "
                        "Siempre incluye ORDER BY y LIMIT para acotar el resultado."
                    ),
                },
            },
            "required": ["parquet_id", "sql"],
        },
    },
}


@register("query_parquet", _SCHEMA)
async def query_parquet(
    state: "AgentState",
    parquet_id: str,
    sql: str,
) -> dict[str, Any]:
    entries = load_parquet_catalog()
    dataset = get_dataset(entries, parquet_id)

    if dataset is None:
        available = [e.id for e in entries]
        return {
            "error": f"parquet_id desconocido: {parquet_id!r}. Usa list_parquets para encontrar el id correcto.",
            "available_ids": available[:30],
            "hint": "Llama list_parquets con una descripción de los datos que buscas.",
        }

    parquet_dir = get_parquet_dir()
    parquet_path = dataset.parquet_path(parquet_dir)

    if not parquet_path.exists():
        return {
            "error": f"Archivo no encontrado en disco: {parquet_path}",
            "parquet_id": parquet_id,
        }

    resolved_sql = sql.replace("{parquet_path}", str(parquet_path))

    try:
        rows = await asyncio.to_thread(_run_duckdb, resolved_sql)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Error ejecutando query sobre {parquet_id!r}: {exc}",
            "parquet_id": parquet_id,
            "hint": (
                "Verifica nombres de columnas con comillas dobles si tienen espacios. "
                f"Columnas disponibles: {[c.name for c in dataset.columns]}"
            ),
        }

    truncated = len(rows) >= _MAX_ROWS
    last_date = _last_date(rows)
    return {
        "parquet_id": parquet_id,
        "name": dataset.name,
        "unit": dataset.unit,
        "segment": dataset.segment,
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
            f"Resultado truncado a {_MAX_ROWS} filas. "
            "Añade filtros de fecha o LIMIT más restrictivo."
            if truncated else None
        ),
    }
