"""Ejecución de SQL DuckDB sobre los parquets de snapshots.

Helper compartido por ``execute_query`` y las analytics tools. Abre una
conexión ``:memory:`` por llamada — las llamadas concurrentes no comparten
el cursor del singleton de ``duckdb.sql()`` — y serializa los tipos que no
son JSON-nativos (date/datetime → ISO, float → redondeado).

Síncrono: las tools lo invocan vía ``asyncio.to_thread`` para no bloquear
el event loop.
"""

from __future__ import annotations

from typing import Any

# Techo duro de filas para una query del catálogo: acota el tamaño del
# resultado tanto en el endpoint /v1/query como en la tool execute_query.
MAX_ROWS = 500


def run_duckdb(sql: str) -> list[dict]:
    """Ejecuta ``sql`` con DuckDB y retorna las filas como lista de dicts."""
    import duckdb

    with duckdb.connect(":memory:") as con:
        rel = con.sql(sql)
        columns = [d[0] for d in rel.description]
        return [
            {col: serialize_value(val) for col, val in zip(columns, row, strict=True)}
            for row in rel.fetchall()
        ]


def serialize_value(val: Any) -> Any:
    """Convierte un valor de DuckDB a un tipo JSON-serializable."""
    if val is None:
        return None
    if isinstance(val, float):
        return round(val, 6)
    iso = getattr(val, "isoformat", None)  # date / datetime
    if callable(iso):
        return iso()
    return val
