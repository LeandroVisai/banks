"""Catalog endpoints: exponen el ``parquet_catalog`` vía REST para alimentar
gráficos del frontend sin pasar por el agente (latencia baja, sin coste LLM).

Reusa exactamente el mismo helper que ``execute_query``
(``application/agent/tools/_parquet_query.fetch_rows_from_dataset``) — la SQL
la arma siempre la capa de infra a partir de parámetros validados; el cliente
HTTP no aporta SQL.

Endpoints:
  - ``GET /v1/catalog``              → lista los datasets del parquet_catalog
  - ``GET /v1/query/{dataset_id}``   → fetch parametrizado, retorna filas
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from banks_rag.application.agent.tools._parquet_query import (
    fetch_rows_from_dataset,
)
from banks_rag.infrastructure.sql.duckdb_runner import MAX_ROWS as _MAX_ROWS
from banks_rag.infrastructure.sql.duckdb_runner import last_date_in_rows as _last_date
from banks_rag.infrastructure.sql.parquet_catalog_loader import load_parquet_catalog
from banks_rag.interface.api.schemas import (
    DatasetColumn,
    DatasetEntry,
    DatasetListResponse,
    QueryResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["catalog"])


@router.get("/catalog", response_model=DatasetListResponse)
async def list_catalog() -> DatasetListResponse:
    """Lista los datasets del parquet_catalog con esquema para el sidebar."""
    entries = await asyncio.to_thread(load_parquet_catalog)
    items = [
        DatasetEntry(
            id=e.id,
            file=e.file,
            name=e.name,
            description=e.description.strip(),
            segment=e.segment,
            unit=e.unit,
            date_range=e.date_range,
            columns=[
                DatasetColumn(name=c.name, type=c.type, values=c.values)
                for c in e.columns
            ],
        )
        for e in entries
    ]
    return DatasetListResponse(n_entries=len(items), entries=items)


@router.get("/query/{dataset_id}", response_model=QueryResponse)
async def execute_catalog_query(
    dataset_id: str,
    fecha_inicio: str | None = Query(default=None, description="ISO YYYY-MM-DD"),
    fecha_fin: str | None = Query(default=None, description="ISO YYYY-MM-DD"),
    limit: int | None = Query(default=None, ge=1, le=_MAX_ROWS),
) -> QueryResponse:
    """Fetch parametrizado del dataset. La SQL la arma el helper safe."""
    try:
        dataset, rows, date_col, select_cols = await fetch_rows_from_dataset(
            dataset_id,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Error ejecutando dataset %s", dataset_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error ejecutando dataset {dataset_id!r}: {exc}",
        ) from exc

    truncated = len(rows) >= _MAX_ROWS
    return QueryResponse(
        dataset_id=dataset.id,
        name=dataset.name,
        unit=dataset.unit,
        segment=dataset.segment,
        date_column=date_col,
        columns=select_cols,
        rows=rows,
        n_rows=len(rows),
        last_date_in_data=_last_date(rows),
        truncated=truncated,
        truncated_note=(
            f"Resultados truncados a {_MAX_ROWS} filas. "
            "Usa fecha_inicio/fecha_fin más acotados."
            if truncated else None
        ),
    )
