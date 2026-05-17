"""Catalog endpoints: exponen el SQL catalog vía REST para alimentar gráficos.

Reutiliza ``catalog_loader`` + ``duckdb_runner`` para evitar duplicar lógica.
El frontend ``interface2/`` consume estos endpoints sin pasar por el agente
(latencia baja, sin coste de LLM).

Endpoints:
  - ``GET /v1/catalog``           → lista las 23 queries del catálogo
  - ``GET /v1/query/{query_id}``  → ejecuta una query y retorna filas
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from banks_rag.config.paths import ROOT
from banks_rag.infrastructure.sql.catalog_loader import (
    get_entry,
    load_catalog,
    render_sql,
)
from banks_rag.infrastructure.sql.duckdb_runner import MAX_ROWS as _MAX_ROWS
from banks_rag.infrastructure.sql.duckdb_runner import run_duckdb as _run_duckdb
from banks_rag.interface.api.schemas import (
    CatalogEntry,
    CatalogListResponse,
    CatalogParamSpec,
    QueryResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["catalog"])

_SNAPSHOTS_DIR = ROOT / "data_pipeline" / "snapshots"


@router.get("/catalog", response_model=CatalogListResponse)
async def list_catalog() -> CatalogListResponse:
    """Lista las queries del catálogo con metadata para alimentar el sidebar."""
    entries = await asyncio.to_thread(load_catalog)
    items = [
        CatalogEntry(
            query_id=e.query_id,
            name=e.name,
            description=e.description.strip(),
            segment=e.segment,
            tags=e.tags,
            unit=e.unit,
            frequency=e.frequency,
            columns=e.columns,
            params=[
                CatalogParamSpec(
                    name=p.name,
                    type=p.type,
                    default=p.default,
                    description=p.description,
                )
                for p in e.params
            ],
        )
        for e in entries
    ]
    return CatalogListResponse(n_entries=len(items), entries=items)


@router.get("/query/{query_id}", response_model=QueryResponse)
async def execute_catalog_query(
    query_id: str,
    fecha_inicio: str | None = Query(default=None, description="ISO YYYY-MM-DD"),
    fecha_fin: str | None = Query(default=None, description="ISO YYYY-MM-DD"),
    limit: int | None = Query(default=None, ge=1, le=_MAX_ROWS),
) -> QueryResponse:
    """Ejecuta una query del catálogo y retorna filas para alimentar gráficos."""
    entries = await asyncio.to_thread(load_catalog)
    entry = get_entry(entries, query_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"query_id desconocido: {query_id!r}",
        )

    params: dict = {}
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
        log.exception("Error ejecutando %s", query_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error ejecutando {query_id!r}: {exc}",
        ) from exc

    truncated = len(rows) >= _MAX_ROWS
    return QueryResponse(
        query_id=query_id,
        name=entry.name,
        unit=entry.unit,
        frequency=entry.frequency,
        segment=entry.segment,
        columns=entry.columns,
        rows=rows,
        n_rows=len(rows),
        truncated=truncated,
        truncated_note=(
            f"Resultados truncados a {_MAX_ROWS} filas. "
            "Usa fecha_inicio/fecha_fin más acotados."
            if truncated else None
        ),
    )
