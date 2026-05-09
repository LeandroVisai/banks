"""Tools para series de tiempo macro: list_historical_series + get_historical_series.

Delegan a ``data_pipeline.dw_store`` (módulo legacy del proyecto). Si el DW
no está disponible (env ``GET_DATA_PATH`` no configurado), las tools devuelven
errores estructurados que el agente puede mostrar al usuario.

En sub-fase posterior (post Fase 4 SQL catalog), estas tools migrarán al nuevo
``infrastructure/sql/catalog_loader`` para Text-to-SQL agentic con catálogo
curado de las 75 queries de ``querys/Monitor.py``.
"""

from __future__ import annotations

import asyncio
import os
from datetime import date
from typing import TYPE_CHECKING, Any

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# list_historical_series
# ─────────────────────────────────────────────────────────────────────────────

LIST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_historical_series",
        "description": (
            "Lista las series de tiempo macroeconómicas disponibles en el "
            "Data Warehouse. Útil para descubrir qué series existen antes "
            "de consultarlas con get_historical_series. El catálogo se "
            "organiza por categorías (tasas_clp, bonos_clp, fx, commodities, "
            "etc.) y cada serie tiene unidad, frecuencia y fuente."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Filtrar por categoría: tasas_clp, curva_spc_clp, "
                        "curva_spc_uf, bonos_clp, bonos_uf, ust, ois_sofr, "
                        "expectativas_tpm, fx, commodities, forwards_clp, "
                        "tasas_usd_mn, liquidez_mx, microestructura."
                    ),
                },
                "variable": {
                    "type": "string",
                    "description": (
                        "Filtrar por variable de la taxonomía: TASA_INTERES, "
                        "TASAS_LARGO_PLAZO, TIPO_CAMBIO, COMMODITIES, etc."
                    ),
                },
                "tenor": {
                    "type": "string",
                    "description": "Filtrar por tenor (ej. '5Y', '10Y', '3M').",
                },
            },
        },
    },
}


def _import_dw_store():
    """Lazy import. Si el módulo no existe, retorna None."""
    try:
        from data_pipeline import dw_store
        return dw_store
    except ImportError:
        return None


def _catalog_path() -> str | None:
    """Path al catálogo YAML. Configurable via env ``BANKS_CATALOG_PATH``."""
    return os.getenv("BANKS_CATALOG_PATH")


def _get_data_path() -> str | None:
    """Path al módulo Get_Data del DW. Configurable via env ``GET_DATA_PATH``."""
    return os.getenv("GET_DATA_PATH")


@register("list_historical_series", LIST_SCHEMA)
async def list_historical_series(
    state: "AgentState",
    category: str | None = None,
    variable: str | None = None,
    tenor: str | None = None,
) -> dict[str, Any]:
    dw_store = _import_dw_store()
    if dw_store is None:
        return {"error": "data_pipeline.dw_store no disponible en este entorno."}

    catalog_path = _catalog_path()
    rows = await asyncio.to_thread(
        dw_store.list_series,
        category=category, variable=variable, tenor=tenor,
        catalog_path=catalog_path,
    )

    categories_available = None
    if not category:
        cats = await asyncio.to_thread(
            dw_store.list_categories, catalog_path=catalog_path,
        )
        categories_available = [c["category"] for c in cats]

    return {
        "series": rows,
        "n_series": len(rows),
        "categories_available": categories_available,
        "note": (
            "El rango de fechas disponible se muestra al llamar "
            "get_historical_series."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# get_historical_series
# ─────────────────────────────────────────────────────────────────────────────

GET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_historical_series",
        "description": (
            "Trae los datos numéricos de una serie de tiempo directamente "
            "del Data Warehouse. Úsalo cuando necesites cifras concretas "
            "(niveles, variaciones, evolución histórica). Si no especificas "
            "date_from/date_to, retorna las observaciones más recientes "
            "según la frecuencia de la serie."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "series_id": {
                    "type": "string",
                    "description": (
                        "ID de la serie (ej. 'btp_5y', 'usdclp', 'cobre'). "
                        "Usa list_historical_series para descubrir IDs."
                    ),
                },
                "date_from": {
                    "type": "string",
                    "description": "Fecha desde (ISO YYYY-MM-DD).",
                },
                "date_to": {
                    "type": "string",
                    "description": "Fecha hasta (ISO YYYY-MM-DD).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Máximo de obs si no hay rango (default 45).",
                    "default": 45,
                    "minimum": 1,
                    "maximum": 500,
                },
            },
            "required": ["series_id"],
        },
    },
}


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


@register("get_historical_series", GET_SCHEMA)
async def get_historical_series(
    state: "AgentState",
    series_id: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 45,
) -> dict[str, Any]:
    dw_store = _import_dw_store()
    if dw_store is None:
        return {"error": "data_pipeline.dw_store no disponible en este entorno."}

    catalog_path = _catalog_path()
    get_data_path = _get_data_path()
    if not get_data_path:
        return {"error": "GET_DATA_PATH no configurado — DW no disponible."}

    meta = await asyncio.to_thread(
        dw_store.get_series_meta, series_id, catalog_path=catalog_path,
    )
    if meta is None:
        return {
            "error": (
                f"series_id desconocido: {series_id!r}. "
                f"Usa list_historical_series para ver el catálogo."
            ),
        }

    df_from = _parse_date(date_from)
    df_to = _parse_date(date_to)
    use_limit = int(limit) if not (df_from and df_to) else None

    try:
        df = await dw_store.fetch_series(
            series_id,
            get_data_path=get_data_path,
            catalog_path=catalog_path,
            date_from=df_from,
            date_to=df_to,
            limit=use_limit,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"Error consultando DW para {series_id!r}: {e}"}

    if df.empty:
        return {
            "series_id": series_id,
            "name": meta["name"],
            "observations": [],
            "n": 0,
            "message": "Sin datos para el rango solicitado.",
        }

    state.add_series(
        series_id,
        {
            "series_name": meta["name"],
            "unit": meta.get("unit", ""),
            "frequency": meta.get("frequency", ""),
        },
        [{"date": d} for d in df["date"]],
    )

    observations = [
        {
            "date": row["date"].date().isoformat(),
            "value_str": dw_store.format_value(row["value"], meta.get("unit", "")),
            "value": float(row["value"]) if row["value"] is not None else None,
        }
        for _, row in df.iterrows()
    ]

    return {
        "series_id": series_id,
        "name": meta["name"],
        "category": meta.get("category"),
        "unit": meta.get("unit"),
        "frequency": meta.get("frequency"),
        "source": meta.get("source"),
        "tenor": meta.get("tenor"),
        "observations": observations,
        "n": len(observations),
        "first_date": observations[0]["date"],
        "last_date": observations[-1]["date"],
    }
