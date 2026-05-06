"""
Tools para series de tiempo macroeconómicas — consultan el DW (SQL Server)
en tiempo real vía data_pipeline.dw_store.

El catálogo (~100 series) viene de data_pipeline/series_catalog.yaml.
El agente puede explorar por categoría o variable de la taxonomía, y
luego traer los datos numéricos de cualquier serie con get_historical_series.

Tools expuestas:
  - list_historical_series: descubre qué series existen, con filtros
  - get_historical_series:  trae los datos de una serie desde el DW
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Optional

from data_pipeline import dw_store

from .registry import register
from ..settings import settings

if TYPE_CHECKING:
    from ..agent import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# list_historical_series
# ─────────────────────────────────────────────────────────────────────────────

LIST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_historical_series",
        "description": (
            "Lista las series de tiempo macroeconómicas disponibles en el "
            "Data Warehouse. Útil para descubrir qué series existen antes de "
            "consultarlas con get_historical_series. El catálogo se organiza "
            "por categorías (tasas_clp, bonos_clp, fx, commodities, etc.) y "
            "cada serie tiene unidad, frecuencia, fuente y tabla de origen en el DW."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Filtrar por categoría. Categorías disponibles: "
                        "tasas_clp, curva_spc_clp, curva_spc_uf, bonos_clp, "
                        "bonos_uf, ust, ois_sofr, expectativas_tpm, fx, "
                        "commodities, forwards_clp, tasas_usd_mn, liquidez_mx, "
                        "microestructura."
                    ),
                },
                "variable": {
                    "type": "string",
                    "description": (
                        "Filtrar por variable de la taxonomía: TASA_INTERES, "
                        "TASAS_LARGO_PLAZO, TASA_INTERES_MERCADO, TIPO_CAMBIO, "
                        "COMMODITIES, EXPECTATIVAS_INFLACIONARIAS."
                    ),
                },
                "tenor": {
                    "type": "string",
                    "description": (
                        "Filtrar por tenor de la curva (ej. '5Y', '10Y', '3M'). "
                        "Útil cuando buscas un punto específico de la curva."
                    ),
                },
            },
        },
    },
}


@register("list_historical_series", LIST_SCHEMA)
async def list_historical_series(
    state: "AgentState",
    category: Optional[str] = None,
    variable: Optional[str] = None,
    tenor: Optional[str] = None,
) -> dict[str, Any]:
    rows = dw_store.list_series(
        category=category,
        variable=variable,
        tenor=tenor,
        catalog_path=settings.catalog_path,
    )
    categories_available = None
    if not category:
        categories_available = [
            c["category"] for c in dw_store.list_categories(
                catalog_path=settings.catalog_path,
            )
        ]
    return {
        "series": rows,
        "n_series": len(rows),
        "categories_available": categories_available,
        "note": (
            "El rango de fechas disponible se muestra al llamar get_historical_series. "
            "No se incluye cobertura en el listado para mayor velocidad."
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
            "del Data Warehouse (SQL Server). Úsalo cuando necesites cifras "
            "concretas (niveles, variaciones, evolución histórica). "
            "Si no especificas date_from/date_to, retorna las observaciones "
            "más recientes según la frecuencia de la serie."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "series_id": {
                    "type": "string",
                    "description": (
                        "ID de la serie (ej. 'btp_5y', 'usdclp', 'cobre', "
                        "'spc_10y_clp'). Usa list_historical_series para "
                        "descubrirlos."
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
                    "description": "Máximo de observaciones si no hay rango (default 45).",
                    "default": 45,
                    "minimum": 1,
                    "maximum": 500,
                },
            },
            "required": ["series_id"],
        },
    },
}


def _parse_date(s: Optional[str]) -> Optional[date]:
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
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 45,
) -> dict[str, Any]:
    meta = dw_store.get_series_meta(series_id, catalog_path=settings.catalog_path)
    if meta is None:
        return {
            "error": (
                f"series_id desconocido: {series_id!r}. "
                f"Usa list_historical_series para ver el catálogo."
            ),
        }

    if not settings.get_data_path:
        return {"error": "GET_DATA_PATH no configurado en .env — DW no disponible"}

    df_from = _parse_date(date_from)
    df_to = _parse_date(date_to)
    use_limit = limit if not (df_from and df_to) else None

    try:
        df = await dw_store.fetch_series(
            series_id,
            get_data_path=settings.get_data_path,
            catalog_path=settings.catalog_path,
            date_from=df_from,
            date_to=df_to,
            limit=use_limit,
        )
    except Exception as e:
        return {"error": f"Error consultando DW para {series_id!r}: {e}"}

    if df.empty:
        return {
            "series_id": series_id,
            "name": meta["name"],
            "observations": [],
            "n": 0,
            "message": (
                "Sin datos para el rango solicitado. La serie existe en el "
                "catálogo pero no se encontraron observaciones en el DW."
            ),
        }

    # Registrar en el state para persistir en agent_messages.historical_series
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
