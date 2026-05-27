"""Analytics tools — el agente no solo trae datos crudos, los interpreta.

Estas tools se apoyan en el catálogo SQL (``discover_query`` / ``execute_query``
descubren el ``query_id`` y las columnas) y le agregan una capa de cálculo:
variaciones, spreads, estadística descriptiva, detección de anomalías y una
foto consolidada del mercado.

Pensadas para el ``QuantitativeAnalyst`` y el ``MarketAnalyst`` de la
arquitectura multi-agente (Fase B). En la Fase A las usa directamente el
agente actual.

El cálculo numérico vive en ``infrastructure/sql/series_analytics`` (funciones
puras); aquí solo se orquesta DuckDB + catálogo + serialización.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import TYPE_CHECKING, Any

from banks_rag.config.paths import ROOT
from banks_rag.infrastructure.sql.catalog_loader import (
    CatalogEntry,
    get_entry,
    load_catalog,
    render_sql,
)
from banks_rag.infrastructure.sql.duckdb_runner import run_duckdb
from banks_rag.infrastructure.sql.series_analytics import (
    anomaly_check,
    clean_series,
    descriptive_stats,
    spread,
    variation,
)

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState

_SNAPSHOTS_DIR = ROOT / "data_pipeline" / "snapshots"
_DATE_COL = "fecha"          # toda query del catálogo aliasa la fecha como 'fecha'
_ANALYTICS_LIMIT = 500       # techo de filas: cubre ~2 años de data diaria


# ─────────────────────────────────────────────────────────────────────────────
# Helpers compartidos
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_rows(
    query_id: str,
    *,
    fecha_inicio: str,
    fecha_fin: str,
    limit: int = _ANALYTICS_LIMIT,
) -> tuple[tuple[CatalogEntry, list[dict]] | None, dict | None]:
    """Ejecuta una query del catálogo. Retorna ``((entry, rows), None)`` o ``(None, error)``."""
    entries = load_catalog()
    entry = get_entry(entries, query_id)
    if entry is None:
        return None, {
            "error": f"query_id desconocido: {query_id!r}.",
            "available_query_ids": [e.query_id for e in entries],
        }

    sql = render_sql(
        entry,
        _SNAPSHOTS_DIR,
        {"fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin, "limit": limit},
    )
    try:
        rows = await asyncio.to_thread(run_duckdb, sql)
    except Exception as exc:  # noqa: BLE001
        return None, {"error": f"Error ejecutando {query_id!r}: {exc}"}
    return (entry, rows), None


def _extract(entry: CatalogEntry, rows: list[dict], column: str):
    """Valida la columna y limpia la serie. Retorna ``(series, None)`` o ``(None, error)``."""
    data_cols = [c for c in entry.columns if c != _DATE_COL]
    if column not in data_cols:
        return None, {
            "error": f"Columna {column!r} no existe en la query {entry.query_id!r}.",
            "available_columns": data_cols,
        }
    return clean_series(rows, _DATE_COL, column), None


def _register_series(
    state: AgentState,
    entry: CatalogEntry,
    column: str,
    series: list[tuple[str, float]],
) -> None:
    """Registra la serie consultada en el AgentState para trazabilidad."""
    state.add_series(
        f"{entry.query_id}:{column}",
        {"series_name": f"{entry.name} — {column}", "unit": entry.unit,
         "frequency": entry.frequency},
        [{"date": fecha} for fecha, _ in series],
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute_variation
# ─────────────────────────────────────────────────────────────────────────────

_VARIATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compute_variation",
        "description": (
            "Calcula cuánto se movió una serie del catálogo entre dos fechas: "
            "cambio absoluto, variación porcentual, cambio en puntos base y el "
            "rango (mínimo/máximo) del período. Úsala en vez de leer datos "
            "crudos cuando necesites CUANTIFICAR un movimiento. Requiere el "
            "query_id (de discover_query) y la columna (de execute_query)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_id": {
                    "type": "string",
                    "description": "Identificador de la query del catálogo SQL.",
                },
                "column": {
                    "type": "string",
                    "description": (
                        "Columna de la serie a analizar (ej. 'usdclp', "
                        "'btp_10y', 'tib_tasa'). La devuelve execute_query."
                    ),
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": (
                        "Fecha de inicio. ISO YYYY-MM-DD o relativa como "
                        "'-90d'. Default: -90d (últimos 3 meses)."
                    ),
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "Fecha de fin. ISO YYYY-MM-DD u 'hoy'. Default: hoy.",
                },
            },
            "required": ["query_id", "column"],
        },
    },
}


@register("compute_variation", _VARIATION_SCHEMA)
async def compute_variation(
    state: AgentState,
    query_id: str,
    column: str,
    fecha_inicio: str = "-90d",
    fecha_fin: str = "hoy",
) -> dict[str, Any]:
    fetched, err = await _fetch_rows(query_id, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    if err:
        return err
    entry, rows = fetched
    series, err = _extract(entry, rows, column)
    if err:
        return err

    var = variation(series)
    if var is None:
        return {
            "error": (
                f"La serie {query_id}:{column} tiene menos de 2 observaciones "
                f"en el período {fecha_inicio} → {fecha_fin}; no se puede "
                "calcular variación. Amplía el rango de fechas."
            ),
        }

    _register_series(state, entry, column, series)
    return {
        "query_id": query_id,
        "name": entry.name,
        "column": column,
        "unit": entry.unit,
        "frequency": entry.frequency,
        "nota_bps": (
            "cambio_bps es válido solo si la unidad es una tasa/porcentaje."
        ),
        **var,
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_spread
# ─────────────────────────────────────────────────────────────────────────────

_SPREAD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compute_spread",
        "description": (
            "Calcula el spread (diferencia A - B) entre dos series del "
            "catálogo, alineadas por fecha. Sirve para break-even de inflación "
            "(BTP - BTU), spread TIB - tasa de referencia, pendiente de curva "
            "(10A - 2A), etc. Si ambas columnas están en la misma query, "
            "repite el query_id. Retorna el spread actual y su evolución."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_id_a": {"type": "string", "description": "query_id de la serie A."},
                "column_a": {"type": "string", "description": "Columna de la serie A (minuendo)."},
                "query_id_b": {"type": "string", "description": "query_id de la serie B."},
                "column_b": {"type": "string", "description": "Columna de la serie B (sustraendo)."},
                "fecha_inicio": {
                    "type": "string",
                    "description": "ISO YYYY-MM-DD o relativa ('-90d'). Default: -90d.",
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "ISO YYYY-MM-DD u 'hoy'. Default: hoy.",
                },
            },
            "required": ["query_id_a", "column_a", "query_id_b", "column_b"],
        },
    },
}


@register("compute_spread", _SPREAD_SCHEMA)
async def compute_spread(
    state: AgentState,
    query_id_a: str,
    column_a: str,
    query_id_b: str,
    column_b: str,
    fecha_inicio: str = "-90d",
    fecha_fin: str = "hoy",
) -> dict[str, Any]:
    fetched_a, err = await _fetch_rows(query_id_a, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    if err:
        return err
    entry_a, rows_a = fetched_a

    # Si ambas columnas comparten query, se reutilizan las mismas filas.
    if query_id_b == query_id_a:
        entry_b, rows_b = entry_a, rows_a
    else:
        fetched_b, err = await _fetch_rows(
            query_id_b, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin,
        )
        if err:
            return err
        entry_b, rows_b = fetched_b

    series_a, err = _extract(entry_a, rows_a, column_a)
    if err:
        return err
    series_b, err = _extract(entry_b, rows_b, column_b)
    if err:
        return err

    spr = spread(series_a, series_b)
    if spr is None:
        return {
            "error": (
                f"Sin fechas en común entre {query_id_a}:{column_a} y "
                f"{query_id_b}:{column_b} en el período solicitado."
            ),
        }

    _register_series(state, entry_a, column_a, series_a)
    _register_series(state, entry_b, column_b, series_b)
    return {
        "serie_a": {"query_id": query_id_a, "column": column_a, "name": entry_a.name},
        "serie_b": {"query_id": query_id_b, "column": column_b, "name": entry_b.name},
        "definicion": f"spread = {column_a} - {column_b}",
        "unit": entry_a.unit,
        **spr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# get_series_stats
# ─────────────────────────────────────────────────────────────────────────────

_STATS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_series_stats",
        "description": (
            "Estadísticas descriptivas de una serie del catálogo en un período: "
            "media, desviación estándar, mínimo, máximo, último valor y el "
            "percentil del último valor dentro de la distribución del período. "
            "Úsala para situar un dato en su contexto histórico (¿es alto o bajo?)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_id": {"type": "string", "description": "query_id del catálogo."},
                "column": {"type": "string", "description": "Columna de la serie a analizar."},
                "fecha_inicio": {
                    "type": "string",
                    "description": "ISO YYYY-MM-DD o relativa ('-365d'). Default: -365d.",
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "ISO YYYY-MM-DD u 'hoy'. Default: hoy.",
                },
            },
            "required": ["query_id", "column"],
        },
    },
}


@register("get_series_stats", _STATS_SCHEMA)
async def get_series_stats(
    state: AgentState,
    query_id: str,
    column: str,
    fecha_inicio: str = "-365d",
    fecha_fin: str = "hoy",
) -> dict[str, Any]:
    fetched, err = await _fetch_rows(query_id, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    if err:
        return err
    entry, rows = fetched
    series, err = _extract(entry, rows, column)
    if err:
        return err

    stats = descriptive_stats(series)
    if stats is None:
        return {
            "error": (
                f"La serie {query_id}:{column} no tiene observaciones en el "
                f"período {fecha_inicio} → {fecha_fin}."
            ),
        }

    _register_series(state, entry, column, series)
    return {
        "query_id": query_id,
        "name": entry.name,
        "column": column,
        "unit": entry.unit,
        "frequency": entry.frequency,
        "periodo": {"desde": fecha_inicio, "hasta": fecha_fin},
        **stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# detect_anomaly
# ─────────────────────────────────────────────────────────────────────────────

_ANOMALY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_anomaly",
        "description": (
            "Evalúa si el valor más reciente de una serie está fuera de su "
            "rango histórico normal. Calcula media y desviación estándar de la "
            "ventana y reporta el z-score del último dato; lo marca como "
            "anomalía si |z| supera el umbral. Úsala para detectar movimientos "
            "inusuales en curvas, spreads o tipo de cambio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_id": {"type": "string", "description": "query_id del catálogo."},
                "column": {"type": "string", "description": "Columna de la serie a evaluar."},
                "lookback_days": {
                    "type": "integer",
                    "description": "Tamaño de la ventana histórica en días. Default: 180.",
                    "default": 180,
                    "minimum": 10,
                    "maximum": 1095,
                },
                "umbral_desviaciones": {
                    "type": "number",
                    "description": (
                        "Nº de desviaciones estándar para marcar anomalía. "
                        "Default: 2.0."
                    ),
                    "default": 2.0,
                    "minimum": 1.0,
                    "maximum": 5.0,
                },
            },
            "required": ["query_id", "column"],
        },
    },
}


@register("detect_anomaly", _ANOMALY_SCHEMA)
async def detect_anomaly(
    state: AgentState,
    query_id: str,
    column: str,
    lookback_days: int = 180,
    umbral_desviaciones: float = 2.0,
) -> dict[str, Any]:
    lookback_days = max(10, min(int(lookback_days), 1095))
    umbral = max(1.0, min(float(umbral_desviaciones), 5.0))

    fetched, err = await _fetch_rows(
        query_id, fecha_inicio=f"-{lookback_days}d", fecha_fin="hoy",
    )
    if err:
        return err
    entry, rows = fetched
    series, err = _extract(entry, rows, column)
    if err:
        return err

    check = anomaly_check(series, threshold_stds=umbral)
    if check is None:
        return {
            "error": (
                f"La serie {query_id}:{column} tiene menos de 3 observaciones "
                f"en la ventana de {lookback_days} días; sin masa estadística "
                "para evaluar anomalías."
            ),
        }

    _register_series(state, entry, column, series)
    return {
        "query_id": query_id,
        "name": entry.name,
        "column": column,
        "unit": entry.unit,
        "lookback_days": lookback_days,
        **check,
    }


# ─────────────────────────────────────────────────────────────────────────────
# get_market_snapshot
# ─────────────────────────────────────────────────────────────────────────────

# Indicadores clave que monitorea la División de Mercados Financieros.
# (clave, nombre, query_id, column) — todos existen en sql_catalog/catalog.yaml.
_SNAPSHOT_INDICATORS: list[tuple[str, str, str, str]] = [
    ("usdclp", "Tipo de cambio USD/CLP", "usdclp_historico", "usdclp"),
    ("cobre", "Precio del cobre", "precio_cobre", "cobre_usd_lb"),
    ("btp_10y", "Bono nominal BCCh 10 años (BTP)", "curva_btp_clp", "btp_10y"),
    ("btu_10y", "Bono real BCCh 10 años (BTU)", "curva_btu_uf", "btu_10y"),
    ("tib_spread", "Spread TIB-TPM (bps)", "tib_mercado", "tib_spread"),
    ("ust_10y", "US Treasury 10 años", "curva_ust", "ust_10y"),
    ("exp_tpm_3m", "Expectativa TPM implícita 3M (spread MIPR)",
     "expectativas_tpm_mipr", "spread_mipr_3m"),
]

_SNAPSHOT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_market_snapshot",
        "description": (
            "Foto rápida del estado actual de los indicadores clave que "
            "monitorea la División de Mercados Financieros del BCCh: USD/CLP, "
            "cobre, curva soberana (BTP/BTU 10A), break-even de inflación, "
            "tasa interbancaria, US Treasury 10A y expectativa de TPM. Retorna "
            "el último valor de cada uno en una sola llamada. Úsala al inicio "
            "para tener contexto de mercado sin encadenar varias queries."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


async def _latest_indicator(
    state: AgentState, clave: str, nombre: str, query_id: str, column: str,
) -> dict[str, Any]:
    """Último valor disponible de un indicador (ventana corta de 30 días)."""
    fetched, err = await _fetch_rows(
        query_id, fecha_inicio="-30d", fecha_fin="hoy", limit=30,
    )
    if err:
        return {"clave": clave, "nombre": nombre, "valor": None, "nota": err["error"]}
    entry, rows = fetched
    series, col_err = _extract(entry, rows, column)
    if col_err or not series:
        return {"clave": clave, "nombre": nombre, "valor": None,
                "nota": "sin datos recientes"}
    _register_series(state, entry, column, series)
    fecha, valor = series[-1]
    return {
        "clave": clave,
        "nombre": nombre,
        "valor": round(valor, 6),
        "unidad": entry.unit,
        "fecha": fecha,
    }


@register("get_market_snapshot", _SNAPSHOT_SCHEMA)
async def get_market_snapshot(state: AgentState) -> dict[str, Any]:
    indicadores = await asyncio.gather(
        *(_latest_indicator(state, *ind) for ind in _SNAPSHOT_INDICATORS)
    )
    indicadores = list(indicadores)

    # Break-even de inflación 10A = BTP 10A - BTU 10A (derivado).
    por_clave = {ind["clave"]: ind for ind in indicadores}
    btp, btu = por_clave.get("btp_10y"), por_clave.get("btu_10y")
    if btp and btu and btp.get("valor") is not None and btu.get("valor") is not None:
        indicadores.append({
            "clave": "bei_10y",
            "nombre": "Break-even de inflación 10 años (BTP - BTU)",
            "valor": round(btp["valor"] - btu["valor"], 6),
            "unidad": "% anual",
            "fecha": btp.get("fecha"),
            "nota": "Derivado: BTP 10A menos BTU 10A.",
        })

    return {
        "fecha_referencia": date.today().isoformat(),
        "n_indicadores": len(indicadores),
        "indicadores": indicadores,
        "nota": (
            "Último valor disponible de cada serie. Para variaciones o "
            "contexto histórico usa compute_variation / get_series_stats."
        ),
    }
