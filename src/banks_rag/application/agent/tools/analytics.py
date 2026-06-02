"""Analytics tools — el agente no trae datos crudos, los interpreta.

Operan contra ``parquet_catalog.yaml``: el LLM elige un ``dataset_id``
(descubierto vía ``discover_query``) + columna + filtros opcionales (para
datasets long-format donde la serie se identifica por una columna categórica,
ej. ``Tenor="10Y"`` en ``btp_curva``). La SQL la construye internamente la
tool (vía ``_parquet_query.fetch_rows_from_dataset``); el LLM NUNCA escribe SQL.

El cálculo numérico vive en ``infrastructure/sql/series_analytics`` (funciones
puras); aquí solo se orquesta DuckDB + catálogo + serialización.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from banks_rag.infrastructure.sql.duckdb_runner import MAX_ROWS
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset
from banks_rag.infrastructure.sql.series_analytics import (
    anomaly_check,
    clean_series,
    composition,
    composition_wide,
    descriptive_stats,
    spread,
    variation,
)

from ._parquet_query import fetch_rows_from_dataset
from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState

_ANALYTICS_LIMIT = MAX_ROWS   # cubre ~2 años de data diaria

# Filtro opcional para datasets long-format (ej. {"Tenor": "10Y"}). En el JSON
# Schema se expone con `additionalProperties: True` para que el LLM pase pares
# arbitrarios; el helper valida que la columna exista en el catálogo y, si la
# columna declara `values` (enum), que el valor esté en esa lista.
_FILTERS_SCHEMA = {
    "type": "object",
    "description": (
        "Filtros opcionales de igualdad para columnas categóricas (ej. "
        "`{\"Tenor\": \"10Y\"}` para curvas long-format, `{\"Banco\": "
        "\"BCI\"}` para balance). Cada clave debe existir en el esquema "
        "del dataset."
    ),
    "additionalProperties": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Resolución de fechas relativas y helpers
# ─────────────────────────────────────────────────────────────────────────────

_REL_DATE_RE = re.compile(r"^-(\d+)([dm])$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resolve_date(raw: str) -> str:
    """``hoy`` / ``-90d`` / ``-12m`` / ISO → ISO YYYY-MM-DD."""
    s = raw.strip().lower()
    if s in ("hoy", "today"):
        return date.today().isoformat()
    m = _REL_DATE_RE.match(s)
    if m:
        n = int(m.group(1))
        days = n if m.group(2) == "d" else n * 30   # mes ≈ 30 días
        return (date.today() - timedelta(days=days)).isoformat()
    if _ISO_DATE_RE.match(raw):
        return raw
    raise ValueError(
        f"Fecha {raw!r} inválida (usa 'hoy', '-Nd', '-Nm' o ISO YYYY-MM-DD)."
    )


async def _fetch_rows_from_dataset(
    dataset_id: str,
    *,
    column: str,
    fecha_inicio: str,
    fecha_fin: str,
    filters: dict | None = None,
    limit: int = _ANALYTICS_LIMIT,
) -> tuple[tuple[ParquetDataset, list[dict], str] | None, dict | None]:
    """Resuelve fechas relativas, delega en ``_parquet_query`` y empaqueta
    errores como dict. Returns ``((dataset, rows, date_col), None)`` o
    ``(None, error_dict)``."""
    try:
        fi = _resolve_date(fecha_inicio)
        ff = _resolve_date(fecha_fin)
    except ValueError as exc:
        return None, {"error": str(exc)}
    try:
        dataset, rows, date_col, _ = await fetch_rows_from_dataset(
            dataset_id,
            columns=[column],
            fecha_inicio=fi,
            fecha_fin=ff,
            filters=filters,
            limit=limit,
        )
    except ValueError as exc:
        return None, {"error": str(exc), "dataset_id": dataset_id}
    except FileNotFoundError as exc:
        return None, {"error": str(exc), "dataset_id": dataset_id}
    except Exception as exc:
        return None, {
            "error": f"Error ejecutando dataset {dataset_id!r}: {exc}",
        }
    if date_col is None:
        return None, {
            "error": (
                f"Dataset {dataset_id!r} es un snapshot sin columna de fecha: "
                "no admite análisis de serie temporal (variación, spread, "
                "estadística, anomalía). Usa execute_query para ver sus filas."
            ),
            "dataset_id": dataset_id,
        }
    return (dataset, rows, date_col), None


def _extract(
    dataset: ParquetDataset, rows: list[dict], column: str, date_col: str,
) -> tuple[list[tuple[str, float]] | None, dict | None]:
    """Valida que la columna exista en el esquema y limpia la serie."""
    valid = {c.name for c in dataset.columns}
    if column not in valid:
        return None, {
            "error": f"Columna {column!r} no existe en {dataset.id!r}.",
            "available_columns": sorted(valid),
        }
    return clean_series(rows, date_col, column), None


def _series_id(dataset_id: str, column: str, filters: dict | None) -> str:
    """ID estable para `state.add_series` que incluye los filtros."""
    base = f"{dataset_id}:{column}"
    if not filters:
        return base
    flt = ",".join(f"{k}={v}" for k, v in sorted(filters.items()))
    return f"{base}:{flt}"


def _register_series(
    state: AgentState,
    dataset: ParquetDataset,
    column: str,
    filters: dict | None,
    series: list[tuple[str, float]],
) -> None:
    """Registra la serie en ``state.series_used`` para trazabilidad."""
    sid = _series_id(dataset.id, column, filters)
    name_suffix = (
        " (" + ", ".join(f"{k}={v}" for k, v in sorted(filters.items())) + ")"
        if filters else ""
    )
    state.add_series(
        sid,
        {
            "series_name": f"{dataset.name} — {column}{name_suffix}",
            "unit": dataset.unit,
        },
        [{"date": fecha, "value": valor} for fecha, valor in series],
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute_variation
# ─────────────────────────────────────────────────────────────────────────────

_VARIATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compute_variation",
        "description": (
            "Calcula cuánto se movió una serie entre dos fechas: cambio "
            "absoluto, variación porcentual, cambio en puntos base y el "
            "rango (mín/máx) del período. Úsala para CUANTIFICAR un "
            "movimiento. Requiere el dataset_id (de discover_query) y la "
            "columna; para datasets long-format (ej. btp_curva) pasa "
            "`filters={\"Tenor\": \"10Y\"}` para seleccionar la serie."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID del dataset (obtenido con discover_query).",
                },
                "column": {
                    "type": "string",
                    "description": (
                        "Columna numérica a analizar (ej. 'CLP', 'Cobre', "
                        "'Valor'). Usa el nombre EXACTO del esquema."
                    ),
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": (
                        "Fecha de inicio: ISO YYYY-MM-DD o relativa "
                        "('-90d', '-12m'). Default: -90d."
                    ),
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "Fecha de fin: ISO YYYY-MM-DD u 'hoy'. Default: hoy.",
                },
                "filters": _FILTERS_SCHEMA,
            },
            "required": ["dataset_id", "column"],
        },
    },
}


@register("compute_variation", _VARIATION_SCHEMA)
async def compute_variation(
    state: AgentState,
    dataset_id: str,
    column: str,
    fecha_inicio: str = "-90d",
    fecha_fin: str = "hoy",
    filters: dict | None = None,
) -> dict[str, Any]:
    fetched, err = await _fetch_rows_from_dataset(
        dataset_id, column=column, fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin, filters=filters,
    )
    if err:
        return err
    dataset, rows, date_col = fetched
    series, err = _extract(dataset, rows, column, date_col)
    if err:
        return err

    var = variation(series)
    if var is None:
        return {
            "error": (
                f"La serie {_series_id(dataset_id, column, filters)} tiene "
                f"menos de 2 observaciones en {fecha_inicio} → {fecha_fin}; "
                "no se puede calcular variación. Amplía el rango de fechas."
            ),
        }

    _register_series(state, dataset, column, filters, series)
    return {
        "dataset_id": dataset_id,
        "name": dataset.name,
        "column": column,
        "filters": filters,
        "unit": dataset.unit,
        "nota_bps": "cambio_bps es válido solo si la unidad es tasa/porcentaje.",
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
            "Calcula el spread (A - B) entre dos series alineadas por fecha. "
            "Sirve para break-even (BTP - BTU), spreads TIB-TPM, pendientes "
            "de curva (10A - 2A), etc. Si ambas series están en el mismo "
            "dataset long-format (mismo dataset_id), usa filters_a y "
            "filters_b para distinguirlas (ej. filters_a={\"Tenor\":\"10Y\"}, "
            "filters_b={\"Tenor\":\"2Y\"})."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id_a": {"type": "string", "description": "dataset_id de la serie A."},
                "column_a": {"type": "string", "description": "Columna de la serie A (minuendo)."},
                "dataset_id_b": {"type": "string", "description": "dataset_id de la serie B."},
                "column_b": {"type": "string", "description": "Columna de la serie B (sustraendo)."},
                "fecha_inicio": {
                    "type": "string",
                    "description": "ISO YYYY-MM-DD o relativa ('-90d'). Default: -90d.",
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "ISO YYYY-MM-DD u 'hoy'. Default: hoy.",
                },
                "filters_a": _FILTERS_SCHEMA,
                "filters_b": _FILTERS_SCHEMA,
            },
            "required": ["dataset_id_a", "column_a", "dataset_id_b", "column_b"],
        },
    },
}


@register("compute_spread", _SPREAD_SCHEMA)
async def compute_spread(
    state: AgentState,
    dataset_id_a: str,
    column_a: str,
    dataset_id_b: str,
    column_b: str,
    fecha_inicio: str = "-90d",
    fecha_fin: str = "hoy",
    filters_a: dict | None = None,
    filters_b: dict | None = None,
) -> dict[str, Any]:
    fetched_a, err = await _fetch_rows_from_dataset(
        dataset_id_a, column=column_a, fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin, filters=filters_a,
    )
    if err:
        return err
    dataset_a, rows_a, date_col_a = fetched_a

    # Si ambas series comparten dataset Y filtros, no vale la pena re-fetch.
    same_source = dataset_id_b == dataset_id_a and filters_a == filters_b
    if same_source:
        dataset_b, rows_b, date_col_b = dataset_a, rows_a, date_col_a
    else:
        fetched_b, err = await _fetch_rows_from_dataset(
            dataset_id_b, column=column_b, fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin, filters=filters_b,
        )
        if err:
            return err
        dataset_b, rows_b, date_col_b = fetched_b

    series_a, err = _extract(dataset_a, rows_a, column_a, date_col_a)
    if err:
        return err
    series_b, err = _extract(dataset_b, rows_b, column_b, date_col_b)
    if err:
        return err

    spr = spread(series_a, series_b)
    if spr is None:
        return {
            "error": (
                f"Sin fechas en común entre "
                f"{_series_id(dataset_id_a, column_a, filters_a)} y "
                f"{_series_id(dataset_id_b, column_b, filters_b)} en el período."
            ),
        }

    _register_series(state, dataset_a, column_a, filters_a, series_a)
    _register_series(state, dataset_b, column_b, filters_b, series_b)
    return {
        "serie_a": {
            "dataset_id": dataset_id_a, "column": column_a,
            "name": dataset_a.name, "filters": filters_a,
        },
        "serie_b": {
            "dataset_id": dataset_id_b, "column": column_b,
            "name": dataset_b.name, "filters": filters_b,
        },
        "definicion": f"spread = {column_a} - {column_b}",
        "unit": dataset_a.unit,
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
            "Estadística descriptiva de una serie en un período: media, "
            "desviación estándar, mínimo, máximo, último valor y percentil "
            "del último valor en la distribución del período. Para situar un "
            "dato en su contexto histórico."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "column": {"type": "string"},
                "fecha_inicio": {
                    "type": "string",
                    "description": "ISO o relativa ('-365d'). Default: -365d.",
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "ISO u 'hoy'. Default: hoy.",
                },
                "filters": _FILTERS_SCHEMA,
            },
            "required": ["dataset_id", "column"],
        },
    },
}


@register("get_series_stats", _STATS_SCHEMA)
async def get_series_stats(
    state: AgentState,
    dataset_id: str,
    column: str,
    fecha_inicio: str = "-365d",
    fecha_fin: str = "hoy",
    filters: dict | None = None,
) -> dict[str, Any]:
    fetched, err = await _fetch_rows_from_dataset(
        dataset_id, column=column, fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin, filters=filters,
    )
    if err:
        return err
    dataset, rows, date_col = fetched
    series, err = _extract(dataset, rows, column, date_col)
    if err:
        return err

    stats = descriptive_stats(series)
    if stats is None:
        return {
            "error": (
                f"La serie {_series_id(dataset_id, column, filters)} no "
                f"tiene observaciones en {fecha_inicio} → {fecha_fin}."
            ),
        }

    _register_series(state, dataset, column, filters, series)
    return {
        "dataset_id": dataset_id,
        "name": dataset.name,
        "column": column,
        "filters": filters,
        "unit": dataset.unit,
        "periodo": {"desde": fecha_inicio, "hasta": fecha_fin},
        **stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_composition
# ─────────────────────────────────────────────────────────────────────────────

_COMPOSITION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compute_composition",
        "description": (
            "Calcula la COMPOSICIÓN (% por categoría) de un monto a una fecha "
            "de corte. Úsala para preguntas de cartera / allocation / "
            "distribución (ej. '% de la cartera AFP en Chile vs. extranjero', "
            "'composición de activos por banco'). NO inventes porcentajes: esta "
            "tool los calcula. Dos modos según el esquema (usa discover_query "
            "para ver las columnas):\n"
            "- LONG: una columna categórica + una de monto → pasa "
            "`category_column` + `value_column` (ej. allocation: "
            "category_column='Tipo_instrumento', value_column='Monto_USD').\n"
            "- WIDE: una columna por categoría → pasa `value_columns` con la "
            "lista de columnas (ej. allocation_int_nac: "
            "value_columns=['Nacional','Extranjero'])."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID del dataset (obtenido con discover_query).",
                },
                "value_column": {
                    "type": "string",
                    "description": "Modo LONG: columna numérica a sumar (ej. 'Monto_USD').",
                },
                "category_column": {
                    "type": "string",
                    "description": (
                        "Modo LONG: columna categórica por la que desglosar "
                        "(ej. 'Tipo_instrumento', 'Banco', 'Tipo_fondo')."
                    ),
                },
                "value_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Modo WIDE: lista de columnas numéricas, una por "
                        "categoría (ej. ['Nacional','Extranjero'])."
                    ),
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": "ISO YYYY-MM-DD o relativa ('-365d'). Default: -365d.",
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "ISO u 'hoy'. Default: hoy. Usa la última fecha del rango.",
                },
                "filters": _FILTERS_SCHEMA,
            },
            "required": ["dataset_id"],
        },
    },
}


@register("compute_composition", _COMPOSITION_SCHEMA)
async def compute_composition(
    state: AgentState,
    dataset_id: str,
    value_column: str | None = None,
    category_column: str | None = None,
    value_columns: list[str] | None = None,
    fecha_inicio: str = "-365d",
    fecha_fin: str = "hoy",
    filters: dict | None = None,
) -> dict[str, Any]:
    # Determina el modo y las columnas a traer.
    wide = bool(value_columns)
    long = bool(category_column and value_column)
    if not (wide or long):
        return {
            "error": (
                "Indica el modo: LONG (category_column + value_column) o WIDE "
                "(value_columns con la lista de columnas por categoría)."
            ),
        }
    fetch_cols = list(value_columns) if wide else [category_column, value_column]

    try:
        fi = _resolve_date(fecha_inicio)
        ff = _resolve_date(fecha_fin)
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        dataset, rows, date_col, _ = await fetch_rows_from_dataset(
            dataset_id,
            columns=fetch_cols,
            fecha_inicio=fi,
            fecha_fin=ff,
            filters=filters,
            limit=_ANALYTICS_LIMIT,
        )
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc), "dataset_id": dataset_id}
    except Exception as exc:
        return {"error": f"Error ejecutando dataset {dataset_id!r}: {exc}"}

    if wide:
        comp = composition_wide(rows, date_col, list(value_columns))
        label = f"{', '.join(value_columns)}"
        series_value_name = value_columns[0]
    else:
        comp = composition(rows, date_col, category_column, value_column)
        label = f"{value_column} por {category_column}"
        series_value_name = value_column

    if comp is None:
        return {
            "error": (
                f"Sin datos para componer ({label}) en {dataset_id!r} "
                f"({fecha_inicio} → {fecha_fin}). Verifica las columnas y el rango."
            ),
        }

    # Traza: registra la serie de valor en la fecha de corte (para series_used).
    state.add_series(
        _series_id(dataset_id, series_value_name, filters),
        {"series_name": f"{dataset.name} — {label}", "unit": dataset.unit},
        [{"date": comp["fecha"]}],
    )
    return {
        "dataset_id": dataset_id,
        "name": dataset.name,
        "unit": dataset.unit,
        "mode": "wide" if wide else "long",
        "category_column": category_column,
        "value_column": value_column,
        "value_columns": value_columns,
        "filters": filters,
        **comp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_aggregate
# ─────────────────────────────────────────────────────────────────────────────

_AGGREGATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compute_aggregate",
        "description": (
            "Suma / total / posición NETA a una fecha de corte (con signo). NO "
            "es porcentaje — para shares usa compute_composition. NUNCA sumes a "
            "mano. Tres usos:\n"
            "- TOTAL de una columna: pasa `value_column` (ej. DV01 total de la "
            "cartera AFP: value_column='dv01').\n"
            "- TOTAL por grupo: `value_column` + `group_by` (ej. DV01 por "
            "moneda: value_column='dv01', group_by='moneda'; stock por sector: "
            "value_column='Stock', group_by='Serie').\n"
            "- NETO de varias columnas: `value_columns` (ej. flujo neto = "
            "Spot+Forward: value_columns=['Spot','Forward'], filtrando el sector "
            "con filters)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID del dataset (obtenido con discover_query).",
                },
                "value_column": {
                    "type": "string",
                    "description": "Columna numérica a sumar (modos TOTAL y TOTAL por grupo).",
                },
                "group_by": {
                    "type": "string",
                    "description": (
                        "Columna categórica para desglosar la suma (ej. "
                        "'moneda', 'Serie', 'Fondo'). Opcional; requiere value_column."
                    ),
                },
                "value_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modo NETO: columnas a sumar entre sí (ej. ['Spot','Forward']).",
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": "ISO YYYY-MM-DD o relativa ('-365d'). Default: -365d.",
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "ISO u 'hoy'. Default: hoy. Usa la última fecha del rango.",
                },
                "filters": _FILTERS_SCHEMA,
            },
            "required": ["dataset_id"],
        },
    },
}


@register("compute_aggregate", _AGGREGATE_SCHEMA)
async def compute_aggregate(
    state: AgentState,
    dataset_id: str,
    value_column: str | None = None,
    group_by: str | None = None,
    value_columns: list[str] | None = None,
    fecha_inicio: str = "-365d",
    fecha_fin: str = "hoy",
    filters: dict | None = None,
) -> dict[str, Any]:
    # Resolver modo y columnas a traer.
    if group_by and value_column:
        mode, fetch_cols = "grouped", [group_by, value_column]
    elif value_columns:
        mode, fetch_cols = "net", list(value_columns)
    elif value_column:
        mode, fetch_cols = "total", [value_column]
    else:
        return {
            "error": (
                "Indica qué sumar: `value_column` (total), `value_column`+"
                "`group_by` (total por grupo) o `value_columns` (neto de columnas)."
            ),
        }

    try:
        fi = _resolve_date(fecha_inicio)
        ff = _resolve_date(fecha_fin)
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        dataset, rows, date_col, _ = await fetch_rows_from_dataset(
            dataset_id,
            columns=fetch_cols,
            fecha_inicio=fi,
            fecha_fin=ff,
            filters=filters,
            limit=_ANALYTICS_LIMIT,
        )
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc), "dataset_id": dataset_id}
    except Exception as exc:
        return {"error": f"Error ejecutando dataset {dataset_id!r}: {exc}"}

    if mode == "grouped":
        comp = composition(rows, date_col, group_by, value_column)
        series_name = value_column
    else:  # "net" o "total" → sumar columnas (wide)
        cols = value_columns if mode == "net" else [value_column]
        comp = composition_wide(rows, date_col, cols)
        series_name = (value_columns or [value_column])[0]

    if comp is None:
        return {
            "error": (
                f"Sin datos para agregar en {dataset_id!r} ({fecha_inicio} → "
                f"{fecha_fin}). Verifica columnas y rango."
            ),
        }

    state.add_series(
        _series_id(dataset_id, series_name, filters),
        {"series_name": f"{dataset.name} — {series_name} (agregado)", "unit": dataset.unit},
        [{"date": comp["fecha"]}],
    )

    # Presenta totales/neto (no %): renombra `breakdown` según el modo y omite share.
    componentes = [
        {"grupo" if mode == "grouped" else "columna": b["categoria"], "valor": b["valor"]}
        for b in comp["breakdown"]
    ]
    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "name": dataset.name,
        "unit": dataset.unit,
        "mode": mode,
        "fecha": comp["fecha"],
        "filters": filters,
    }
    if mode == "net":
        out["neto"] = comp["total"]
        out["componentes"] = componentes
    elif mode == "grouped":
        out["total"] = comp["total"]
        out["por_grupo"] = componentes
    else:  # total
        out["total"] = comp["total"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# detect_anomaly
# ─────────────────────────────────────────────────────────────────────────────

_ANOMALY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_anomaly",
        "description": (
            "Evalúa si el último valor de una serie está fuera de su rango "
            "histórico normal vía z-score sobre la ventana lookback_days. "
            "Útil para detectar movimientos inusuales."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "column": {"type": "string"},
                "lookback_days": {
                    "type": "integer",
                    "description": "Tamaño de ventana histórica en días. Default: 180.",
                    "default": 180,
                    "minimum": 10,
                    "maximum": 1095,
                },
                "umbral_desviaciones": {
                    "type": "number",
                    "description": "Nº de desviaciones para marcar anomalía. Default: 2.0.",
                    "default": 2.0,
                    "minimum": 1.0,
                    "maximum": 5.0,
                },
                "filters": _FILTERS_SCHEMA,
            },
            "required": ["dataset_id", "column"],
        },
    },
}


@register("detect_anomaly", _ANOMALY_SCHEMA)
async def detect_anomaly(
    state: AgentState,
    dataset_id: str,
    column: str,
    lookback_days: int = 180,
    umbral_desviaciones: float = 2.0,
    filters: dict | None = None,
) -> dict[str, Any]:
    lookback_days = max(10, min(int(lookback_days), 1095))
    umbral = max(1.0, min(float(umbral_desviaciones), 5.0))

    fetched, err = await _fetch_rows_from_dataset(
        dataset_id, column=column,
        fecha_inicio=f"-{lookback_days}d", fecha_fin="hoy", filters=filters,
    )
    if err:
        return err
    dataset, rows, date_col = fetched
    series, err = _extract(dataset, rows, column, date_col)
    if err:
        return err

    check = anomaly_check(series, threshold_stds=umbral)
    if check is None:
        return {
            "error": (
                f"La serie {_series_id(dataset_id, column, filters)} tiene "
                f"menos de 3 observaciones en {lookback_days} días; sin masa "
                "estadística para evaluar anomalías."
            ),
        }

    _register_series(state, dataset, column, filters, series)
    return {
        "dataset_id": dataset_id,
        "name": dataset.name,
        "column": column,
        "filters": filters,
        "unit": dataset.unit,
        "lookback_days": lookback_days,
        **check,
    }


# ─────────────────────────────────────────────────────────────────────────────
# get_market_snapshot
# ─────────────────────────────────────────────────────────────────────────────

# Indicadores clave que monitorea la División de Mercados. Tupla:
# (clave, nombre, dataset_id, column, filters_or_None). Los IDs y columnas
# son los del parquet_catalog: BTP/BTU/MIPR son long-format y requieren un
# filtro por Tenor. UST 10Y no tiene dataset equivalente en parquet_catalog,
# por eso se omite (el panorama degrada elegante si algo falta).
_SNAPSHOT_INDICATORS: list[tuple[str, str, str, str, dict | None]] = [
    ("usdclp", "Tipo de cambio USD/CLP",
     "clp_monto", "CLP", None),
    ("cobre", "Precio del cobre (USc/lb)",
     "cobre_dxy", "Cobre", None),
    ("btp_10y", "Bono nominal BCCh 10A (BTP)",
     "btp_curva", "Valor", {"Tenor": "10Y"}),
    ("btu_10y", "Bono real BCCh 10A (BTU)",
     "btu_curva", "Valor", {"Tenor": "10Y"}),
    ("tib_spread", "Spread TIB-TPM (pb)",
     "tib_monto_transado", "Spread TIB-TPM", None),
    ("exp_tpm_3m", "Expectativa TPM impl. 3M (spread MIPR CL-US)",
     "mipr_spread_hist", "Valor", {"Tenor": "3M"}),
]

_SNAPSHOT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_market_snapshot",
        "description": (
            "Foto rápida del estado actual de los indicadores clave que "
            "monitorea la División de Mercados del BCCh: USD/CLP, cobre, "
            "curva soberana (BTP/BTU 10A), break-even de inflación, "
            "interbancario, expectativa de TPM. Retorna el último valor de "
            "cada uno en una sola llamada — úsala al inicio para tener "
            "contexto de mercado sin encadenar varias queries."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


async def _latest_indicator(
    state: AgentState,
    clave: str,
    nombre: str,
    dataset_id: str,
    column: str,
    filters: dict | None,
) -> dict[str, Any]:
    """Último valor disponible de un indicador (ventana de 30 días)."""
    fetched, err = await _fetch_rows_from_dataset(
        dataset_id, column=column, fecha_inicio="-30d", fecha_fin="hoy",
        filters=filters, limit=60,
    )
    if err:
        return {"clave": clave, "nombre": nombre, "valor": None,
                "nota": err["error"]}
    dataset, rows, date_col = fetched
    series, col_err = _extract(dataset, rows, column, date_col)
    if col_err or not series:
        return {"clave": clave, "nombre": nombre, "valor": None,
                "nota": (col_err or {}).get("error", "sin datos recientes")}
    _register_series(state, dataset, column, filters, series)
    fecha, valor = series[-1]
    return {
        "clave": clave,
        "nombre": nombre,
        "dataset_id": dataset_id,
        "valor": round(valor, 6),
        "unidad": dataset.unit,
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
