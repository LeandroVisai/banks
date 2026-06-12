"""Tool ``plot_series`` — grafica una o más series de tiempo de un dataset del
catálogo de parquets, devolviendo un **spec Vega-Lite** que el frontend renderiza.

Filosofía idéntica a ``execute_query``: el LLM elige dataset, columnas y tipo de
gráfico; la TOOL construye el spec Vega-Lite (declarativo, validable, sin código
ejecutable). El spec se acumula en ``AgentState.charts`` (igual que
``series_used``/``chunks_seen``) y llega al frontend vía ``ChatResponse.charts``.

Soporta dos formas de datos del catálogo:
  - "largo": Fecha + columna numérica + columna categórica (p.ej. btp_curva:
    Fecha/Variable/Valor) → ``color`` por la categoría.
  - "ancho": Fecha + varias columnas numéricas → ``fold`` de esas columnas en
    series.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from banks_rag.domain.agent.chart_types import vega_mark

from ._parquet_query import fetch_rows_from_dataset, find_date_column  # noqa: F401
from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState

# Tipos DuckDB que tratamos como numéricos (graficables en el eje Y).
_NUMERIC_TYPES = frozenset({
    "DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC",
    "BIGINT", "INTEGER", "INT", "SMALLINT", "HUGEINT", "TINYINT", "UBIGINT",
})
_DATE_TYPES = frozenset({"TIMESTAMP", "DATE", "TIMESTAMP_NS", "TIMESTAMP_S", "TIMESTAMP_MS"})
_MARKS = frozenset({"line", "area", "bar", "point"})
_MAX_POINTS = 1000  # tope de puntos embebidos en el spec (no inflar la respuesta)


_SCHEMA = {
    "type": "function",
    "function": {
        "name": "plot_series",
        "description": (
            "Grafica una serie de tiempo de un dataset del catálogo de parquets "
            "(tipo de cambio, curvas de bonos, spreads, liquidez, etc.) y la "
            "devuelve como gráfico para mostrar al usuario. Úsala cuando el "
            "usuario pide 'graficar', 'plotear', 'muéstrame la evolución/curva' "
            "de una serie. Primero usa discover_query para el `dataset_id`; la "
            "tool arma el gráfico — tú solo eliges dataset, columnas y tipo. "
            "Tras graficar, describe en texto la tendencia que muestra."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "ID del dataset (de discover_query).",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Columnas a graficar (nombres EXACTOS del esquema). Si "
                        "omites, se grafican las columnas numéricas del dataset."
                    ),
                },
                "fecha_inicio": {"type": "string", "description": "Fecha desde (ISO YYYY-MM-DD)."},
                "fecha_fin": {"type": "string", "description": "Fecha hasta (ISO YYYY-MM-DD)."},
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "area", "bar", "point"],
                    "description": (
                        "Tipo de gráfico. Si lo omites se usa el tipo "
                        "canónico que el catálogo declara para el dataset "
                        "(recomendado: deja que el catálogo decida)."
                    ),
                },
                "title": {"type": "string", "description": "Título del gráfico (opcional)."},
            },
            "required": ["dataset_id"],
        },
    },
}


def _isoval(value: Any) -> Any:
    """Convierte date/datetime a ISO string; deja el resto igual."""
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else value


def _classify_columns(dataset, select_cols: list[str], date_col: str) -> tuple[list[str], list[str]]:
    """Separa las columnas (sin la fecha) en numéricas y categóricas según el esquema."""
    by_name = {c.name: c for c in dataset.columns}
    numeric, categorical = [], []
    for name in select_cols:
        if name == date_col:
            continue
        col = by_name.get(name)
        if col is None:
            continue
        t = col.type.upper()
        if t in _NUMERIC_TYPES:
            numeric.append(name)
        elif t not in _DATE_TYPES:
            categorical.append(name)
    return numeric, categorical


def _build_vegalite(
    *, title: str, mark: str, data: list[dict], date_col: str,
    value_cols: list[str], cat_col: str | None,
) -> dict:
    """Construye el spec Vega-Lite (v5) para las series. La tool lo arma — el
    LLM nunca escribe el JSON a mano."""
    enc: dict[str, Any] = {
        "x": {"field": date_col, "type": "temporal", "title": "Fecha"},
    }
    transform: list[dict] = []
    if cat_col is not None:
        # Forma larga: una columna de valor + categoría -> color por categoría.
        enc["y"] = {"field": value_cols[0], "type": "quantitative", "title": value_cols[0]}
        enc["color"] = {"field": cat_col, "type": "nominal", "title": cat_col}
    elif len(value_cols) == 1:
        enc["y"] = {"field": value_cols[0], "type": "quantitative", "title": value_cols[0]}
    else:
        # Forma ancha: varias columnas numéricas -> fold a (serie, valor).
        transform.append({"fold": value_cols, "as": ["serie", "valor"]})
        enc["y"] = {"field": "valor", "type": "quantitative", "title": "Valor"}
        enc["color"] = {"field": "serie", "type": "nominal", "title": "Serie"}

    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "data": {"values": data},
        "mark": {"type": mark, "point": mark == "line", "tooltip": True},
        "encoding": enc,
        "width": "container",
        "height": 320,
    }
    if transform:
        spec["transform"] = transform
    return spec


@register("plot_series", _SCHEMA)
async def plot_series(
    state: "AgentState",
    dataset_id: str,
    columns: list[str] | None = None,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    chart_type: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    try:
        dataset, rows, date_col, select_cols = await fetch_rows_from_dataset(
            dataset_id,
            columns=columns,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            limit=_MAX_POINTS,
        )
    except ValueError as exc:
        return {"error": str(exc), "dataset_id": dataset_id}
    except FileNotFoundError as exc:
        return {"error": str(exc), "dataset_id": dataset_id}

    if date_col is None:
        return {
            "error": (
                f"Dataset {dataset_id!r} es un snapshot sin columna de fecha: no "
                "es una serie de tiempo graficable. Usa execute_query para verlo."
            ),
            "dataset_id": dataset_id,
        }
    if not rows:
        return {"error": f"Sin datos para {dataset_id!r} en ese rango.", "dataset_id": dataset_id}

    numeric, categorical = _classify_columns(dataset, select_cols, date_col)
    if not numeric:
        return {
            "error": (
                f"Dataset {dataset_id!r} no tiene columnas numéricas para "
                f"graficar. Columnas disponibles: {select_cols}."
            ),
            "dataset_id": dataset_id,
        }
    # Color por categoría solo si hay UNA columna numérica + alguna categórica.
    cat_col = categorical[0] if (len(numeric) == 1 and categorical) else None
    keep = [date_col, *numeric] + ([cat_col] if cat_col else [])
    data = [{k: _isoval(r.get(k)) for k in keep} for r in rows]

    # El tipo canónico lo decide el catálogo (gráfico consistente con el
    # tablero); el LLM solo puede forzar una marca simple explícita. La marca
    # Vega-Lite se deriva del tipo canónico (stacked_area → area, *_bar → bar;
    # el apilado lo resuelve el encoding `color` de Vega por defecto).
    if chart_type in _MARKS:
        canonical = chart_type
        mark = chart_type
    else:
        canonical = dataset.chart_type or "line"
        mark = vega_mark(canonical)
    chart_title = title or f"{dataset.name}"
    spec = _build_vegalite(
        title=chart_title, mark=mark, data=data,
        date_col=date_col, value_cols=numeric, cat_col=cat_col,
    )

    chart_id = state.add_chart({
        "dataset_id": dataset.id,
        "title": chart_title,
        "chart_type": canonical,
        "mark": mark,
        "unit": dataset.unit,
        "n_points": len(data),
        "spec": spec,
    })

    # Resumen compacto para el LLM (NO el spec completo) + cifras citables.
    last_row = rows[0]  # fetch viene ordenado por fecha DESC
    last_values = {c: last_row.get(c) for c in numeric}
    return {
        "chart_id": chart_id,
        "dataset_id": dataset.id,
        "name": dataset.name,
        "unit": dataset.unit,
        "chart_type": canonical,
        "mark": mark,
        "series": numeric if cat_col is None else f"{numeric[0]} por {cat_col}",
        "n_points": len(data),
        "last_date_in_data": _isoval(last_row.get(date_col)),
        "last_values": last_values,
        "note": (
            f"Gráfico {chart_id} generado y enviado al frontend para su "
            "renderizado. Descríbelo en tu respuesta (tendencia, nivel actual, "
            f"cambios) referenciándolo como 'gráfico {chart_id}'."
        ),
    }
