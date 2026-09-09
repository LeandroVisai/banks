"""Tipos de gráfico canónicos del catálogo de parquets y su reducción a
familias renderizables.

El ``chart_type`` de cada dataset en ``sql_catalog/parquet_catalog.yaml``
viene del diccionario del tablero (``market_monitor_table``,
``bar_neto_cum_interactive``, ``stacked_area``, ...). Esos nombres describen
el render *completo* del tablero original; este repo los reduce de forma
determinista a una FAMILIA base que el frontend de chat y los specs
Vega-Lite saben dibujar, de modo que el gráfico construido sea consistente
con el que correspondería a ese dataset (área apilada, barra agrupada, etc.)
sin reimplementar cada renderer interactivo.

Fuente de verdad única: las tools (``plot_series``, ``execute_query``) y el
estado del agente importan de aquí — no duplicar este mapping en otros
módulos.
"""

from __future__ import annotations

# Familias base que el frontend sabe renderizar.
BASE_FAMILIES: frozenset[str] = frozenset({
    "line",         # serie temporal simple / curvas por tenor
    "area",         # área (apilada cuando hay categoría)
    "bar",          # barras simples (categórico o temporal corto)
    "grouped_bar",  # barras agrupadas por categoría
    "stacked_bar",  # barras apiladas por categoría
    "point",        # dispersión
    "pie",          # composición (torta)
    "table",        # tabla-monitor (no graficable como serie)
})

# chart_type canónico del catálogo → familia base.
# Los tipos "*_interactive"/"*_dual" describen renders compuestos del tablero;
# aquí se reducen a su componente principal.
CHART_FAMILY: dict[str, str] = {
    # líneas y curvas
    "line": "line",
    "multi_line_dual": "line",
    "dual_line_interactive": "line",
    "line_cum_interactive": "line",
    "var_line_interactive": "line",
    "fx_var_line_interactive": "line",
    "spc_curve": "line",
    "curve_snapshot": "line",
    "variation_line_interactive": "line",
    "cumulative_line_interactive": "line",
    # Curva por tenor con una fecha elegible (spread BTP-UST/SPC): el eje X es
    # el plazo, no el tiempo — misma lectura que `spc_curve`.
    "variation_per_date_selected": "line",
    # Vela OHLC (Informe Cambiario AM). El informe curado la dibuja nativa
    # (svg_chart._render_candlestick); fuera de ahí degrada a línea, que es la
    # lectura correcta de un cierre diario.
    "candlestick": "line",
    # áreas (apiladas)
    "stacked_area": "area",
    "stacked_area_line": "area",
    # barras simples
    "bar": "bar",
    "variation_bar": "bar",
    "variation_bar_interactive": "bar",
    "relative_bar": "bar",
    "fx_var_bar_interactive": "bar",
    "hist_range": "bar",
    "flujo_cambiario": "bar",
    "flujo_spot_variacion": "bar",
    "bar_scatter_dual": "bar",
    # barras agrupadas
    "grouped_bar": "grouped_bar",
    "mipr_var": "grouped_bar",
    "spc_ois_var": "grouped_bar",
    # barras apiladas (incluye los acumulados barra+neto del tablero)
    "stacked_bar": "stacked_bar",
    "stacked_bar_interactive": "stacked_bar",
    "bar_neto_cum_interactive": "stacked_bar",
    "pos_bar_delta_interactive": "stacked_bar",
    "pos_delta_interactive": "stacked_bar",
    "ffmm_cum_interactive": "stacked_bar",
    "ffmm_stock_cum_interactive": "stacked_bar",
    "nr_deriv_cumulative_interactive": "stacked_bar",
    "nr_spot_cumulative_interactive": "stacked_bar",
    "fx_pos_cumulative_interactive": "stacked_bar",
    "stacked_neto_interactive": "stacked_bar",
    "stock_cum_interactive": "stacked_bar",
    # dispersión
    "scatter": "point",
    "fx_retorno_scatter_interactive": "point",
    # composición
    "pie": "pie",
    # Treemap (informe curado ipc): jerarquía de dos niveles, área=tamaño,
    # color=escala divergente. El agente genérico no dibuja jerarquías; se
    # reduce a "pie" (misma familia semántica: composición de un total).
    "treemap": "pie",
    # tablas-monitor (color de mercados): no son una serie graficable
    "market_monitor_table": "table",
}

# Marca Vega-Lite por familia (para plot_series). Las familias de barra
# comparten la marca "bar"; el apilado/agrupado lo resuelve el encoding.
VEGA_MARK_BY_FAMILY: dict[str, str] = {
    "line": "line",
    "area": "area",
    "bar": "bar",
    "grouped_bar": "bar",
    "stacked_bar": "bar",
    "point": "point",
    "pie": "bar",    # plot_series no grafica tortas: degrade a barra
    "table": "line",  # tabla-monitor: si se fuerza un plot, línea neutra
}


def chart_family(chart_type: str | None) -> str:
    """Reduce un ``chart_type`` canónico del catálogo a su familia base.

    Tipos desconocidos (catálogo más nuevo que el código) degradan a
    ``"line"`` en vez de fallar."""
    return CHART_FAMILY.get((chart_type or "").strip(), "line")


def vega_mark(chart_type: str | None) -> str:
    """Marca Vega-Lite para un ``chart_type`` canónico del catálogo."""
    return VEGA_MARK_BY_FAMILY.get(chart_family(chart_type), "line")
