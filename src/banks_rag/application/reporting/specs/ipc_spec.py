"""Spec curado del "Informe Post IPC" (familia ipc).

Réplica del informe mensual que el DOMA publica el día que el INE divulga el IPC.
El original vive en ``Nuevo_Repo/IPC/Post IPC/`` y se arma con Plotly + Jinja:
un notebook calcula el análisis y un publicador lo maqueta. Acá se trae al flujo
estándar de informes curados —mismos gráficos, mismo orden, dibujados con el
renderer SVG del repo— leyendo parquets como el resto de las familias.

El orden, los títulos de sección y los de cada tarjeta se calcaron de la versión
final publicada (``versiones_finales/IPC.html``, agosto 2026): son sus 15 gráficos
en sus 6 secciones, en el mismo orden.

  Dato del mes                   esperado vs efectivo + variaciones por agregado  (2)
  Esperado vs efectivo           incidencia esperada vs efectiva por división     (1)
  Difusión inflacionaria         general + bienes SV + servicios SV, con banda    (3)
  Canasta e incidencias          canasta + grupo + incidencia 12m + histórico     (4)
  Expectativas y Fundamentales   petróleo/TC, seguros y compensaciones            (4)
  Detalle por producto           la canasta del mes, por incidencia               (1)

De dónde salen los datos
------------------------
Los parquets los generan los tres extractores de ``scripts/ingest/``:

    from_excel.py     canasta del INE (4 bases), diccionario y lo que carga el
                      operador en "Expectativas IPC.xlsx"
    from_sql.py       compensaciones, seguros de inflación y fundamentales, del DW
    from_notebook.py  las series del BCCh que solo existen vía su API: analíticos
                      F074, volátiles G073 y las bandas de bienes/servicios SV

Un bloque cuyo parquet todavía no esté sale como tarjeta "sin serie en el
parquet" en su lugar, y la estructura del informe se ve completa igual.

Tres desvíos deliberados respecto del original
----------------------------------------------
1. **El treemap de la canasta** SÍ se dibuja (``svg_chart._render_treemap``,
   ``PlotData.kind='hierarchy'`` — layout squarified, ver el módulo), pero en
   DOS niveles (división → grupo) en vez de los cuatro del original (división →
   grupo → clase → producto): con 283 productos individuales el rectángulo de
   cada uno mediría un puñado de píxeles y ni el color ni la etiqueta se leerían
   en un correo. División+grupo (13+46 rectángulos) sí entran legibles. El área
   sigue siendo la ponderación y el color la variación mensual, igual que el
   ``px.treemap(color_continuous_scale='RdBu_r')`` original.

2. **Las bandas** (difusión, bienes SV, servicios SV) llevaban en el original una
   línea por año además de la banda. Acá van como rango: banda + promedio + el
   año en curso. Se pierde la comparación con 2024 y 2025; se gana un gráfico que
   se lee de un vistazo, que es lo que pide el formato.

3. **La tabla de detalle** era interactiva (filtro, búsqueda, orden por columna).
   Acá se recorta a los 25 productos de mayor incidencia del mes. La canasta
   completa sigue en ``ipc_canasta_mes.parquet`` para quien quiera consultarla.

Fuera de esos puntos no se agrega ni se quita ningún gráfico: el informe
tiene las mismas 15 piezas que el original.

Esta es la ÚNICA pieza a editar para reordenar o retitular el informe. Las
transforms ``ipc_*`` viven en ``series_transforms.py`` y el render en
``curated_report.py``.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import (
    STATUS_MVP,
    FamilyReportSpec,
    ReportBlock,
)

# Secciones: los ``label`` de nav_groups del informe original, en su orden.
_S_RESUMEN = "Dato del mes"
_S_EXPECTATIVAS = "Esperado vs efectivo"
_S_DIFUSION = "Difusión inflacionaria"
_S_CANASTA = "Canasta e incidencias"
_S_FUNDAMENTALES = "Expectativas y Fundamentales"
_S_DETALLE = "Detalle por producto"

_BLOCKS: tuple[ReportBlock, ...] = (
    # ── Dato del mes ─────────────────────────────────────────────────────────
    ReportBlock(
        section=_S_RESUMEN, title="IPC esperado vs efectivo",
        unit="%", chart="grouped_bar", status=STATUS_MVP,
        source_id="ipc_esperado_vs_efectivo", transform="wide_monthly_bars",
        params={"include": ["Seguros", "EOF", "EEE", "Bloomberg"], "overlay": ["Efectivo"]},
        text_slot="ipc:esperado_efectivo",
        note="El efectivo va superpuesto: donde queda fuera de las barras, el mes sorprendió.",
        full_width=True,
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Variaciones del mes por agregado",
        unit="%", chart="grouped_bar", status=STATUS_MVP,
        source_id="ipc_variaciones_mes", transform="snapshot_grouped",
        params={"group": "Agregado", "values": ["Mes anterior", "Mes actual"]},
        no_text=True,
    ),

    # ── Expectativas ─────────────────────────────────────────────────────────
    ReportBlock(
        section=_S_EXPECTATIVAS, title="Incidencias: feedback de mercado vs efectivo",
        unit="pp", chart="hist_range", status=STATUS_MVP,
        source_id="ipc_incidencias_feedback", transform="ipc_rango_categoria",
        params={"actual_label": "Efectivo INE"},
        text_slot="ipc:incidencias",
        note="La caja es el rango mínimo-máximo entre instituciones; el marcador, el dato del INE.",
        full_width=True,
    ),

    # ── Difusión ─────────────────────────────────────────────────────────────
    ReportBlock(
        section=_S_DIFUSION, title="Difusión IPC general · banda histórica",
        unit="%", chart="hist_range", status=STATUS_MVP,
        source_id="ipc_difusion_banda", transform="ipc_rango_categoria",
        params={"actual_label": "Año en curso"},
        text_slot="ipc:difusion",
        note="Porcentaje de productos con variación mensual positiva. Banda 2010-2021.",
        full_width=True,
    ),
    ReportBlock(
        section=_S_DIFUSION, title="Bienes sin volátiles · banda",
        unit="%", chart="hist_range", status=STATUS_MVP,
        source_id="ipc_banda_bienes_sv", transform="ipc_rango_categoria",
        params={"actual_label": "Año en curso"},
        no_text=True, note="Rango p10-p90 de 2010-2020, serie empalmada del BCCh (G073).",
    ),
    ReportBlock(
        section=_S_DIFUSION, title="Servicios sin volátiles · banda",
        unit="%", chart="hist_range", status=STATUS_MVP,
        source_id="ipc_banda_servicios_sv", transform="ipc_rango_categoria",
        params={"actual_label": "Año en curso"},
        no_text=True, note="Rango p10-p90 de 2010-2020, serie empalmada del BCCh (G073).",
    ),
    ReportBlock(
        section=_S_CANASTA, title="Canasta por división, grupo y producto",
        unit="%", chart="treemap", status=STATUS_MVP,
        source_id="ipc_canasta_treemap", transform="ipc_treemap",
        text_slot="ipc:canasta",
        note="El área es la ponderación en la canasta; el color, la variación mensual "
             "(rojo sube, azul baja).",
        full_width=True,
    ),
    ReportBlock(
        section=_S_CANASTA, title="Variación IPC por grupo",
        unit="%", chart="grouped_bar", status=STATUS_MVP,
        source_id="ipc_grupos", transform="wide_monthly_bars",
        no_text=True,
    ),
    ReportBlock(
        section=_S_CANASTA, title="Incidencia 12 meses por división",
        unit="pp", chart="grouped_bar", status=STATUS_MVP,
        source_id="ipc_incidencia_12m_division", transform="snapshot_grouped",
        params={"group": "Nombre", "values": ["Incidencia 12 Meses (%)"]},
        no_text=True,
    ),
    ReportBlock(
        section=_S_CANASTA, title="Variación histórica del mes",
        unit="%", chart="grouped_bar", status=STATUS_MVP,
        source_id="ipc_variacion_historica_mes", transform="snapshot_grouped",
        params={"group": "Año",
                "values": ["Variación Mensual (%)", "Promedio Histórico",
                           "Promedio Histórico Ex"],
                "overlay": ["Promedio Histórico", "Promedio Histórico Ex"]},
        no_text=True,
        note="Los dos puntos son el promedio histórico del mes y el que excluye 2020-2022.",
    ),

    # ── Expectativas y fundamentales ─────────────────────────────────────────
    ReportBlock(
        section=_S_FUNDAMENTALES, title="Fundamentales: Petróleo y Tipo de Cambio",
        unit="CLP/USD", chart="dual_axis", status=STATUS_MVP,
        source_id="ipc_fundamentales", transform="cam_lines",
        params={"right_axis": ["PETRO"], "right_unit": "USD/bbl", "right_style": "line"},
        text_slot="ipc:fundamentales",
    ),
    ReportBlock(
        section=_S_FUNDAMENTALES, title="Variación: Seguros de Inflación",
        unit="pb", chart="grouped_bar", status=STATUS_MVP,
        source_id="ipc_variacion_seguros", transform="cam_signed_bars",
        params={"category": "Tenor", "value": "Variacion_pb",
                "pos_label": "Corrección al alza", "neg_label": "Corrección a la baja"},
        no_text=True,
        note="Cambio entre las dos últimas fechas: hacia dónde movió el mercado cada vencimiento.",
    ),
    ReportBlock(
        section=_S_FUNDAMENTALES, title="Expectativas de Inflación",
        unit="%", chart="line", status=STATUS_MVP,
        source_id="ipc_seguros_inflacion", transform="wide_lines",
        no_text=True,
    ),
    ReportBlock(
        section=_S_FUNDAMENTALES, title="Expectativas de Inflación a largo plazo",
        unit="%", chart="line", status=STATUS_MVP,
        source_id="ipc_compensaciones", transform="wide_lines",
        no_text=True, note="Compensación inflacionaria implícita en swaps a 2, 5 y 10 años.",
    ),

    # ── Detalle ──────────────────────────────────────────────────────────────
    ReportBlock(
        section=_S_DETALLE, title="Canasta del último mes · variación e incidencia por producto",
        unit="%", chart="heatmap_table", status=STATUS_MVP,
        source_id="ipc_canasta_mes", transform="ipc_tabla_canasta",
        params={"top": 25},
        text_slot="ipc:detalle",
        full_width=True,
    ),
)

IPC_SPEC = FamilyReportSpec(
    family="ipc",
    title="Informe Post IPC",
    blocks=_BLOCKS,
    # El IPC es MENSUAL: no hay corte semanal común que anclar. Cada bloque se
    # ancla al máximo de su propio parquet, que es lo correcto cuando conviven
    # series mensuales (la canasta del INE, los analíticos del BCCh) con series
    # diarias (los seguros de inflación y los fundamentales, que se leen al día
    # de la publicación).
    share_weekly_cutoff=False,
)

__all__ = ["IPC_SPEC"]
