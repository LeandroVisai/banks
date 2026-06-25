"""Spec curado del "Informe AFP" (familia afp).

Réplica del informe real del BCCh (tableros Allocation / Renta Fija / Mercado
cambiario / Derivados de tasas / Traspasos y rentabilidad): los gráficos que se
pueden reproducir desde los parquets locales, en el ORDEN narrativo del informe y
agrupados por tema. Cada bloque mapea a un parquet + una transform + un tipo de
gráfico objetivo.

Sólo se incluyen bloques con dato reproducible; los gráficos del informe que viven
de fuentes externas o de series que no tenemos (retornos por AFP, ranking de
lugares, AUM por AFP, posición AFP en SPC con agentes locales) se OMITEN a
propósito.

Esta es la ÚNICA pieza a editar para ajustar el informe afp. Las transforms
genéricas (``snapshot_grouped``, ``snapshot_stacked``, ``latest_snapshot``,
``category_series``, …) viven en ``series_transforms.py``.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import (
    STATUS_MVP,
    FamilyReportSpec,
    ReportBlock,
)

_S_ALLOC = "Allocation y patrimonio"
_S_RF = "Renta Fija (DCV)"
_S_FX = "Mercado cambiario"
_S_TASAS = "Derivados de tasas (swaps)"
_S_ATTR = "Atribución de retorno"

_BLOCKS: tuple[ReportBlock, ...] = (
    # ── Allocation y patrimonio ──────────────────────────────────────────────
    ReportBlock(
        section=_S_ALLOC, title="Allocation nacional vs. extranjero y patrimonio (AUM)",
        unit="% del AUM", chart="dual_axis", status=STATUS_MVP,
        source_id="allocation_int_nac", transform="straight_series",
        params={"right_axis": ["AUM"], "right_unit": "US$ Mill."},
        note="*AUM en el eje derecho (US$ Mill.); allocation en el izquierdo (%)",
    ),
    ReportBlock(
        section=_S_ALLOC, title="Stock de fondos por tipo (A-E)",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="stock_fondo_afp", transform="straight_series",
    ),
    # Traspaso entre multifondos — VARIACIÓN: barras agrupadas por fondo (A-E) con el
    # flujo acumulado de la última semana y del último mes, lado a lado. Va ARRIBA del
    # gráfico diario para leer la variación semanal/mensual de un vistazo.
    ReportBlock(
        section=_S_ALLOC, title="Traspaso de fondos entre multifondos (acumulado semanal y mensual)",
        unit="US$ Mill.", chart="grouped_bar", status=STATUS_MVP,
        source_id="movimientos_fondos", transform="window_accum_by_cat",
        params={"category": "fondo", "value": "flujos_usd",
                "order": ["A", "B", "C", "D", "E"],
                "windows": [["Δ T-7", 7], ["Δ T-30", 30]]},
        note="*flujo acumulado por tipo de fondo: última semana vs. último mes",
    ),
    # Vista NETA: las dos ventanas (semanal / mensual) como columnas APILADAS por
    # fondo (mismos colores que el gráfico diario). El alto neto de cada columna es
    # el flujo neto de la ventana; muestra cómo quedaron los fondos entre sí.
    ReportBlock(
        section=_S_ALLOC, title="Traspaso de fondos entre multifondos (neto apilado por fondo)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="movimientos_fondos", transform="window_accum_stacked_by_cat",
        params={"category": "fondo", "value": "flujos_usd",
                "order": ["A", "B", "C", "D", "E"],
                "windows": [["Δ T-7", 7], ["Δ T-30", 30]]},
        note="*composición del flujo neto por fondo: semanal vs. mensual",
        no_text=True,  # comentario único en el bloque de variación de arriba
    ),
    # Traspaso entre multifondos: barra apilada DIVERGENTE por día (flujos diarios
    # por fondo A-E), réplica de "Traspaso de fondos de FP" del tablero.
    ReportBlock(
        section=_S_ALLOC, title="Traspaso de fondos entre multifondos (diario)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="movimientos_fondos", transform="window_stacked_by_cat",
        params={"category": "fondo", "value": "flujos_usd",
                "order": ["A", "B", "C", "D", "E"], "last_n": 14},
        note="*flujos diarios por tipo de fondo, últimas ~2 semanas",
        no_text=True,  # comentario único en el bloque de variación de arriba
    ),
    # ── Renta Fija (DCV) ─────────────────────────────────────────────────────
    ReportBlock(
        section=_S_RF, title="Stock DCV de AFP por instrumento",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="stock_nivel_afp", transform="straight_series",
    ),
    ReportBlock(
        section=_S_RF, title="Stock DCV de AFP por tramo de plazo",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="stock_bucket_afp", transform="straight_series",
    ),
    ReportBlock(
        section=_S_RF, title="Composición del portafolio DCV",
        unit="US$ Mill.", chart="pie", status=STATUS_MVP,
        source_id="stock_nivel_afp", transform="latest_snapshot",
        params={"category": "Tipo", "value": "Stock_USD"},
        note="*composición al último día con datos",
    ),
    ReportBlock(
        section=_S_RF, title="Variación DCV (Δ T-7 / Δ T-30 por instrumento y plazo)",
        unit="US$ Mill.", chart="heatmap_table", status=STATUS_MVP,
        source_id="variacion_stock_afp", transform="dcv_heatmap",
    ),
    ReportBlock(
        section=_S_RF, title="Variación semanal DCV por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="variacion_stock_afp", transform="stacked_by_bucket",
        params={"window": "7d", "net_as_overlay": True},
        note="*variación de la última semana por plazo; Neto como punto",
    ),
    ReportBlock(
        section=_S_RF, title="Composición del portafolio DCV por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="variacion_stock_afp", transform="composition_by_bucket",
    ),
    # ── Mercado cambiario ────────────────────────────────────────────────────
    # Flujo de la semana por AFP: barra apilada (Spot + Forward) + Neto (=Spot+Forward)
    # como punto, réplica de "Flujos cambiarios" del tablero.
    ReportBlock(
        section=_S_FX, title="Flujo cambiario por AFP (spot + forward)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="cambiario_afp", transform="snapshot_grouped",
        params={"group": "Sector_contraparte", "values": ["Spot", "Forward", "Neto"],
                "overlay": ["Neto"],
                "order": ["Habitat", "Provida", "Uno", "Cuprum", "Capital", "Modelo", "Planvital"]},
        note="*última semana; Neto = Spot + Forward como punto",
    ),
    # Posición acumulada spot vs derivados: área apilada DIVERGENTE (spot compra +,
    # derivados venden -) + Neto en línea. Flujos diarios → cumsum desde ene.
    ReportBlock(
        section=_S_FX, title="Posición spot y derivados acumulada",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="spot_derivados_afp", transform="wide_lines",
        params={"include": ["Spot", "Derivados"], "accumulate": "cumsum",
                "window": "ytd", "net": "auto"},
        note="*acumulado desde ene.; Neto = spot + derivados",
    ),
    # ── Derivados de tasas (swaps) ───────────────────────────────────────────
    ReportBlock(
        section=_S_TASAS, title="MtM de swaps por tipo de fondo",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="mtm_afp", transform="category_series",
        params={"category": "fondo", "value": "mtm",
                "order": ["A", "B", "C", "D", "E"], "net": "auto"},  # + línea Neto = suma de fondos
    ),
    ReportBlock(
        section=_S_TASAS, title="DV01 proyectado en swap por moneda",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="dv01_spc_afp", transform="filter_fund",
        params={"funds": ["CLP", "US$"]},  # se excluye UF a pedido
    ),
    # ── Atribución de retorno ────────────────────────────────────────────────
    ReportBlock(
        section=_S_ATTR, title="Atribución de retorno por clase de activos (por fondo)",
        unit="%", chart="stacked_bar", status=STATUS_MVP,
        source_id="attribution", transform="snapshot_stacked", scale=100.0,
        params={"x": "fondo", "series": "Clase", "value": "Valor",
                "x_order": ["A", "B", "C", "D", "E"], "total_overlay": True},
        note="*contribución al retorno por clase de activo; Total como punto",
    ),
)

AFP_SPEC = FamilyReportSpec(
    family="afp",
    title="Informe AFP",
    blocks=_BLOCKS,
)
