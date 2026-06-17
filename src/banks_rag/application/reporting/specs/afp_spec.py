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
    ReportBlock(
        section=_S_ALLOC, title="Traspaso de cotizantes entre fondos",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="movimientos_fondos", transform="straight_series",
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
        section=_S_RF, title="Variación DCV (Δ T-5 / Δ T-20 por instrumento y plazo)",
        unit="US$ Mill.", chart="heatmap_table", status=STATUS_MVP,
        source_id="variacion_stock_afp", transform="dcv_heatmap",
    ),
    ReportBlock(
        section=_S_RF, title="Variación semanal DCV por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="variacion_stock_afp", transform="stacked_by_bucket",
        params={"window": "7d"},
    ),
    ReportBlock(
        section=_S_RF, title="Composición del portafolio DCV por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="variacion_stock_afp", transform="composition_by_bucket",
    ),
    # ── Mercado cambiario ────────────────────────────────────────────────────
    ReportBlock(
        section=_S_FX, title="Flujo cambiario por AFP (spot / forward / neto)",
        unit="US$ Mill.", chart="grouped_bar", status=STATUS_MVP,
        source_id="cambiario_afp", transform="snapshot_grouped",
        params={"group": "Sector_contraparte", "values": ["Spot", "Forward", "Neto"]},
        note="*última semana con datos",
    ),
    ReportBlock(
        section=_S_FX, title="Posición spot y derivados acumulada",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="spot_derivados_afp", transform="straight_series",
    ),
    # ── Derivados de tasas (swaps) ───────────────────────────────────────────
    ReportBlock(
        section=_S_TASAS, title="MtM de swaps por tipo de fondo",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="mtm_afp", transform="straight_series",
    ),
    ReportBlock(
        section=_S_TASAS, title="DV01 proyectado en swap por moneda",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="dv01_spc_afp", transform="straight_series",
    ),
    # ── Atribución de retorno ────────────────────────────────────────────────
    ReportBlock(
        section=_S_ATTR, title="Atribución de retorno por clase de activos (por fondo)",
        unit="%", chart="stacked_bar", status=STATUS_MVP,
        source_id="attribution", transform="snapshot_stacked", scale=100.0,
        params={"x": "fondo", "series": "Clase", "value": "Valor",
                "x_order": ["A", "B", "C", "D", "E"]},
        note="*contribución al retorno por clase de activo, último corte",
    ),
)

AFP_SPEC = FamilyReportSpec(
    family="afp",
    title="Informe AFP",
    blocks=_BLOCKS,
)
