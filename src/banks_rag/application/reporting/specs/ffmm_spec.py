"""Spec curado del "Informe Fondos Mutuos" (familia ffmm).

Réplica de las capturas 1-11 del informe real: 31 bloques en 8 secciones, en el
ORDEN del informe. Cada bloque mapea a un parquet + una transformación + un tipo
de gráfico objetivo. Las marcas de estado (MVP/EXP/SKIP) las dibuja o difiere el
builder; ver el plan y ``report_spec`` para el significado.

Esta es la ÚNICA pieza a editar para ajustar el informe ffmm (orden, títulos,
qué parquet alimenta cada gráfico). Las demás familias = otro archivo análogo.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import (
    STATUS_EXP,
    STATUS_MVP,
    STATUS_SKIP,
    FamilyReportSpec,
    ReportBlock,
)

_S_FLUJOS = "Flujos y Retornos"
_S_CARTERAS = "Carteras DCV"
_S_AUM = "AUM"
_S_PORTAFOLIO = "Portafolio DCV"
_S_RENT = "Rentabilidad"
_S_ALLOC = "Allocation Carteras mensuales"
_S_DUR = "Duración Carteras mensuales"
_S_VARALLOC = "Variación en allocation carteras mensuales"
_S_FX = "Mercado cambiario"
_S_SUBASTAS = "Subastas y Vencimientos PDBC"

_BLOCKS: tuple[ReportBlock, ...] = (
    # ── Flujos y Retornos (imagen 1) ─────────────────────────────────────────
    ReportBlock(
        section=_S_FLUJOS, title="Variación Patrimonio efectivo por tipo de fondo",
        unit="US$ Mill.", chart="grouped_bar", status=STATUS_EXP,
        source_id="flujos_ffmm", transform="monthly_sum_by_fund",
        params={"funds": ["Tipo 2", "Tipo 3", "Tipo 6"], "months": 6},
    ),
    ReportBlock(
        section=_S_FLUJOS, title="Rentabilidad por tipo de fondo (Δ7d / Δ30d)",
        unit="%", chart="grouped_bar", status=STATUS_EXP,
        source_id="retornos_fondo_ffmm", transform="window_returns",
        params={"windows": ["7d", "30d"]},
    ),
    ReportBlock(  # acumulado (cumsum por fondo)
        section=_S_FLUJOS, title="Flujos acumulados por fondo",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="flujos_acum_ffmm", transform="accumulated", params={"window": "y2", "accumulate": "cumsum"},
    ),
    ReportBlock(
        section=_S_FLUJOS, title="Flujos diarios por tipo de fondo",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="flujos_fondo_ffmm", transform="straight_series",
    ),
    # ── Carteras DCV (imagen 2) ──────────────────────────────────────────────
    ReportBlock(
        section=_S_CARTERAS, title="Variación Acumulada YtD — BTP, BTU, BB y Otros",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="stock_nivel_ffmm", transform="accumulated",
        params={"types": ["BTP", "BTU", "BB", "Otros"], "window": "ytd", "accumulate": "rebase"},
    ),
    # ── AUM (imágenes 2-3) — sin parquet de patrimonio/cuotas → SKIP ─────────
    ReportBlock(
        section=_S_AUM, title="Variación Patrimonio neto (7d / 30d / YTD)",
        unit="US$ Mill.", chart="grouped_bar", status=STATUS_SKIP,
        note="*Datos hasta el 8/6 · *Data corresponde a fondos en CLP",
        text_slot="ffmm:aum",
    ),
    ReportBlock(
        section=_S_AUM, title="Variación cuotas en circulación (7d / 30d / YTD)",
        unit="Mill. cuotas", chart="grouped_bar", status=STATUS_SKIP,
    ),
    ReportBlock(
        section=_S_AUM, title="Patrimonio Efectivo",
        unit="US$ Mill.", chart="line", status=STATUS_SKIP,
    ),
    ReportBlock(
        section=_S_AUM, title="Variación acumulada Patrimonio Efectivo",
        unit="US$ Mill.", chart="line", status=STATUS_SKIP,
    ),
    # ── Portafolio DCV (imágenes 4-5) ────────────────────────────────────────
    ReportBlock(
        section=_S_PORTAFOLIO, title="Fechas de corte DCV (T, T-5, T-20)",
        chart="heatmap_table", status=STATUS_EXP,
        source_id="stock_nivel_ffmm", transform="dcv_cut_dates",
    ),
    ReportBlock(
        section=_S_PORTAFOLIO, title="Variación DCV (Δ T-5 / Δ T-20 por instrumento y plazo)",
        unit="US$ Mill.", chart="heatmap_table", status=STATUS_EXP,
        source_id="variacion_stock_ffmm", transform="dcv_heatmap",
    ),
    ReportBlock(
        section=_S_PORTAFOLIO, title="Variación semanal DCV por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_EXP,
        source_id="variacion_stock_ffmm", transform="stacked_by_bucket",
        params={"window": "7d"},
    ),
    ReportBlock(
        section=_S_PORTAFOLIO, title="Variación Acumulada YtD — DAP y PDBC",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="dap_pdbc_ffmm", transform="accumulated", params={"window": "ytd", "accumulate": "rebase"},
    ),
    ReportBlock(
        section=_S_PORTAFOLIO, title="Variación Acumulada YtD — BTP, BTU, BB y Otros",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="stock_nivel_ffmm", transform="accumulated",
        params={"types": ["BTP", "BTU", "BB", "Otros"], "window": "ytd", "accumulate": "rebase"},
    ),
    ReportBlock(
        section=_S_PORTAFOLIO, title="Composición Portafolio DCV por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_EXP,
        source_id="variacion_stock_ffmm", transform="composition_by_bucket",
    ),
    ReportBlock(  # ── MVP · torta ──
        section=_S_PORTAFOLIO, title="Composición Portafolio DCV",
        unit="%", chart="pie", status=STATUS_MVP,
        source_id="dcv_composicion_ffmm", transform="snapshot_composition",
    ),
    # ── Rentabilidad (imágenes 6-7) ──────────────────────────────────────────
    ReportBlock(
        section=_S_RENT, title="Rentabilidad por tipo de fondo (Δ7d / Δ30d / ΔYtD)",
        unit="%", chart="grouped_bar", status=STATUS_EXP,
        source_id="retornos_fondo_ffmm", transform="window_returns",
        params={"windows": ["7d", "30d", "ytd"]},
        note="*Datos hasta el 8/6",
    ),
    ReportBlock(
        section=_S_RENT, title="Rentabilidad mensual por tipo de fondo",
        unit="%", chart="grouped_bar", status=STATUS_EXP,
        source_id="retornos_fondo_ffmm", transform="monthly_returns",
        params={"funds": ["Tipo 1", "Tipo 2", "Tipo 3"], "months": 8},
    ),
    ReportBlock(  # ── MVP ──
        section=_S_RENT, title="Rentabilidad por tipo de fondo YTD",
        unit="%", chart="line", status=STATUS_MVP,
        source_id="retorno_acum_ffmm", transform="straight_series",
    ),
    # ── Allocation Carteras mensuales (imágenes 7-8) ─────────────────────────
    *[
        ReportBlock(
            section=_S_ALLOC, title=f"Allocation {fund}",
            unit="Mill US$.", chart="stacked_area", status=STATUS_EXP,
            source_id="var_cartera_mensual_ffmm", transform="allocation_by_fund",
            params={"fund": fund},
            note="*datos con carteras al cierre de Abril · *solo instrumentos locales"
            if fund == "Tipo 1" else "",
        )
        for fund in ("Tipo 1", "Tipo 2", "Tipo 3", "Tipo 6")
    ],
    ReportBlock(
        section=_S_ALLOC, title="Cartera mensual por instrumento (variante tipo_f)",
        unit="Mill US$.", chart="stacked_area", status=STATUS_EXP,
        source_id="var_cartera_mensual_ffmm_tipo_f", transform="straight_series",
    ),
    # ── Duración Carteras mensuales (imagen 8) ───────────────────────────────
    ReportBlock(  # ── MVP ──
        section=_S_DUR, title="Duración por tipo de fondo (T1 / T2)",
        unit="Años", chart="line", status=STATUS_MVP,
        source_id="duracion_ffmm", transform="filter_fund",
        params={"funds": ["Tipo 1", "Tipo 2"]},
        note="*datos con carteras al cierre de Abril",
    ),
    ReportBlock(  # ── MVP ──
        section=_S_DUR, title="Duración por tipo de fondo (T3 / T6)",
        unit="Años", chart="line", status=STATUS_MVP,
        source_id="duracion_ffmm", transform="filter_fund",
        params={"funds": ["Tipo 3", "Tipo 6"]},
    ),
    ReportBlock(  # ── MVP ──
        section=_S_DUR, title="DV01 por fondo",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="dv01_ffmm", transform="straight_series",
    ),
    # ── Variación en allocation carteras mensuales (imágenes 9-10) ───────────
    *[
        ReportBlock(
            section=_S_VARALLOC, title=f"Variación Mes y YtD — {fund}",
            unit="US$ Mill.", chart="grouped_bar", status=STATUS_EXP,
            source_id="var_cartera_mensual_ffmm", transform="monthly_var_alloc",
            params={"fund": fund},
            note="*datos con carteras al cierre de Abril" if fund == "Tipo 1" else "",
        )
        for fund in ("Tipo 1", "Tipo 2", "Tipo 3", "Tipo 6")
    ],
    # ── Mercado cambiario (imagen 10) ────────────────────────────────────────
    ReportBlock(  # ── MVP · acumulado (cumsum) ──
        section=_S_FX, title="Flujos spot acumulados",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="flujos_spot_ffmm", transform="accumulated", params={"window": "y2", "accumulate": "cumsum"},
        note="*Datos hasta el 08-jun",
    ),
    ReportBlock(
        section=_S_FX, title="Flujos spot mensuales",
        unit="US$ Mill.", chart="bar_time", status=STATUS_EXP,
        source_id="flujos_spot_ffmm", transform="monthly_diff",
    ),
    ReportBlock(
        section=_S_FX, title="Posición Spot y Derivados — FFMM",
        unit="MM USD", chart="line", status=STATUS_MVP,
        source_id="posicion_spot_derivados_ffmm", transform="straight_series",
    ),
    # ── Subastas y Vencimientos PDBC (imagen 11) — sin parquet → SKIP ────────
    ReportBlock(
        section=_S_SUBASTAS, title="Subastas PDBC (Vencimiento / Subasta / Ratio)",
        unit="US$ Mill.", chart="dual_line", status=STATUS_SKIP,
        text_slot="ffmm:subastas",
    ),
)

FFMM_SPEC = FamilyReportSpec(
    family="ffmm",
    title="Informe Fondos Mutuos",
    blocks=_BLOCKS,
)
