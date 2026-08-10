"""Spec curado del "Informe Cambiario AM" (familia cambiarioam).

Réplica del dashboard que hoy arma el DOMA con Plotly + Jinja
(``Nuevo Informe/paquete_cambiario/``), traído al flujo estándar de informes
curados: mismos gráficos, mismas tablas y MISMO ORDEN, pero dibujados con el
renderer SVG del repo y sin su plantilla HTML propia. El orden de los bloques y
sus títulos salen de ``nav_groups`` del original, sección por sección:

  Drivers del día        tabla de mercado + 3 gráficos + tabla de la vela   (6)
  CLP · Análisis         7 gráficos + tabla de percentiles                  (8)
  No Residentes          4 gráficos + tabla de percentiles                  (5)
  Monedas & Carry        6 gráficos                                         (6)
  Cobre                  6 gráficos + tabla de percentiles                  (7)
  Dólar · DXY            3 gráficos + tabla de percentiles                  (4)
  Expectativas           5 gráficos                                         (5)
  Otros                  6 gráficos                                         (6)

Dos diferencias deliberadas con el original, ambas por consistencia con el resto
de los informes del repo:

  - las tablas que el dashboard cuelga DEBAJO de un gráfico (percentiles S/R,
    máximo/mínimo de la vela) son acá un bloque propio ``no_text`` justo después
    del gráfico, porque un bloque = una pieza; y
  - los paneles de monedas base 100 no llevan los botones 1M/3M/6M/1A/YTD: la
    ventana la fija el spec (``months``), que es la vista por defecto del
    original (dos meses).

**Los datos todavía no están en ``data_pipeline/parquet/``.** Los 41 parquets los
genera ``scripts/build_cambiario_parquets.py`` desde el Excel "Datos BI Informe
Cambiario.xlsm" y siete consultas al DW; sus entradas ya viven en
``sql_catalog/parquet_catalog.yaml`` (segmento ``cambiarioam``). Mientras no se
copien, cada bloque sale como tarjeta "sin serie en el parquet" en su posición —
la estructura del informe se ve completa igual. Cuando lleguen, correr
``scripts/refresh_catalog_dates.py`` y revisar que las COLUMNAS reales coincidan
con las declaradas en el catálogo: si el Excel cambió un encabezado, lo que hay
que ajustar es el ``mapping`` del script de extracción, no este spec.

Esta es la ÚNICA pieza a editar para reordenar o retitular el informe. Las
transforms ``cam_*`` viven en ``series_transforms.py`` y el render en
``svg_chart.py`` (la vela OHLC es ``_render_candlestick``).
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import (
    STATUS_MVP,
    FamilyReportSpec,
    ReportBlock,
)

# Secciones: los ``label`` de nav_groups del dashboard original, en su orden.
_S_RESUMEN = "Drivers del día"
_S_CLP = "CLP · Análisis"
_S_NR = "No Residentes"
_S_MONEDAS = "Monedas & Carry"
_S_COBRE = "Cobre"
_S_DXY = "Dólar · DXY"
_S_EXPECTATIVAS = "Expectativas"
_S_OTROS = "Otros"

# Ventana de los paneles de monedas base 100: la vista por defecto del original
# (``default_window_start`` = máximo - 2 meses).
_BASE100_MESES = 2

# Ventana de los gráficos de soporte/resistencia: ``_last_year`` del original.
_SR_MESES = 12


def _sr_pair(
    section: str, title: str, source_id: str, unit: str, *, note: str = "",
) -> tuple[ReportBlock, ...]:
    """Gráfico de soportes/resistencias + su tabla de percentiles.

    Los dos bloques comparten ``source_id`` y ventana, así el nivel que dice la
    tabla es exactamente el que dibuja la línea. La tabla va ``no_text``: el
    comentario del corte se escribe una sola vez, en el gráfico."""
    return (
        ReportBlock(
            section=section, title=title, unit=unit, chart="line", status=STATUS_MVP,
            source_id=source_id, transform="cam_sr_percentiles",
            params={"months": _SR_MESES}, note=note,
        ),
        ReportBlock(
            section=section, title=f"{title} — niveles", unit=unit,
            chart="heatmap_table", status=STATUS_MVP,
            source_id=source_id, transform="cam_sr_table",
            params={"months": _SR_MESES}, no_text=True,
        ),
    )


def _base100(title: str, source_id: str, *, full_width: bool = False) -> ReportBlock:
    return ReportBlock(
        section=_S_MONEDAS, title=title, unit="Índice base 100", chart="line",
        status=STATUS_MVP, source_id=source_id, transform="cam_base100",
        params={"months": _BASE100_MESES}, full_width=full_width,
    )


def _curve(title: str, source_id: str, unit: str, series_label: str) -> ReportBlock:
    return ReportBlock(
        section=_S_EXPECTATIVAS, title=title, unit=unit, chart="line", status=STATUS_MVP,
        source_id=source_id, transform="cam_contract_curve",
        params={"series_label": series_label},
    )


# Tabla "Snapshot de Mercado · Drivers" de la portada. ``clp`` es el signo con
# que cada driver mueve al peso, tal como lo declara ``_DRIVER_CLP`` del original:
# cobre y diferencial de tasas al alza APRECIAN el peso; dólar global y petróleo
# lo DEPRECIAN; la punta forward se informa sin dirección.
_SNAPSHOT_INDICADORES = [
    {"column": "Cobre", "label": "Cobre", "unit": "Comex · ¢/lb", "decimals": 2,
     "mode": "pct", "clp": +1},
    {"column": "DXY", "label": "DXY", "unit": "Índice dólar", "decimals": 3,
     "mode": "pct", "clp": -1},
    {"column": "SPC 1Y", "minus": "OIS 1Y", "factor": 100.0, "label": "Dif. SPC–OIS",
     "unit": "Spread tasa · pb", "decimals": 1, "mode": "bp", "clp": +1},
    {"column": "Petróleo", "label": "Petróleo", "unit": "Brent · USD/bbl", "decimals": 2,
     "mode": "pct", "clp": -1},
    {"column": "Punta FWD 1M", "label": "Pto. Fwd 30D", "unit": "CLP · puntos",
     "decimals": 2, "mode": "pts", "clp": 0},
]


_BLOCKS: tuple[ReportBlock, ...] = (
    # ── Resumen · Drivers del día ────────────────────────────────────────────
    ReportBlock(
        section=_S_RESUMEN, title="Snapshot de Mercado · Drivers",
        unit="niveles", chart="heatmap_table", status=STATUS_MVP,
        source_id="cam_drivers_snapshot", transform="cam_market_table",
        params={"indicators": _SNAPSHOT_INDICADORES},
        full_width=True,
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Variación Monedas en el día",
        unit="%", chart="stacked_bar", status=STATUS_MVP,
        source_id="cam_monedas_variacion_dia", transform="cam_signed_bars",
        params={"category": "Pais", "value": "Variacion"},
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Monedas Intradía",
        unit="Índice base 100", chart="line", status=STATUS_MVP,
        source_id="cam_clp_intradia", transform="cam_lines",
        params={"keep_time": True},
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Evolución tipo de cambio",
        unit="CLP/USD", chart="candlestick", status=STATUS_MVP,
        source_id="cam_clp_ohlc", transform="cam_candlestick",
        params={"months": 3},
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Evolución tipo de cambio — resumen de la sesión",
        unit="CLP/USD", chart="heatmap_table", status=STATUS_MVP,
        source_id="cam_clp_ohlc", transform="cam_candle_table",
        params={"months": 3}, no_text=True,
    ),

    # ── Tipo de Cambio · CLP - Análisis ──────────────────────────────────────
    ReportBlock(
        section=_S_CLP, title="Medias Móviles",
        unit="CLP/USD", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_clp_medias_moviles", transform="cam_moving_averages",
        # Histórico COMPLETO (desde 2019), como el dashboard original: la MA200
        # necesita recorrido para que el cruce con la MA50 se vea en contexto.
        params={"base": "CLP Cierre", "right": "Monto transado (MM5d)",
                "right_axis": ["Monto transado (MM5d)"], "right_unit": "MM USD"},
        note="*MA50 en oro y MA200 en rojo (cruce dorado/de la muerte); el resto "
             "de las medias móviles queda en gris de solo contexto.",
    ),
    ReportBlock(
        section=_S_CLP, title="Bollinger CLP",
        unit="CLP/USD", chart="line", status=STATUS_MVP,
        source_id="cam_clp_bollinger", transform="cam_bollinger",
        params={"period": 20, "mult": 2.0, "months": 12},
    ),
    ReportBlock(
        section=_S_CLP, title="RSI CLP",
        unit="RSI", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_rsi_clp", transform="cam_lines",
        params={"right_axis": ["RSI CLP"], "right_unit": "RSI", "right_style": "line"},
    ),
    *_sr_pair(_S_CLP, "S/R CLP", "cam_clp_sr", "CLP/USD"),
    ReportBlock(
        section=_S_CLP, title="Volatilidad CLP",
        unit="CLP/USD", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_clp_volatilidad", transform="cam_lines",
        params={"right_axis": ["Volatilidad"], "right_unit": "%", "right_style": "line"},
    ),
    ReportBlock(
        section=_S_CLP, title="Fixing de la banca",
        unit="MM USD", chart="stacked_bar", status=STATUS_MVP,
        source_id="cam_fixing_bancos", transform="cam_fixing_stacked",
        params={"agent": "Informante", "sector": "Sector", "value": "Pos_neta"},
    ),
    ReportBlock(
        section=_S_CLP, title="Histograma CLP",
        unit="Frecuencia (días)", chart="grouped_bar", status=STATUS_MVP,
        source_id="cam_clp_distribucion", transform="cam_histogram",
        params={"x": "Tramo CLP", "value": "Frecuencia", "series_label": "Días en el tramo"},
    ),

    # ── No Residentes ────────────────────────────────────────────────────────
    ReportBlock(
        section=_S_NR, title="Derivados",
        unit="CLP/USD", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_pos_no_residentes", transform="cam_lines",
        # El CLP tiene historia desde 2019; la posición de no residentes recién
        # desde 2022 — sin "align_from" el CLP dibujaría 3 años de fondo con la
        # otra serie vacía, como si el gráfico comparara un período que en
        # realidad solo tiene una serie.
        params={"right_axis": ["CLPBODM Index"], "right_unit": "MM USD", "right_style": "line",
                "align_from": ["CLPBODM Index"]},
        full_width=True,
    ),
    ReportBlock(
        section=_S_NR, title="Vol 1W",
        unit="%", chart="line", status=STATUS_MVP,
        source_id="cam_vol_implicita_1w", transform="cam_lines",
    ),
    ReportBlock(
        section=_S_NR, title="Punta forward",
        unit="CLP/USD", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_fwd_clp", transform="cam_lines",
        params={"right_axis": ["Punta FWD 1M"], "right_unit": "Puntos", "right_style": "line"},
    ),
    *_sr_pair(_S_NR, "S/R NR", "cam_pos_nr_sr", "MM USD",
              note="*Signo invertido respecto del dato crudo: un valor más alto es "
                   "posición offshore más larga en dólares."),

    # ── Monedas & Carry ──────────────────────────────────────────────────────
    _base100("Monedas LATAM", "cam_monedas_latam", full_width=True),
    _base100("Monedas Emergentes", "cam_monedas_emergentes"),
    _base100("Monedas Commodities", "cam_monedas_comparables"),
    _base100("Monedas G10", "cam_monedas_g10"),
    ReportBlock(
        section=_S_MONEDAS, title="Monedas vs USD 1 mes",
        unit="%", chart="stacked_bar", status=STATUS_MVP,
        source_id="cam_monedas_variacion_30d", transform="cam_signed_bars",
        params={"category": "Moneda", "value": "Variacion 30d", "ascending": True,
                "pos_label": "Depreciación vs USD", "neg_label": "Apreciación vs USD"},
        note="*Variación de la paridad contra el dólar: positiva = la moneda se deprecia.",
    ),
    ReportBlock(
        section=_S_MONEDAS, title="Carry trade",
        unit="%", chart="stacked_bar", status=STATUS_MVP,
        source_id="cam_carry_trade", transform="cam_signed_bars",
        params={"category": "Pais", "value": "Carry",
                "pos_label": "Carry positivo", "neg_label": "Carry negativo"},
    ),

    # ── Drivers · Cobre ──────────────────────────────────────────────────────
    ReportBlock(
        section=_S_COBRE, title="Cobre vs CLP",
        unit="USD/lb", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_cobre_clp", transform="cam_lines",
        params={"right_axis": ["CLP Cierre"], "right_unit": "CLP/USD", "right_style": "line"},
        full_width=True,
    ),
    *_sr_pair(_S_COBRE, "S/R Cobre", "cam_cobre_sr", "USD/lb"),
    ReportBlock(
        section=_S_COBRE, title="RSI Cobre",
        unit="RSI", chart="line", status=STATUS_MVP,
        source_id="cam_rsi_cobre", transform="cam_rsi_bands",
        params={"levels": [70, 30]},
    ),
    ReportBlock(
        section=_S_COBRE, title="Inventarios COMEX",
        unit="USD/lb", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_inventarios_comex", transform="cam_lines",
        # Ejes invertidos a propósito: el original pinta el inventario como BARRAS
        # y el precio como línea. El renderer no hace barras temporales en doble
        # eje, así que el inventario va al eje DERECHO como área —el equivalente
        # visual más cercano— y el precio queda de línea a la izquierda.
        params={"right_axis": ["Inventarios COMEX"], "right_unit": "Toneladas", "left_unit": "USD/lb"},
    ),
    ReportBlock(
        section=_S_COBRE, title="Inventarios Londres",
        unit="USD/lb", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_inventarios_lme", transform="cam_lines",
        # Mismo criterio que Inventarios COMEX (ver arriba).
        params={"right_axis": ["Inventarios LME"], "right_unit": "Toneladas", "left_unit": "USD/lb"},
    ),
    ReportBlock(
        section=_S_COBRE, title="Spread COMEX-LME",
        unit="USD/lb", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_cobre_comex_lme", transform="cam_lines",
        params={"right_axis": ["Spread COMEX–Londres"], "right_unit": "USD/lb", "right_style": "line"},
    ),

    # ── Drivers · Dólar - DXY ────────────────────────────────────────────────
    ReportBlock(
        section=_S_DXY, title="DXY vs CLP",
        unit="Índice", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_dxy_clp", transform="cam_lines",
        params={"right_axis": ["CLP Cierre"], "right_unit": "CLP/USD", "right_style": "line"},
        full_width=True,
    ),
    *_sr_pair(_S_DXY, "S/R DXY", "cam_dxy_sr", "Índice"),
    ReportBlock(
        section=_S_DXY, title="RSI DXY",
        unit="RSI", chart="line", status=STATUS_MVP,
        source_id="cam_rsi_dxy", transform="cam_rsi_bands",
        params={"levels": [70, 30]},
    ),

    # ── Expectativas ─────────────────────────────────────────────────────────
    ReportBlock(
        section=_S_EXPECTATIVAS, title="Expectativas de TPM US - CL",
        unit="%", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_tpm_spread", transform="cam_spread_expanding",
        params={"minuend": "TPM Chile 1Y", "subtrahend": "TPM US 1Y", "scale": 100.0,
                "right_axis": ["Spread CL–US (pb)", "Promedio histórico"],
                "right_unit": "pb", "right_style": "line",
                "spread_label": "Spread CL–US (pb)"},
    ),
    ReportBlock(
        section=_S_EXPECTATIVAS, title="Puntas FWD Chile",
        unit="Puntos", chart="line", status=STATUS_MVP,
        source_id="cam_puntos_forward", transform="cam_lines",
        params={"months": 12},
    ),
    _curve("Curva de contratos: Cobre COMEX", "cam_curva_hga", "¢/lb", "HGA · COMEX"),
    _curve("Curva de contratos: Cobre Londres", "cam_curva_lma", "USD/t", "LMA · Londres"),
    _curve("Curva de contratos: Petróleo", "cam_curva_cl1", "USD/bbl", "CL1 · WTI"),

    # ── Otros ────────────────────────────────────────────────────────────────
    ReportBlock(
        section=_S_OTROS, title="Gamma Proxy",
        unit="MM USD", chart="grouped_bar", status=STATUS_MVP,
        source_id="cam_gamma_proxy", transform="cam_histogram",
        params={"x": "Strike", "value": "Gamma", "series_label": "Gamma proxy"},
    ),
    ReportBlock(
        section=_S_OTROS, title="Heatmap Gamma Proxy",
        unit="MM USD", chart="heatmap_table", status=STATUS_MVP,
        source_id="cam_gamma_heatmap", transform="cam_gamma_heatmap",
        params={"row": "Strike", "value": "Gamma", "row_title": "Strike"},
        # SVG (no tabla HTML): se auto-ajusta al ancho de su tarjeta como
        # cualquier otro gráfico, así comparte fila con "Gamma Proxy" pese a
        # tener ~12 columnas — igual que el ``go.Heatmap`` interactivo del
        # dashboard original, pero sin JS.
        note="*verde = mayor concentración de gamma.",
    ),
    ReportBlock(
        section=_S_OTROS, title="Tasas implícitas",
        unit="%", chart="line", status=STATUS_MVP,
        source_id="cam_tasas_implicitas", transform="cam_lines",
        params={"months": 12},
    ),
    ReportBlock(
        section=_S_OTROS, title="Terms Of Trade GS",
        unit="Índice", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_tot_gs", transform="cam_lines",
        params={"right_axis": ["CLP Cierre"], "right_unit": "CLP/USD", "right_style": "line"},
    ),
    ReportBlock(
        section=_S_OTROS, title="Terms Of Trade CT",
        unit="Índice", chart="dual_axis", status=STATUS_MVP,
        source_id="cam_tot_ct", transform="cam_lines",
        params={"right_axis": ["CLP Cierre"], "right_unit": "CLP/USD", "right_style": "line"},
    ),
    ReportBlock(
        section=_S_OTROS, title="Tipo de Cambio Real",
        unit="Índice", chart="line", status=STATUS_MVP,
        source_id="cam_tcr", transform="cam_lines",
    ),
)

CAMBIARIOAM_SPEC = FamilyReportSpec(
    family="cambiarioam",
    title="Informe Cambiario AM",
    blocks=_BLOCKS,
    # El corte común existe SOLO para anclar el fixing a la fecha del informe.
    # ``cam_fixing_bancos`` se abre por fecha de FIXING (el vencimiento del
    # forward), que vive en el FUTURO: su máximo son los forwards a dos meses,
    # no la sesión que describe el resto del informe. Anclando la sección al mín
    # de los máximos de sus datasets de alta frecuencia (todos los del CLP, que
    # cierran en la última sesión), la transform toma el fixing de ESE día.
    #
    # Solo ``cam_fixing_stacked`` lee ``weekly_asof``; el resto de las transforms
    # ``cam_*`` lo ignora y sigue anclado al máximo de su propio parquet, que es
    # lo correcto en un informe de apertura donde el intradía llega al minuto y
    # los términos de intercambio son semanales.
    weekly_anchor_sections=(_S_CLP,),
    # Único informe con layout "grid": réplica del dashboard 2 columnas del
    # original (gráfico junto a gráfico, tabla de percentiles pegada bajo SU
    # gráfico). El resto de las familias no lo declara y sigue en "stack".
    layout="grid",
)
