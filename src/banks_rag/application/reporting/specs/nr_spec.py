"""Spec curado del "Informe No Residentes" (familia nr).

Réplica FIEL del correo real del BCCh ("Informe No Residentes", remitente DACE)
y de sus tableros (Flujos Derivados / Renta Fija / Posición SPC / Flujos Spot).
Los htmls de referencia viven en ``html_informes/NR_embedded.html``: la narrativa
recorre Mercado Derivados → Renta Fija → SPC → Flujos Spot, y cada mercado muestra
sus gráficos. El patrón gráfico dominante del informe es **apilado divergente**
(positivos arriba / negativos abajo) con una serie **Neto** superpuesta (línea en
series temporales, punto en barras por categoría) — no una línea por serie.

Sólo se incluyen bloques con dato reproducible desde ``data_pipeline/parquet/``;
los gráficos del informe que viven de fuentes externas (atractivo de carry trade
CLP, posicionamiento BNP, índice GBI-EM de JP Morgan, "Chile vs otras economías")
no tienen parquet y se OMITEN a propósito.

Esta es la ÚNICA pieza a editar para ajustar el informe nr (orden, títulos, qué
parquet alimenta cada gráfico y su tipo). Las transforms genéricas
(``category_series``, ``wide_lines``, ``window_grouped``, ``wide_window_bars``,
``accumulated``) viven en ``series_transforms.py``; el renderer de apilados
divergentes + overlay Neto vive en ``svg_chart.py``.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import (
    STATUS_MVP,
    FamilyReportSpec,
    ReportBlock,
)

_S_CAMBIARIO = "Mercado Cambiario"
_S_RF = "Mercado de Renta Fija"
_S_SPC = "Mercado SPC"
_S_EXTRA = "Chile vs otras economías"

_BLOCKS: tuple[ReportBlock, ...] = (
    # ── Mercado de Derivados ─────────────────────────────────────────────────
    # Posición histórica = stock por tramo de plazo apilado + Neto (suma) en línea.
    ReportBlock(
        section=_S_CAMBIARIO, title="Posición cambiaria histórica de no residentes",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="posicion_nr_derivados", transform="wide_lines",
        params={"overlay": ["Neto"]},
        note="*posición por tramo de plazo; Neto = suma de tramos",
    ),


    # Cambio de la semana: barra apilada divergente (Suscripción arriba,
    # Vencimiento RESTA → se negocia → baja) y Neto = Suscripción - Vencimiento
    # como punto. Réplica de "Cambio posición Derivados cambiarios".
    ReportBlock(
        section=_S_CAMBIARIO, title="Cambio posición derivados cambiarios (última semana)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="var_pos_derivados", transform="window_grouped",
        params={"group": "Plazo", "values": ["Suscripcion", "Vencimiento"],
                "window_days": 7, "include_net": True, "net_as_overlay": True,
                
                "order": ["1 a 90 dias", "91 a 360 dias", "Mayor a 360 dias"]},
        note="*suma de la última semana por tramo; Neto = Suscripción - Vencimiento",
    ),
    ReportBlock(
        section=_S_CAMBIARIO, title="Flujos acumulados en derivados por instrumento",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="flujos_acumulados_derivados", transform="category_series",
        params={"filter_col": "Institucion", "filter_val": "Total",
                "category": "Instrumento", "value": "Net",
                "accumulate": "cumsum", "window": "ytd"},  # Net es flujo diario → cumsum = acumulado YtD
        note="*flujo acumulado (suma corrida) del total de agentes, desde ene.",
    ),
    # Variación MENSUAL de la posición en derivados por plazo (flujos diarios →
    # suma por mes); barra apilada divergente + Neto punto.
    ReportBlock(
        section=_S_CAMBIARIO, title="Variación posición derivados mensual",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_posicion_derivados", transform="wide_monthly_bars",
        params={"include": ["1 a 90 dias", "91 a 360 dias", "Mayor a 360 dias"],
                "overlay": ["Neto"], "months": 12},
        note="*suma mensual de la variación por tramo; Neto como punto",
    ),
    # Cambio por TIPO de instrumento (último mes): suscripción (+) / vencimiento (-)
    # apilados + Neto punto. Réplica de "Cambio posición por tipo de derivados".
    ReportBlock(
        section=_S_CAMBIARIO, title="Cambio de posición por tipo de derivados (último mes)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="susc_vcto_agente_instrumento", transform="window_grouped_long",
        params={"group": "Instrumento", "type_col": "Tipo", "value": "Monto",
                "pos": "Suscripción", "neg": "Vencimiento", "window_days": 30},
        note="*suma del último mes por instrumento; Neto = suscripción - vencimiento",
    ),
    # Variación semanal de la posición por AGENTE (top 12): suscripción / vencimiento
    # apilados + Neto punto. Réplica de "Variación posición NR semanal".
    ReportBlock(
        section=_S_CAMBIARIO, title="Variación posición NR semanal NR por agente",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="spot_susc_vcto_agente", transform="window_grouped_long",
        params={"group": "Institucion", "type_col": "Tipo", "value": "Monto",
                "pos": "Suscripción", "neg": "Vencimiento",
                "exclude_types": ["Spot"], "window_days": 7, "top_n": 12},
        note="*12 agentes con mayor variación semanal; Neto = suscripción - vencimiento",
    ),

    ReportBlock(
        section=_S_CAMBIARIO, title="Flujos Spot acumulados NR (afecto / no afecto)",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="flujo_spot_nr", transform="wide_lines",
        params={"include": ["Afecto", "No Afecto"], "accumulate": "cumsum",
                "window": "y2", "net": "auto"},
        note="*flujos acumulados (suma corrida) desde inicios del año anterior",
    ),
    ReportBlock(
        section=_S_CAMBIARIO, title="Flujos Spot acumulados por Afecto y no Afecto",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="spot_acumulado_agente", transform="category_series",
        params={"filter_col": "Institucion", "filter_val": "Total",
                "category": "Afecto_derivado", "value": "Net",
                "accumulate": "cumsum", "window": "ytd"},
        note="*acumulado YTD, Sí: Afecto a derivados y No: No afecto a derivados",
    ),


    # ── Mercado de Renta Fija (BTP en DCV) ───────────────────────────────────


    # OJO: el parquet variacion_rfl_dcv_nr YA viene acumulado YtD (es un nivel que
    # arranca ~489 en ene y llega ~3.750 en jun, no un flujo diario). Se grafica
    # TAL CUAL (straight): hacerle cumsum lo doble-acumulaba a ~277.000. Réplica de
    # "NR: Var. Acu. Stock BTP en DCV (YtD)" del tablero.
    ReportBlock(
        section=_S_RF, title="Variación acumulada Stock BTP en DCV (YtD)",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="variacion_rfl_dcv_nr", transform="straight_series",
    ),


    #ReportBlock(
    #    section=_S_RF, title="Posición de no residentes en BTP por plazo",
    #    unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
    #    source_id="posicion_rfl_nr", transform="straight_series",
    #),
 

    #Filtrar por fecha>="2020-01-01"
    ReportBlock(
        section=_S_RF, title="Cuenta financiera: Tenencia BTP agente NR",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="nr_tenencia_cf_btp", date_from= "2020-01-01", transform="category_series",
        params={"category": "Tramo", "value": "mmusd",
                "order": ["Menor a 2Y", "Entre 2 y 5Y", "Entre 6 y 10Y", "Entre 11 y 20Y", "Mayor a 20Y"]},
        note="*stock por tramo de madurez; área apilada",
        no_text=True,
    ),

    # Cuenta Financiera_ Variacion stock NR Mensual:
    ReportBlock(
        section=_S_RF, title="Cuenta Financiera: Variación stock BTP NR mensual",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_cf_btp", transform="wide_monthly_bars",
        params={"overlay":["Neto"],"months": 18},
        note="*variación mensual por BTP/BTU desde ene-2025 (sin Neto)",
        no_text=True,
    ),


    ReportBlock(
        section=_S_RF, title="Cuenta Financiera: Tenencia agentes NR por tipo de instrumento (BTP / BTU)",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="nr_tenencia_cf_soberanos", transform="category_series",
        params={"category": "Tipo", "value": "mmusd", "order": ["BTP", "BTU"]},
        note="*stock por tipo de bono soberano (BTP nominal / BTU en UF); área apilada",
        no_text=True,
    ),


    # Cuenta Financiera_ Variacion stock NR mensual por instrumento:

    ReportBlock(
        section=_S_RF, title="Cuenta Financiera: Variación stock NR mensual por instrumento",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_cf_soberanos", transform="wide_monthly_bars",
        params={"overlay":["Neto"],"months": 18},
        note="*variación mensual por instrumento desde ene-2025 (sin Neto)",
        no_text=True,
    ),


    # ── Mercado SPC (tasas) ──────────────────────────────────────────────────
    # Parquet en US$ mil millones → scale 1000 para mostrar en US$ Mill. (como el
    # informe). Apilado DIVERGENTE: lo que pagan fija va arriba, lo que reciben
    # fija va abajo; el Neto (suma) cruza el cero como línea.
    # OJO: el informe enviado OMITE el tramo corto "1 a 90 dias" en este gráfico
    # (es enorme y negativo, ~-34.000, y descuadra el eje). Sin él, el Neto de los
    # 3 tramos largos da ~+9.000 como en el correo. Replicamos esa vista.
    ReportBlock(
        section=_S_SPC, title="Variación acumulada de no residentes en SPC nominal",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="variacion_nr_spc", transform="wide_lines",
        params={"overlay": ["Neto"], "accumulate": "rebase"},
        note="*tramos ≥90d; pagan fija (+) / reciben fija (-); Neto = suma de TODOS los tramos",
    ),

    ReportBlock(
        section=_S_SPC, title="Variación posición NR en SPC YTD (todos los tramos)",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="variacion_nr_spc", transform="wide_lines",
        params={"overlay": ["Neto"], "window": "ytd", "accumulate": "rebase"},
        note="*variación acumulada desde ene; incluye el tramo 1 a 90 días; Neto = suma de tramos",
        no_text=True,
    ),


    ReportBlock(
        section=_S_SPC, title="Variación diaria posición NR en SPC nominal (últimas semanas)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_posicion_spc", transform="wide_window_bars",
        params={"overlay": ["Neto"], "last_n": 15},
        note="*variación diaria por tramo, últimos ~15 días hábiles; Neto como punto",
        no_text=True,
    ),

    ReportBlock(
        section=_S_SPC, title="Variación mensual posición NR en SPC nominal por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_posicion_spc", transform="wide_monthly_bars",
        params={"include": ["1 a 90 dias", "91 a 360 dias", "Entre 1 y 2Y", "Mayor a 2Y"],
                "months": 18},
        note="*suma mensual de la variación por tramo desde ene-2025 (sin Neto)",
        no_text=True,
    ),




    ReportBlock(
        section=_S_EXTRA, title="Tasas 10Y GBI",
        unit="puntos base", chart="hist_range", status=STATUS_MVP,
        source_id="gbi_10y_tasas_hist", transform="gbi_rendimiento_range_sin_base",
        params={"order": ["USD | AA-", "CZK | AA-", "CLP | A", "PLN | A-", "PEN | BBB+",
                          "HUF | BBB", "MXN | BBB-", "COP | BB+", "BRL | BB-"],
                "window": "ytd"},
        note="*En puntos base; caja = rango histórico de la ventana, "
             "línea = promedio, punto = valor de hoy",
        no_text=True,
    ),

    ReportBlock(
        section=_S_EXTRA, title="Spread de tasas vs UST10Y",
        unit="puntos base", chart="hist_range", status=STATUS_MVP,
        source_id="gbi_10y_spread_hist", transform="gbi_rendimiento_range_sin_base",
        params={"order": ["USD | AA-", "CZK | AA-", "CLP | A", "PLN | A-", "PEN | BBB+",
                          "HUF | BBB", "MXN | BBB-", "COP | BB+", "BRL | BB-"],
                "window": "ytd"},
        note="*En puntos base; caja = rango histórico de la ventana, "
             "línea = promedio, punto = valor de hoy",
        no_text=True,
    ),
    ReportBlock(
        section=_S_EXTRA, title="Rendimiento monedas comparables (índice, base 100 = inicio de año)",
        unit="Índice (base 100)", chart="hist_range", status=STATUS_MVP,
        source_id="gbi_index", transform="gbi_rendimiento_range",
        params={"order": ["USD  |  AA", "CZK  |  AA", "CLP  |  A", "PLN  |  A-", "PEN  |  BBB+",
                          "HUF  |  BBB", "MXN  |  BBB-", "COP  |  BB+", "BRL  |  BB-"],
                "window": "ytd"},
        note="*índice base 100 YTD, Arriba DEPRECIACIÓN - Abajo APRECIACIÓN, caja = rango histórico de la ventana, "
             "línea = promedio, punto = valor de hoy",
        no_text=True,
    ),
    ReportBlock(
        section=_S_EXTRA, title="Retorno acumulado FX vs. tasas GBI — monedas comparables (YtD)",
        unit="%", chart="point", status=STATUS_MVP,
        source_id="retorno_fx_tasas", transform="fx_tasas_scatter",
        params={"window": "ytd", "highlight": "CLP","fx_prefix":"fx_","rate_prefix":"rates_",
                "x_label": "Retorno FX (%, + = apreciación)", "y_label": "Retorno tasas (%)"},
        note="*retorno acumulado desde inicio de año (suma de variaciones %/ diarias); "
             "punto relleno = país doméstico",
        no_text=True,
    ),


)

#["BRL  |  BB-", "CLP  |  A", "COP  |  BB+", "CZK  |  AA", "HUF  |  BBB", "MXN  |  BBB-", "PEN  |  BBB+", "PLN  |  A-", ]

NR_SPEC = FamilyReportSpec(
    family="nr",
    title="Informe No Residentes",
    blocks=_BLOCKS,
    # Los parquets NR cierran en fechas distintas. Sin corte común, el TEXTO (y los
    # gráficos) usan el máximo de SU propio parquet, no la fecha del de menor fecha.
    share_weekly_cutoff=False,
)
