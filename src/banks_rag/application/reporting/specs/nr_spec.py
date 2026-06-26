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

_S_DERIV = "Mercado de Derivados"
_S_RF = "Mercado de Renta Fija"
_S_SPC = "Mercado SPC (tasas)"
_S_SPOT = "Flujos Spot"

_BLOCKS: tuple[ReportBlock, ...] = (
    # ── Mercado de Derivados ─────────────────────────────────────────────────
    # Posición histórica = stock por tramo de plazo apilado + Neto (suma) en línea.
    ReportBlock(
        section=_S_DERIV, title="Posición cambiaria histórica de no residentes",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="posicion_nr_derivados", transform="category_series",
        params={"category": "Plazo", "value": "Valor", "net": "auto",
                "order": ["1 a 90 dias", "91 a 360 dias", "Mayor a 360 dias"]},
        note="*posición por tramo de plazo; Neto = suma de tramos",
    ),
    # Cambio de la semana: barra apilada divergente (Suscripción arriba,
    # Vencimiento RESTA → se negocia → baja) y Neto = Suscripción - Vencimiento
    # como punto. Réplica de "Cambio posición Derivados cambiarios".
    ReportBlock(
        section=_S_DERIV, title="Cambio en la posición de derivados cambiarios (última semana)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="var_pos_derivados", transform="window_grouped",
        params={"group": "Plazo", "values": ["Suscripcion", "Vencimiento"],
                "window_days": 7, "include_net": True, "net_as_overlay": True,
                "negate": ["Vencimiento"],
                "order": ["1 a 90 días", "91 a 360 días", "Mayor a 360 días"]},
        note="*suma de la última semana por tramo; Neto = Suscripción - Vencimiento",
    ),
    ReportBlock(
        section=_S_DERIV, title="Flujos acumulados en derivados por instrumento",
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
        section=_S_DERIV, title="Variación mensual de la posición NR en derivados",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_posicion_derivados", transform="wide_monthly_bars",
        params={"include": ["1 a 90 días", "91 a 360 días", "Mayor a 360 días"],
                "overlay": ["Neto"], "months": 12},
        note="*suma mensual de la variación por tramo; Neto como punto",
    ),
    # Cambio por TIPO de instrumento (último mes): suscripción (+) / vencimiento (-)
    # apilados + Neto punto. Réplica de "Cambio posición por tipo de derivados".
    ReportBlock(
        section=_S_DERIV, title="Cambio de posición en derivados por instrumento (último mes)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="susc_vcto_agente_instrumento", transform="window_grouped_long",
        params={"group": "Instrumento", "type_col": "Tipo", "value": "Monto",
                "pos": "Suscripción", "neg": "Vencimiento", "window_days": 30},
        note="*suma del último mes por instrumento; Neto = suscripción - vencimiento",
    ),
    # Variación semanal de la posición por AGENTE (top 12): suscripción / vencimiento
    # apilados + Neto punto. Réplica de "Variación posición NR semanal".
    ReportBlock(
        section=_S_DERIV, title="Variación semanal de la posición NR por agente",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="spot_susc_vcto_agente", transform="window_grouped_long",
        params={"group": "Institucion", "type_col": "Tipo", "value": "Monto",
                "pos": "Suscripción", "neg": "Vencimiento",
                "exclude_types": ["Spot"], "window_days": 7, "top_n": 12},
        note="*12 agentes con mayor variación semanal; Neto = suscripción - vencimiento",
    ),
    # ── Mercado de Renta Fija (BTP en DCV) ───────────────────────────────────
    ReportBlock(
        section=_S_RF, title="Posición de no residentes en BTP por plazo",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="posicion_rfl_nr", transform="straight_series",
    ),
    # OJO: el parquet variacion_rfl_dcv_nr YA viene acumulado YtD (es un nivel que
    # arranca ~489 en ene y llega ~3.750 en jun, no un flujo diario). Se grafica
    # TAL CUAL (straight): hacerle cumsum lo doble-acumulaba a ~277.000. Réplica de
    # "NR: Var. Acu. Stock BTP en DCV (YtD)" del tablero.
    ReportBlock(
        section=_S_RF, title="Variación acumulada del stock de BTP en DCV (YtD)",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="variacion_rfl_dcv_nr", transform="straight_series",
    ),
    # ── Mercado SPC (tasas) ──────────────────────────────────────────────────
    # Parquet en US$ mil millones → scale 1000 para mostrar en US$ Mill. (como el
    # informe). Apilado DIVERGENTE: lo que pagan fija va arriba, lo que reciben
    # fija va abajo; el Neto (suma) cruza el cero como línea.
    # OJO: el informe enviado OMITE el tramo corto "1 a 90 dias" en este gráfico
    # (es enorme y negativo, ~-34.000, y descuadra el eje). Sin él, el Neto de los
    # 3 tramos largos da ~+9.000 como en el correo. Replicamos esa vista.
    ReportBlock(
        section=_S_SPC, title="Posición de no residentes en SPC nominal",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="posicion_nr_spc", transform="category_series",  # scale 1000 ahora en el catálogo (chart Y texto)
        params={"category": "Plazos_D", "value": "Monto_USD", "net": "auto",
                "order": ["91 a 360 dias", "Entre 1 y 2Y", "Mayor a 2Y"]},
        note="*tramos ≥90d; pagan fija (+) / reciben fija (-); Neto = suma de tramos",
    ),
    # Variación acumulada YtD = nivel rebaseado al inicio del año (el parquet trae
    # niveles); apilado divergente + Neto en línea.
    ReportBlock(
        section=_S_SPC, title="Variación acumulada de la posición NR en SPC (YtD)",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="nr_var_posicion_spc", transform="wide_lines",  # scale 1000 ahora en el catálogo
        params={"overlay": ["Neto"], "window": "ytd", "accumulate": "rebase"},
    ),
    # Variación DIARIA = diferencia día-a-día del nivel (últimos días); barra
    # apilada divergente por día + Neto como punto.
    ReportBlock(
        section=_S_SPC, title="Variación diaria de la posición NR en SPC por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_posicion_spc", transform="wide_window_bars",  # scale 1000 ahora en el catálogo
        params={"overlay": ["Neto"], "last_n": 6, "diff": True},
        note="*variación diaria (Δ día anterior) de los últimos días; Neto como punto",
    ),
    # ── Flujos Spot ──────────────────────────────────────────────────────────
    # Flujos diarios → cumsum (≈2 años) para leer el flujo ACUMULADO; apilado
    # afecto/no afecto + Neto (suma) en línea.
    ReportBlock(
        section=_S_SPOT, title="Flujos spot acumulados de no residentes (afecto / no afecto)",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="flujo_spot_nr", transform="wide_lines",
        params={"include": ["Afecto", "No Afecto"], "accumulate": "cumsum",
                "window": "y2", "net": "auto"},
        note="*flujos acumulados (suma corrida) desde inicios del año anterior",
    ),
    ReportBlock(
        section=_S_SPOT, title="Flujos spot acumulados por agente (total)",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="spot_acumulado_agente", transform="category_series",
        params={"filter_col": "Institucion", "filter_val": "Total",
                "category": "Afecto_derivado", "value": "Net",
                "accumulate": "cumsum", "window": "ytd"},
        note="*acumulado YtD del total de agentes",
    ),
)

NR_SPEC = FamilyReportSpec(
    family="nr",
    title="Informe No Residentes",
    blocks=_BLOCKS,
    # Los parquets NR cierran en fechas distintas. Sin corte común, el TEXTO (y los
    # gráficos) usan el máximo de SU propio parquet, no la fecha del de menor fecha.
    share_weekly_cutoff=False,
)
