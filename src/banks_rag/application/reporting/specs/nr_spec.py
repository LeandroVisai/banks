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
_S_EXTRA = "Gráficos Extras"

_BLOCKS: tuple[ReportBlock, ...] = (
    # ── Mercado de Derivados ─────────────────────────────────────────────────
    # Posición histórica = stock por tramo de plazo apilado + Neto (suma) en línea.
    # OJO: el parquet es ANCHO (una columna por tramo + Neto), NO largo. Con
    # ``category_series`` (category="Plazo") DuckDB no encuentra la columna y el
    # bloque caía a placeholder en silencio.
    ReportBlock(
        section=_S_DERIV, title="Posición cambiaria histórica de no residentes",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="posicion_nr_derivados", transform="wide_lines",
        params={"include": ["1 a 90 dias", "91 a 360 dias", "Mayor a 360 dias"],
                "overlay": ["Neto"]},
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
    # "posicion_nr_spc" (long: Plazos_D/Monto_USD) se descontinuó en el tablero;
    # "nr_var_posicion_spc" (wide: un tramo por columna + Neto) es su reemplazo
    # — mismo dataset que alimenta los dos bloques siguientes. A diferencia del
    # descontinuado, ningún tramo domina el eje (magnitudes parejas entre
    # tramos), así que ya no hace falta excluir "1 a 90 dias" para que el
    # gráfico sea legible: se incluyen los 4 tramos + Neto (overlay), igual que
    # en los bloques de abajo.
    ReportBlock(
        section=_S_SPC, title="Posición de no residentes en SPC nominal",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="nr_var_posicion_spc", transform="wide_lines",  # scale 1000 ahora en el catálogo
        params={"overlay": ["Neto"]},
        note="*pagan fija (+) / reciben fija (-); Neto = suma de tramos",
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
    # ── Gráficos Extras ──────────────────────────────────────────────────────
    # Réplica de los gráficos Plotly del analista (capturas), inferidos SÓLO del
    # código — no del parquet. Cada bloque reproduce un pipeline pandas→px con la
    # transform declarativa equivalente. Son vistas de APÉNDICE: dibujan el gráfico
    # sin párrafo del LLM (``no_text``). Tres fuentes (``variacion_nr_spc``,
    # ``nr_tenencia_cf_btp``, ``nr_tenencia_cf_soberanos``) aún NO están en el
    # catálogo/parquet: hasta que se agreguen (con su esquema y unidad reales)
    # el builder las muestra como tarjeta placeholder, no rompe el informe.
    #
    # Captura 1 — "Var posicion NR en SPC Nominal":
    #   read_parquet(variacion_nr_spc).set_index(Fecha).diff()
    #     .query("Fecha>='2026-01-01'").cumsum().drop(columns=Neto)
    #     .pipe(px.bar).add_scatter(y=Neto, line black)
    #   diff().cumsum() es una identidad telescópica == nivel - nivel_base, i.e. la
    #   VARIACIÓN ACUMULADA YtD (rebase al inicio de año). Barras diarias densas
    #   (~140) por tramo + Neto en línea == área apilada divergente + Neto overlay.
    #   → wide_lines(accumulate=rebase, window=ytd, overlay=[Neto]) sobre stacked_area.
    ReportBlock(
        section=_S_EXTRA, title="Variación acumulada YtD de la posición NR en SPC nominal (todos los tramos)",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="variacion_nr_spc", transform="wide_lines",
        params={"overlay": ["Neto"], "window": "ytd", "accumulate": "rebase"},
        note="*variación acumulada desde ene (diff·cumsum ≡ rebase); incluye el tramo 1 a 90 días; Neto = suma de tramos",
        no_text=True,
    ),
    # Captura 2 — "Var posicion NR en SPC Nominal Diaria":
    #   read_parquet(nr_var_posicion_spc).query("Fecha>='2026-06-18'")
    #     .set_index(Fecha).pipe(px.bar)
    #   Barras apiladas divergentes por DÍA de los valores TAL CUAL (sin diff),
    #   últimas ~3 semanas hábiles. Reproducimos "los últimos N días" con last_n
    #   (una fecha fija se desactualiza al regenerar el informe). Neto va como punto
    #   superpuesto (patrón Neto del informe), no como 5ª barra apilada.
    #   → wide_window_bars(last_n≈15, sin diff, overlay=[Neto]) sobre stacked_bar.
    ReportBlock(
        section=_S_EXTRA, title="Variación diaria de la posición NR en SPC nominal (últimas semanas)",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_posicion_spc", transform="wide_window_bars",
        params={"overlay": ["Neto"], "last_n": 15},
        note="*variación diaria por tramo, últimos ~15 días hábiles; Neto como punto",
        no_text=True,
    ),
    # Captura 3 — "Var posicion Mensual":
    #   read_parquet(nr_var_posicion_spc).query("Fecha>='2025-01-01'")
    #     .groupby(Grouper(Fecha, freq=ME)).sum().drop(columns=Neto).pipe(px.bar)
    #   Suma mensual por tramo desde ene-2025 (~18 meses), SIN Neto (se dropea, no
    #   hay overlay). → wide_monthly_bars(months=18, include=4 tramos) sobre
    #   stacked_bar. El ``include`` omite Neto; si el parquet local no trae aún el
    #   tramo "1 a 90 dias", la transform lo salta (usa las value_cols reales).
    ReportBlock(
        section=_S_EXTRA, title="Variación mensual de la posición NR en SPC nominal por plazo",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="nr_var_posicion_spc", transform="wide_monthly_bars",
        params={"include": ["1 a 90 dias", "91 a 360 dias", "Entre 1 y 2Y", "Mayor a 2Y"],
                "months": 18},
        note="*suma mensual de la variación por tramo desde ene-2025 (sin Neto)",
        no_text=True,
    ),
    # Captura 4 — "Agregar en Renta Fija no residente" (BTP por tramo):
    #   read_parquet(nr_tenencia_cf_btp).pivot(index=Fecha, columns=Tramo,
    #     values=mmusd).pipe(px.area)
    #   Parquet LARGO (Fecha, Tramo, mmusd) → pivot → área apilada (todo positivo,
    #   sin Neto). → category_series(category=Tramo, value=mmusd) sobre stacked_area.
    #   ``order`` en madurez ascendente (más limpio que el orden por defecto de px).
    ReportBlock(
        section=_S_EXTRA, title="Tenencia NR en renta fija soberana (BTP) por tramo de plazo",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="nr_tenencia_cf_btp", transform="category_series",
        params={"category": "Tramo", "value": "mmusd",
                "order": ["Menor a 2Y", "Entre 2 y 5Y", "Entre 6 y 10Y", "Entre 11 y 20Y", "Mayor a 20Y"]},
        note="*stock por tramo de madurez; área apilada",
        no_text=True,
    ),
    # Captura 5 — "Agregar en Renta Fija no residente" (soberanos por tipo):
    #   read_parquet(nr_tenencia_cf_soberanos).pivot(index=Fecha, columns=Tipo,
    #     values=mmusd).pipe(px.area)
    #   Parquet LARGO (Fecha, Tipo∈{BTP,BTU}, mmusd) → pivot → área apilada.
    #   → category_series(category=Tipo, value=mmusd) sobre stacked_area.
    ReportBlock(
        section=_S_EXTRA, title="Tenencia NR en renta fija soberana por tipo (BTP / BTU)",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="nr_tenencia_cf_soberanos", transform="category_series",
        params={"category": "Tipo", "value": "mmusd", "order": ["BTP", "BTU"]},
        note="*stock por tipo de bono soberano (BTP nominal / BTU en UF); área apilada",
        no_text=True,
    ),
    # Captura 6 — "Rendimiento monedas" (tablero de comparables GBI):
    #   read_parquet(gbi_index).sort_values(Fecha).query("Fecha>'2026-01-01'")
    #     .assign(Base_100=100*Valor/groupby(Tenor).Valor.transform('first'),
    #             Minimo=groupby(Tenor).Base_100.transform('min'),
    #             Maximo=groupby(Tenor).Base_100.transform('max'),
    #             Mean=groupby(Tenor).Base_100.transform('mean'))
    #   → última fila por Tenor: caja [Mínimo,Máximo] ("Rango histórico") + tick
    #   Promedio + punto "Hoy" (Base_100) etiquetado con su valor, eje X = Tenor
    #   ("País | Rating" — es un parquet LARGO Fecha/Tenor/Valor, "Tenor" es el
    #   nombre real de la columna categórica en el parquet fotografiado, no un
    #   plazo). NINGÚN gráfico existente dibuja esta forma (caja de rango + 2
    #   overlays por categoría) → tipo de gráfico NUEVO ``hist_range`` (kind
    #   "range", ver ``svg_chart._render_range_band``) + transform NUEVA
    #   ``gbi_rendimiento_range`` (rebase a 100 = ratio, no resta — distinto de
    #   ``_accumulate(mode="rebase")`` que ya usa el resto del informe).
    #   ``order``: por calidad crediticia DESCENDENTE (los comparables más sólidos
    #   primero — USD/CZK, más cerca de Chile — y los de mayor riesgo al final).
    #   Orden pedido explícitamente por el analista (prioriza legibilidad: lo más
    #   relevante para comparar contra CLP entra primero, sin desplazarse).
    ReportBlock(
        section=_S_EXTRA, title="Rendimiento monedas comparables (índice, base 100 = inicio de año)",
        unit="Índice (base 100)", chart="hist_range", status=STATUS_MVP,
        source_id="gbi_index", transform="gbi_rendimiento_range",
        params={"order": ["USD | AA", "CZK | AA", "CLP | A", "PLN | A-", "PEN | BBB+",
                          "HUF | BBB", "MXN | BBB-", "COP | BB+", "BRL | BB-"],
                "window": "ytd"},
        note="*índice rebasado a 100 al inicio del año; caja = rango histórico de la ventana, "
             "línea = promedio, punto = valor de hoy",
        no_text=True,
    ),
    # Captura 7 — "Retorno FX y tasas GBI" (mismo tablero de comparables):
    #   data = read_parquet(retorno_fx_Tasas).set_index(Fecha).pct_change()
    #     .query("Fecha>'2026-01-01'").cumsum()
    #   fx = data.unstack()...query('Pais.str.contains("fx")')...assign(retorno_fx=-1*...)
    #   rates = data.unstack()...query('Pais.str.contains("tasas")')
    #   rates.merge(fx, on='Pais').pipe(px.scatter, x='retorno_fx', y='retorno_tasas', text='Pais')
    #   → snapshot al último dato: dispersión x=retorno FX acumulado YtD (%, signo
    #   invertido: nivel FX es moneda-local-por-USD, así que negar da la
    #   convención "positivo = apreciación", igual que el tablero), y=retorno tasas
    #   acumulado YtD (%) — un punto por país, un país destacado ("hoy"/doméstico).
    #   Parquet ANCHO con columnas PREFIJADAS ``fx_{PAIS}``/``rates_{PAIS}`` (país en
    #   MAYÚSCULA, niveles) — confirmado contra el catálogo real (id
    #   ``retorno_fx_tasas``, chart_type ``fx_retorno_scatter_interactive``, unit
    #   "Porcentaje", 15 países en ambos lados: CLP/COP/MXN/PEN/BRL/CNY/MYR/THB/
    #   IDR/CZK/PLN/TRY/ZAR/HUF/USD). Tampoco existe este tipo de gráfico
    #   (dispersión x/y etiquetada con cruce en 0) → tipo NUEVO ``point`` (kind
    #   "scatter", ver ``svg_chart._render_scatter_labeled``) + transform NUEVA
    #   ``fx_tasas_scatter`` (retorno = SUMA de variaciones % diarias desde el
    #   inicio de año, aditivo — réplica fiel de ``pct_change().cumsum()`` del
    #   tablero, no la fórmula compuesta/geométrica de las otras transforms de
    #   rentabilidad de este módulo). ``fx_retorno_scatter_interactive`` → familia
    #   "point" mapeado en ``domain/agent/chart_types.py`` (no estaba).
    #   ``highlight="clp"``: CLP está confirmado entre los 15 países del parquet;
    #   es el país doméstico del informe, el único candidato razonable para el
    #   punto destacado ("hoy") de la captura.
    ReportBlock(
        section=_S_EXTRA, title="Retorno acumulado FX vs. tasas — monedas comparables (YtD)",
        unit="%", chart="point", status=STATUS_MVP,
        source_id="retorno_fx_tasas", transform="fx_tasas_scatter",
        params={"window": "ytd", "highlight": "clp", "fx_prefix": "fx_", "rate_prefix": "rates_",
                "x_label": "Retorno FX (%, + = apreciación)", "y_label": "Retorno tasas (%)"},
        note="*retorno acumulado desde inicio de año (suma de variaciones % diarias); "
             "punto relleno = país doméstico",
        no_text=True,
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
