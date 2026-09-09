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

_S_ALLOC = "Allocation"
_S_FLUJOS = "Flujos Y Retornos"
_S_RF = "Renta Fija (DCV)"
_S_FX = "Mercado cambiario"
_S_TASAS = "Derivados de tasas (swaps)"
_S_ATTR = "Atribución de retorno"

_BLOCKS: tuple[ReportBlock, ...] = (

    ReportBlock(
                section=_S_FLUJOS, title="Flujo por fondos (diario)",
                unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                source_id="movimientos_fondos", transform="window_stacked_by_cat",
                params={"category": "Fondo", "value": "Flujos_usd",
                        "order": ["A", "B", "C", "D", "E"], "last_n": 14},
                note="*flujos diarios por tipo de fondo, últimas ~2 semanas",
                no_text=True,  # comentario único en el bloque de variación de arriba
            ),

    ReportBlock(
                        section=_S_FLUJOS, title="Cambio de AUM por Fondos",
                        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
                        source_id="afp_var_aum_fondos",date_from="2026-01-01", transform="wide_daily_diff_ytd",
                        params={"columns": ["A","B","C","D","E",]},
                        note="US $ Mill; YTD",
                        no_text=True,  # comentario único en el bloque de variación de arriba
                    ),


    ReportBlock(
                    section=_S_FLUJOS, title="Retornos acumulados por fondo",
                    unit="%", chart="line", status=STATUS_MVP,
                    source_id="retorno_acum_afp", transform="ytd_return_geom",
                    params={"funds": "all"},  # este gráfico muestra TODOS los fondos disponibles
                    note="*acumulada (geométrica) desde inicios de 2026",
                ),

    # Retorno mensual (ya viene compuesto en el parquet, NO un índice a diferenciar)
    # del consolidado de la industria (AFP="Total"), desagregado por fondo (A-E) →
    # barras agrupadas, X = mes. Réplica de "Retorno de fondos de FP mensuales".
    ReportBlock(
            section=_S_FLUJOS, title="Retorno de fondos de FP mensuales",
            unit="%", chart="grouped_bar", status=STATUS_MVP,
            source_id="retorno_mensual_afp", transform="monthly_bars_by_cat",
            params={"category": "Fondo", "value": "Retorno",
                    "filter_col": "AFP", "filter_val": "Total",
                    "order": ["A", "B", "C", "D", "E"], "months": 10},
        ),

    # Rentabilidad YTD (compuesta geométricamente desde los retornos mensuales) por
        # AFP real (se excluye el consolidado "Total") y fondo → barras agrupadas,
        # X = fondo, una serie por AFP. Réplica de "Retornos por AFP y tipo de fondo".
    ReportBlock(
            section=_S_FLUJOS, title="Retornos por AFP y tipo de fondo",
            unit="%", chart="grouped_bar", status=STATUS_MVP,
            source_id="retorno_mensual_afp", transform="ytd_grouped_by_cat",
            params={"category": "AFP", "group": "Fondo", "value": "Retorno",
                        "exclude": ["Total"], "order": ["A", "B", "C", "D", "E"]},
            note="*YTD, compuesto de los retornos mensuales",
            ),


    ReportBlock(
            section=_S_FLUJOS, title="Stock de fondos por tipo (A-E)",
            unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
            source_id="stock_fondo_afp", transform="straight_series",
        ),
        # Traspaso entre multifondos — VARIACIÓN: barras agrupadas por fondo (A-E) con el
        # flujo acumulado de la última semana y del último mes, lado a lado. Va ARRIBA del
        # gráfico diario para leer la variación semanal/mensual de un vistazo.
        ReportBlock(
            section=_S_FLUJOS, title="Flujo de fondos (acumulado semanal y mensual)",
            unit="US$ Mill.", chart="grouped_bar", status=STATUS_MVP,
            source_id="movimientos_fondos", transform="window_accum_by_cat",
            params={"category": "Fondo", "value": "Flujos_usd",
                    "order": ["A", "B", "C", "D", "E"],
                    "windows": [["Δ T-7", 7], ["Δ T-30", 30]]},
            note="*flujo acumulado por tipo de fondo: última semana vs. último mes",
        ),
        # Vista NETA: las dos ventanas (semanal / mensual) como columnas APILADAS por
        # fondo (mismos colores que el gráfico diario). El alto neto de cada columna es
        # el flujo neto de la ventana; muestra cómo quedaron los fondos entre sí.
        ReportBlock(
            section=_S_FLUJOS, title="Flujo de fondos (neto apilado por fondo)",
            unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
            source_id="movimientos_fondos", transform="window_accum_stacked_by_cat",
            params={"category": "Fondo", "value": "Flujos_usd",
                    "order": ["A", "B", "C", "D", "E"],
                    "windows": [["Δ T-7", 7], ["Δ T-30", 30]]},
            note="*composición del flujo neto por fondo: semanal vs. mensual",
            no_text=True,  # comentario único en el bloque de variación de arriba
        ),


    
    # Traspaso entre multifondos: barra apilada DIVERGENTE por día (flujos diarios
    # por fondo A-E), réplica de "Traspaso de fondos de FP" del tablero.
    
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
                    section=_S_RF, title="Variación diaria del stock DCV de AFP por instrumento",
                    unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                    source_id="stock_nivel_afp", transform="window_stacked_diff_by_cat",
                    params={"category": "Tipo", "value": "Stock_USD",
                            "order": ["BB", "BE", "BTP", "BTU"], "last_n": 14, "net": True},
                    note="*variación diaria (nivel t vs. t-1) por instrumento, últimas ~2 semanas",
                    no_text=True,  # comentario único en el bloque de stock de arriba
                ),

    ReportBlock(
        section=_S_RF, title="Variación semanal DCV por sector",
        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
        source_id="variacion_sector_todos", transform="stacked_by_bucket",
        params={"window": "7d", "net_as_overlay": True,
                "buckets": ["BB", "BE", "BTP", "BTU"]},
        note="*variación de la última semana por plazo; Neto como punto",
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
        source_id="cambiario_afp", transform="window_pivot_grouped",
        params={"group": "Sector_contraparte", "type_col": "Tipo",
                "value": "Monto", "values": ["Spot","Forward","Neto"],
                "overlay": ["Neto"], "window_days": 7,
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

    ReportBlock(
            section=_S_FX, title="Flujos Spot",
            unit="US$ Mill.", chart="line", status=STATUS_MVP,
            source_id="afp_var_acum_spot", transform="category_series",
            params={"category": "AFP", "value": "Monto",
                    "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
        ),

    ReportBlock(
            section=_S_FX, title="Posición derivados", #Identificar porque no aparece plan vital
            unit="US$ Mill.", chart="line", status=STATUS_MVP,
            source_id="afp_var_acum_derivados", transform="category_series",
            params={"category": "AFP", "value": "Monto",
                    "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
        ),
 
    ReportBlock(
                section=_S_FX, title="Posición derivados",
                unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
                source_id="afp_posicion_derivados", transform="category_series",
                params={"category":"Plazo","value":"Posicion",
                       "include": ["1 a 90 dias", "91 a 360 dias", "Mayor a 360 dias"],
                        "net": "auto"},
                        note="",  # + línea Neto = suma de fondos
            ),
    
    # AUM (RFI) queda de ÁREA en el eje izquierdo (natural, 0 a positivo);
    # Posición cambiaria va de LÍNEA en el eje derecho, INVERTIDO (right_invert):
    # la serie es negativa y así se mueve visualmente CON el AUM en vez de en
    # espejo — réplica de "AUM Renta Fija internacional vs posición cambiaria".
    ReportBlock(
            section=_S_FX, title="AUM Renta Fija internacional vs. posición cambiaria",
            unit="Millones de USD", chart="dual_axis", status=STATUS_MVP,
            source_id="afp_aum_posicion_cambiaria", transform="straight_series",
            params={"right_axis": ["Posicion"], "right_unit": "Millones de USD",
                    "right_style": "line", "right_invert": True, "left_style": "area"},
            note="*AUM (RFI) en el eje izquierdo; Posición cambiaria en el eje derecho, invertido",
        ),





    # ── Derivados de tasas (swaps) ───────────────────────────────────────────
    ReportBlock(
        section=_S_TASAS, title="MtM de swaps por tipo de moneda",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="mtm_afp_moneda", transform="category_series",date_from= "2023-01-01",
        params={"category": "Moneda", "value": "MtM",
                "order": ["CLP", "USD"], "net": "False"},  # + línea Neto = suma de fondos
    ),
    ReportBlock(
        section=_S_TASAS, title="DV01 proyectado en swap por moneda",date_from= "2023-01-01",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="dv01_spc_afp", transform="category_series",
        params={"category":"moneda","value":"dv01","order": ["USD","CLP"]},
    ),


    #ReportBlock(
    #        section=_S_TASAS, title="Posición SPC en dólares",
    #        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
    #        source_id="afp_posicion_swap_usd", transform="category_series",date_from= "2023-01-01",
    #        params={"category":"Plazo","value":"mmusd",
    #                "order": ["10Y", "1Y", "2Y", "5Y"], "net": "auto"},
    #        note="*acumulado desde ene.; Neto = spot + derivados",
    #    ),

    ReportBlock(
                section=_S_TASAS, title="Posición SPC nominal con agentes locales",
                unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
                source_id="afp_posicion_spc", transform="category_series",date_from= "2023-01-01",
                params={"category":"Plazo","value":"mmusd",
                            "order": ["1 a 90 dias", "91 a 360 dias", "Entre 1 y 2Y", "Mayor a 2Y"], "net": "auto"},
                note="*acumulado desde ene.; Neto = spot + derivados",
                ),



    ReportBlock(
        section=_S_TASAS, title="Variación acumulada en SPC nominal con agentes locales",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="afp_variacion_spc", transform="wide_daily_diff_ytd",
        params={
            "columns": [
                "1 a 90 dias",
                "91 a 360 dias",
                "Entre 1 y 2Y",
                "Mayor a 2Y",
                "Neto",
            ],
            "overlay": ["Neto"],
        },
        note=(
            "Variación diaria acumulada YTD por tramo de plazo. "
            "Neto se presenta como línea superpuesta."
        ),
    ),



    ReportBlock(
                section=_S_TASAS, title="Var. Mensual Posición AFP en SPC nominal",
                unit="Millones de USD", chart="stacked_bar", status=STATUS_MVP,
                source_id="afp_variacion_spc", transform="wide_monthly_diff_bars",
                params={"include": ["1 a 90 dias", "91 a 360 dias", "Entre 1 y 2Y", "Mayor a 2Y"],
                        "overlay": ["Neto"], "months": 10},
                note="*variación mes a mes por tramo (corte vs. corte previo); Neto como punto",
            ),


    ReportBlock(
                    section=_S_TASAS, title="Posiciones Swap en dólares por tramo de plazo",
                    unit="Millones de USD", chart="stacked_area", status=STATUS_MVP,
                    source_id="afp_posicion_swap_usd", transform="category_series",date_from= "2023-01-01",
                    params={"category":"Plazo","value":"mmusd",
                            "order": ["Menor a 1Y", "Menor a 2Y", "Menor a 3Y", "Menor a 5Y", "Menor a 10Y", "Mayor a 10Y"], "net": "auto"},
    ),




    
 # ── Allocation ──────────────────────────────────────────────
    ReportBlock(
        section=_S_ALLOC, title="Allocation nacional vs. extranjero y patrimonio (AUM)",
        unit="% del AUM", chart="dual_axis", status=STATUS_MVP,
        source_id="allocation_int_nac", transform="straight_series",
        params={"right_axis": ["AUM"], "right_unit": "US$ Mill."},
        note="*AUM en el eje derecho (US$ Mill.); allocation en el izquierdo (%)",
    ),

    ReportBlock(
            section=_S_ALLOC, title="Allocation activos",
            unit="% del AUM", chart="dual_axis", status=STATUS_MVP,
            source_id="allocation_categoria", transform="straight_series",
            params={"right_axis": ["AUM"], "right_unit": "US$ Mill."},
            note="*AUM en el eje derecho (US$ Mill.); allocation por activo en el izquierdo (%)",
        ),

    ReportBlock(
        section=_S_ALLOC,
        title="Allocation activos - Fondo A",
        unit="% del AUM",
        chart="line",
        status=STATUS_MVP,
        source_id="allocation_fondo_categoria",date_from= "2015-01-01",
        transform="allocation_wide_by_fund",
        params={
            "fund": "A",
            "columns": ["RFN", "RVN", "RFI", "RVI", "Otros"],
            "include_aum": False},
        note=(
            "allocation por clase de activo en el eje izquierdo (%).*"
            ),
        ),


    ReportBlock(
                section=_S_ALLOC,
                title="Allocation activos - Fondo B",
                unit="% del AUM",
                chart="line",
                status=STATUS_MVP,
                source_id="allocation_fondo_categoria",date_from= "2015-01-01",
                transform="allocation_wide_by_fund",
                params={
                    "fund": "B",
                    "columns": ["RFN", "RVN", "RFI", "RVI", "Otros"],
                    "include_aum": False},
                note=(
                    "allocation por clase de activo en el eje izquierdo (%).*"
                    ),
                ),
                
    ReportBlock(
                    section=_S_ALLOC,
                    title="Allocation activos - Fondo C",
                    unit="% del AUM",
                    chart="line",
                    status=STATUS_MVP,
                    source_id="allocation_fondo_categoria",date_from= "2015-01-01",
                    transform="allocation_wide_by_fund",
                    params={
                        "fund": "C",
                        "columns": ["RFN", "RVN", "RFI", "RVI", "Otros"],
                        "include_aum": False},
                    note=(
                        "allocation por clase de activo en el eje izquierdo (%).*"
                        ),
                    ),

    ReportBlock(
                        section=_S_ALLOC,
                        title="Allocation activos - Fondo D",
                        unit="% del AUM",
                        chart="line",
                        status=STATUS_MVP,
                        source_id="allocation_fondo_categoria",date_from= "2015-01-01",
                        transform="allocation_wide_by_fund",
                        params={
                            "fund": "D",
                            "columns": ["RFN", "RVN", "RFI", "RVI", "Otros"],
                            "include_aum": False},
                        note=(
                            "allocation por clase de activo en el eje izquierdo (%).*"
                            ),
                        ),


    ReportBlock(
                section=_S_ALLOC,
                title="Allocation activos - Fondo E",
                unit="% del AUM",
                chart="line",
                status=STATUS_MVP,
                source_id="allocation_fondo_categoria",date_from= "2015-01-01",
                transform="allocation_wide_by_fund",
                params={
                                 "fund": "E",
                                "columns": ["RFN", "RVN", "RFI", "RVI", "Otros"],
                                "include_aum": False},
                            note=(
                                "allocation por clase de activo en el eje izquierdo (%).*"
                                ),
                            ),





    # ── Atribución de retorno ────────────────────────────────────────────────
    #ReportBlock(
    #    section=_S_ATTR, title="Atribución de retorno por clase de activos (por fondo)",
    #    unit="%", chart="stacked_bar", status=STATUS_MVP,
    #    source_id="attribution", transform="snapshot_stacked",  # scale 100 ahora en el catálogo
    #    params={"x": "fondo", "series": "Clase", "value": "Valor",
    #            "x_order": ["A", "B", "C", "D", "E"], "total_overlay": True},
    #    note="*contribución al retorno por clase de activo; Total como punto",
    #),
)

AFP_SPEC = FamilyReportSpec(
    family="afp",
    title="Informe AFP",
    blocks=_BLOCKS,
    # Los parquets de AFP cierran en fechas distintas (p.ej. movimientos al 09-06,
    # DCV al 10-06). Sin corte común cada gráfico se ancla al máximo de SU parquet,
    # evitando arrastrar todo a la fecha del parquet con menor fecha.
    share_weekly_cutoff=False,
    # Solo "Allocation y patrimonio" en grilla de 2 columnas (gráfico junto a
    # gráfico, como el informe cambiario): el resto del informe sigue apilado
    # (``layout`` default "stack"). 11 bloques → 5 pares + el último a fila
    # completa (impar), regla automática de ``_wide_block_ids``.
    grid_sections=frozenset({_S_FLUJOS,_S_RF,_S_ALLOC,_S_TASAS,_S_FX}),
)
