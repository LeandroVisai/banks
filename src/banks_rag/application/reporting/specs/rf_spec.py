"""Spec curado del "Informe Renta Fija" (familia rf).

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

_S_INVERSIONES = "Inversiones Globales"
_S_CURVAS = "Variacióne Curvas"
_S_FLUJOS = "Flujos BTP-BTU"
_S_FLUJOS_BB_BC = "Flujos Bonos Bancarios y Corporativos"
_S_INTERMEDIACION = "Intermediación Financiera"
_S_VARAGENTEDCV = "Variación acumulada DCV por Agente"
_S_CURVASHISTORICAS = "Variación acumulada DCV por Agente"



_BLOCKS: tuple[ReportBlock, ...] = (

    ReportBlock(
            section=_S_INVERSIONES, title="Portafolio por agente",
            unit= "US$ Mill.", chart="heatmap_table", status=STATUS_MVP,
            source_id="variacion_instrumento_todos_plazo", transform="dcv_portfolio_table",
            note="*Duración: promedio ponderado por monto en la fila/columna Total.",
        ),



    ReportBlock(
                section=_S_CURVAS, title="Curvas BTP %", #Identificar porque no aparece plan vital
                unit="US$ Mill.", chart="line", status=STATUS_MVP,
                source_id="btp_curva", transform="",
                params={"category": "AFP", "value": "Monto",
                        "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
            ),

    ReportBlock(
                section=_S_CURVAS, title="Curvas BTU %", #Identificar porque no aparece plan vital
                unit="US$ Mill.", chart="line", status=STATUS_MVP,
                source_id="btu_curva", transform="",
                params={"category": "AFP", "value": "Monto",
                        "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
            ),

    ReportBlock(
                    section=_S_CURVAS, title="Breakeven inflación %", #Identificar porque no aparece plan vital
                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                    source_id="bei_curve", transform="",
                    params={"category": "AFP", "value": "Monto",
                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                ),


    ReportBlock(
                    section=_S_CURVAS, title="Variación Semanal Curvas CLP y US (pb)", #Identificar porque no aparece plan vital
                    unit="US$ Mill.", chart="groupbed_bar", status=STATUS_MVP,
                    source_id="", transform="",
                    params={"category": "AFP", "value": "Monto",
                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                ),

    ReportBlock(
                    section=_S_CURVAS, title="Spread BTP-UST por plazo (pb)", #Identificar porque no aparece plan vital
                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                    source_id="curva_spread_btp_ust", transform="",
                    params={"category": "AFP", "value": "Monto",
                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                ),


    ReportBlock(
                    section=_S_CURVAS, title="Swap Spread BTP pb", #Identificar porque no aparece plan vital
                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                    source_id="Curva spread BTP-SPC", transform="",
                    params={"category": "AFP", "value": "Monto",
                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                ),


    ReportBlock(
                    section=_S_CURVAS, title="Swap Spread BTU pb", #Identificar porque no aparece plan vital
                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                    source_id="", transform="",
                    params={"category": "AFP", "value": "Monto",
                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                ),


    ReportBlock(
                    section=_S_CURVAS, title="Curva Swap vs BTP %", #Identificar porque no aparece plan vital
                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                    source_id="spc_curve-btp_curva", transform="",
                    params={"category": "AFP", "value": "Monto",
                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                ),


    ReportBlock(
                        section=_S_FLUJOS, title="Flujos BTP Variación T-5 (USD Mill.)", #Identificar porque no aparece plan vital
                        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                        source_id="", transform="",
                        params={"category": "AFP", "value": "Monto",
                                "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                    ),

    ReportBlock(
                        section=_S_FLUJOS, title="Flujos BTU Variación T-5 (USD Mill.)", #Identificar porque no aparece plan vital
                        unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                        source_id="", transform="",
                        params={"category": "AFP", "value": "Monto",
                                "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                    ),

    ReportBlock(
                            section=_S_FLUJOS_BB_BC, title="Flujos BB Variación T-5 (USD Mill.)", #Identificar porque no aparece plan vital
                            unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                            source_id="", transform="",
                            params={"category": "AFP", "value": "Monto",
                                    "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                        ),
    
    ReportBlock(
                            section=_S_FLUJOS_BB_BC, title="Flujos BC Variación T-5 (USD Mill.)", #Identificar porque no aparece plan vital
                            unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                            source_id="", transform="",
                            params={"category": "AFP", "value": "Monto",
                                    "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                        ),


    ReportBlock(
                                section=_S_INTERMEDIACION, title="Flujos PDBC Variación T-5 (USD Mill.)", #Identificar porque no aparece plan vital
                                unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                                source_id="", transform="",
                                params={"category": "AFP", "value": "Monto",
                                        "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                            ),
    ReportBlock(
                                section=_S_INTERMEDIACION, title="Flujos DAP CLP Variación T-5 (USD Mill.)", #Identificar porque no aparece plan vital
                                    unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                                    source_id="", transform="",
                                    params={"category": "AFP", "value": "Monto",
                                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                ),
    ReportBlock(
                                section=_S_INTERMEDIACION, title="Flujos DAP UF Variación T-5 (USD Mill.)", #Identificar porque no aparece plan vital
                                    unit="US$ Mill.", chart="stacked_bar", status=STATUS_MVP,
                                    source_id="", transform="",
                                    params={"category": "AFP", "value": "Monto",
                                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                ),


    ReportBlock(
                                section=_S_VARAGENTEDCV, title="Fondos de Pensiones (US$ Mill.)", #Identificar porque no aparece plan vital
                                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                    source_id="", transform="",
                                    params={"category": "AFP", "value": "Monto",
                                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                    ),
    ReportBlock(
                                section=_S_VARAGENTEDCV, title="Fondos Mutuos (USD Mill.)", #Identificar porque no aparece plan vital
                                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                    source_id="", transform="",
                                    params={"category": "AFP", "value": "Monto",
                                    "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                    ),

    ReportBlock(
                                section=_S_VARAGENTEDCV, title="NR (US$ Mill.)", #Identificar porque no aparece plan vital
                                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                    source_id="", transform="",
                                    params={"category": "AFP", "value": "Monto",
                                            "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                        ),
    ReportBlock(
                                section=_S_VARAGENTEDCV, title="Bancos (USD Mill.)", #Identificar porque no aparece plan vital
                                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                    source_id="", transform="",
                                    params={"category": "AFP", "value": "Monto",
                                   "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                        ),


    ReportBlock(
                                section=_S_CURVASHISTORICAS, title="Pendientes BTP", #Identificar porque no aparece plan vital
                                    unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                    source_id="", transform="",
                                    params={"category": "AFP", "value": "Monto",
                                     "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                    ),

    ReportBlock(
                                section=_S_CURVASHISTORICAS, title="Swap Spread", #Identificar porque no aparece plan vital
                                        unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                        source_id="", transform="",
                                        params={"category": "AFP", "value": "Monto",
                                               "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                                ),
    ReportBlock(
                                section=_S_CURVASHISTORICAS, title="Spread BTP-UST", #Identificar porque no aparece plan vital
                                        unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                        source_id="", transform="",
                                        params={"category": "AFP", "value": "Monto",
                                        "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                                    ),
    ReportBlock(
                                section=_S_CURVASHISTORICAS, title="Tasa UF Bono Bancario y Corp (%)", #Identificar porque no aparece plan vital
                                            unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                                source_id="", transform="",
                                                params={"category": "AFP", "value": "Monto",
                                                "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                                    ),

    ReportBlock(
                                section=_S_CURVASHISTORICAS, title="Spread Bono Bancario AAA (pb)", #Identificar porque no aparece plan vital
                                                unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                                    source_id="", transform="",
                                                    params={"category": "AFP", "value": "Monto",
                                                    "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                                        ),

    ReportBlock(
                                section=_S_CURVASHISTORICAS, title="Spread BC - SPC (pb)", #Identificar porque no aparece plan vital
                                        unit="US$ Mill.", chart="line", status=STATUS_MVP,
                                        source_id="", transform="",
                                        params={"category": "AFP", "value": "Monto",
                                        "net": "False","window":"ytd","accumulate":"cumsum"},  # + línea Neto = suma de fondos
                                ),

                    


# _S_CURVASHISTORICAS



)

RF_SPEC = FamilyReportSpec(
    family="rf",
    title="Informe Renta Fija",
    blocks=_BLOCKS,
    # Los parquets de AFP cierran en fechas distintas (p.ej. movimientos al 09-06,
    # DCV al 10-06). Sin corte común cada gráfico se ancla al máximo de SU parquet,
    # evitando arrastrar todo a la fecha del parquet con menor fecha.
    share_weekly_cutoff=False,
    # Solo "Allocation y patrimonio" en grilla de 2 columnas (gráfico junto a
    # gráfico, como el informe cambiario): el resto del informe sigue apilado
    # (``layout`` default "stack"). 11 bloques → 5 pares + el último a fila
    # completa (impar), regla automática de ``_wide_block_ids``.
    #grid_sections=frozenset({}),
)
