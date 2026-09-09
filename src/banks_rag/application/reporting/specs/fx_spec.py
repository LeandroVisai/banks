"""Spec curado del "Informe Flujos Cambiarios" (familia fx).

Réplica del correo real del BCCh (remitente DACE), cuyas capturas —tomadas EN
ORDEN— viven en ``data_pipeline/Tipos de informe/Informe Flujos Cambiario``
(``1.jpg`` … ``7.jpg``). El orden de los bloques, sus TÍTULOS (literales, con la
unidad entre paréntesis tal como el original) y la forma de cada gráfico son los
del correo: 22 bloques = 2 tablas + 20 gráficos, en 3 secciones.

  RESUMEN GENERAL                    tabla + Gráficos N°1 … N°8.1   (11)
  Flujos SDR Forward FX USD - CLP    3 cortes de transacciones SDR   (3)
  NO RESIDENTES                      Gráficos N°9 … N°15 + Tabla N°1 (8)

Convención de signo del mercado cambiario: **positivo = compra de dólares,
negativo = venta**. El patrón gráfico dominante es apilado divergente (positivos
arriba / negativos abajo) con un Neto superpuesto como punto.

**Bloques sin parquet** (``STATUS_SKIP``): se mantienen EN SU POSICIÓN del
informe como tarjeta "Sin datos" con la nota de qué serie falta, para que el
correo conserve la numeración del original y se vea qué habría que traer del
servidor. Verificado contra las columnas REALES de los 174 parquets:

  - N°6 — tramo de PRECIO de suscripciones/vencimientos: ninguna columna de
    precio (los `Tramo` del catálogo son de plazo: "1Y", "Menor a 2Y").
  - SDR (3) — no hay dataset de transacciones SDR.
  - N°12 — clasificación especulativo / carry / cobertura.
  - N°14 y N°15 — fixing cruzado banco x agente offshore.

Dos diferencias de COBERTURA (el gráfico existe y es correcto, pero el universo
del parquet es más chico que el del correo); van anotadas en el bloque:

  - Tabla resumen: el correo abre Spot en afecto/no-afecto y Derivados en
    NDF/resto; ``flujo_cambiario`` solo trae ``Spot`` y ``Forward`` por sector.

Esta es la ÚNICA pieza a editar para ajustar el informe fx. Las transforms
genéricas viven en ``series_transforms.py``; el render en ``svg_chart.py``.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import (
    STATUS_MVP,
    STATUS_SKIP,
    FamilyReportSpec,
    ReportBlock,
)

# Secciones (banners del correo, literales).
_S_TABLA = "TABLA DE FLUJOS"
_S_RESUMEN = "RESUMEN GENERAL"
_S_SDR = "Flujos SDR Forward FX USD - CLP"
_S_NR = "NO RESIDENTES"

# Sector del parquet → nombre del correo, en el orden de la tabla del original.
_SECTOR_LABELS = {
    "AFP": "AFP",
    "Bancos": "Bancos",
    "BCCh": "BCCh",
    "CS": "Cia Seguros",
    "CB": "Corredora De Bolsa",
    "Emp_financiera": "Empresa Financiera",
    "Mineras": "Empresa Minera",
    "Emp_real": "Empresa Real",
    "FFMM": "FFMM",
    "NR": "No Residentes",
    "Persona_natural": "Personas Naturales",
    "Otros": "Otros",
    "TGR": "TGR",
    
}
_SECTOR_ORDER = list(_SECTOR_LABELS.values())

# Leyenda de los gráficos por sector (nombres largos del correo).
_SECTOR_SERIES = {
    "AFP": "AFP", "BCCh": "BCCh", "CS": "CSV", "CB": "CB", "FFMM": "FFMM",
    "NR": "NR", "TGR": "TGR", "Emp_financiera": "Empresas Financieras",
    "Mineras": "Empresas Mineras", "Emp_real": "Empresas Reales",
}
_SECTOR_SERIES_ORDER = list(_SECTOR_SERIES)

# Instrumento del parquet → nombre del correo (leyenda de N°3 / N°4 / N°10).
_INSTRUMENT_LABELS = {
    "FWD": "Forward",
    "FXS": "Fx swap",
    "CSS": "CCS",
    "CALL": "Opción call",
    "PUT": "Opción put",
    "FWD_obs": "Forward observado"
}
_INSTRUMENT_ORDER = list(_INSTRUMENT_LABELS)

# Tramos de plazo del derivado, del más corto al más largo (eje X de N°11 y N°13).
_PLAZO_ORDER = ["1 a 7 dias", "8 a 30 dias", "31 a 90 dias", "91 a 180 dias",
                "181 a 360 dias", "361 a 720 dias", ">720 dias"]

_SUSC_VCTO_LABELS = {"Suscripcion": "Suscripciones", "Vencimiento": "Vencimientos"}

# Agentes de spot_susc_vcto_agente.parquet (columna Institucion), sin Total (va
# aparte como fila de cierre): para que Tabla N°1 los liste TODOS aunque un
# agente no haya operado en la ventana leída (queda en 0), no solo los que
# aparecen en el corte.
_SPOT_SUSC_VCTO_AGENTS = [
    "Air Products and Chemicals", "BBVA", "BNP Paribas", "Bank of America",
    "Barclays", "Bryan Whitfield Miller .", "Caixa",
    "Canada Pension Plan Investme", "Citibank", "Credit Agricole", "Deutsche",
    "Euroclear", "Goldman Sachs", "HSBC", "Itau", "J.P. Morgan Securities Llc",
    "JP Morgan", "Merrill Lynch", "Morgan Stanley", "Natixis", "Otros_static",
    "Santander", "Scotiabank", "Societe Generale", "Standard Chartered", "TD",
    "UBS", "Wells Fargo",
]

# Tramos de plazo SDR (sdr_ndf_plazo / sdr_ndf_precio_plazo), del más corto al
# más largo (eje X de los bloques SDR); no vienen ordenados en el parquet.
_SDR_PLAZO_ORDER = ["1 semana", "2 semanas", "1 mes", "2 meses", "3 meses",
                    "6 meses", "9 meses", "1 año"]

_BLOCKS: tuple[ReportBlock, ...] = (
    # ═══ RESUMEN GENERAL ═════════════════════════════════════════════════════
    # Tabla que abre el correo (sin número): ancla numérica del informe; el resto
    # de los gráficos desagrega estas mismas cifras.
    ReportBlock(
        section=_S_TABLA, title="Resumen de flujos por sector (US$ MM)",
        chart="heatmap_table", status=STATUS_MVP,
        source_id="flujos_por_sector_ultimo_dia", transform="fx_sector_flow_table",
        params={"sector": "Sector", "category": "Categoria", "value": "Monto",
                "source_id_5d": "flujos_por_sector_ultimos_cinco_dias",
                "labels": _SECTOR_LABELS, "order": _SECTOR_ORDER},
        note="*positivo = compra de dólares, negativo = venta. Spot abierto en "
             "no afecto/afecto y Derivados en NDF/resto, con su columna de "
             "consolidación, para el día y el acumulado de 5 días",
    ),
    # N°1 y N°2: una línea por sector acumulando el flujo del último mes. Todas
    # nacen en 0 el primer día de la ventana (anchor_zero), como el original.
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°1: Spot acumulado por sectores (US$ MM)",
        chart="line", status=STATUS_MVP,
        source_id="flujo_cambiario", transform="category_series",
        params={"category": "Sector", "value": "Spot",
                "accumulate": "cumsum", "window": "d30", "anchor_zero": True,
                "order": _SECTOR_SERIES_ORDER, "labels": _SECTOR_SERIES},
        note="*flujo spot acumulado de los últimos 30 días",
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°2: Derivados acumulado por sectores (US$ MM)",
        chart="line", status=STATUS_MVP,
        source_id="flujo_cambiario", transform="category_series",
        params={"category": "Sector", "value": "Forward",
                "accumulate": "cumsum", "window": "d30", "anchor_zero": True,
                "order": _SECTOR_SERIES_ORDER, "labels": _SECTOR_SERIES},
        note="*flujo en derivados acumulado de los últimos 30 días",
    ),
    # N°3 y N°4: mismo corte (último día) abierto por instrumento, con columna
    # Total. ``susc_neta_sector_instrumento`` ya trae Sector directo (sin agente
    # offshore intermedio), en el orden del correo vía ``group_order``.
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°3: Suscripciones netas derivados (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="susc_neta_sector_instrumento", transform="window_stacked_two_cat",
        params={"group": "Sector", "series": "Instrumento", "value": "Suscripcion", "window_days": 1,
                "labels": _INSTRUMENT_LABELS, "series_order": _INSTRUMENT_ORDER,
                "group_order": list(_SECTOR_LABELS), "total_label": "Total"},
        note="*suscripciones del último día, apiladas por instrumento; Neto como punto",
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°4: Vencimientos netos derivados (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="susc_neta_sector_instrumento", transform="window_stacked_two_cat",
        params={"group": "Sector", "series": "Instrumento", "value": "Vencimiento", "window_days": 1,
                "labels": _INSTRUMENT_LABELS, "series_order": _INSTRUMENT_ORDER,
                "group_order": list(_SECTOR_LABELS), "total_label": "Total"},
        note="*vencimientos del último día, apilados por instrumento; Neto como punto",
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°5: Spot por tramo de precio (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="spot_bucket_sector", transform="snapshot_stacked",
        params={"x": "Bucket", "series": "Sector", "value": "Monto",
                "labels": _SECTOR_LABELS, "series_order": _SECTOR_ORDER,
                },
        note="*monto spot transado por tramo de tipo de cambio (Bucket) y sector",
    ),
    ReportBlock( 
        section=_S_RESUMEN,
        title="Gráfico N°6: Suscripciones por tramo de precio y vencimientos (US$ MM)",
        source_id="susc_bucket_tc_sector", chart="stacked_bar", status=STATUS_MVP,
        transform="snapshot_stacked",
        params={"x": "Bucket", "series": "Sector", "value": "Pos_neta",
            "labels": _SECTOR_LABELS, "series_order": _SECTOR_ORDER},
        note="*posiciones netas por tramo de tipo de cambio (Bucket) y sector",
    ),
    # N°7: quién queda con el fixing NDF del próximo vencimiento, por banco
    # informante, abierto por sector contraparte.
    ReportBlock(
    section=_S_RESUMEN, title="Gráfico N°7: Próximo fixing por banco - NDF (US$ MM)",
    chart="stacked_bar", status=STATUS_MVP,
    source_id="fixing_proximo_contraparte", transform="snapshot_stacked",
    params={"x": "Informante", "series": "Contraparte", "value": "Monto",
            "labels": _SECTOR_LABELS, "series_order": _SECTOR_ORDER,
            "exclude_series": ["Bancos"], "total_overlay": True},
    note="*por banco informante, apilado por sector contraparte; Total como punto",
    ),
    # N°8: cada Fixing futuro como una barra, apilada por Contraparte.
    ReportBlock(
    section=_S_RESUMEN,
    title="Gráfico N°8: Últimos y próximos fixing de la banca - NDF (US$ MM)",
    chart="stacked_bar", status=STATUS_MVP,
    source_id="fixing_futuro_contraparte", transform="snapshot_stacked",
    params={"x": "Fixing", "series": "Contraparte", "value": "Monto",
            "labels": _SECTOR_SERIES, "series_order": _SECTOR_SERIES_ORDER,
            "total_overlay": True},
    note="*una barra por fecha de fixing futuro, apilada por sector contraparte; Total como punto",
    ),
    # N°7.1: mismo parquet que N°8.1, filtrado a la fecha de fixing futura más próxima.
    ReportBlock(
    section=_S_RESUMEN, title="Gráfico N°7.1. Próximo fixing según tipo de Instrumento",
    chart="stacked_bar", status=STATUS_MVP,
    source_id="fixing_futuro_informante", transform="snapshot_stacked",
    params={"x": "Informante", "series": "Instrumento", "value": "Monto",
            "date_col": "Fixing", "date_mode": "nearest", "total_overlay": True},
    note="*por banco informante, apilado por instrumento. Total como punto",
    ),
    ReportBlock(
    section=_S_RESUMEN, title="Gráfico N°8.1 Fixing NDF según banco",
    chart="stacked_bar", status=STATUS_MVP,
    source_id="fixing_futuro_informante", transform="snapshot_stacked",
    params={"x": "Fixing", "series": "Informante", "value": "Monto",
            "total_overlay": True},
    note="*una barra por fecha de fixing futuro, apilada por banco informante; Total como punto",
    ),

    # ═══ Flujos SDR Forward FX USD - CLP ═════════════════════════════════════
    ReportBlock(
        section=_S_SDR, title="Monto transado según bucket (usd)",
        chart="grouped_bar", source_id="sdr_ndf_plazo", status=STATUS_MVP,
        transform="snapshot_grouped",
        params={"group": "Bucket", "values": ["Monto"], "order": _SDR_PLAZO_ORDER},
        note="*transacciones SDR forward USD-CLP por bucket de plazo",
    ),
    ReportBlock(
        section=_S_SDR, title="Monto transado según fecha de vencimiento (usd)",
        chart="grouped_bar", source_id="sdr_ndf_vencimiento_semana", status=STATUS_MVP,
        transform="daily_wide_stacked",
        params={"include": ["Monto"], "from_start": True, "last_n": 10, "net": False},
        note="*monto SDR forward por fecha de vencimiento operada",
    ),
    ReportBlock(
        section=_S_SDR, title="Precio promedio transacciones",
        chart="curve", source_id="sdr_ndf_precio_plazo", status=STATUS_MVP,
        transform="snapshot_grouped",
        params={"group": "Bucket", "values": ["Precio"], "order": _SDR_PLAZO_ORDER,
                "zero_base": False},
        note="*precio promedio pactado por bucket de plazo en SDR",
    ),
    # ═══ NO RESIDENTES ═══════════════════════════════════════════════════════
    # N°9: la jornada de los NR en derivados, día a día. Suscripciones arriba,
    # vencimientos abajo, Posición (neta) como punto.
    ReportBlock(
        section=_S_NR, title="Gráficos N°9: Posición derivados (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="var_pos_derivados", transform="daily_wide_stacked",
        params={"include": ["Suscripcion", "Vencimiento"],
                "last_n": 7, "labels": _SUSC_VCTO_LABELS, "net_label": "Posición"},
        note="*variación neta del día",
        full_width=True,
    ),
    # Tabla N°1: qué agente offshore movió la posición, en 4 ventanas, Spot y
    # Derivados por separado (dos mini-tablas), con todos los agentes del
    # catálogo aunque no hayan operado en la ventana (quedan en 0).
    ReportBlock(
        section=_S_NR, title="Tabla N°1: Posición derivados (US$ MM)",
        chart="heatmap_table", status=STATUS_MVP,
        source_id="spot_susc_vcto_agente", transform="fx_agent_delta_table",
        params={"agent": "Institucion", "value": "Monto", "type_col": "Tipo",
            "pos": "Suscripción", "neg": "Vencimiento", "spot": "Spot", "net": False,
            "windows": [1, 5], "exclude_agents": ["Total"],
            "all_agents": _SPOT_SUSC_VCTO_AGENTS},
        note="*Spot y Derivados (suscripción + vencimiento) por separado, monto "
             ". Incluye todos los Agentes NR",
        full_width=True,
    ),
    # N°10: stock acumulado DESDE EL INICIO de la serie por instrumento; la
    # ventana solo recorta la vista al último mes, sin reiniciar el acumulado.
    ReportBlock(
        section=_S_NR, title="Gráfico N°10: Suscripciones brutas derivados (US$ MM)",
        chart="stacked_area", status=STATUS_MVP,
        source_id="nr_susc_neta_instrumento", transform="category_series",
        params={"category": "Instrumento", "value": "Pos_neta", "window": "d30",
                "labels": _INSTRUMENT_LABELS, "order": _INSTRUMENT_ORDER},
        note="*monto suscrito acumulado desde el inicio de la serie por "
             "instrumento; se muestra el movimiento del stock del último mes",
    ),
    # N°11: el último día por TRAMO DE PLAZO del contrato. Suscripción suma y
    # vencimiento resta, así el alto neto de la barra es la variación del tramo.
    ReportBlock(
        section=_S_NR, title="Gráfico N°11: Posición por plazo derivados (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="posicion_derivados_plazo", transform="window_grouped",
        params={"group": "Plazo", "values": ["Suscripcion", "Vencimiento"],
                "window_days": 0, "include_net": True, "net_as_overlay": True, "order": _PLAZO_ORDER,
                "labels": _SUSC_VCTO_LABELS},
        note="*último día por tramo; Neto = Suscripciones - Vencimientos",
    ),   

    ReportBlock(
            section=_S_NR, title="Gráfico N°12: Posición NR todos los derivados (US$)",
            unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
            source_id="posicion_nr_derivados", date_from= "2022-01-01", transform="wide_lines",
            params={"overlay": ["Neto"]},
            note="*posición por tramo de plazo; Neto = suma de tramos",
        ),

    # N°13: la misma posición por plazo, acumulada a lo largo del mes. ## REVISAR URGENTE.
    ReportBlock(
        section=_S_NR, title="Gráfico N°13: Posición acumulada por plazos derivados (US$ MM)", 
        chart="line", status=STATUS_MVP,
        source_id="posicion_derivados_plazo", transform="category_series",
        params={"category": "Plazo", "value":"Suscripcion","value":"Vencimiento",
                "net": "False", "accumulate": "cumsum", "window": "d30",
                "anchor_zero": True, "order": _PLAZO_ORDER},
        note="*posición neta (suscripción - vencimiento) acumulada por tramo, últimos "
             "30 días; Neto = suma de tramos",
    ),

    ReportBlock(
            section=_S_NR, title="Gráfico N°14: Último fixing de la banca por agente - NDF (US$ MM)",
            chart="stacked_bar", status=STATUS_MVP,
            source_id="fixing_nr_t_1", transform="snapshot_stacked",
            params={"x": "Informante", "series": "Nombre", "value": "Monto", "total_overlay": True},
            note="*por banco informante, apilado por sector contraparte; Total como punto",
            ),

    ReportBlock(
        section=_S_NR, title="Gráfico N°15: Próximo fixing de la banca por agente - NDF (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="fixing_nr_proximo", transform="snapshot_stacked",
        params={"x": "Informante", "series": "Nombre", "value": "Monto", "total_overlay": True},
        note="*por banco informante, apilado por sector contraparte; Total como punto",
        ),


)

FX_SPEC = FamilyReportSpec(
    family="fx",
    title="Informe Flujos Cambiarios",
    blocks=_BLOCKS,
    share_weekly_cutoff=False,
    grid_sections=frozenset({
            _S_TABLA, _S_RESUMEN ,_S_SDR,_S_NR
        }),

)

