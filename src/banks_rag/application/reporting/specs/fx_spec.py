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

  - N°5 y N°6  — tramo de PRECIO de la transacción: ninguna columna de precio
    (los ``Tramo`` del catálogo son de plazo: "1Y", "Menor a 2Y").
  - N°7.1 y N°8.1 — fixing abierto por instrumento y por fecha-banco: los dos
    parquets de fixing solo abren por sector contraparte.
  - SDR (3) — no hay dataset de transacciones SDR.
  - N°12 — clasificación especulativo / carry / cobertura.
  - N°14 y N°15 — fixing cruzado banco x agente offshore.

Dos diferencias de COBERTURA (el gráfico existe y es correcto, pero el universo
del parquet es más chico que el del correo); van anotadas en el bloque:

  - N°3 y N°4: el correo abre el eje X por sector; el único parquet con la
    apertura suscripción/vencimiento por instrumento es de no residentes y su
    ``Institucion`` son los agentes offshore contraparte.
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
_S_RESUMEN = "RESUMEN GENERAL"
_S_SDR = "Flujos SDR Forward FX USD - CLP"
_S_NR = "NO RESIDENTES"

# Sector del parquet → nombre del correo, en el orden de la tabla del original.
_SECTOR_LABELS = {
    "AFP": "AFP",
    "CS": "CIA SEGUROS",
    "CB": "CORREDORA DE BOLSA",
    "Emp_financiera": "EMPRESA FINANCIERA",
    "Mineras": "EMPRESA MINERA",
    "Emp_real": "EMPRESA REAL",
    "FFMM": "FFMM",
    "NR": "NO RESIDENTES",
    "Persona_natural": "PERSONAS NATURALES",
    "Otros": "OTROS",
    "BCCh": "BCCH",
    "TGR": "TGR",
    "Bancos": "BANCOS",
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
    "CSS": "CCS",
    "FWD": "Forward",
    "FWD_obs": "Forward observado",
    "FXS": "Fx swap",
    "CALL": "Opción call",
    "PUT": "Opción put",
}
_INSTRUMENT_ORDER = list(_INSTRUMENT_LABELS)

# Tramos de plazo del derivado, del más corto al más largo (eje X de N°11 y N°13).
_PLAZO_ORDER = ["1 a 7 dias", "8 a 30 dias", "31 a 90 dias", "91 a 180 dias",
                "181 a 360 dias", "361 a 720 dias", ">720 dias"]

_SUSC_VCTO_LABELS = {"Suscripcion": "Suscripciones", "Vencimiento": "Vencimientos"}

_BLOCKS: tuple[ReportBlock, ...] = (
    # ═══ RESUMEN GENERAL ═════════════════════════════════════════════════════
    # Tabla que abre el correo (sin número): ancla numérica del informe; el resto
    # de los gráficos desagrega estas mismas cifras.
    ReportBlock(
        section=_S_RESUMEN, title="Resumen de flujos por sector (US$ MM)",
        chart="heatmap_table", status=STATUS_MVP,
        source_id="flujo_cambiario", transform="fx_sector_flow_table",
        params={"sector": "Sector", "spot": "Spot", "deriv": "Forward", "days": 5,
                "labels": _SECTOR_LABELS, "order": _SECTOR_ORDER},
        note="*positivo = compra de dólares, negativo = venta. El correo abre además "
             "Spot en afecto/no-afecto y Derivados en NDF/resto: esa apertura no está "
             "en el parquet (solo Spot y Forward por sector)",
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
        note="*flujo spot acumulado (suma corrida) de los últimos 30 días",
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°2: Derivados acumulado por sectores (US$ MM)",
        chart="line", status=STATUS_MVP,
        source_id="flujo_cambiario", transform="category_series",
        params={"category": "Sector", "value": "Forward",
                "accumulate": "cumsum", "window": "d30", "anchor_zero": True,
                "order": _SECTOR_SERIES_ORDER, "labels": _SECTOR_SERIES},
        note="*flujo en derivados acumulado (suma corrida) de los últimos 30 días",
    ),
    # N°3 y N°4: mismo corte (último día) abierto por instrumento, con columna
    # Total. El eje X es el agente offshore, no el sector (ver docstring).
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°3: Suscripciones netas derivados (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="susc_vcto_agente_instrumento", transform="window_stacked_two_cat",
        params={"group": "Institucion", "series": "Instrumento", "value": "Monto",
                "filter_col": "Tipo", "filter_val": "Suscripción", "window_days": 1,
                "labels": _INSTRUMENT_LABELS, "series_order": _INSTRUMENT_ORDER,
                "exclude_groups": ["Total"], "total_label": "Total"},
        note="*suscripciones del último día, apiladas por instrumento; Neto como punto. "
             "Eje X = agente offshore: el corte por sector del correo no está en el catálogo",
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°4: Vencimientos netos derivados (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="susc_vcto_agente_instrumento", transform="window_stacked_two_cat",
        params={"group": "Institucion", "series": "Instrumento", "value": "Monto",
                "filter_col": "Tipo", "filter_val": "Vencimiento", "window_days": 1,
                "labels": _INSTRUMENT_LABELS, "series_order": _INSTRUMENT_ORDER,
                "exclude_groups": ["Total"], "total_label": "Total"},
        note="*vencimientos del último día, apilados por instrumento; Neto como punto",
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°5: Spot por tramo de precio (US$ MM)",
        chart="stacked_bar", status=STATUS_SKIP,
        note="Falta parquet: spot del día abierto por TRAMO DE PRECIO (<924, 924-926, …) "
             "y sector. Ningún parquet trae el precio de la transacción; los `Tramo` del "
             "catálogo son de plazo.",
    ),
    ReportBlock(
        section=_S_RESUMEN,
        title="Gráfico N°6: Suscripciones por tramo de precio y vencimientos (US$ MM)",
        chart="stacked_bar", status=STATUS_SKIP,
        note="Falta parquet: suscripciones de derivados por TRAMO DE PRECIO pactado y "
             "sector, más la columna de vencimientos.",
    ),
    # N°7: quién queda con el fixing NDF del próximo vencimiento, por banco
    # informante, abierto por sector contraparte.
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°7: Próximo fixing por banco - NDF (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="fixing_banca_sector", transform="wide_row_stacked",
        params={"row": "Institucion", "overlay": ["Neto"], "exclude_rows": ["Total"],
                "labels": _SECTOR_SERIES},
        note="*por banco informante, apilado por sector contraparte; Neto como punto",
    ),
    # N°8: el mismo fixing agregado, comparado contra los cortes previos.
    ReportBlock(
        section=_S_RESUMEN,
        title="Gráfico N°8: Últimos y próximos fixing de la banca - NDF (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="fixing_por_fecha", transform="wide_row_stacked",
        params={"row": "Temporalidad", "overlay": ["Neto"], "labels": _SECTOR_SERIES,
                "row_order": ["Hoy", "Ayer", "1 semana", "2 semanas", "1 mes"]},
        note="*el parquet trae cortes RELATIVOS (hoy / ayer / 1 semana / 2 semanas / "
             "1 mes), no las fechas de fixing futuras del correo original",
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°7.1. Próximo fixing según tipo de Instrumento",
        chart="stacked_bar", status=STATUS_SKIP,
        note="Falta parquet: fixing por banco abierto por INSTRUMENTO (forward / FX swap / "
             "CCS / opciones). `fixing_banca_sector` solo abre por sector contraparte.",
    ),
    ReportBlock(
        section=_S_RESUMEN, title="Gráfico N°8.1 Fixing NDF según banco",
        chart="stacked_bar", status=STATUS_SKIP,
        note="Falta parquet: fixing por FECHA de vencimiento futura y banco. "
             "`fixing_por_fecha` agrega por temporalidad relativa, sin abrir por banco.",
    ),
    # ═══ Flujos SDR Forward FX USD - CLP ═════════════════════════════════════
    ReportBlock(
        section=_S_SDR, title="Monto transado según bucket (usd)",
        chart="bar_time", status=STATUS_SKIP,
        note="Falta parquet: transacciones SDR forward USD-CLP por bucket de plazo "
             "(1 semana, 2 semanas, 1 mes, …).",
    ),
    ReportBlock(
        section=_S_SDR, title="Monto transado según fecha de vencimiento (usd)",
        chart="bar_time", status=STATUS_SKIP,
        note="Falta parquet: monto SDR forward por fecha de vencimiento operada.",
    ),
    ReportBlock(
        section=_S_SDR, title="Precio promedio transacciones",
        chart="line", status=STATUS_SKIP,
        note="Falta parquet: precio promedio pactado por bucket de plazo en SDR.",
    ),
    # ═══ NO RESIDENTES ═══════════════════════════════════════════════════════
    # N°9: la jornada de los NR en derivados, día a día. Suscripciones arriba,
    # vencimientos abajo, Posición (neta) como punto.
    ReportBlock(
        section=_S_NR, title="Gráficos N°9: Posición derivados (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="var_pos_derivados", transform="daily_wide_stacked",
        params={"include": ["Suscripcion", "Vencimiento"], "negate": ["Vencimiento"],
                "last_n": 10, "labels": _SUSC_VCTO_LABELS, "net_label": "Posición"},
        note="*últimas 10 jornadas; el vencimiento resta, la Posición (punto) es la "
             "variación neta del día",
    ),
    # Tabla N°1: qué agente offshore movió la posición, en 4 ventanas.
    ReportBlock(
        section=_S_NR, title="Tabla N°1: Posición derivados (US$ MM)",
        chart="heatmap_table", status=STATUS_MVP,
        source_id="susc_vcto_agente_instrumento", transform="fx_agent_delta_table",
        params={"agent": "Institucion", "value": "Monto", "type_col": "Tipo",
                "pos": "Suscripción", "neg": "Vencimiento",
                "windows": [1, 5, 10, 20], "exclude_agents": ["Total"]},
        note="*variación neta (suscripción - vencimiento) acumulada de las últimas N "
             "jornadas con dato. El parquet agrupa los agentes menores en `Otros`",
    ),
    # N°10: composición diaria del monto BRUTO suscrito, con el promedio del período.
    ReportBlock(
        section=_S_NR, title="Gráfico N°10: Suscripciones brutas derivados (US$ MM)",
        chart="stacked_area", status=STATUS_MVP,
        source_id="susc_vcto_agente_instrumento", transform="category_series",
        params={"category": "Instrumento", "value": "Monto",
                "filter_col": "Tipo", "filter_val": "Suscripción",
                "abs": True, "window": "d30", "mean_overlay": True,
                "labels": _INSTRUMENT_LABELS, "order": _INSTRUMENT_ORDER},
        note="*monto BRUTO suscrito por día e instrumento (últimos 30 días); la línea "
             "marca el promedio diario del período",
    ),
    # N°11: el último día por TRAMO DE PLAZO del contrato. Suscripción suma y
    # vencimiento resta, así el alto neto de la barra es la variación del tramo.
    ReportBlock(
        section=_S_NR, title="Gráfico N°11: Posición por plazo derivados (US$ MM)",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="posicion_derivados_plazo", transform="window_grouped",
        params={"group": "Plazo", "values": ["Suscripcion", "Vencimiento"],
                "window_days": 1, "include_net": True, "net_as_overlay": True,
                "negate": ["Vencimiento"], "order": _PLAZO_ORDER,
                "labels": _SUSC_VCTO_LABELS},
        note="*último día por tramo; Neto = Suscripciones - Vencimientos",
    ),
    ReportBlock(
        section=_S_NR, title="Gráfico N°12: Posición NR todos los derivados (US$)",
        chart="stacked_area", status=STATUS_SKIP,
        note="Falta parquet: clasificación de la posición NR por MOTIVO (especulativo / "
             "carry / cobertura). `posicion_derivados_plazo.Modalidad` solo distingue "
             "compensación de entrega física, que no es lo mismo.",
    ),
    # N°13: la misma posición por plazo, acumulada a lo largo del mes.
    ReportBlock(
        section=_S_NR, title="Gráfico N°13: Posición acumulada por plazos derivados (US$ MM)",
        chart="line", status=STATUS_MVP,
        source_id="posicion_derivados_plazo", transform="category_series",
        params={"category": "Plazo", "value": "Suscripcion", "value_neg": "Vencimiento",
                "net": "auto", "accumulate": "cumsum", "window": "d30",
                "anchor_zero": True, "order": _PLAZO_ORDER},
        note="*posición neta (suscripción - vencimiento) acumulada por tramo, últimos "
             "30 días; Neto = suma de tramos",
    ),
    ReportBlock(
        section=_S_NR,
        title="Gráfico N°14: Último fixing de la banca por agente - NDF (US$ MM)",
        chart="stacked_bar", status=STATUS_SKIP,
        note="Falta parquet: fixing cruzado banco informante x AGENTE offshore "
             "(Citibank, JP Morgan, …). El catálogo abre el fixing por sector, no por "
             "contraparte nominada.",
    ),
    ReportBlock(
        section=_S_NR,
        title="Gráfico N°15: Próximo fixing de la banca por agente - NDF (US$ MM)",
        chart="stacked_bar", status=STATUS_SKIP,
        note="Falta parquet: mismo cruce que el N°14 para el próximo fixing.",
    ),
)

FX_SPEC = FamilyReportSpec(
    family="fx",
    title="Informe Flujos Cambiarios",
    blocks=_BLOCKS,
    # SIN corte común: cada bloque se ancla al máximo de SU propio parquet, para
    # que el informe muestre el último dato disponible de cada serie. El corte
    # común (mín de los máximos) arrastraba TODO el informe a la fecha del parquet
    # más atrasado y dejaba fuera datos que sí existían en los demás.
    share_weekly_cutoff=False,
)
