"""Spec curado del "Informe No Residentes" (familia nr).

Réplica del informe real del BCCh (correo "Informe No Residentes" + tableros
Flujos Derivados / Posición SPC / Renta Fija / Flujos Spot): los gráficos que se
pueden reproducir desde los parquets locales, en el ORDEN narrativo del informe y
agrupados por mercado. Cada bloque mapea a un parquet + una transform + un tipo de
gráfico objetivo.

Sólo se incluyen bloques con dato reproducible; los gráficos del informe que viven
de fuentes externas (carry trade, posicionamiento BNP, índice GBI EM de JP Morgan,
"Chile vs otras economías") no tienen parquet y se OMITEN a propósito.

Esta es la ÚNICA pieza a editar para ajustar el informe nr (orden, títulos, qué
parquet alimenta cada gráfico). Las transforms genéricas (``category_series``,
``wide_lines``, ``window_grouped``) viven en ``series_transforms.py``.
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
    ReportBlock(
        section=_S_DERIV, title="Posición cambiaria histórica de no residentes",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="posicion_nr_derivados", transform="category_series",
        params={"category": "Plazo", "value": "Valor",
                "order": ["1 a 90 dias", "91 a 360 dias", "Mayor a 360 dias"]},
    ),
    ReportBlock(
        section=_S_DERIV, title="Cambio en la posición de derivados cambiarios (última semana)",
        unit="US$ Mill.", chart="grouped_bar", status=STATUS_MVP,
        source_id="var_pos_derivados", transform="window_grouped",
        params={"group": "Plazo", "values": ["Suscripcion", "Vencimiento"],
                "window_days": 7, "include_net": True,
                "order": ["1 a 90 días", "91 a 360 días", "Mayor a 360 días"]},
        note="*suma de la última semana con datos por tramo de plazo",
    ),
    ReportBlock(
        section=_S_DERIV, title="Flujos acumulados en derivados por instrumento",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="flujos_acumulados_derivados", transform="category_series",
        params={"filter_col": "Institucion", "filter_val": "Total",
                "category": "Instrumento", "value": "Net"},
        note="*acumulado del total de agentes, desde ene.",
    ),
    # ── Mercado de Renta Fija (BTP en DCV) ───────────────────────────────────
    ReportBlock(
        section=_S_RF, title="Posición de no residentes en BTP por plazo",
        unit="US$ Mill.", chart="stacked_area", status=STATUS_MVP,
        source_id="posicion_rfl_nr", transform="straight_series",
    ),
    ReportBlock(
        section=_S_RF, title="Variación semanal del stock de BTP en DCV",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="variacion_rfl_dcv_nr", transform="straight_series",
    ),
    ReportBlock(
        section=_S_RF, title="Variación acumulada del stock de BTP en DCV (YtD)",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="variacion_rfl_dcv_nr", transform="accumulated",
        params={"window": "ytd", "accumulate": "cumsum"},
    ),
    # ── Mercado SPC (tasas) ──────────────────────────────────────────────────
    # Datos del parquet en US$ mil millones → scale 1000 para mostrar en US$ Mill.
    # (como el informe BCCh). Multi-línea: el área apilada recortaría los valores
    # negativos (reciben fija), que aquí son parte de la lectura.
    ReportBlock(
        section=_S_SPC, title="Posición de no residentes en SPC nominal",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="posicion_nr_spc", transform="straight_series", scale=1000.0,
    ),
    ReportBlock(
        section=_S_SPC, title="Variación de la posición NR en SPC por plazo",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="nr_var_posicion_spc", transform="wide_lines", scale=1000.0,
        params={"include": ["91 a 360 dias", "Entre 1 y 2Y", "Mayor a 2Y", "Neto"]},
    ),
    # ── Flujos Spot ──────────────────────────────────────────────────────────
    # El parquet trae flujos diarios → cumsum (ventana ≈ últimos 2 años) para
    # leer el flujo ACUMULADO, como el informe BCCh.
    ReportBlock(
        section=_S_SPOT, title="Flujos spot acumulados de no residentes (afecto / no afecto)",
        unit="US$ Mill.", chart="line", status=STATUS_MVP,
        source_id="flujo_spot_nr", transform="wide_lines",
        params={"include": ["Afecto", "No Afecto"], "accumulate": "cumsum", "window": "y2"},
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
)
