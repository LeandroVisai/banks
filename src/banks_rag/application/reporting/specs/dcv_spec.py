"""Spec curado del "Informe Stocks Depósito Central de Valores" (familia dcv).

Réplica del correo real del BCCh, cuyas capturas —tomadas EN ORDEN— viven en
``data_pipeline/Tipos de informe/Informe DCV`` (``1.jpg`` … ``6.jpg``). El orden
de los bloques, sus TÍTULOS y la forma de cada pieza son los del correo:
22 bloques = 10 tablas + 12 gráficos, en 10 secciones (banners del original).

  Portafolio por agente          tabla + tenencia + 2 duraciones     (4)   1.jpg
  Próximos Vencimientos          8 tablas por agente + gráfico       (2)   2.jpg
  Todos los instrumentos         tabla + apilado por tramo           (2)   3.jpg
  Bancos                         tabla + apilado por tramo           (2)   3.jpg
  Fondos de Pensiones y AFC      tabla + apilado por tramo           (2)   4.jpg
  Fondos Mutuos                  tabla + apilado por tramo           (2)   4.jpg
  Compañías de Seguros           tabla + apilado por tramo           (2)   5.jpg
  Mandantes y Depósitos de Val.  tabla + apilado por tramo           (2)   5.jpg
  Corredores de Bolsa y BV       tabla + apilado por tramo           (2)   6.jpg
  Otros                          tabla + apilado por tramo           (2)   6.jpg

Todo sale de ``variacion_stock_todos`` (Fecha x Bucket x Tipo x Sector x Moneda),
el ÚNICO parquet que abre el stock DCV por agente e instrumento a la vez: tabla y
gráfico de cada bloque comparten fuente y corte, así no pueden descuadrarse. La
EXCEPCIÓN es la duración (ver abajo): sale de dos parquets APARTE, mismo grano.

**Duración por agente e instrumento** — ``duracion_iif``/``duracion_rf``
(Fecha x Tipo x Moneda x Sector → Duracion), un SNAPSHOT cada uno (solo la fecha
de hoy, sin histórico). Alimentan la columna "Dur." de la tabla de portafolio
(``dcv_portfolio_table``, promedio ponderado por monto en Total) y los dos
gráficos de dispersión categórica "Duración agentes IIF/RF"
(``dcv_duration_scatter`` → ``chart="point"`` sobre un ``PlotData`` de
``kind='grouped'``; ver ``_render_grouped_dots`` en ``svg_chart.py``).

**Próximos vencimientos por agente** — 3 parquets SNAPSHOT más (``vencimientos_
hoy``/``vencimientos_t_mas_uno``/``vencimientos_cinco_dias``, mismo grano Tipo x
Sector x Moneda) + ``vencimientos_futuros_instrumento`` (mensual, ya en el
catálogo) filtrado al mes en curso, dan las columnas T / T+1 / Acum 5d. / Mes de
las 8 mini-tablas (``dcv_upcoming_maturities_table`` → ``render_dcv_maturities_
grid``). Dos diferencias de COBERTURA frente al correo (anotadas en el bloque):
el dato trae Acum 5d., no Acum 7d.; y ninguno de los 4 parquets abre "Otros" en
Soberano/Bancario/Corporativo (el correo sí) — queda como una sola fila "RF".

**Bloques sin parquet** (``STATUS_SKIP``): se mantienen EN SU POSICIÓN del correo
como tarjeta "Falta el parquet", para conservar la estructura del original y dejar
a la vista qué falta traer del servidor. Su ``note`` empieza con ``"Falta parquet:"``
y dice QUÉ serie es. Verificado contra las columnas REALES de los parquets:

  - Mandantes y Depósitos de Valores, y Corredores de Bolsa y Bolsa de Valores
    (5.jpg y 6.jpg) — el ``Sector`` del parquet solo distingue AFP, Bancos, CS,
    FFMM y Otros; ambos agentes vienen agregados dentro de "Otros" y no se pueden
    separar.

Dos diferencias de COBERTURA (el bloque existe y es correcto, pero el universo
del parquet es más chico que el del correo); van anotadas en el bloque:

  - Los tramos de plazo del parquet son 5 (Menor a 1Y, 1-2Y, 2-5Y, 5-10Y, Mayor a
    10Y) contra los 8 del correo (<=30d, <=90d, <=180d, <=360d, <=2a, <=5a, <=10a,
    >10a): el corte fino dentro del primer año no está en el dato.
  - "Vencimientos Totales" se dibuja con deuda bancaria (Bonos + DAP); el correo
    apila además PDBC y abre los DAP por moneda.

Esta es la ÚNICA pieza a editar para ajustar el informe dcv. Las transforms viven
en ``series_transforms.py`` (``dcv_portfolio_table``, ``dcv_bucket_table``,
``dcv_snapshot_stacked``, ``dcv_duration_scatter``, ``dcv_upcoming_maturities_
table``); el render en ``svg_chart.py``.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import (
    STATUS_MVP,
    STATUS_SKIP,
    FamilyReportSpec,
    ReportBlock,
)

# Parquet maestro: stock DCV por fecha x tramo x instrumento x agente x moneda.
_SRC = "variacion_instrumento_todos_plazo"
_UNIT = "US$ Mill."

# Secciones (banners del correo, literales y en orden).
_S_PORTAFOLIO = "Portafolio por agente"
_S_VENCIMIENTOS = "Próximos Vencimientos"

# Nota compartida por los 8 bloques de tramo: el dato no tiene el corte fino
# intra-anual del correo.
_NOTE_TRAMOS = (
    "*Tramos del dato: 5 (Menor a 1Y, 1-2Y, 2-5Y, 5-10Y, Mayor a 10Y). "
    "El correo abre 8 tramos con corte fino bajo 1 año, que el parquet no trae."
)


def _agent_blocks(section: str, sector: str | None) -> tuple[ReportBlock, ...]:
    """Los dos bloques de un agente: tabla instrumento x tramo + su apilado.

    ``sector`` es el valor de la columna ``Sector`` del parquet (``None`` = todos
    los agentes). El gráfico va ``no_text``: comenta el bloque de la tabla, que es
    la pieza principal de la sección; si no, cada agente saldría con dos párrafos
    sobre exactamente el mismo corte."""
    params = {"sector": sector} if sector else {}
    return (
        ReportBlock(
            section=section, title=f"{section} — stock por instrumento y tramo de plazo",
            unit=_UNIT, chart="heatmap_table", status=STATUS_MVP,
            source_id=_SRC, transform="dcv_bucket_table", params=params,
            note=_NOTE_TRAMOS,
        ),
        ReportBlock(
            section=section, title=f"{section} — distribución por tramo de plazo",
            unit=_UNIT, chart="stacked_bar", status=STATUS_MVP,
            source_id=_SRC, transform="dcv_snapshot_stacked",
            params={**params, "x": "Bucket", "series": "Tipo"},
            no_text=True,
        ),
    )


def _skip_agent_blocks(section: str, missing: str) -> tuple[ReportBlock, ...]:
    """Agente que el parquet no separa: se conservan sus dos bloques en posición,
    como tarjeta "Falta el parquet" con el motivo."""
    return (
        ReportBlock(
            section=section, title=f"{section} — stock por instrumento y tramo de plazo",
            unit=_UNIT, chart="heatmap_table", status=STATUS_SKIP, note=missing,
        ),
        ReportBlock(
            section=section, title=f"{section} — distribución por tramo de plazo",
            unit=_UNIT, chart="stacked_bar", status=STATUS_SKIP, note=missing,
        ),
    )


_NO_SECTOR = (
    "Falta parquet: el Sector del dato solo distingue AFP, Bancos, CS, FFMM y "
    "Otros; este agente viene agregado dentro de \"Otros\"."
)

_BLOCKS: tuple[ReportBlock, ...] = (
    # ── 1.jpg — Portafolio por agente ────────────────────────────────────────
    ReportBlock(
        section=_S_PORTAFOLIO, title="Portafolio por agente",
        unit=_UNIT, chart="heatmap_table", status=STATUS_MVP,
        source_id=_SRC, transform="dcv_portfolio_table",
        note="*Duración: promedio ponderado por monto en la fila/columna Total.",
    ),
    ReportBlock(
        section=_S_PORTAFOLIO, title="Tenencia agentes por instrumento",
        unit=_UNIT, chart="stacked_bar", status=STATUS_MVP,
        source_id=_SRC, transform="dcv_snapshot_stacked",
        params={"x": "Tipo", "series": "Sector"},
        no_text=True,
    ),
    ReportBlock(
        section=_S_PORTAFOLIO, title="Duración agentes IIF",
        unit="Años", chart="point", status=STATUS_MVP,
        source_id="duracion_iif", transform="dcv_duration_scatter",
    ),
    ReportBlock(
        section=_S_PORTAFOLIO, title="Duración agentes RF",
        unit="Años", chart="point", status=STATUS_MVP,
        source_id="duracion_rf", transform="dcv_duration_scatter",
    ),
    # ── 2.jpg — Próximos Vencimientos ────────────────────────────────────────
    ReportBlock(
        section=_S_VENCIMIENTOS,
        title="Próximos vencimientos por agente (T, T+1, Acum. 5d., Mes)",
        unit="Millones de USD", chart="heatmap_table", status=STATUS_MVP,
        source_id="vencimientos_hoy", transform="dcv_upcoming_maturities_table",
        note="*Acum. 5d. (el correo trae Acum. 7d.; no hay parquet a 7 días). "
             "Mes = total programado del mes en curso vía vencimientos_futuros_"
             "instrumento (no una ventana móvil de 30 días; no distingue lo ya "
             "vencido del mes de lo que falta). RF: el dato no abre Soberano/"
             "Bancario/Corporativo.",
    ),
    ReportBlock(
        section=_S_VENCIMIENTOS, title="Vencimientos Totales",
        unit="Millones USD", chart="stacked_bar", status=STATUS_MVP,
        source_id="perfil_vencimiento_dp_bb", transform="daily_wide_stacked",
        params={"from_start": True, "last_n": 45, "net": False},
        note="*Solo deuda bancaria (bonos y DAP); el correo apila además PDBC y "
             "abre los DAP por moneda.",
    ),
    # ── 3.jpg … 6.jpg — Distribución por tramo de plazo, un agente por sección ─
    *_agent_blocks("Todos los instrumentos", None),
    *_agent_blocks("Bancos", "Bancos"),
    *_agent_blocks("Fondos de Pensiones y AFC", "AFP"),
    *_agent_blocks("Fondos Mutuos", "FFMM"),
    *_agent_blocks("Compañías de Seguros", "CS"),
    *_agent_blocks("Mandantes", "Mandantes"),
    *_agent_blocks("Corredores de Bolsa", "CB"),
    *_agent_blocks("Otros", "Otros"),
)

DCV_SPEC = FamilyReportSpec(
    family="dcv",
    title="Informe Stocks Depósito Central de Valores",
    blocks=_BLOCKS,
    # Todo el informe es un CORTE del mismo día (stocks al cierre), no ventanas
    # semanales: cada bloque se ancla al máximo de su propio parquet. Forzar un
    # corte común arrastraría las tablas de stock a la fecha del perfil de
    # vencimientos, que vive en el futuro.
    share_weekly_cutoff=False,
    grid_sections=frozenset({"Todos los instrumentos","Bancos","Fondos de Pensiones y AFC","Fondos Mutuos","Compañías de Seguros","Mandantes","Corredores de Bolsa","Otros"})
)
