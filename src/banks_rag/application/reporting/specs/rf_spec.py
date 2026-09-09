"""Spec curado del "Informe Renta Fija" (familia rf).

Réplica del correo real del BCCh: Tabla N°1 de inversiones globales, la sección de
curvas (BTP / BTU / breakeven / swap spreads), los flujos por instrumento apilados
por agente, la intermediación financiera y las curvas históricas — en el ORDEN del
correo y agrupados por tema. Cada bloque mapea a un parquet del catálogo + una
transform + un tipo de gráfico.

Los bloques cuyo dato NO existe en los parquets locales quedan igual, en su posición,
con ``STATUS_SKIP``: el informe conserva la estructura del correo y la ``note`` dice
qué serie falta traer. Es a propósito — borrarlos escondería el hueco.

Diferencias conocidas contra el correo real (el dato que tenemos no llega tan fino):

- Los AGENTES del correo son Banco / FP y AFC / FFMM / CSV / Otros / Mandantes y DV /
  NR; el parquet DCV agregado trae ``Sector`` ∈ {AFP, Bancos, CB, CS, FFMM, Mandantes,
  Otros}: no incluye NR y "Mandantes y DV" está partido en CB + Mandantes. NR sí tiene
  su propia tenencia (``nr_tenencia_cf_soberanos``, la que grafica el informe nr), que
  es la que usa el panel de variación acumulada.
- "BC" (bono corporativo) en el correo es ``Tipo='BE'`` (bono de empresa) en el parquet.
- Los flujos BTP/BTU del correo abren por AÑO de vencimiento (2026…>2040); el parquet
  solo trae 5 tramos de plazo, así que el eje X son tramos.
- La intermediación del correo abre por plazo 7d/30d/90d/180d/360d. Eso existe solo
  para DAP (``dap_flujos_plazo``); PDBC sale del DCV, con sus tramos.

Esta es la ÚNICA pieza a editar para ajustar el informe rf. Las transforms viven en
``series_transforms.py``; el render en ``svg_chart.py``.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import (
    STATUS_MVP,
    STATUS_SKIP,
    FamilyReportSpec,
    ReportBlock,
)

_S_INVERSIONES = "Inversiones Globales"
_S_CURVAS = "Variación Curvas"
_S_FLUJOS = "Flujos BTP-BTU"
_S_FLUJOS_BB_BC = "Flujos Bonos Bancarios y Corporativos"
_S_INTERMEDIACION = "Intermediación Financiera"
_S_VAR_AGENTE = "Variación acumulada DCV por Agente"
_S_CURVAS_HIST = "Curvas Históricas"

# Agentes en el orden del correo (Banco, FP y AFC, FFMM, CSV, Otros, Mandantes y DV),
# con el nombre que usa el parquet. "CB" (corredores de bolsa) va al final: el correo
# no lo abre como fila propia. "NR" no está en el enum de ``Sector`` del DCV agregado
# — su tenencia vive en su propio parquet (ver el bloque NR más abajo).
_AGENTES = ["Bancos", "AFP", "FFMM", "CS", "Otros", "Mandantes", "CB"]

# Los flujos del correo comparan contra T-5 (una semana hábil).
_FLUJO_PARAMS = {"window": "5d", "series_col": "Sector", "order": _AGENTES}


def _flujo(section: str, title: str, tipo: str, *, moneda: str | None = None,
           unit: str = "USD Mill.", note: str = "") -> ReportBlock:
    """Un gráfico de flujos: Δ T-5 del stock DCV de UN instrumento, eje X = tramo de
    plazo, apilado por agente. Todos salen del MISMO parquet; lo que los distingue es
    el filtro (``Tipo``, y para el DAP también ``Moneda``)."""
    filters: dict[str, str] = {"Tipo": tipo}
    if moneda:
        filters["Moneda"] = moneda
    return ReportBlock(
        section=section, title=title, unit=unit, chart="stacked_bar", status=STATUS_MVP,
        source_id="variacion_instrumento_todos_plazo", transform="stacked_by_bucket",
        params={**_FLUJO_PARAMS, "filters": filters},
        note=note,
    )



def _acumulado(title: str, source_id: str, *, value: str = "Stock_USD",
               window: str = "d120", note: str = "") -> ReportBlock:
    """Variación acumulada del stock DCV de un agente, una línea por instrumento.

    Cada agente tiene su parquet dedicado y DIARIO (``stock_nivel_afp``,
    ``stock_nivel_ffmm``, ``stock_nivel_bancos``: Fecha x Tipo x Stock_USD) — son los
    mismos que usan los informes de esas familias. Sale mejor que filtrar el DCV
    agregado por ``Sector``: ya viene al grano, sin sumar sobre plazo y moneda.

    ``accumulate="rebase"`` (no ``cumsum``): el parquet trae NIVELES de stock, así que
    la variación acumulada es nivel(t) - nivel(t0). Un cumsum sobre un nivel sumaría
    stocks día tras día y daría un número sin sentido."""
    return ReportBlock(
        section=_S_VAR_AGENTE, title=title, unit="US$ Mill.", chart="line",
        status=STATUS_MVP, source_id=source_id, transform="category_series",
        params={"category": "Tipo", "value": value,
                "accumulate": "rebase", "window": window,
                "order": ['BB', 'BE', 'BTP', 'BTU', 'DAP', 'Otros', 'PDBC','Letras MdH']},
        note=note,
    )


_BLOCKS: tuple[ReportBlock, ...] = (
    # ── Tabla N°1: Inversiones globales ──────────────────────────────────────
    # ``variacion_instrumento_todos_plazo`` (Fecha, Bucket, Tipo, Sector, Moneda,
    # Stock_USD), sumado sobre Bucket: filas = Sector (agente), columnas =
    # instrumento (con moneda solo si el instrumento trae más de una, p.ej. "DAP
    # CLP"/"DAP UF" — PDBC/BTP/BTU son de una sola moneda y quedan sin sufijo).
    # USD/EURO se excluyen (no son la inversión "global" en pesos/UF que muestra
    # esta tabla); BCCh también se excluye completo. BE en CLP igual se saca, pero
    # BE en UF SÍ va, rotulado "BC" (el bono corporativo del correo).
    # ``days=[1, 5]`` reemplaza el Δ T-7 / Δ T-30 del informe DCV por el Δ T-1 / Δ T-5
    # de este.
    ReportBlock(
        section=_S_INVERSIONES, title="Tabla N°1: Inversiones globales",
        unit="MM USD$", chart="heatmap_table", status=STATUS_MVP,
        source_id="variacion_instrumento_todos_plazo", transform="dcv_heatmap_moneda",
        params={"days": [1, 5],
                "exclude_moneda": ["USD", "EURO"],
                "exclude_tipo": ["BCCh"],
                "exclude_tipo_moneda": [["BE", "CLP"]],
                "label_overrides": {"BE": {"UF": "BC"}},
                "moneda_order": ["CLP", "UF"],
                # Columnas en el orden del correo (liquidez), no alfabético.
                "order_cols": ["PDBC", "DAP", "BTP", "BTU", "BB", "BE", "Letras MdH", "Otros"],
                "order_rows": _AGENTES},
        note="*Información DCV actualizada a T-1, por lo que las variaciones indicadas "
             "son respecto a esa fecha.",
        full_width=True,
    ),


    # ── Variación Curvas ─────────────────────────────────────────────────────
    # Curvas por PLAZO en dos cortes (hoy vs t-5): eje X categórico, no temporal.
    # ``chart="curve"`` las dibuja como líneas sobre ese eje (``_render_grouped_lines``).
    ReportBlock(
        section=_S_CURVAS, title="Curva BTP %", unit="%", chart="curve", status=STATUS_MVP,
        source_id="btp_curva", transform="curve_by_tenor",
        params={"window": "5d", "tenor_col": "Tenor", "labels": ["Hoy", "t-5"]},
        full_width=True,
    ),
    ReportBlock(
        section=_S_CURVAS, title="Curva BTU %", unit="%", chart="curve", status=STATUS_MVP,
        source_id="btu_curva", transform="curve_by_tenor",
        params={"window": "5d", "tenor_col": "Tenor", "labels": ["Hoy", "t-5"]},
    ),
    ReportBlock(
        section=_S_CURVAS, title="Breakeven Inflación %", unit="%", chart="curve",
        status=STATUS_MVP, source_id="bei_curve", transform="curve_by_tenor",
        params={"window": "5d", "labels": ["Hoy", "t-5"]},
        note="*El parquet trae 4 plazos (2Y/5Y/10Y/20Y); el correo abre 7.",
    ),

    ReportBlock(
        section=_S_CURVAS, title="Variación Semanal Curvas CLP y US (pb)", unit="pb",
        chart="grouped_bar", status=STATUS_MVP,
        source_id="spc_ois_var", transform="window_stacked_two_cat",
        params={"group": "Tenor", "series": "Serie", "value": "Valor",
                "window_days": 8, "net": False,
                "group_order": ["3M", "6M", "9M", "12M", "18M", "2Y", "5Y", "10Y"]},
        note="*El correo compara Δ UST / Δ BTP / Δ SPC; el parquet disponible trae SPC y OIS.",
    ),
    ReportBlock(
        section=_S_CURVAS, title="Spread BTP-UST por plazo (pb)", unit="pb", chart="curve",
        status=STATUS_MVP, source_id="curva_spread_btp_ust", transform="curve_by_tenor",
        params={"window": "5d", "tenor_col": "Tenor", "labels": ["Hoy", "t-5"]},
    ),
    ReportBlock(
        section=_S_CURVAS, title="Swap Spread BTP (pb)", unit="pb", chart="curve",
        status=STATUS_MVP, source_id="curva_spread_btp_spc", transform="curve_by_tenor",
        params={"window": "5d", "tenor_col": "Tenor", "labels": ["Hoy", "t-5"]},
    ),
    ReportBlock(
        section=_S_CURVAS, title="Swap Spread BTU (pb)", unit="pb", chart="curve",
        status=STATUS_SKIP,
        note="Falta el parquet: curva de spread BTU-SPC por plazo (O BTU).",
    ),

    ReportBlock(
        section=_S_CURVAS, title="Curva Swap vs BTP (%)", unit="%", chart="curve",
        status=STATUS_MVP, source_id="spc_ois_var", transform="spc_btp_curve",
        params={"other_file": "btp_plazo.parquet", "tenors": ["2Y", "5Y", "10Y", "20Y"]},
        note="*SPC no trae tramo 20Y; esa curva queda con un punto menos que la BTP.",
    ),

    # ── Flujos BTP-BTU ───────────────────────────────────────────────────────

    ReportBlock(
        section=_S_FLUJOS, title="Flujos BTP ΔT-5 (USD Mill.)", unit="USD Mill.", chart="stacked_bar"
        ,status=STATUS_MVP, source_id="variacion_sector_todos_año", transform="stacked_by_bucket",
        params={"x_col":"Plazo","window": "7d", "series_col": "Sector", "order": _AGENTES, "filters": {"Tipo":"BTP"}},
        note="BTP: Δ T-5 del stock DCV por tramo de plazo, apilado por agente.",
    ),

    ReportBlock(
            section=_S_FLUJOS, title="Flujos BTU ΔT-5 (USD Mill.)", unit="USD Mill.", chart="stacked_bar"
            ,status=STATUS_MVP, source_id="variacion_sector_todos_año", transform="stacked_by_bucket",
            params={"x_col":"Plazo","window": "7d", "series_col": "Sector", "order": _AGENTES, "filters": {"Tipo":"BTU"}},
            note="BTU: Δ T-5 del stock DCV por tramo de plazo, apilado por agente.",
        ),


    # ── Flujos Bonos Bancarios y Corporativos ────────────────────────────────
    ReportBlock(
            section=_S_FLUJOS_BB_BC, title="Flujos BB ΔT-5 (USD Mill.)", unit="USD Mill.", chart="stacked_bar"
            ,status=STATUS_MVP, source_id="variacion_instrumento_todos_plazo", transform="stacked_by_bucket",
            params={"x_col":"Bucket","window": "7d", "series_col": "Sector", "order": _AGENTES, "filters": {"Tipo":"BB"}},
            note="BB: Δ T-5 del stock DCV por tramo de plazo, apilado por agente.",
        ),
    
        ReportBlock(
                section=_S_FLUJOS_BB_BC, title="Flujos BE ΔT-5 (USD Mill.)", unit="USD Mill.", chart="stacked_bar"
                ,status=STATUS_MVP, source_id="variacion_instrumento_todos_plazo", transform="stacked_by_bucket",
                params={"x_col":"Bucket","window": "7d", "series_col": "Sector", "order": _AGENTES, "filters": {"Tipo":"BE"}},
                note="BE: Δ T-5 del stock DCV por tramo de plazo, apilado por agente.",
            ),

    

    # ── Intermediación Financiera ────────────────────────────────────────────
    _flujo(_S_INTERMEDIACION, "Flujos PDBC ΔT-5 (USD Mill.)", "PDBC",
           note="*Pendiente por plazo 7d/30d/90d/180d/360d; el DCV trae tramos."),
 
    ReportBlock(
        section=_S_INTERMEDIACION, title="Flujos DAP $ (MM USD)", unit="MM USD",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="dap_flujos_plazo", transform="window_stacked_two_cat",
        params={"group": "Bucket", "series": "Sector", "value": "Monto",
                "filter_col": "Moneda", "filter_val": "Pesos", "window_days": 1, "net": False,
                "group_order": ["14", "30", "60", "90", "180", "360", "540", "720", ">720"]},
    ),
    ReportBlock(
        section=_S_INTERMEDIACION, title="Flujos DAP UF (MM USD)", unit="MM USD",
        chart="stacked_bar", status=STATUS_MVP,
        source_id="dap_flujos_plazo", transform="window_stacked_two_cat",
        params={"group": "Bucket", "series": "Sector", "value": "Monto",
                "filter_col": "Moneda", "filter_val": "UF", "window_days": 1, "net": False,
                "group_order": ["14", "30", "60", "90", "180", "360", "540", "720", ">720"]},
    ),

    # ── Variación acumulada DCV por Agente ───────────────────────────────────
    _acumulado("Fondos de Pensiones (US$ Mill., YoY)", "stock_nivel_afp"),
    _acumulado("Fondos Mutuos (US$ Mill., YoY)", "stock_nivel_ffmm"),
    # NR no está en el ``Sector`` del DCV agregado, pero sí tiene su propia tenencia de
    # soberanos (la misma que grafica el informe nr). Es MENSUAL y solo BTP/BTU, así
    # que la ventana es más larga para que la línea tenga puntos suficientes.
    _acumulado("NR (US$ Mill., YoY)", "nr_tenencia_cf_soberanos", value="mmusd",
               window="y2",
               note="*Tenencia NR de soberanos: mensual y solo BTP/BTU (el DCV agregado "
                    "no abre No Residentes)."),
    _acumulado("Bancos (US$ Mill., YoY)", "stock_nivel_bancos"),

    # ── Curvas Históricas ────────────────────────────────────────────────────
    ReportBlock(
        section=_S_CURVAS_HIST, title="Pendientes BTP", unit="%", chart="line",
        status=STATUS_MVP, source_id="pendiente_btp_btu", date_from="2025-01-01", transform="category_series",
        params={"category": "Serie", "value": "Valor",
                "order": ["BTP_10-5", "BTP_10-2"],
                "labels": {"BTP_10-5": "BTP 2-5", "BTP_10-2": "BTP 2-10"}},
    ),
    ReportBlock(
        section=_S_CURVAS_HIST, title="Swap Spread", unit="pb", chart="line",
        status=STATUS_MVP, source_id="spread_btp_spc", date_from="2024-01-01", transform="category_series",
        params={"category": "Serie", "value": "Valor", "order": ["5Y", "10Y"],
                "labels": {"5Y": "BTP 5", "10Y": "BTP 10"}},
    ),
    ReportBlock(
        section=_S_CURVAS_HIST, title="Spread BTP-UST", unit="pb", chart="line",
        status=STATUS_MVP, source_id="spread_btp_ust", date_from="2024-01-01", 
        transform="category_series",
        params={"category": "Serie", "value": "Valor", "order": ["5Y", "10Y"],
                "labels": {"5Y": "Spread BTP-UST 5 yr", "10Y": "Spread BTP-UST 10 yr"}},
    ),
    ReportBlock(
        section=_S_CURVAS_HIST, title="Tasa UF Bono Bancario y Corp (%)", unit="%",
        chart="line", status=STATUS_SKIP,
        note="Falta en SQL: tasa UF de bono bancario AA y de empresa A a 5-7 años "
             "(el catálogo trae montos transados de BB/BC, no sus tasas).",
    ),
    ReportBlock(
        section=_S_CURVAS_HIST, title="Spread Bono Bancario AAA (pb)", unit="pb",
        chart="line", status=STATUS_SKIP,
        note="Falta en SQL: spread del bono bancario contra SPC y contra BTU.",
    ),
    ReportBlock(
        section=_S_CURVAS_HIST, title="Spread BC - SPC (pb)", unit="pb",
        chart="line", status=STATUS_SKIP,
        note="Falta en SQL: spread del bono de empresa UF-AA y UF-A a 5-7 años "
             "contra SPC.",
    ),
)

RF_SPEC = FamilyReportSpec(
    family="rf",
    title="Informe Renta Fija",
    blocks=_BLOCKS,
    # Los parquets vienen de tableros distintos (DCV cierra a T-1, las curvas al día,
    # el DAP es mensual): sin corte común cada bloque se ancla al máximo de SU parquet
    # en vez de arrastrar todo a la fecha del más atrasado.
    share_weekly_cutoff=False,

    grid_sections=frozenset({
        _S_CURVAS, _S_FLUJOS, _S_FLUJOS_BB_BC, _S_INTERMEDIACION,
        _S_VAR_AGENTE, _S_CURVAS_HIST,
    }),
)
