"""Transformaciones parquet → ``PlotData`` para los bloques del informe curado.

Cada bloque (``ReportBlock.transform``) nombra una de estas funciones. Todas leen
del parquet REAL (vía ``compute_series`` / ``detect_roles``), nunca de agregados
del LLM — el gráfico es verificación independiente de la prosa.

Las transforms con forma propia ya implementada (serie natural, filtro por fondo,
allocation/variación por instrumento) devuelven su ``PlotData``. Las que aún no
tienen su agregación exacta (mensual, ventanas, acumulado YtD, etc.) quedan
REGISTRADAS como ``None``: el builder igual dibuja esos bloques como VISTA
PRELIMINAR (la serie natural del parquet) y refina el tipo de gráfico en la 2ª
iteración. Solo las que no tienen equivalente como serie (tablas-heatmap) terminan
en placeholder.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import duckdb

from banks_rag.domain.agent.chart_types import chart_family
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset

from .parquet_facts import (
    _MAX_PLOT_CATEGORIES,
    _MAX_PLOT_SERIES,
    HtmlTable,
    PlotData,
    PlotSeries,
    _aggregate_by_category,
    _downsample,
    _read_rows,
    _read_series_rows,
    compute_series,
    detect_roles,
    geom_return,
    geom_ytd,
    weekly_delta,
)

log = logging.getLogger(__name__)

# Firma de una transform: (dataset, parquet_dir, params) → PlotData | HtmlTable | None.
Transform = Callable[[ParquetDataset, Path, dict], "PlotData | HtmlTable | None"]


# ── Transforms MVP (implementadas) ───────────────────────────────────────────

def straight_series(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Serie natural del parquet tal cual la detecta ``compute_series``
    (multi-línea para categóricas/wide, línea simple, o composición snapshot)."""
    return compute_series(dataset, parquet_dir)


def snapshot_composition(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Composición de un corte transversal (sin fecha) → barras de composición."""
    return compute_series(dataset, parquet_dir)


def filter_fund(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Serie temporal restringida a ciertas categorías (``params['funds']``),
    respetando ese orden. P.ej. Duración solo Tipo 1 / Tipo 2."""
    funds = list(params.get("funds") or [])
    if not funds:
        return compute_series(dataset, parquet_dir)
    return compute_series(dataset, parquet_dir, category_filter=funds)


def _fund_instrument_by_inst(
    dataset: ParquetDataset, parquet_dir: Path, fund: str | None,
) -> dict[str, list[tuple[str, float]]] | None:
    """Para parquets con DOS categóricas (instrumento x tipo de fondo): filtra por
    ``fund`` (la columna de fondo) y devuelve ``{instrumento: [(iso, valor)]}``
    (niveles mensuales). ``None`` si el parquet no tiene esa forma."""
    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        if roles.date_col is None or len(roles.category_cols) < 2 or not roles.value_cols:
            return None
        fund_col = next((c for c in roles.category_cols if "fondo" in c.lower()), roles.category_cols[-1])
        inst_col = next((c for c in roles.category_cols if c != fund_col), roles.category_cols[0])
        val_col = roles.value_cols[0]
        rows = _read_series_rows(
            con, parquet_path, date_col=roles.date_col,
            columns=[roles.date_col, fund_col, inst_col, val_col],
        )
        if fund:
            rows = [r for r in rows if str(r.get(fund_col)) == fund]
        return _aggregate_by_category(rows, roles.date_col, inst_col, val_col)
    finally:
        con.close()


def allocation_by_fund(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Allocation de UN tipo de fondo: niveles por instrumento sobre el tiempo
    (multi-línea; el bloque la dibuja como área apilada)."""
    by_inst = _fund_instrument_by_inst(dataset, parquet_dir, params.get("fund"))
    if by_inst is None:
        return compute_series(dataset, parquet_dir)
    top = sorted(by_inst, key=lambda c: abs(by_inst[c][-1][1]) if by_inst[c] else 0.0, reverse=True)[:_MAX_PLOT_SERIES]
    series = [PlotSeries(label=c, points=_downsample(by_inst[c])) for c in top if by_inst[c]]
    plot = PlotData(dataset.id, chart_family(dataset.chart_type), "timeseries", dataset.unit, series)
    return None if plot.is_empty() else plot


def monthly_var_alloc(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Variación Mes y YtD de la cartera de UN tipo de fondo, por instrumento.

    Barras agrupadas: eje X = instrumento, dos series Mes (último mes vs anterior)
    y YtD (último mes vs inicio del año). Replica los cuatro paneles del informe.
    """
    by_inst = _fund_instrument_by_inst(dataset, parquet_dir, params.get("fund"))
    if not by_inst:
        return None
    mes: dict[str, float] = {}
    ytd: dict[str, float] = {}
    for inst, s in by_inst.items():
        if not s:
            continue
        last_iso, last_v = s[-1]
        prev_v = s[-2][1] if len(s) >= 2 else 0.0
        in_year = [v for iso, v in s if iso[:4] == last_iso[:4]]
        base = in_year[0] if in_year else s[0][1]
        mes[inst] = last_v - prev_v
        ytd[inst] = last_v - base
    insts = sorted(ytd, key=lambda k: abs(ytd[k]), reverse=True)[:_MAX_PLOT_SERIES + 2]
    # Fechas explícitas: Mes = mes previo → último; YtD = inicio de año → último.
    all_dates = sorted({iso for s in by_inst.values() for iso, _ in s})
    note = ""
    if all_dates:
        last = all_dates[-1]
        prev = all_dates[-2] if len(all_dates) >= 2 else last
        ystart = next((d for d in all_dates if d[:4] == last[:4]), all_dates[0])
        note = (f"Mes: {_fmt_date(prev)} → {_fmt_date(last)}  ·  "
                f"YtD: {_fmt_date(ystart)} → {_fmt_date(last)}")
    return _grouped(dataset.id, dataset.unit, {"Mes": mes, "YtD": ytd}, insts, ["Mes", "YtD"],
                    date_note=note)


# ── Transforms que producen datos "grouped" (barras agrupadas/apiladas) ───────

_MONTHS_ES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def _fmt_date(iso: str) -> str:
    """``YYYY-MM-DD`` → ``DD-mmm-YYYY`` (legible: 10-jun-2026)."""
    try:
        d = date.fromisoformat(iso[:10])
    except ValueError:
        return iso[:10]
    return f"{d.day:02d}-{_MONTHS_ES[d.month - 1]}-{d.year}"


def _month_key(iso: str) -> str:
    return iso[:7]  # YYYY-MM


def _month_label(month_key: str) -> str:
    return f"{_MONTHS_ES[int(month_key[5:7]) - 1]}{month_key[2:4]}"


def _grouped(
    dataset_id: str, unit: str,
    series_dict: dict[str, dict[str, float]], cat_order: list[str], series_order: list[str],
    *, overlay: tuple[str, ...] = (), date_note: str = "",
) -> PlotData | None:
    """``{serie: {categoria: valor}}`` → PlotData ``kind='grouped'`` (eje X =
    categorías, una serie por color). ``overlay`` marca series que se dibujan
    superpuestas (punto "Neto"/"Total" por categoría) en vez de barra. ``date_note``
    hace explícitas las fechas/ventanas que cubre el gráfico (eje X no temporal)."""
    series = [
        PlotSeries(label=sl, points=[(c, series_dict[sl].get(c, 0.0)) for c in cat_order])
        for sl in series_order
    ]
    plot = PlotData(dataset_id, "grouped_bar", "grouped", unit, series, overlay=overlay,
                    date_note=date_note)
    return None if not cat_order or plot.is_empty() else plot


def _read_fund_series(dataset: ParquetDataset, parquet_dir: Path) -> dict[str, list[tuple[str, float]]]:
    """``{categoria: [(iso, valor)]}`` de un parquet (fecha + 1 categórica + valor)."""
    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return {}
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        if roles.date_col is None or not roles.category_cols or not roles.value_cols:
            return {}
        cat_col, val_col = roles.category_cols[0], roles.value_cols[0]
        rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, cat_col, val_col])
        return _aggregate_by_category(rows, roles.date_col, cat_col, val_col)
    finally:
        con.close()


def _win_meta(win: str) -> tuple[str, int | None]:
    """``"7d"`` → ("Δ7d", 7); ``"ytd"`` → ("ΔYtD", None=desde 1-ene)."""
    w = win.strip().lower()
    if w == "ytd":
        return "ΔYtD", None
    days = {"7d": 7, "30d": 30, "90d": 90}.get(w, 30)
    return f"Δ{w}", days


def _compound_pct(i_start_pct: float, i_end_pct: float) -> float:
    """Retorno COMPUESTO (%) entre dos valores de un índice de retorno acumulado
    expresado en PORCENTAJE (``retornos_fondo_ffmm``: 39,28 ↔ i=0,3928):
    ``(1+i_end)/(1+i_start)-1``. Diferencia GEOMÉTRICA del retorno, no la resta del
    índice (que sobreestima al crecer el índice)."""
    denom = 1.0 + i_start_pct / 100.0
    return ((1.0 + i_end_pct / 100.0) / denom - 1.0) * 100.0 if denom != 0 else 0.0


def _geom_window_pct(points_pct: list[tuple[str, float]], start: str, last: str, *, ytd: bool) -> float:
    """Retorno COMPUESTO (%) de una ventana sobre un índice de retorno acumulado en
    PORCENTAJE. Saca el retorno diario por diferencia geométrica y lo compone,
    reusando las MISMAS primitivas que el gráfico de rentabilidad acumulada
    (``geom_return``/``geom_ytd``, que operan en fracción) para que ambos coincidan;
    para YtD la base es el cierre del año previo, idéntico a ``ytd_return_geom``."""
    if not points_pct:
        return 0.0
    frac = [(iso, v / 100.0) for iso, v in points_pct]  # % → fracción para 1+i
    if ytd:
        ser = geom_ytd(frac, start)
        return ser[-1][1] * 100.0 if ser else 0.0
    r = geom_return(frac, start, last)
    return r * 100.0 if r is not None else 0.0


def window_returns(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Rentabilidad por ventana (Δ7d/Δ30d/ΔYtD) y tipo de fondo → barras agrupadas
    (eje X = fondo, una serie por ventana). Retorno COMPUESTO (geométrico) del índice
    de retorno acumulado, NO la resta del índice."""
    funds = params.get("funds") or ["Tipo 1", "Tipo 2", "Tipo 3", "Tipo 6"]
    windows = params.get("windows") or ["7d", "30d"]
    by_fund = _read_fund_series(dataset, parquet_dir)
    if not by_fund:
        return None
    last = max(p[0] for s in by_fund.values() for p in s)
    series_dict: dict[str, dict[str, float]] = {}
    labels: list[str] = []
    note_parts: list[str] = []
    for win in windows:
        label, days = _win_meta(win)
        labels.append(label)
        start = f"{last[:4]}-01-01" if days is None else (date.fromisoformat(last[:10]) - timedelta(days=days)).isoformat()
        series_dict[label] = {f: _geom_window_pct(by_fund.get(f, []), start, last, ytd=days is None) for f in funds}
        note_parts.append(f"{label}: {_fmt_date(start)} → {_fmt_date(last)}")
    return _grouped(dataset.id, "%", series_dict, funds, labels,
                    date_note="  ·  ".join(note_parts))


# ── Rentabilidad acumulada GEOMÉTRICA (índice de retorno) ─────────────────────
#
# retorno_acum_ffmm es un índice de retorno acumulado GEOMÉTRICO en fracción desde
# 2019 (1+i_t = ∏(1+r_s)). El retorno diario se despeja con la fórmula inversa
# r_t = (1+i_t)/(1+i_{t-1})-1 y el acumulado de una ventana es el COMPUESTO
# (1+i_t)/(1+i_base)-1 — NO la resta del índice (que sobreestima: para YTD da
# ~2,5% vs ~1,8% real). Estas funciones lo computan en FRACCIÓN; curated aplica
# value_scale 100 → %.

_DEFAULT_FUNDS = ["Tipo 1", "Tipo 2", "Tipo 3", "Tipo 6"]


def ytd_return_geom(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Rentabilidad acumulada YTD por fondo, GEOMÉTRICA, desde el índice de retorno
    acumulado ancho (col por fondo). Despeja el diario y lo recompone desde el 1-ene
    del último año con datos. ``params['funds']`` filtra (default Tipo 1/2/3/6);
    ``"all"`` incluye TODOS los fondos disponibles del parquet (sin tope)."""
    from banks_rag.infrastructure.sql import series_analytics as sa

    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None
    funds = params.get("funds")
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        if funds == "all":
            cols = list(roles.value_cols)  # TODOS los fondos disponibles (sin tope)
        else:
            wanted = funds or _DEFAULT_FUNDS
            cols = [c for c in wanted if c in roles.value_cols] or roles.value_cols[:_MAX_PLOT_SERIES]
        rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, *cols])
    finally:
        con.close()

    last = max((r[roles.date_col] for r in rows if r.get(roles.date_col)), default="")
    if not last:
        return None
    ytd_start = f"{last[:4]}-01-01"
    series: list[PlotSeries] = []
    for c in cols:
        ytd = geom_ytd(sa.clean_series(rows, roles.date_col, c), ytd_start)
        if ytd:
            series.append(PlotSeries(label=c, points=_downsample(ytd)))
    if not series:
        return None
    note = f"Rentabilidad acumulada (geométrica) desde {_fmt_date(ytd_start)} → {_fmt_date(last)}"
    return PlotData(dataset.id, chart_family(dataset.chart_type), "timeseries",
                    dataset.unit, series, date_note=note)


def _month_end_values(points: list[tuple[str, float]]) -> dict[str, float]:
    """``{month_key: último valor del mes}`` (índice a fin de mes; puntos ordenados)."""
    me: dict[str, float] = {}
    for iso, v in points:
        me[_month_key(iso)] = v
    return me


def _monthly_sum(by_fund: dict[str, list[tuple[str, float]]], funds: list[str], months: int) -> tuple[dict, list[str]]:
    """Suma por mes por fondo (para FLUJOS); devuelve ``({fondo:{label_mes:val}},
    [labels de los últimos N meses])``."""
    per_fund: dict[str, dict[str, float]] = {}
    all_keys: set[str] = set()
    for f in funds:
        agg: dict[str, float] = {}
        for iso, v in by_fund.get(f, []):
            k = _month_key(iso)
            agg[k] = agg.get(k, 0.0) + v
            all_keys.add(k)
        per_fund[f] = agg
    keys = sorted(all_keys)[-months:]
    out = {f: {_month_label(k): per_fund[f].get(k, 0.0) for k in keys} for f in funds}
    return out, [_month_label(k) for k in keys]


def monthly_sum_by_fund(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Flujo mensual (suma) por tipo de fondo → barras agrupadas (X = mes)."""
    funds = params.get("funds") or ["Tipo 1", "Tipo 3", "Tipo 6"]
    by_fund = _read_fund_series(dataset, parquet_dir)
    if not by_fund:
        return None
    series_dict, labels = _monthly_sum(by_fund, funds, params.get("months", 6))
    last = max((iso for s in by_fund.values() for iso, _ in s), default="")
    note = f"Suma por mes · datos hasta {_fmt_date(last)}" if last else ""
    return _grouped(dataset.id, dataset.unit, series_dict, labels, funds, date_note=note)


def monthly_returns(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Rentabilidad mensual por tipo de fondo → barras agrupadas (X = mes).

    Retorno COMPUESTO del mes desde el índice de retorno acumulado (geométrico):
    ``(1+i_finmes)/(1+i_finmes_previo)-1`` — saca el retorno diario por diferencia
    geométrica y lo compone en el mes, NO la resta del índice acumulado.
    """
    funds = params.get("funds") or ["Tipo 1", "Tipo 2", "Tipo 3"]
    months = params.get("months", 8)
    by_fund = _read_fund_series(dataset, parquet_dir)
    if not by_fund:
        return None
    me = {f: _month_end_values(by_fund.get(f, [])) for f in funds}
    all_keys = sorted({k for d in me.values() for k in d})
    keys = all_keys[-months:]
    sd: dict[str, dict[str, float]] = {}
    for f in funds:
        col: dict[str, float] = {}
        for k in keys:
            i = all_keys.index(k)
            prev = all_keys[i - 1] if i > 0 else None
            cur, base = me[f].get(k), (me[f].get(prev) if prev else None)
            col[_month_label(k)] = _compound_pct(base, cur) if cur is not None and base is not None else 0.0
        sd[f] = col
    last = max((iso for s in by_fund.values() for iso, _ in s), default="")
    note = f"Retorno por mes · datos hasta {_fmt_date(last)}" if last else ""
    return _grouped(dataset.id, "%", sd, [_month_label(k) for k in keys], funds, date_note=note)


_BUCKET_ORDER = ("Menor a 1Y", "Entre 1 y 2Y", "Entre 2 y 5Y", "Entre 5 y 10Y", "Mayor a 10Y")


def _order_buckets(keys) -> list[str]:
    return sorted(keys, key=lambda b: _BUCKET_ORDER.index(b) if b in _BUCKET_ORDER else 99)


def _bucket_tipo_series(
    dataset: ParquetDataset, parquet_dir: Path,
) -> tuple[dict[str, dict[str, list[tuple[str, float]]]], str] | tuple[None, None]:
    """``{bucket: {instrumento: [(iso, valor)]}}`` + última fecha, de un parquet
    con plazo (Bucket) x instrumento (Tipo) x valor (suma sobre Moneda u otras)."""
    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None, None
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        if roles.date_col is None or len(roles.category_cols) < 2 or not roles.value_cols:
            return None, None
        cats = roles.category_cols
        bucket_col = next((c for c in cats if c.lower() in ("bucket", "plazo")), cats[0])
        tipo_col = next((c for c in cats if c != bucket_col and "moneda" not in c.lower()), cats[1])
        val_col = roles.value_cols[0]
        rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, bucket_col, tipo_col, val_col])
    finally:
        con.close()

    nested: dict[str, dict[str, dict[str, float]]] = {}
    last = ""
    for r in rows:
        b, t, f, raw = r.get(bucket_col), r.get(tipo_col), r.get(roles.date_col), r.get(val_col)
        if b is None or t is None or f is None or raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        per = nested.setdefault(str(b), {}).setdefault(str(t), {})
        per[str(f)] = per.get(str(f), 0.0) + v
        last = max(last, str(f))
    out = {b: {t: sorted(d.items()) for t, d in tipos.items()} for b, tipos in nested.items()}
    return out, last


def _top_tipos(sd: dict[str, dict[str, float]]) -> list[str]:
    tot = {t: sum(abs(x) for x in vals.values()) for t, vals in sd.items()}
    return sorted(tot, key=lambda t: tot[t], reverse=True)[:_MAX_PLOT_SERIES]


def composition_by_bucket(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Composición DCV por plazo a la fecha de corte → barras apiladas
    (X = plazo, una serie por instrumento)."""
    bt, last = _bucket_tipo_series(dataset, parquet_dir)
    if not bt:
        return None
    buckets = _order_buckets(bt)
    sd: dict[str, dict[str, float]] = {}
    for b in buckets:
        for t, s in bt[b].items():
            if s:
                sd.setdefault(t, {})[b] = s[-1][1]
    tipos = _top_tipos(sd)
    note = f"Corte: {_fmt_date(last)}" if last else ""
    return _grouped(dataset.id, dataset.unit, {t: sd[t] for t in tipos}, buckets, tipos,
                    date_note=note)


def stacked_by_bucket(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Variación de la ventana (default 7d) del stock DCV por plazo → barras
    apiladas (X = plazo, una serie por instrumento). ``net_as_overlay`` agrega un
    punto "Neto" = suma de instrumentos por plazo (como el informe)."""
    bt, last = _bucket_tipo_series(dataset, parquet_dir)
    if not bt:
        return None
    days = {"7d": 7, "30d": 30}.get(str(params.get("window", "7d")), 7)
    # Corte semanal común del informe (si lo hay) en vez del máximo de este parquet:
    # así la variación de la barra coincide con la del texto. weekly_delta replica
    # exactamente la base at-or-before de antes cuando asof == last.
    asof = str(params.get("weekly_asof") or last)
    start = (date.fromisoformat(asof[:10]) - timedelta(days=days)).isoformat()
    buckets = _order_buckets(bt)
    sd: dict[str, dict[str, float]] = {}
    for b in buckets:
        for t, s in bt[b].items():
            if not s:
                continue
            wd = weekly_delta(s, asof, days=days, is_flow=False)
            if wd is not None:
                sd.setdefault(t, {})[b] = wd["cambio_absoluto"]
    tipos = _top_tipos(sd)
    out = {t: sd[t] for t in tipos}
    series_order = list(tipos)
    overlay: tuple[str, ...] = ()
    if params.get("net_as_overlay"):
        out["Neto"] = {b: sum(sd[t].get(b, 0.0) for t in tipos) for b in buckets}
        series_order.append("Neto")
        overlay = ("Neto",)
    note = f"Variación {_fmt_date(start)} → {_fmt_date(asof)}"
    return _grouped(dataset.id, dataset.unit, out, buckets, series_order, overlay=overlay,
                    date_note=note)


def monthly_diff(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Flujo mensual neto (suma por mes) de una serie simple → barras (X = mes)."""
    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None
    from banks_rag.infrastructure.sql import series_analytics as sa
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, roles.value_cols[0]])
        series = sa.clean_series(rows, roles.date_col, roles.value_cols[0])
    finally:
        con.close()
    by_month: dict[str, float] = {}
    for iso, v in series:
        by_month[_month_key(iso)] = by_month.get(_month_key(iso), 0.0) + v
    keys = sorted(by_month)[-params.get("months", 12):]
    sd = {"Flujo mensual": {_month_label(k): by_month[k] for k in keys}}
    last = series[-1][0] if series else ""
    note = f"Suma por mes · datos hasta {_fmt_date(last)}" if last else ""
    return _grouped(dataset.id, dataset.unit, sd, [_month_label(k) for k in keys], ["Flujo mensual"],
                    date_note=note)


def _window_start(last_iso: str, window: str | None) -> str | None:
    """Fecha de inicio de la ventana de acumulación. ``ytd`` = 1-ene del último
    año con datos; ``y2`` = 1-ene del año anterior (≈18 meses, como el informe);
    ``None``/otro = sin recorte (acumula toda la serie)."""
    year = int(last_iso[:4])
    if window == "ytd":
        return f"{year}-01-01"
    if window == "y2":
        return f"{year - 1}-01-01"
    return None


def _accumulate(points: list[tuple[str, float]], mode: str = "auto") -> list[tuple[str, float]]:
    """Convierte una serie en su versión ACUMULADA según ``mode``:

    - ``cumsum`` → suma acumulada (para FLUJOS: sin esto el gráfico muestra los
      flujos del período, no el acumulado).
    - ``rebase`` → valor menos el valor inicial de la ventana (para NIVELES: es
      la VARIACIÓN ACUMULADA respecto del inicio).
    - ``auto`` → cumsum si la serie oscila (>25% negativos), si no rebase. Frágil
      cuando una ventana de flujos queda toda positiva; preferir el modo explícito.
    """
    if not points:
        return points
    if mode == "auto":
        neg_frac = sum(1 for _d, v in points if v < 0) / len(points)
        mode = "cumsum" if neg_frac > 0.25 else "rebase"
    if mode == "cumsum":
        out: list[tuple[str, float]] = []
        total = 0.0
        for d, v in points:
            total += v
            out.append((d, total))
        return out
    base = points[0][1]
    return [(d, v - base) for d, v in points]


def accumulated_series(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Serie(s) ACUMULADA(s) leídas a resolución completa del parquet.

    Para cualquier bloque cuyo título dice "acumulado": construye la serie por
    categoría (o simple), la recorta a la ventana (``params['window']``, default
    ``ytd``), aplica ``_accumulate`` ANTES del downsample (el cumsum debe correr
    sobre todos los puntos, no sobre la muestra) y devuelve el multi-línea.
    ``params['types']``/``['funds']`` filtra/ordena categorías.
    """
    from banks_rag.infrastructure.sql import series_analytics as sa

    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        if roles.date_col is None or not roles.value_cols:
            return compute_series(dataset, parquet_dir)
        val_col = roles.value_cols[0]
        if roles.category_cols:
            cat_col = roles.category_cols[0]
            rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, cat_col, val_col])
            by_cat = _aggregate_by_category(rows, roles.date_col, cat_col, val_col)
        else:
            rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, val_col])
            by_cat = {val_col: sa.clean_series(rows, roles.date_col, val_col)}

        flt = params.get("types") or params.get("funds")
        if flt:
            order = [k for k in flt if k in by_cat]
        else:
            order = sorted(by_cat, key=lambda c: abs(by_cat[c][-1][1]) if by_cat[c] else 0.0, reverse=True)[:_MAX_PLOT_SERIES]

        start = None
        all_pts = [p for s in by_cat.values() for p in s]
        if all_pts:
            start = _window_start(max(p[0] for p in all_pts), params.get("window", "ytd"))

        mode = params.get("accumulate", "auto")
        series: list[PlotSeries] = []
        for k in order:
            pts = by_cat.get(k) or []
            if start:
                pts = [p for p in pts if p[0] >= start]
            if not pts:
                continue
            series.append(PlotSeries(label=k, points=_downsample(_accumulate(pts, mode))))
        last = max((p[0] for s in by_cat.values() for p in s), default="")
        note = ""
        if mode == "cumsum" and start:
            note = f"Acumulado (suma corrida) desde {_fmt_date(start)}"
        elif mode == "cumsum" and last:
            note = f"Acumulado (suma corrida) hasta {_fmt_date(last)}"
        elif start:
            note = f"Variación acumulada desde {_fmt_date(start)} → {_fmt_date(last)}"
        plot = PlotData(dataset.id, chart_family(dataset.chart_type), "timeseries",
                        dataset.unit, series, date_note=note)
        return None if plot.is_empty() else plot
    finally:
        con.close()


# ── Transforms genéricas reutilizables (NR / AFP) ────────────────────────────


def category_series(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Serie temporal por una categoría EXPLÍCITA, con casteo numérico del valor.

    Para parquets donde ``detect_roles`` no sirve por sí solo: la columna de valor
    llega como VARCHAR (números como texto → no se detecta) o hay varias categóricas
    y hay que fijar cuál agrupa. ``_aggregate_by_category`` castea con ``float``.

    params: ``category`` (col del eje de color), ``value`` (col numérica),
    ``filter_col``/``filter_val`` (opcional, restringe filas, p.ej. Institucion=Total),
    ``order`` (orden/selección de categorías), ``net`` (``"auto"`` añade una serie
    superpuesta "Neto" = suma de las categorías por fecha, para apilados divergentes).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    cat, val = params.get("category"), params.get("value")
    if not cat or not val:
        return None
    fcol, fval = params.get("filter_col"), params.get("filter_val")
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None:
            return None
        cols = [roles.date_col, cat, val] + ([fcol] if fcol else [])
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=cols)
    finally:
        con.close()
    if fcol and fval is not None:
        rows = [r for r in rows if str(r.get(fcol)) == str(fval)]
    by_cat = _aggregate_by_category(rows, roles.date_col, cat, val)
    # Ventana + acumulado (cumsum/rebase) ANTES del downsample, como accumulated_series.
    mode, window = params.get("accumulate"), params.get("window")
    acc_start = None
    if mode or window:
        start = None
        if window:
            all_pts = [p for s in by_cat.values() for p in s]
            if all_pts:
                start = _window_start(max(p[0] for p in all_pts), window)
        acc_start = start
        for c, pts in list(by_cat.items()):
            if start:
                pts = [p for p in pts if p[0] >= start]
            by_cat[c] = _accumulate(pts, mode) if mode else pts
    order = params.get("order")
    if order:
        cats = [c for c in order if c in by_cat]
    else:
        cats = sorted(by_cat, key=lambda c: abs(by_cat[c][-1][1]) if by_cat[c] else 0.0, reverse=True)[:_MAX_PLOT_SERIES]
    series = [PlotSeries(label=c, points=_downsample(by_cat[c])) for c in cats if by_cat[c]]
    overlay: tuple[str, ...] = ()
    if str(params.get("net") or "").lower() == "auto" and series:
        net_by_date: dict[str, float] = {}
        for c in cats:
            for d, v in by_cat[c]:
                net_by_date[d] = net_by_date.get(d, 0.0) + v
        net_pts = sorted(net_by_date.items())
        if net_pts:
            series.append(PlotSeries(label="Neto", points=_downsample(net_pts)))
            overlay = ("Neto",)
    note = ""
    if mode == "cumsum" and acc_start:
        note = f"Acumulado (suma corrida) desde {_fmt_date(acc_start)}"
    elif acc_start:
        note = f"Acumulado desde {_fmt_date(acc_start)}"
    plot = PlotData(dataset.id, chart_family(dataset.chart_type), "timeseries", dataset.unit,
                    series, overlay=overlay, date_note=note)
    return None if plot.is_empty() else plot


def wide_lines(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Multi-línea de un parquet 'ancho' (varias columnas de valor), seleccionando
    o excluyendo columnas y respetando su orden.

    params: ``include`` (lista en orden; default = todas), ``exclude`` (lista),
    ``accumulate`` (``cumsum``/``rebase`` opcional sobre cada columna),
    ``overlay`` (lista de columnas que se dibujan superpuestas como línea "Neto"
    sobre el área apilada divergente, en vez de apilarse).
    """
    from banks_rag.infrastructure.sql import series_analytics as sa

    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        exclude = set(params.get("exclude") or [])
        wanted = params.get("include") or roles.value_cols
        cols = [c for c in wanted if c in roles.value_cols and c not in exclude]
        if not cols:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, *cols])
    finally:
        con.close()
    mode, window = params.get("accumulate"), params.get("window")
    series: list[PlotSeries] = []
    for c in cols[:_MAX_PLOT_SERIES]:
        pts = sa.clean_series(rows, roles.date_col, c)
        if window and pts:
            start = _window_start(pts[-1][0], window)
            if start:
                pts = [p for p in pts if p[0] >= start]
        if mode:
            pts = _accumulate(pts, mode)
        if pts:
            series.append(PlotSeries(label=c, points=_downsample(pts)))
    overlay = tuple(c for c in (params.get("overlay") or []) if any(s.label == c for s in series))
    if not overlay and str(params.get("net") or "").lower() == "auto" and series:
        net_by_date: dict[str, float] = {}
        for s in series:
            for d, v in s.points:
                net_by_date[d] = net_by_date.get(d, 0.0) + v
        net_pts = sorted(net_by_date.items())
        if net_pts:
            series.append(PlotSeries(label="Neto", points=_downsample(net_pts)))
            overlay = ("Neto",)
    # Ventana del acumulado (eje X temporal): hace explícitas las fechas que considera.
    note = ""
    if mode and series:
        all_isos = [iso for s in series for iso, _ in s.points]
        if all_isos:
            last_iso = max(all_isos)
            start_iso = (window and _window_start(last_iso, window)) or min(all_isos)
            label = "Acumulado (suma corrida)" if mode == "cumsum" else "Acumulado"
            note = f"{label} {_fmt_date(start_iso)} → {_fmt_date(last_iso)}"
    plot = PlotData(dataset.id, chart_family(dataset.chart_type), "timeseries", dataset.unit,
                    series, overlay=overlay, date_note=note)
    return None if plot.is_empty() else plot


def window_grouped(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras agrupadas de un CORTE temporal: suma de columnas de valor sobre la
    última ventana, agrupada por una categoría (eje X).

    Replica los gráficos "cambio de posición de la última semana por tramo":
    una serie por cada columna de ``values`` (p.ej. Suscripción/Vencimiento) y,
    opcionalmente, una serie ``Neto`` con la suma.

    params: ``group`` (col categórica del eje X), ``values`` (cols de valor),
    ``window_days`` (int, default 7), ``order`` (orden del eje X),
    ``include_net`` (bool), ``negate`` (cols cuyo signo se invierte antes de apilar,
    p.ej. ``["Vencimiento"]`` para que el vencimiento reste de la posición y el
    Neto = Suscripción - Vencimiento salga del propio apilado),
    ``net_as_overlay`` (bool: dibuja el Neto como punto superpuesto, no como barra).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    group = params.get("group")
    values = list(params.get("values") or [])
    if not group or not values:
        return None
    negate = set(params.get("negate") or [])
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, group, *values])
    finally:
        con.close()
    isos = [str(r[roles.date_col]) for r in rows if r.get(roles.date_col)]
    if not isos:
        return None
    last = max(isos)
    days = int(params.get("window_days", 7))
    start = (date.fromisoformat(last[:10]) - timedelta(days=days)).isoformat()
    agg: dict[str, dict[str, float]] = {}
    for r in rows:
        d, g = r.get(roles.date_col), r.get(group)
        if d is None or g is None or str(d) < start:
            continue
        per = agg.setdefault(str(g), {})
        for v in values:
            try:
                fv = float(r.get(v))
            except (TypeError, ValueError):
                continue
            per[v] = per.get(v, 0.0) + (-fv if v in negate else fv)
    if not agg:
        return None
    cats = [c for c in (params.get("order") or sorted(agg)) if c in agg]
    sd: dict[str, dict[str, float]] = {v: {c: agg[c].get(v, 0.0) for c in cats} for v in values}
    series_order = list(values)
    overlay: tuple[str, ...] = ()
    if params.get("include_net"):
        sd["Neto"] = {c: sum(agg[c].get(v, 0.0) for v in values) for c in cats}
        series_order.append("Neto")
        if params.get("net_as_overlay"):
            overlay = ("Neto",)
    return _grouped(dataset.id, dataset.unit, sd, cats, series_order, overlay=overlay,
                    date_note=f"Suma {_fmt_date(start)} → {_fmt_date(last)}")


def _daymon(iso: str) -> str:
    """``YYYY-MM-DD`` → ``DD-MM`` (etiqueta compacta del eje X por día)."""
    try:
        d = date.fromisoformat(iso[:10])
    except ValueError:
        return iso[:10]
    return f"{d.day:02d}-{d.month:02d}"


def wide_window_bars(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras apiladas DIVERGENTES por DÍA (corte de los últimos N días) de un
    parquet 'ancho': eje X = fecha (últimos N), una serie por columna de valor;
    la(s) columna(s) en ``overlay`` se dibujan como punto "Neto" superpuesto.

    Réplica de "Var. Diaria Posición … por plazo" del BCCh.
    params: ``include`` (cols a apilar; default = todas menos overlay),
    ``overlay`` (cols superpuestas, p.ej. ``["Neto"]``), ``last_n`` (int, default 6),
    ``diff`` (bool: si el parquet trae NIVELES, dibuja la variación día-a-día =
    diferencia con el día anterior, no el nivel).
    """
    from banks_rag.infrastructure.sql import series_analytics as sa

    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        overlay_cols = [c for c in (params.get("overlay") or []) if c in roles.value_cols]
        include = params.get("include") or [c for c in roles.value_cols if c not in overlay_cols]
        cols = [c for c in include if c in roles.value_cols and c not in overlay_cols]
        if not cols:
            return None
        wanted = cols + overlay_cols
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, *wanted])
    finally:
        con.close()
    do_diff = bool(params.get("diff"))
    by_col: dict[str, dict[str, float]] = {}
    for c in wanted:
        pts = sa.clean_series(rows, roles.date_col, c)  # ya viene ordenado por fecha
        if do_diff:
            pts = [(pts[i][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))]
        by_col[c] = dict(pts)
    all_dates = sorted({d for c in wanted for d in by_col[c]})
    sel = all_dates[-int(params.get("last_n", 6)):]
    if not sel:
        return None
    labels = [_daymon(d) for d in sel]
    sd: dict[str, dict[str, float]] = {
        c: {labels[i]: by_col[c].get(sel[i], 0.0) for i in range(len(sel))} for c in wanted
    }
    series_order = list(cols)
    overlay: tuple[str, ...] = ()
    if overlay_cols:
        series_order.append(overlay_cols[0])
        overlay = (overlay_cols[0],)
    kind_txt = "Variación diaria" if do_diff else "Por día"
    note = f"{kind_txt} · {_fmt_date(sel[0])} → {_fmt_date(sel[-1])}"
    return _grouped(dataset.id, dataset.unit, sd, labels, series_order, overlay=overlay,
                    date_note=note)


def window_stacked_by_cat(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras apiladas DIVERGENTES por DÍA (últimos N) de un parquet LARGO
    (fecha + categoría + valor): eje X = día, una serie por categoría; opcional
    punto "Neto" = suma por día.

    Réplica de "Traspaso de fondos de FP" del BCCh (flujos diarios por fondo A-E).
    params: ``category`` / ``value`` (si faltan, ``detect_roles``), ``last_n``
    (int, default 14), ``order`` (orden de categorías), ``net`` (bool).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        cat = params.get("category") or (roles.category_cols[0] if roles.category_cols else None)
        val = params.get("value") or (roles.value_cols[0] if roles.value_cols else None)
        if roles.date_col is None or not cat or not val:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, cat, val])
    finally:
        con.close()
    by_cat = _aggregate_by_category(rows, roles.date_col, cat, val)  # {cat: [(iso, v)]}
    by_cat = {c: dict(s) for c, s in by_cat.items()}
    all_dates = sorted({d for s in by_cat.values() for d in s})
    sel = all_dates[-int(params.get("last_n", 14)):]
    if not sel:
        return None
    labels = [_daymon(d) for d in sel]
    cats = [c for c in (params.get("order") or sorted(by_cat)) if c in by_cat]
    sd: dict[str, dict[str, float]] = {
        c: {labels[i]: by_cat[c].get(sel[i], 0.0) for i in range(len(sel))} for c in cats
    }
    series_order = list(cats)
    overlay: tuple[str, ...] = ()
    if params.get("net"):
        sd["Neto"] = {labels[i]: sum(by_cat[c].get(sel[i], 0.0) for c in cats) for i in range(len(sel))}
        series_order.append("Neto")
        overlay = ("Neto",)
    note = f"Por día · {_fmt_date(sel[0])} → {_fmt_date(sel[-1])}"
    return _grouped(dataset.id, dataset.unit, sd, labels, series_order, overlay=overlay,
                    date_note=note)


def _accum_windows(
    dataset: ParquetDataset, parquet_dir: Path, params: dict,
) -> tuple[list[str], dict[str, dict[str, float]], list[tuple[str, str, str]]] | None:
    """Núcleo compartido del flujo ACUMULADO por categoría en ventanas.

    Lee un parquet LARGO (fecha + categoría + valor) y suma el flujo de cada
    ventana por categoría. Devuelve ``(cats, accum, spans)`` donde
    ``accum = {ventana_label: {cat: suma}}`` y ``spans = [(label, start, last)]``;
    ``None`` si el parquet no tiene la forma esperada.

    params: ``category`` / ``value`` (si faltan, ``detect_roles``), ``order``
    (orden de categorías), ``windows`` (lista de ``[label, días]``; default
    ``[["Variación semanal", 7], ["Variación mensual", 30]]``).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        cat = params.get("category") or (roles.category_cols[0] if roles.category_cols else None)
        val = params.get("value") or (roles.value_cols[0] if roles.value_cols else None)
        if roles.date_col is None or not cat or not val:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, cat, val])
    finally:
        con.close()
    by_cat = _aggregate_by_category(rows, roles.date_col, cat, val)  # {cat: [(iso, v)]}
    cats = [c for c in (params.get("order") or sorted(by_cat)) if c in by_cat]
    if not cats:
        return None
    last = max(iso for s in by_cat.values() for iso, _ in s)
    windows = params.get("windows") or [["Variación semanal", 7], ["Variación mensual", 30]]
    accum: dict[str, dict[str, float]] = {}
    spans: list[tuple[str, str, str]] = []
    for label, days in windows:
        start = (date.fromisoformat(last[:10]) - timedelta(days=int(days))).isoformat()
        accum[label] = {c: sum(v for iso, v in by_cat.get(c, []) if iso >= start) for c in cats}
        spans.append((label, start, last))
    return cats, accum, spans


def window_accum_by_cat(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Flujo ACUMULADO por categoría (fondo) en ventanas → barras AGRUPADAS:
    eje X = categoría (A-E), una serie por ventana = SUMA de los flujos de la
    ventana. Vista de variación semanal / mensual del mismo dato que el gráfico
    diario (``window_stacked_by_cat``): para cada fondo, el traspaso neto acumulado
    de la última semana y del último mes, lado a lado.

    params: ver ``_accum_windows``.
    """
    res = _accum_windows(dataset, parquet_dir, params)
    if res is None:
        return None
    cats, accum, spans = res
    series_order = [label for label, _, _ in spans]
    note_parts = [
        f"{label}: {_fmt_date(start)} → {_fmt_date(last)}"
        for label, start, last in spans
    ]
    return _grouped(dataset.id, dataset.unit, accum, cats, series_order,
                    date_note="  ·  ".join(note_parts))


def window_accum_stacked_by_cat(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Flujo ACUMULADO APILADO POR FONDO en ventanas → barras APILADAS DIVERGENTES:
    eje X = ventana (semanal / mensual), una serie por FONDO (A-E, mismos colores
    que el gráfico diario). El alto neto de cada columna ES el flujo neto de la
    ventana y la composición muestra cómo aportó cada fondo. Vista "neta" que cierra
    el bloque de traspasos: cómo quedaron los fondos entre sí en la semana vs. el mes.

    params: ver ``_accum_windows`` (las categorías del parquet pasan a SER las
    series apiladas; las ventanas pasan a ser el eje X).
    """
    res = _accum_windows(dataset, parquet_dir, params)
    if res is None:
        return None
    cats, accum, spans = res  # cats = fondos; accum = {ventana: {fondo: suma}}
    win_labels = [label for label, _, _ in spans]
    # Transpone: una serie por fondo, su valor en cada ventana (eje X).
    series_dict = {f: {w: accum[w].get(f, 0.0) for w in win_labels} for f in cats}
    note_parts = [
        f"{label}: {_fmt_date(start)} → {_fmt_date(last)}"
        for label, start, last in spans
    ]
    return _grouped(dataset.id, dataset.unit, series_dict, win_labels, list(cats),
                    date_note="  ·  ".join(note_parts))


def window_grouped_long(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras apiladas DIVERGENTES por categoría de un parquet LARGO con columna de
    TIPO (Suscripción/Vencimiento): suma de la última ventana por ``group``,
    apilando la suscripción (+) y el vencimiento (-, se resta de la posición), con
    Neto (= suscripción - vencimiento) como punto. Top-N grupos por |Neto|.

    Réplica de "Cambio posición por tipo de derivados" (group=Instrumento) y
    "Variación posición NR semanal" (group=Institucion) del tablero.
    params: ``group`` (eje X), ``type_col`` (col de tipo), ``pos`` / ``neg``
    (valores de tipo que suman / restan), ``value``, ``window_days`` (default 7),
    ``top_n`` (opcional), ``exclude_types`` (valores de type_col a ignorar).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    group, tcol, val = params.get("group"), params.get("type_col"), params.get("value")
    pos, neg = params.get("pos"), params.get("neg")
    if not group or not tcol or not val or not pos or not neg:
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, group, tcol, val])
    finally:
        con.close()
    isos = [str(r[roles.date_col]) for r in rows if r.get(roles.date_col)]
    if not isos:
        return None
    start = (date.fromisoformat(max(isos)[:10]) - timedelta(days=int(params.get("window_days", 7)))).isoformat()
    exclude = set(params.get("exclude_types") or [])
    agg: dict[str, dict[str, float]] = {}
    for r in rows:
        d, g, t = r.get(roles.date_col), r.get(group), r.get(tcol)
        if d is None or g is None or str(d) < start or str(t) in exclude:
            continue
        try:
            v = float(r.get(val))
        except (TypeError, ValueError):
            continue
        per = agg.setdefault(str(g), {pos: 0.0, neg: 0.0})
        if str(t) == pos:
            per[pos] += v
        elif str(t) == neg:
            per[neg] -= v  # el vencimiento resta de la posición
    if not agg:
        return None
    net = {g: per[pos] + per[neg] for g, per in agg.items()}
    cats = sorted(agg, key=lambda g: abs(net[g]), reverse=True)
    if params.get("top_n"):
        cats = cats[: int(params["top_n"])]
    sd = {pos: {g: agg[g][pos] for g in cats}, neg: {g: agg[g][neg] for g in cats},
          "Neto": {g: net[g] for g in cats}}
    note = f"Suma {_fmt_date(start)} → {_fmt_date(max(isos))}"
    return _grouped(dataset.id, dataset.unit, sd, cats, [pos, neg, "Neto"], overlay=("Neto",),
                    date_note=note)


def wide_monthly_bars(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras apiladas por MES (suma del mes) de un parquet ANCHO: eje X = mes, una
    serie por columna de valor; la(s) columna(s) en ``overlay`` van como punto.

    Réplica de "Variación posición derivados mensual" del tablero.
    params: ``include`` (cols a apilar; default = todas menos overlay),
    ``overlay`` (cols superpuestas, p.ej. ``["Neto"]``), ``months`` (int, default 12).
    """
    from banks_rag.infrastructure.sql import series_analytics as sa

    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        overlay_cols = [c for c in (params.get("overlay") or []) if c in roles.value_cols]
        include = params.get("include") or [c for c in roles.value_cols if c not in overlay_cols]
        cols = [c for c in include if c in roles.value_cols and c not in overlay_cols]
        if not cols:
            return None
        wanted = cols + overlay_cols
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, *wanted])
    finally:
        con.close()
    by_col_month: dict[str, dict[str, float]] = {c: {} for c in wanted}
    for c in wanted:
        for iso, v in sa.clean_series(rows, roles.date_col, c):
            mk = _month_key(iso)
            by_col_month[c][mk] = by_col_month[c].get(mk, 0.0) + v
    all_months = sorted({m for c in wanted for m in by_col_month[c]})[-int(params.get("months", 12)):]
    if not all_months:
        return None
    labels = [_month_label(m) for m in all_months]
    sd = {c: {labels[i]: by_col_month[c].get(all_months[i], 0.0) for i in range(len(all_months))} for c in wanted}
    series_order = list(cols)
    overlay: tuple[str, ...] = ()
    if overlay_cols:
        series_order.append(overlay_cols[0])
        overlay = (overlay_cols[0],)
    last_iso = max((iso for c in wanted for iso, _ in sa.clean_series(rows, roles.date_col, c)), default="")
    note = f"Suma por mes · datos hasta {_fmt_date(last_iso)}" if last_iso else ""
    return _grouped(dataset.id, dataset.unit, sd, labels, series_order, overlay=overlay,
                    date_note=note)


def snapshot_grouped(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras agrupadas de un parquet SIN fecha (corte transversal): eje X = una
    columna categórica y una serie por cada columna de valor.

    Para "Flujo cambiario por AFP" (Spot/Forward apilados + Neto por AFP).
    params: ``group`` (col del eje X), ``values`` (cols de valor), ``order``,
    ``overlay`` (cols que van como punto superpuesto, p.ej. ``["Neto"]``),
    ``title_col`` (col cuyo texto trae la ventana de fechas tras un "·",
    p.ej. ``_title_override`` = "… · 03-06 al 10-06-2026" → date_note).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    group = params.get("group")
    values = list(params.get("values") or [])
    if not group or not values:
        return None
    title_col = params.get("title_col")
    read_cols = [group, *values] + ([title_col] if title_col else [])
    con = duckdb.connect()
    try:
        rows = _read_rows(con, path, date_col=None, columns=read_cols)
    finally:
        con.close()
    agg: dict[str, dict[str, float]] = {}
    for r in rows:
        g = r.get(group)
        if g is None:
            continue
        per = agg.setdefault(str(g), {})
        for v in values:
            try:
                per[v] = per.get(v, 0.0) + float(r.get(v))
            except (TypeError, ValueError):
                continue
    if not agg:
        return None
    cats = [c for c in (params.get("order") or sorted(agg)) if c in agg]
    sd = {v: {c: agg[c].get(v, 0.0) for c in cats} for v in values}
    overlay = tuple(c for c in (params.get("overlay") or []) if c in values)
    # Ventana de fechas: el parquet no tiene columna fecha, pero la trae embebida en
    # una columna de título (tras un "·"). La mostramos como date_note del corte.
    date_note = ""
    if title_col:
        raw = next((r.get(title_col) for r in rows if r.get(title_col)), None)
        if raw:
            txt = str(raw)
            date_note = txt.split("·", 1)[1].strip() if "·" in txt else txt.strip()
    return _grouped(dataset.id, dataset.unit, sd, cats, values, overlay=overlay, date_note=date_note)


def snapshot_stacked(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras (apilables) de un parquet SIN fecha: eje X = una categórica y una
    serie por cada valor de OTRA categórica.

    Para "Atribución por clase de activos" (X = fondo, una serie por Clase).
    params: ``x`` (categórica del eje X), ``series`` (categórica del color),
    ``value`` (col de valor), ``x_order``/``series_order``, ``total_overlay``
    (bool: agrega un punto "Total" = suma de las clases por X, como el informe).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    x, scol, val = params.get("x"), params.get("series"), params.get("value")
    if not x or not scol or not val:
        return None
    con = duckdb.connect()
    try:
        rows = _read_rows(con, path, date_col=None, columns=[x, scol, val])
    finally:
        con.close()
    agg: dict[str, dict[str, float]] = {}
    xs: list[str] = []
    ss: list[str] = []
    for r in rows:
        xv, sv = r.get(x), r.get(scol)
        if xv is None or sv is None:
            continue
        try:
            v = float(r.get(val))
        except (TypeError, ValueError):
            continue
        agg.setdefault(str(sv), {})[str(xv)] = agg.setdefault(str(sv), {}).get(str(xv), 0.0) + v
        if str(xv) not in xs:
            xs.append(str(xv))
        if str(sv) not in ss:
            ss.append(str(sv))
    if not agg:
        return None
    xcats = [c for c in (params.get("x_order") or sorted(xs)) if c in xs]
    series_order = [s for s in (params.get("series_order") or sorted(ss)) if s in ss]
    overlay: tuple[str, ...] = ()
    if params.get("total_overlay"):
        agg["Total"] = {c: sum(agg[s].get(c, 0.0) for s in series_order) for c in xcats}
        series_order = [*series_order, "Total"]
        overlay = ("Total",)
    return _grouped(dataset.id, dataset.unit, agg, xcats, series_order, overlay=overlay)


def latest_snapshot(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Composición del ÚLTIMO corte temporal de un parquet con fecha: suma del
    valor por categoría en la fecha más reciente → snapshot (torta / barras).

    Para "Composición del portafolio DCV" (torta del stock por instrumento al
    último día). params: ``category`` (col del color), ``value`` (col de valor);
    si faltan, se infieren de ``detect_roles``.
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None:
            return None
        cat = params.get("category") or (roles.category_cols[0] if roles.category_cols else None)
        val = params.get("value") or (roles.value_cols[0] if roles.value_cols else None)
        if not cat or not val:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, cat, val])
    finally:
        con.close()
    dates = [str(r[roles.date_col]) for r in rows if r.get(roles.date_col)]
    if not dates:
        return None
    last = max(dates)
    sums: dict[str, float] = {}
    for r in rows:
        if str(r.get(roles.date_col)) != last:
            continue
        cv = r.get(cat)
        if cv is None:
            continue
        try:
            sums[str(cv)] = sums.get(str(cv), 0.0) + float(r.get(val))
        except (TypeError, ValueError):
            continue
    sums = {k: v for k, v in sums.items() if v > 0}  # torta: aportes positivos
    if not sums:
        return None
    ordered = sorted(sums.items(), key=lambda kv: abs(kv[1]), reverse=True)[:_MAX_PLOT_CATEGORIES]
    return PlotData(
        dataset.id, chart_family(dataset.chart_type), "snapshot", dataset.unit,
        [PlotSeries(label=cat, points=[(k, round(v, 6)) for k, v in ordered])],
        date_note=f"Corte: {_fmt_date(last)}",
    )


# ── Transforms de tabla con color (heatmap DCV) ──────────────────────────────


def _distinct_dates_sorted(con: duckdb.DuckDBPyConnection, parquet_path: Path, date_col: str) -> list[str]:
    """Fechas distintas del parquet como ISO strings, ordenadas ASC."""
    src = f"read_parquet('{parquet_path.as_posix()}')"
    rows = con.execute(
        f"SELECT DISTINCT TRY_CAST({date_col} AS DATE) AS d "
        f"FROM {src} WHERE d IS NOT NULL ORDER BY d"
    ).fetchall()
    return [str(r[0]) for r in rows]


def _cut_indices(
    dates: list[str], asof: str | None = None,
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Devuelve ``(display, value)``, cada uno ``(T, T-7, T-30)``.

    - ``display``: las fechas que se MUESTRAN en la tabla (encabezados).
    - ``value``: las fechas CON dato para buscar el stock (ultima fecha <= cada display).

    Sin ``asof``: ``T`` = maximo del parquet y ``T-7``/``T-30`` la ultima fecha con
    dato <= (T - 7 / - 30 dias naturales) -> ``display == value`` (comportamiento previo;
    variacion a 1 semana / 1 mes de calendario, no por posicion de dia habil).

    Con ``asof`` (corte COMUN del informe): ``T`` = ``asof`` y ``T-7``/``T-30`` =
    ``asof`` - 7 / - 30 dias. Asi TODAS las tablas muestran los MISMOS T, T-7, T-30 que
    el texto y que flujos, aunque el parquet DCV no tenga dato justo en el corte (p.ej.
    el corte cae en fin de semana): el VALOR se resuelve at-or-before, pero la fecha que
    se cita es la del corte. ``dates`` viene ordenado ASC."""
    def _aob(ref: str) -> str:
        return next((d for d in reversed(dates) if d <= ref), dates[0])

    if asof:
        t_d = date.fromisoformat(asof[:10])
        disp = (asof, (t_d - timedelta(days=7)).isoformat(), (t_d - timedelta(days=30)).isoformat())
        return disp, (_aob(disp[0]), _aob(disp[1]), _aob(disp[2]))

    t = dates[-1]
    t_d = date.fromisoformat(t[:10])
    data = (t, _aob((t_d - timedelta(days=7)).isoformat()), _aob((t_d - timedelta(days=30)).isoformat()))
    return data, data


def dcv_cut_dates(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Tabla de fechas de corte DCV (T, T-7, T-30): stock por instrumento +
    deltas coloreados (variación a 1 semana y 1 mes). Lee ``stock_nivel_ffmm``
    (Fecha, Tipo, Stock_USD)."""
    from .svg_chart import render_dcv_cut_table

    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        if roles.date_col is None or not roles.category_cols or not roles.value_cols:
            return None
        cat_col, val_col = roles.category_cols[0], roles.value_cols[0]

        dates = _distinct_dates_sorted(con, parquet_path, roles.date_col)
        if len(dates) < 3:
            return None
        (t_disp, t7_disp, t30_disp), (t_d, t7_d, t30_d) = _cut_indices(dates, params.get("weekly_asof"))

        in_clause = ", ".join(f"DATE '{d}'" for d in {t_d, t7_d, t30_d})
        src = f"read_parquet('{parquet_path.as_posix()}')"
        rows = con.execute(
            f"SELECT TRY_CAST({roles.date_col} AS DATE) AS d, {cat_col}, SUM({val_col}) "
            f"FROM {src} WHERE TRY_CAST({roles.date_col} AS DATE) IN ({in_clause}) "
            f"GROUP BY d, {cat_col} ORDER BY {cat_col}"
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return None

    by_tipo: dict[str, dict[str, float]] = {}
    for d, tipo, val in rows:
        if val is None:
            continue
        by_tipo.setdefault(str(tipo), {})[str(d)] = float(val)

    if not by_tipo:
        return None

    tipos = sorted(by_tipo)
    # Valores en las fechas CON dato (at-or-before); encabezado con las fechas del corte.
    data = {
        t: (by_tipo[t].get(t_d), by_tipo[t].get(t7_d), by_tipo[t].get(t30_d))
        for t in tipos
    }
    return HtmlTable(
        html=render_dcv_cut_table(tipos, (t_disp, t7_disp, t30_disp), data, unit=dataset.unit),
        dataset_id=dataset.id,
    )


def dcv_heatmap(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Variacion del stock DCV (Delta T-7 / Delta T-30) por instrumento y plazo
    (variación a 1 semana y 1 mes).
    Lee ``variacion_stock_ffmm`` (Fecha, Bucket, Tipo, Moneda, Stock_USD)."""
    from .svg_chart import render_dcv_heatmap_tables

    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        if roles.date_col is None or len(roles.category_cols) < 2 or not roles.value_cols:
            return None
        cats = roles.category_cols
        bucket_col = next((c for c in cats if c.lower() in ("bucket", "plazo")), cats[0])
        tipo_col = next((c for c in cats if c != bucket_col and "moneda" not in c.lower()), cats[1])
        val_col = roles.value_cols[0]

        dates = _distinct_dates_sorted(con, parquet_path, roles.date_col)
        if len(dates) < 3:
            return None
        # display = (T, T-7, T-30) para rotular el span de cada matriz; value = fechas
        # CON dato (at-or-before del corte) para calcular los deltas Δ T-7 / Δ T-30.
        (t_disp, t7_disp, t30_disp), (t_iso, t7_iso, t30_iso) = _cut_indices(dates, params.get("weekly_asof"))

        in_clause = ", ".join(f"DATE '{d}'" for d in {t_iso, t7_iso, t30_iso})
        src = f"read_parquet('{parquet_path.as_posix()}')"
        rows = con.execute(
            f"SELECT TRY_CAST({roles.date_col} AS DATE) AS d, {bucket_col}, {tipo_col}, SUM({val_col}) "
            f"FROM {src} WHERE TRY_CAST({roles.date_col} AS DATE) IN ({in_clause}) "
            f"GROUP BY d, {bucket_col}, {tipo_col} ORDER BY {tipo_col}, {bucket_col}"
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return None

    cell: dict[tuple[str, str], dict[str, float]] = {}
    tipos_seen: list[str] = []
    buckets_seen: list[str] = []
    for d, bucket, tipo, val in rows:
        if val is None:
            continue
        k = (str(tipo), str(bucket))
        if k not in cell:
            cell[k] = {}
            if str(tipo) not in tipos_seen:
                tipos_seen.append(str(tipo))
            if str(bucket) not in buckets_seen:
                buckets_seen.append(str(bucket))
        cell[k][str(d)] = float(val)

    if not cell:
        return None

    tipos = sorted(tipos_seen)
    buckets = _order_buckets(buckets_seen)
    delta7: dict[str, dict[str, float | None]] = {t: {} for t in tipos}
    delta30: dict[str, dict[str, float | None]] = {t: {} for t in tipos}
    for (tipo, bucket), by_date in cell.items():
        vt = by_date.get(t_iso)
        vt7 = by_date.get(t7_iso)
        vt30 = by_date.get(t30_iso)
        if vt is not None and vt7 is not None:
            delta7[tipo][bucket] = vt - vt7
        if vt is not None and vt30 is not None:
            delta30[tipo][bucket] = vt - vt30

    label7 = f"Δ T-7 · {_fmt_date(t7_disp)} → {_fmt_date(t_disp)}"
    label30 = f"Δ T-30 · {_fmt_date(t30_disp)} → {_fmt_date(t_disp)}"
    return HtmlTable(
        html=render_dcv_heatmap_tables(tipos, buckets, delta7, delta30, unit=dataset.unit,
                                       label7=label7, label30=label30),
        dataset_id=dataset.id,
    )


# ── Registro: nombre → transform (None = declarada, pendiente de 2ª iteración) ─

_REGISTRY: dict[str, Transform | None] = {
    # Implementadas (producen una serie graficable como línea/composición).
    "straight_series": straight_series,
    "snapshot_composition": snapshot_composition,
    "filter_fund": filter_fund,
    "allocation_by_fund": allocation_by_fund,
    "monthly_var_alloc": monthly_var_alloc,
    "accumulated": accumulated_series,
    "window_returns": window_returns,
    "ytd_return_geom": ytd_return_geom,
    "monthly_sum_by_fund": monthly_sum_by_fund,
    "monthly_returns": monthly_returns,
    "composition_by_bucket": composition_by_bucket,
    "stacked_by_bucket": stacked_by_bucket,
    "monthly_diff": monthly_diff,
    # Genéricas reutilizables (NR / AFP).
    "category_series": category_series,
    "wide_lines": wide_lines,
    "window_grouped": window_grouped,
    "wide_window_bars": wide_window_bars,
    "wide_monthly_bars": wide_monthly_bars,
    "window_stacked_by_cat": window_stacked_by_cat,
    "window_accum_by_cat": window_accum_by_cat,
    "window_accum_stacked_by_cat": window_accum_stacked_by_cat,
    "window_grouped_long": window_grouped_long,
    "snapshot_grouped": snapshot_grouped,
    "snapshot_stacked": snapshot_stacked,
    "latest_snapshot": latest_snapshot,
    # Tablas con color condicional (heatmap DCV): implementadas.
    "dcv_cut_dates": dcv_cut_dates,
    "dcv_heatmap": dcv_heatmap,
}


def get_transform(name: str | None) -> Transform | None:
    """Resuelve el nombre a su función; ``None`` si no está implementada todavía
    (o no existe). El builder distingue ambos casos por presencia en el registro."""
    return _REGISTRY.get(name or "")


def is_known(name: str | None) -> bool:
    """True si el nombre está REGISTRADO (implementado o declarado pendiente)."""
    return (name or "") in _REGISTRY
