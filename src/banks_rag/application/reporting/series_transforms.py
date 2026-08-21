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
import re
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
    punto "Neto" = suma de instrumentos por plazo (como el informe).

    Nota: en parquets sin columna ``Bucket``/``Plazo`` real (ej.
    ``variacion_sector_todos``, que solo tiene ``Tipo``/``Sector``),
    ``_bucket_tipo_series`` usa la primera categoría como eje X ("bucket") y la
    segunda como serie apilada — acá el eje X termina siendo ``Tipo`` y la serie
    ``Sector``. Por eso hay DOS filtros independientes:

    - ``buckets`` (opcional): lista de valores del eje X a incluir/ordenar (ej.
      instrumentos ``["BB", "BE", "BTP", "BTU"]`` cuando el eje X es ``Tipo``).
    - ``order`` (opcional): lista de series (segunda categoría, ej. ``Sector``)
      a incluir/ordenar. Si no se pasa, se usan las top series por magnitud
      (``_top_tipos``), igual que antes.
    """
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
    if params.get("buckets"):
        wanted_buckets = [b for b in params["buckets"] if b in buckets]
        if wanted_buckets:
            buckets = wanted_buckets
    sd: dict[str, dict[str, float]] = {}
    for b in buckets:
        for t, s in bt[b].items():
            if not s:
                continue
            wd = weekly_delta(s, asof, days=days, is_flow=False)
            if wd is not None:
                sd.setdefault(t, {})[b] = wd["cambio_absoluto"]
    default_tipos = _top_tipos(sd)
    tipos = [t for t in (params.get("order") or default_tipos) if t in sd] or default_tipos
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
    ``dN`` (p.ej. ``d30``) = últimos N días naturales, para los acumulados de
    ventana corta del informe de flujos cambiarios (el eje X del correo real cubre
    ~1 mes); ``None``/otro = sin recorte (acumula toda la serie)."""
    year = int(last_iso[:4])
    if window == "ytd":
        return f"{year}-01-01"
    if window == "y2":
        return f"{year - 1}-01-01"
    if window and re.fullmatch(r"d\d+", window):
        last = date.fromisoformat(last_iso[:10])
        return str(last - timedelta(days=int(window[1:])))
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
    ``value_neg`` (col que RESTA de ``value``: posición neta = suscripción -
    vencimiento), ``abs`` (magnitud bruta, sin netear dentro del día),
    ``filter_col``/``filter_val`` (opcional, restringe filas, p.ej. Institucion=Total),
    ``order`` (orden/selección de categorías; sin él se toman las ``max_series``
    mayores por |último valor|), ``labels`` (renombra las categorías al nombre del
    informe), ``net`` (``"auto"`` añade una serie superpuesta "Neto" = suma de las
    categorías por fecha, para apilados divergentes), ``mean_overlay`` (línea
    horizontal en el promedio del total diario), ``anchor_zero`` (con
    ``accumulate="cumsum"``: todas las series arrancan en 0 el primer día de la
    ventana, como los acumulados del informe).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    cat, val = params.get("category"), params.get("value")
    if not cat or not val:
        return None
    fcol, fval = params.get("filter_col"), params.get("filter_val")
    # ``value_neg``: segunda columna que RESTA (posición neta = suscripción -
    # vencimiento). Sin esto un "acumulado de posición" solo sumaría una pata y
    # mostraría una serie que no es la posición.
    vneg = params.get("value_neg")
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None:
            return None
        cols = [roles.date_col, cat, val] + ([fcol] if fcol else []) + ([vneg] if vneg else [])
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=cols)
    finally:
        con.close()
    if fcol and fval is not None:
        rows = [r for r in rows if str(r.get(fcol)) == str(fval)]
    if vneg:
        rows = [{**r, val: _num(r.get(val)) - _num(r.get(vneg))} for r in rows]
    if params.get("abs"):
        # Flujo BRUTO (el informe grafica "suscripciones brutas"): magnitud operada,
        # sin netear compras contra ventas dentro del mismo día.
        rows = [{**r, val: abs(_num(r.get(val)))} for r in rows]
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
        # ``anchor_zero``: el primer día de la ventana vale 0 y la acumulación corre
        # desde el siguiente, que es como el informe dibuja los acumulados (todas las
        # series nacen del mismo origen y son comparables entre sí).
        #
        # El ancla es una fecha COMÚN a todas las series, no el primer punto de cada
        # una: una categoría que no operó el primer día empieza más tarde, y anclarla
        # en SU primer punto le descontaría un día que a las demás no — las series
        # dejarían de ser comparables, que es justo lo que el anclaje busca.
        anchor = bool(params.get("anchor_zero")) and mode == "cumsum"
        windowed = {
            c: ([p for p in pts if p[0] >= start] if start else pts)
            for c, pts in by_cat.items()
        }
        anchor_iso = min(
            (pts[0][0] for pts in windowed.values() if pts), default=None,
        ) if anchor else None
        for c, pts in windowed.items():
            if mode:
                pts = _accumulate(pts, mode)
                if anchor_iso is not None:
                    # valor acumulado EN la fecha ancla (0 si la serie aún no operaba)
                    base = next((v for d, v in pts if d == anchor_iso), 0.0)
                    pts = [(d, v - base) for d, v in pts]
            by_cat[c] = pts
    order = params.get("order")
    if order:
        cats = [c for c in order if c in by_cat]
    else:
        top = int(params.get("max_series") or _MAX_PLOT_SERIES)
        cats = sorted(by_cat, key=lambda c: abs(by_cat[c][-1][1]) if by_cat[c] else 0.0, reverse=True)[:top]
    # ``labels``: nombre del parquet → nombre del informe (p.ej. Emp_real → Empresas
    # reales). Solo cambia la etiqueta visible; el orden y la agregación usan la clave.
    labels = dict(params.get("labels") or {})
    series = [
        PlotSeries(label=labels.get(c, c), points=_downsample(by_cat[c]))
        for c in cats if by_cat[c]
    ]
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
    # ``mean_overlay``: línea horizontal en el promedio del total diario (el
    # "Promedio" que el informe dibuja sobre las suscripciones brutas, para leer
    # si la jornada estuvo sobre o bajo lo habitual del período).
    if params.get("mean_overlay") and series:
        total_by_date: dict[str, float] = {}
        for c in cats:
            for d, v in by_cat[c]:
                total_by_date[d] = total_by_date.get(d, 0.0) + v
        if total_by_date:
            avg = sum(total_by_date.values()) / len(total_by_date)
            label = str(params.get("mean_label") or "Promedio")
            series.append(PlotSeries(
                label=label, points=_downsample([(d, avg) for d in sorted(total_by_date)]),
            ))
            overlay = (*overlay, label)
    note = ""
    if mode == "cumsum" and acc_start:
        note = f"Acumulado (suma corrida) desde {_fmt_date(acc_start)}"
    elif acc_start:
        note = f"Acumulado desde {_fmt_date(acc_start)}"
    plot = PlotData(dataset.id, chart_family(dataset.chart_type), "timeseries", dataset.unit,
                    series, overlay=overlay, date_note=note)
    return None if plot.is_empty() else plot

def wide_daily_diff_ytd(
    dataset: ParquetDataset,
    parquet_dir: Path,
    params: dict,
) -> PlotData | None:
    """
    Convierte columnas de niveles diarios en variaciones diarias y luego
    acumula las variaciones desde el comienzo del último año disponible.

    Parquet esperado:
        Fecha
        1 a 90 dias
        91 a 360 dias
        Entre 1 y 2Y
        Mayor a 2Y
        Neto

    Para cada serie:
        diferencia_diaria[t] = nivel[t] - nivel[t-1]
        acumulado_ytd[t] = suma de las diferencias diarias desde la base

    Esto equivale a:
        acumulado_ytd[t] = nivel[t] - nivel_base
    """
    parquet_path = dataset.parquet_path(parquet_dir)

    if not parquet_path.exists():
        return None

    requested_columns = list(
        params.get("columns")
        or [
            "1 a 90 dias",
            "91 a 360 dias",
            "Entre 1 y 2Y",
            "Mayor a 2Y",
            "Neto",
        ]
    )

    overlay_columns = tuple(params.get("overlay") or ["Neto"])

    con = duckdb.connect()

    try:
        describe_rows = con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{parquet_path.as_posix()}')
            """
        ).fetchall()

        available_columns = {str(row[0]) for row in describe_rows}

        if "Fecha" not in available_columns:
            log.warning(
                "Dataset %s no contiene la columna Fecha",
                dataset.id,
            )
            return None

        value_columns = [
            column
            for column in requested_columns
            if column in available_columns
        ]

        missing_columns = [
            column
            for column in requested_columns
            if column not in available_columns
        ]

        if missing_columns:
            log.warning(
                "Dataset %s no contiene estas columnas: %s",
                dataset.id,
                missing_columns,
            )

        if not value_columns:
            log.warning(
                "Dataset %s no tiene columnas numéricas graficables",
                dataset.id,
            )
            return None

        value_select = ", ".join(
            f'TRY_CAST("{column}" AS DOUBLE) AS "{column}"'
            for column in value_columns
        )

        query = f"""
            SELECT
                TRY_CAST("Fecha" AS DATE) AS fecha,
                {value_select}
            FROM read_parquet('{parquet_path.as_posix()}')
            WHERE TRY_CAST("Fecha" AS DATE) IS NOT NULL
            ORDER BY fecha
        """

        rows = con.execute(query).fetchall()

    finally:
        con.close()

    if not rows:
        log.warning(
            "Dataset %s no contiene observaciones válidas",
            dataset.id,
        )
        return None

    # El YTD se calcula para el año de la última observación.
    last_date = rows[-1][0]
    target_year = last_date.year
    year_start = date(target_year, 1, 1)

    previous_rows = [
        row
        for row in rows
        if row[0] < year_start
    ]

    ytd_rows = [
        row
        for row in rows
        if year_start <= row[0] <= last_date
    ]

    if not ytd_rows:
        log.warning(
            "Dataset %s no contiene datos para el YTD de %s",
            dataset.id,
            target_year,
        )
        return None

    # Idealmente se utiliza la última observación del año anterior.
    # Si no existe, se utiliza la primera observación del año actual.
    base_row = previous_rows[-1] if previous_rows else ytd_rows[0]

    series: list[PlotSeries] = []

    for column_index, column in enumerate(value_columns, start=1):
        base_value = base_row[column_index]

        if base_value is None:
            first_valid_row = next(
                (
                    row
                    for row in ytd_rows
                    if row[column_index] is not None
                ),
                None,
            )

            if first_valid_row is None:
                continue

            base_value = first_valid_row[column_index]

        base_value = float(base_value)
        previous_value = base_value
        accumulated_value = 0.0
        points: list[tuple[str, float]] = []

        for row in ytd_rows:
            current_raw = row[column_index]

            if current_raw is None:
                continue

            current_value = float(current_raw)

            # Paso 1: diferencia diaria.
            daily_difference = current_value - previous_value

            # Paso 2: acumulación YTD.
            accumulated_value += daily_difference

            points.append(
                (
                    str(row[0]),
                    accumulated_value,
                )
            )

            previous_value = current_value

        if points:
            series.append(
                PlotSeries(
                    label=column,
                    points=_downsample(points),
                )
            )

    valid_overlays = tuple(
        label
        for label in overlay_columns
        if any(series_item.label == label for series_item in series)
    )

    plot = PlotData(
        dataset.id,
        "stacked_area",
        "timeseries",
        dataset.unit,
        series,
        overlay=valid_overlays,
        date_note=(
            f"Variación diaria acumulada YTD "
            f"desde el corte base {base_row[0]} "
            f"hasta {last_date}"
        ),
    )

    return None if plot.is_empty() else plot


def _day_month_label(iso: str) -> str:
    """``YYYY-MM-DD`` → ``29abr26`` (día + mes abreviado + año corto). Como
    ``_month_label`` pero con el DÍA: estos reportes no siempre cortan a fin de
    mes calendario, así que el eje X muestra la fecha real del corte, no el mes."""
    d = date.fromisoformat(iso[:10])
    return f"{d.day}{_MONTHS_ES[d.month - 1]}{iso[2:4]}"


def wide_monthly_diff_bars(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Variación MES A MES de columnas ANCHAS que vienen como NIVEL ACUMULADO (no
    un flujo a sumar): para cada columna, resta el nivel del corte contra el del
    corte anterior → barras APILADAS (una serie por columna; ``overlay``
    superpuesta como punto). Es la versión "sin acumular después" de
    ``wide_daily_diff_ytd`` (que sí encadena el diff diario en un cumsum YTD) —
    acá cada barra es la variación de UN período nomás, para leerlas de a una
    (réplica de "Var. Mensual Posición AFP en SPC nominal" del tablero).

    Agrupa por MES calendario tomando la última observación de cada uno (soporta
    tanto un corte mensual regular como uno irregular con varias filas por mes) y
    etiqueta el eje X con la fecha REAL de ese corte (``29abr26``), no solo el mes
    — el reporte no siempre cae a fin de mes.

    params: ``include`` (cols a apilar; default = todas menos overlay), ``overlay``
    (cols superpuestas como punto, p.ej. ``["Neto"]``), ``months`` (int, default 10).
    """
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

    # Una fila representativa por mes: la última observación (por fecha) del mes.
    by_month: dict[str, dict] = {}
    for r in rows:
        iso = r.get(roles.date_col)
        if iso is None:
            continue
        mk = _month_key(iso)
        prev = by_month.get(mk)
        if prev is None or iso > prev[roles.date_col]:
            by_month[mk] = r
    month_keys = sorted(by_month)
    if len(month_keys) < 2:
        return None
    # El primer mes no tiene un mes previo con el que diferenciar.
    diff_keys = month_keys[1:][-int(params.get("months", 10)):]

    sd: dict[str, dict[str, float]] = {c: {} for c in wanted}
    labels: list[str] = []
    for k in diff_keys:
        i = month_keys.index(k)
        cur_row, prev_row = by_month[k], by_month[month_keys[i - 1]]
        label = _day_month_label(str(cur_row[roles.date_col]))
        labels.append(label)
        for c in wanted:
            cur_v, prev_v = cur_row.get(c), prev_row.get(c)
            if cur_v is None or prev_v is None:
                continue
            try:
                sd[c][label] = float(cur_v) - float(prev_v)
            except (TypeError, ValueError):
                continue
    if not labels:
        return None

    series_order = list(cols)
    overlay: tuple[str, ...] = ()
    if overlay_cols:
        series_order.append(overlay_cols[0])
        overlay = (overlay_cols[0],)
    last_iso = str(by_month[month_keys[-1]][roles.date_col])
    note = f"Variación mes a mes (corte vs. corte previo) · datos hasta {_fmt_date(last_iso)}"
    return _grouped(dataset.id, dataset.unit, sd, labels, series_order, overlay=overlay, date_note=note)


def monthly_bars_by_cat(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Valor MENSUAL ya calculado en el parquet (NO un índice a diferenciar, p.ej.
    un retorno mensual) por categoría → barras agrupadas (X = mes, una serie por
    categoría). A diferencia de ``monthly_returns``/``monthly_sum_by_fund``
    (parquets ANCHOS, una columna por fondo), lee un parquet LARGO con la categoría
    en una columna — útil cuando además hay que FILTRAR otra columna (ej. un
    parquet Fecha x AFP x Fondo→Retorno, filtrando AFP="Total" para el consolidado).

    params: ``category`` (col de la serie/eje de color), ``value`` (col numérica),
    ``filter_col``/``filter_val`` (opcional, restringe filas), ``order`` (orden de
    las categorías/series; sin él, alfabético), ``months`` (int, default 10).
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

    by_cat: dict[str, dict[str, float]] = {}
    last = ""
    for r in rows:
        c, iso, raw = r.get(cat), r.get(roles.date_col), r.get(val)
        if c is None or iso is None or raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        by_cat.setdefault(str(c), {})[_month_key(iso)] = v  # última fila vista por (mes, categoría)
        last = max(last, iso)
    if not by_cat or not last:
        return None
    cats = [c for c in (params.get("order") or sorted(by_cat)) if c in by_cat] or sorted(by_cat)
    all_keys = sorted({k for d in by_cat.values() for k in d})
    keys = all_keys[-int(params.get("months", 10)):]
    sd = {c: {_month_label(k): by_cat[c].get(k, 0.0) for k in keys} for c in cats}
    note = f"Datos hasta {_fmt_date(last)}"
    return _grouped(dataset.id, dataset.unit, sd, [_month_label(k) for k in keys], cats, date_note=note)


def ytd_grouped_by_cat(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Rentabilidad YTD compuesta GEOMÉTRICAMENTE desde valores MENSUALES ya
    calculados (no un índice a diferenciar), cruzando DOS categóricas → barras
    agrupadas (eje X = ``group``, una serie por ``category``). Para cada
    combinación (categoría, grupo) compone los valores del año en curso:
    ``prod(1 + v_i/100) - 1``.

    params: ``category`` (col de la serie/color, ej. AFP), ``group`` (col del eje
    X, ej. Fondo), ``value`` (col numérica, retorno mensual en %), ``exclude``
    (valores de ``category`` a excluir, ej. un consolidado "Total"), ``order``
    (orden del eje X), ``year`` (default: el año del último dato).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    cat, group, val = params.get("category"), params.get("group"), params.get("value")
    if not cat or not group or not val:
        return None
    exclude = {str(x) for x in (params.get("exclude") or [])}
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, cat, group, val])
    finally:
        con.close()

    by_cat_group: dict[str, dict[str, list[tuple[str, float]]]] = {}
    last = ""
    for r in rows:
        c, g, iso, raw = r.get(cat), r.get(group), r.get(roles.date_col), r.get(val)
        if c is None or g is None or iso is None or raw is None or str(c) in exclude:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        by_cat_group.setdefault(str(c), {}).setdefault(str(g), []).append((str(iso), v))
        last = max(last, str(iso))
    if not last:
        return None
    year = str(params.get("year") or last[:4])
    all_groups = sorted({g for d in by_cat_group.values() for g in d})
    groups = [g for g in (params.get("order") or all_groups) if g in all_groups]
    cats = sorted(by_cat_group)
    sd: dict[str, dict[str, float]] = {}
    for c in cats:
        col: dict[str, float] = {}
        for g in groups:
            pts = sorted(p for p in by_cat_group[c].get(g, []) if p[0][:4] == year)
            if not pts:
                continue
            total = 1.0
            for _iso, v in pts:
                total *= 1.0 + v / 100.0
            col[g] = (total - 1.0) * 100.0
        sd[c] = col
    note = f"YTD {year} (compuesto de retornos mensuales) · datos hasta {_fmt_date(last)}"
    return _grouped(dataset.id, dataset.unit, sd, groups, cats, date_note=note)


def wide_lines(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Multi-línea de un parquet 'ancho' (varias columnas de valor), seleccionando
    o excluyendo columnas y respetando su orden.

    params: ``include`` (lista en orden; default = todas menos ``overlay``),
    ``exclude`` (lista), ``accumulate`` (``cumsum``/``rebase`` opcional sobre cada
    columna), ``overlay`` (columnas que se dibujan superpuestas como línea "Neto"
    sobre el área apilada divergente, en vez de apilarse; se agregan solas, no hace
    falta repetirlas en ``include``).
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
        # ``overlay`` se resuelve APARTE de ``include``, igual que en
        # ``wide_monthly_bars``: pedir el Neto como superpuesto no debe obligar a
        # listarlo también entre las series apiladas (antes, si faltaba en
        # ``include``, el gráfico perdía la línea del Neto sin avisar).
        overlay_cols = [c for c in (params.get("overlay") or []) if c in roles.value_cols]
        wanted = params.get("include") or [c for c in roles.value_cols if c not in overlay_cols]
        cols = [c for c in wanted if c in roles.value_cols and c not in exclude]
        cols += [c for c in overlay_cols if c not in cols]
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
    # ``labels``: nombre de la columna → nombre del informe (Suscripcion →
    # Suscripciones). Solo cambia la leyenda; la agregación usa la columna real.
    labels = dict(params.get("labels") or {})
    sd: dict[str, dict[str, float]] = {
        labels.get(v, v): {c: agg[c].get(v, 0.0) for c in cats} for v in values
    }
    series_order = [labels.get(v, v) for v in values]
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


def window_stacked_diff_by_cat(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Variación DIARIA (nivel[t] - nivel[t-1]) de un parquet LARGO de STOCK
    (fecha + categoría + nivel) → barras apiladas DIVERGENTES por día (últimos
    N), una serie por categoría; opcional punto "Neto" = suma del día.

    A diferencia de ``window_stacked_by_cat`` (que apila el VALOR crudo del
    parquet, pensado para uno que YA es flujo, ej. ``movimientos_fondos``/
    ``Flujos_usd``): acá el valor de entrada es un NIVEL/stock (ej.
    ``stock_nivel_afp``/``Stock_USD``) y la función DIFERENCIA día a día antes
    de apilar, así la barra muestra cuánto cambió cada categoría, no el stock
    acumulado.

    params: ``category`` / ``value`` (si faltan, ``detect_roles``), ``last_n``
    (días de VARIACIÓN a mostrar, default 14 — internamente se lee un día extra
    de nivel para poder diferenciar el primero de la ventana), ``order`` (orden
    de categorías), ``net`` (bool, agrega "Neto" como punto superpuesto).
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
    by_cat = _aggregate_by_category(rows, roles.date_col, cat, val)  # {cat: [(iso, nivel)]}
    by_cat = {c: dict(s) for c, s in by_cat.items()}
    all_dates = sorted({d for s in by_cat.values() for d in s})
    last_n = int(params.get("last_n", 14))
    sel = all_dates[-(last_n + 1):]  # un día extra de NIVEL para diferenciar el primero
    if len(sel) < 2:
        return None
    diff_dates = sel[1:]
    labels = [_daymon(d) for d in diff_dates]
    cats = [c for c in (params.get("order") or sorted(by_cat)) if c in by_cat]
    sd: dict[str, dict[str, float]] = {
        c: {
            labels[i]: by_cat[c].get(diff_dates[i], 0.0) - by_cat[c].get(sel[i], 0.0)
            for i in range(len(diff_dates))
        }
        for c in cats
    }
    series_order = list(cats)
    overlay: tuple[str, ...] = ()
    if params.get("net"):
        sd["Neto"] = {
            labels[i]: sum(sd[c][labels[i]] for c in cats) for i in range(len(diff_dates))
        }
        series_order.append("Neto")
        overlay = ("Neto",)
    note = f"Variación diaria (nivel t vs. t-1) · {_fmt_date(diff_dates[0])} → {_fmt_date(diff_dates[-1])}"
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


def window_pivot_grouped(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras apiladas por categoría de un parquet LARGO con columna de TIPO: pivota
    ``type_col`` a una serie por valor y suma la última ventana por ``group``,
    conservando el signo de cada tipo. La(s) serie(s) en ``overlay`` van como punto.

    Réplica de "Flujo cambiario por AFP" tras pasar a formato largo
    (Fecha, Sector_contraparte, Tipo∈{Spot,Forward,Neto}, Monto): apila Spot+Forward
    con Neto (= Spot+Forward, ya provisto como Tipo) como punto. Es la versión larga y
    no-divergente de ``snapshot_grouped`` (cuando el parquet trae la serie completa en
    vez de un corte transversal pre-reducido).
    params: ``group`` (eje X), ``type_col`` (col de tipo), ``values`` (valores de tipo
    → serie, en orden de apilado), ``value`` (col de monto), ``overlay`` (valores que
    van como punto, p.ej. ``["Neto"]``), ``order`` (orden del eje X), ``window_days``
    (default 7), ``weekly_asof`` (corte común EXPLÍCITO; si se omite, se usa el corte
    común implícito = mínimo de los últimos días con dato de cada tipo en ``values``,
    para que ningún tipo quede con ventana truncada/sesgada).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    group, tcol, val = params.get("group"), params.get("type_col"), params.get("value")
    values = list(params.get("values") or [])
    if not group or not tcol or not val or not values:
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
    wanted = set(values)
    # Corte común: la fecha más reciente en que TODOS los tipos pedidos (ej.
    # Spot y Forward) tienen dato. Si se ancla al máximo global (el tipo que
    # reporta más rápido), el/los tipo(s) rezagados quedan con una ventana
    # truncada (menos días reales sumados) y la punta del gráfico sale sesgada
    # hacia el tipo más adelantado. Usando el mínimo de los máximos por tipo,
    # la ventana de ``window_days`` es completa y comparable para todos.
    per_type_last: dict[str, str] = {}
    for r in rows:
        t, d = r.get(tcol), r.get(roles.date_col)
        if t is None or d is None or str(t) not in wanted:
            continue
        per_type_last[str(t)] = max(per_type_last.get(str(t), ""), str(d))
    common_last = min(per_type_last.values()) if per_type_last else max(isos)
    last = str(params.get("weekly_asof") or common_last)
    start = (date.fromisoformat(last[:10]) - timedelta(days=int(params.get("window_days", 7)))).isoformat()
    agg: dict[str, dict[str, float]] = {}
    for r in rows:
        d, g, t = r.get(roles.date_col), r.get(group), r.get(tcol)
        if d is None or g is None or str(t) not in wanted or not (start <= str(d) <= last):
            continue
        try:
            v = float(r.get(val))
        except (TypeError, ValueError):
            continue
        per = agg.setdefault(str(g), {})
        per[str(t)] = per.get(str(t), 0.0) + v
    if not agg:
        return None
    cats = [c for c in (params.get("order") or sorted(agg)) if c in agg] or sorted(agg)
    sd = {v: {c: agg[c].get(v, 0.0) for c in cats} for v in values}
    overlay = tuple(c for c in (params.get("overlay") or []) if c in values)
    note = f"Suma {_fmt_date(start)} → {_fmt_date(last)}"
    return _grouped(dataset.id, dataset.unit, sd, cats, values, overlay=overlay, date_note=note)


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


# ── Informe DCV: portafolio por agente y distribución por tramo de plazo ─────
#
# Las tres transforms de abajo leen el MISMO parquet maestro
# (``variacion_stock_todos``: Fecha x Bucket x Tipo x Sector x Moneda x Stock_USD)
# al ÚLTIMO corte disponible. Es el único dataset que abre el stock DCV por
# agente e instrumento a la vez, así que tabla y gráfico de cada bloque salen de
# la misma fuente y no pueden descuadrarse.

# Sector del parquet → nombre del agente en el correo DCV.
_DCV_AGENTS = {
    "Bancos": "Bancos",
    "AFP": "FP y FC",
    "FFMM": "FFMM",
    "CS": "CSV",
    "Mandantes": "Mandantes",
    "CB": "CB",
    "Otros": "Otros",
}

# Orden de instrumentos del correo (los que no aparezcan en el parquet se omiten;
# los que el parquet traiga y no estén acá van al final, alfabéticos).
_DCV_TIPO_ORDER = ("PDBC", "DAP", "BTP", "BTU", "BCP", "BCU", "BB", "BE", "BCCh", "Letras MdH", "Otros")

# Sufijo de moneda del correo (el original abre DAP/BB/BC por moneda).
_DCV_CCY_SUFFIX = {"CLP": "$", "UF": "UF", "USD": "USD"}

# Duración por instrumento x agente: dos parquets aparte (no ``variacion_...``),
# mismo grano (Tipo x Moneda x Sector), cada uno un SNAPSHOT (solo trae la fecha
# de hoy, sin histórico) — "intermediación financiera" (DAP/PDBC) y "renta fija"
# (el resto de los instrumentos).
_DCV_DURATION_FILES = ("duracion_iif.parquet", "duracion_rf.parquet")


def _dcv_rows(
    dataset: ParquetDataset, parquet_dir: Path, params: dict,
) -> tuple[list[dict], str] | tuple[None, None]:
    """Filas del ÚLTIMO corte del parquet DCV + la fecha de ese corte.

    ``params["sector"]`` filtra a un agente (``None`` = todos). Con
    ``params["weekly_asof"]`` el corte se ancla a la fecha común del informe
    (última fecha con dato <= ese corte), igual que el resto de la familia."""
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None, None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or not roles.value_cols:
            return None, None
        dates = _distinct_dates_sorted(con, path, roles.date_col)
        if not dates:
            return None, None
        asof = params.get("weekly_asof")
        cut = next((d for d in reversed(dates) if d <= asof), dates[0]) if asof else dates[-1]

        cols = [c for c in ("Bucket", "Tipo", "Sector", "Moneda") if c in roles.category_cols]
        val = roles.value_cols[0]
        src = f"read_parquet('{path.as_posix()}')"
        where = f"TRY_CAST({roles.date_col} AS DATE) = DATE '{cut}'"
        sector = params.get("sector")
        if sector:
            if "Sector" not in cols:
                return None, None
            where += f" AND Sector = '{sector}'"
        rows = con.execute(
            f"SELECT {', '.join(cols)}, SUM({val}) AS v FROM {src} "
            f"WHERE {where} GROUP BY {', '.join(cols)}"
        ).df().to_dict("records")
    finally:
        con.close()
    return ([{**r, "v": float(r["v"])} for r in rows if r.get("v") is not None], cut)


def _dcv_instrument_label(rows: list[dict]) -> dict[tuple[str, str], str]:
    """``(Tipo, Moneda) → etiqueta de fila``.

    Un instrumento que existe en MÁS de una moneda se abre en una fila por moneda
    con sufijo (``DAP $`` / ``DAP UF``), como el correo; el que existe en una sola
    queda con el nombre pelado (``PDBC``, ``BTP``). Así la tabla no inventa filas
    vacías ni pierde la apertura por moneda donde sí la hay."""
    ccy_by_tipo: dict[str, set[str]] = {}
    for r in rows:
        if r.get("v"):
            ccy_by_tipo.setdefault(str(r.get("Tipo")), set()).add(str(r.get("Moneda") or ""))
    out: dict[tuple[str, str], str] = {}
    for r in rows:
        tipo, ccy = str(r.get("Tipo")), str(r.get("Moneda") or "")
        multi = len(ccy_by_tipo.get(tipo, set())) > 1
        suffix = _DCV_CCY_SUFFIX.get(ccy, ccy)
        out[(tipo, ccy)] = f"{tipo} {suffix}".strip() if multi and suffix else tipo
    return out


def _dcv_order_instruments(labels: list[str]) -> list[str]:
    """Instrumentos en el orden del correo; los desconocidos, al final."""
    def key(label: str) -> tuple[int, str, str]:
        tipo = label.split(" ")[0]
        rank = _DCV_TIPO_ORDER.index(tipo) if tipo in _DCV_TIPO_ORDER else len(_DCV_TIPO_ORDER)
        return (rank, tipo, label)
    return sorted(dict.fromkeys(labels), key=key)


def _dcv_snapshot_rows(parquet_dir: Path, files: tuple[str, ...] = _DCV_DURATION_FILES) -> list[dict]:
    """Filas ``{Tipo, Moneda, Sector, v}`` del ÚLTIMO corte de cada archivo en
    ``files``. Genérica: mismo grano que ``variacion_instrumento_todos_plazo``
    (Tipo x Moneda x Sector) — se indexa con las MISMAS ``_dcv_instrument_label``/
    ``_DCV_AGENTS`` de la tabla de portafolio. Cada archivo es un SNAPSHOT (trae
    solo la fecha de hoy, sin histórico): se lee su propio último corte, no el
    de ``variacion_...`` — pueden no coincidir exactamente si se refrescan en
    momentos distintos del día. La usan tanto la duración (``duracion_iif``/
    ``duracion_rf``, ``v=Duracion``) como los vencimientos (``vencimientos_hoy``/
    ``vencimientos_t_mas_uno``/``vencimientos_cinco_dias``, ``v=Stock_USD``)."""
    out: list[dict] = []
    for fname in files:
        path = parquet_dir / fname
        if not path.exists():
            continue
        con = duckdb.connect()
        try:
            roles = detect_roles(path, con)
            if roles.date_col is None or not roles.value_cols:
                continue
            dates = _distinct_dates_sorted(con, path, roles.date_col)
            if not dates:
                continue
            cut = dates[-1]
            cols = [c for c in ("Tipo", "Sector", "Moneda") if c in roles.category_cols]
            val = roles.value_cols[0]
            src = f"read_parquet('{path.as_posix()}')"
            rows = con.execute(
                f"SELECT {', '.join(cols)}, {val} AS v FROM {src} "
                f"WHERE TRY_CAST({roles.date_col} AS DATE) = DATE '{cut}'"
            ).df().to_dict("records")
            out.extend({**r, "v": float(r["v"])} for r in rows if r.get("v") is not None)
        finally:
            con.close()
    return out


def dcv_portfolio_table(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Tabla "Portafolio por agente": monto, DURACIÓN y % del portafolio por
    instrumento (filas) y agente (columnas), al último corte. La duración sale
    de ``_DCV_DURATION_FILES`` (parquets aparte, mismo grano): sin ellos la
    columna simplemente no se dibuja (mismo resultado que antes)."""
    from .svg_chart import render_dcv_portfolio_table

    rows, cut = _dcv_rows(dataset, parquet_dir, params)
    if not rows:
        return None
    dur_rows = _dcv_snapshot_rows(parquet_dir, params.get("duration_files") or _DCV_DURATION_FILES)
    labels = _dcv_instrument_label(rows + dur_rows)

    monto: dict[str, dict[str, float]] = {}
    for r in rows:
        agent = _DCV_AGENTS.get(str(r.get("Sector")))
        if agent is None:
            continue
        inst = labels[(str(r.get("Tipo")), str(r.get("Moneda") or ""))]
        monto.setdefault(agent, {})[inst] = monto.setdefault(agent, {}).get(inst, 0.0) + r["v"]
    if not monto:
        return None

    duracion: dict[str, dict[str, float]] = {}
    for r in dur_rows:
        agent = _DCV_AGENTS.get(str(r.get("Sector")))
        key = (str(r.get("Tipo")), str(r.get("Moneda") or ""))
        if agent is None or key not in labels:
            continue
        duracion.setdefault(agent, {})[labels[key]] = r["v"]

    agents = [a for a in _DCV_AGENTS.values() if a in monto]
    instruments = _dcv_order_instruments([i for per in monto.values() for i in per])
    return HtmlTable(
        html=render_dcv_portfolio_table(
            instruments, agents, monto, unit=dataset.unit, asof=_fmt_date(cut),
            duracion=duracion or None,
        ),
        dataset_id=dataset.id,
    )


# ── Informe DCV: Próximos Vencimientos por agente (T / T+1 / Acum 5d. / Mes) ──
#
# Tres parquets SNAPSHOT (mismo grano Tipo x Sector x Moneda que la duración,
# leídos con ``_dcv_snapshot_rows``) + uno mensual (``vencimientos_futuros_
# instrumento``, Año x Mes_label, YA en el catálogo) para el mes en curso. El
# correo real trae Acum 7d./Acum 30d.; el dato disponible es Acum 5d. (no 7) y
# un total MENSUAL sin resolución diaria (no un rolling 30d) — documentado en
# la nota del bloque, mismo criterio que las demás diferencias de cobertura.

_DCV_MATURITY_SNAPSHOT_FILES: dict[str, str] = {
    "T": "vencimientos_hoy.parquet",
    "T+1": "vencimientos_t_mas_uno.parquet",
    "Acum 5d.": "vencimientos_cinco_dias.parquet",
}
_DCV_MATURITY_MONTHLY_FILE = "vencimientos_futuros_instrumento.parquet"
_DCV_MATURITY_MONTHLY_LABEL = "Mes"
# Filas fijas de la tabla de vencimientos (a diferencia del portafolio, acá el
# correo repite el MISMO template de filas en las 8 mini-tablas): PDBC y DAP
# siempre abiertos por moneda; cualquier otro instrumento (los ``vencimientos_*``
# solo declaran "Otros"; el mensual trae más detalle -BB/BCCh/BE/BTP/BTU/Letras
# MdH- que acá se colapsa por consistencia) cae en "RF" — el correo real abre
# esa fila en Soberano/Bancario/Corporativo, apertura que el dato no trae.
_DCV_MATURITY_ROWS = ("PDBC", "DAP $", "DAP UF", "DAP USD", "RF")
_DCV_MES_LABEL_ES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def _dcv_maturity_label(tipo: str, moneda: str) -> str:
    """Tipo/Moneda → fila de "Próximos Vencimientos" (ver ``_DCV_MATURITY_ROWS``:
    PDBC y DAP quedan tal cual — SIEMPRE con sufijo de moneda, a diferencia de
    ``_dcv_instrument_label`` que solo lo agrega si hay más de una moneda —
    porque acá el template de filas es fijo e igual en las 8 mini-tablas;
    cualquier otro instrumento colapsa en "RF"."""
    if tipo == "PDBC":
        return "PDBC"
    if tipo == "DAP":
        suffix = _DCV_CCY_SUFFIX.get(moneda, moneda)
        return f"DAP {suffix}".strip() if suffix else "DAP"
    return "RF"


def _dcv_current_month_rows(parquet_dir: Path, fname: str, as_of: str | None = None) -> list[dict]:
    """Filas ``{Tipo, Sector, Moneda, v}`` de ``vencimientos_futuros_instrumento``
    (mensual: Año x Mes_label) para el mes de ``as_of`` (default: hoy). Total
    PROGRAMADO del mes — el dato no trae resolución diaria, así que no distingue
    lo ya vencido dentro del mes de lo que falta."""
    path = parquet_dir / fname
    if not path.exists():
        return []
    today = date.fromisoformat(as_of[:10]) if as_of else date.today()
    year, mes_label = str(today.year), _DCV_MES_LABEL_ES[today.month - 1]
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        cols = [c for c in ("Tipo", "Sector", "Moneda") if c in roles.category_cols]
        if not cols or not roles.value_cols:
            return []
        val = roles.value_cols[0]
        src = f"read_parquet('{path.as_posix()}')"
        rows = con.execute(
            f"SELECT {', '.join(cols)}, {val} AS v FROM {src} "
            f'WHERE CAST("Año" AS VARCHAR) = \'{year}\' AND "Mes_label" = \'{mes_label}\''
        ).df().to_dict("records")
    finally:
        con.close()
    return [{**r, "v": float(r["v"])} for r in rows if r.get("v") is not None]


def dcv_upcoming_maturities_table(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Tabla "Próximos Vencimientos" por agente: 8 mini-tablas (Totales + 7
    agentes), columnas T / T+1 / Acum 5d. / Mes, filas = instrumento + Total.

    ``dataset`` es solo UNO de los cuatro parquets (``vencimientos_hoy``, el
    bloque del spec); los otros tres se leen directo del ``parquet_dir`` (igual
    que la duración) — ver ``_DCV_MATURITY_SNAPSHOT_FILES``/
    ``_DCV_MATURITY_MONTHLY_FILE``. ``params["as_of"]`` (ISO) fija el "mes en
    curso" de la columna Mes; sin él, hoy."""
    from .svg_chart import render_dcv_maturities_grid

    columns = ["T", "T+1", "Acum 5d.", _DCV_MATURITY_MONTHLY_LABEL]
    data: dict[str, dict[str, dict[str, float]]] = {}
    any_data = False

    for col, fname in _DCV_MATURITY_SNAPSHOT_FILES.items():
        for r in _dcv_snapshot_rows(parquet_dir, (fname,)):
            agent = _DCV_AGENTS.get(str(r.get("Sector")))
            if agent is None:
                continue
            inst = _dcv_maturity_label(str(r.get("Tipo")), str(r.get("Moneda") or ""))
            per_col = data.setdefault(agent, {}).setdefault(col, {})
            per_col[inst] = per_col.get(inst, 0.0) + r["v"]
            any_data = True

    for r in _dcv_current_month_rows(parquet_dir, _DCV_MATURITY_MONTHLY_FILE, params.get("as_of")):
        agent = _DCV_AGENTS.get(str(r.get("Sector")))
        if agent is None:
            continue
        inst = _dcv_maturity_label(str(r.get("Tipo")), str(r.get("Moneda") or ""))
        per_col = data.setdefault(agent, {}).setdefault(_DCV_MATURITY_MONTHLY_LABEL, {})
        per_col[inst] = per_col.get(inst, 0.0) + r["v"]
        any_data = True

    if not any_data:
        return None

    agents = [a for a in _DCV_AGENTS.values() if a in data]
    # "Totales" = suma de los 7 agentes reales, primero en la grilla.
    totales: dict[str, dict[str, float]] = {}
    for a in agents:
        for col, per_inst in data[a].items():
            tot = totales.setdefault(col, {})
            for inst, v in per_inst.items():
                tot[inst] = tot.get(inst, 0.0) + v
    data_with_total = {"Totales": totales, **{a: data[a] for a in agents}}

    return HtmlTable(
        html=render_dcv_maturities_grid(["Totales", *agents], columns, data_with_total, unit=dataset.unit),
        dataset_id=dataset.id,
    )


def dcv_duration_scatter(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Duración por instrumento y agente → dispersión CATEGÓRICA (X = instrumento,
    color = agente, un punto por agente que tenga dato en ese instrumento).
    Réplica de "Duración agentes IIF/RF" del informe DCV.

    El parquet (``duracion_iif``/``duracion_rf``) es un SNAPSHOT — trae solo la
    fecha de hoy, sin histórico — así que se lee su propio último corte, sin
    ventana ni corte común con el resto del informe (igual que las tablas de
    portafolio/tramo, que también son snapshots del stock)."""
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        dates = _distinct_dates_sorted(con, path, roles.date_col)
        if not dates:
            return None
        cut = dates[-1]
        cols = [c for c in ("Tipo", "Sector", "Moneda") if c in roles.category_cols]
        val = roles.value_cols[0]
        src = f"read_parquet('{path.as_posix()}')"
        rows = con.execute(
            f"SELECT {', '.join(cols)}, {val} AS v FROM {src} "
            f"WHERE TRY_CAST({roles.date_col} AS DATE) = DATE '{cut}'"
        ).df().to_dict("records")
    finally:
        con.close()
    rows = [{**r, "v": float(r["v"])} for r in rows if r.get("v") is not None]
    if not rows:
        return None
    labels = _dcv_instrument_label(rows)

    by_agent: dict[str, dict[str, float]] = {}
    insts: list[str] = []
    for r in rows:
        agent = _DCV_AGENTS.get(str(r.get("Sector")))
        if agent is None:
            continue
        inst = labels[(str(r.get("Tipo")), str(r.get("Moneda") or ""))]
        by_agent.setdefault(agent, {})[inst] = r["v"]
        if inst not in insts:
            insts.append(inst)
    if not by_agent:
        return None

    order = _dcv_order_instruments(insts)
    agents = [a for a in _DCV_AGENTS.values() if a in by_agent]
    series = [
        PlotSeries(label=a, points=[(i, by_agent[a][i]) for i in order if i in by_agent[a]])
        for a in agents
    ]
    return PlotData(dataset.id, "grouped_bar", "grouped", dataset.unit, series,
                    date_note=f"Corte: {_fmt_date(cut)}")


def dcv_bucket_table(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Tabla "distribución por tramo de plazo": instrumento (filas) x tramo
    (columnas) al último corte. ``params["sector"]`` acota a un agente."""
    from .svg_chart import render_dcv_bucket_table

    rows, cut = _dcv_rows(dataset, parquet_dir, params)
    if not rows:
        return None
    labels = _dcv_instrument_label(rows)

    matrix: dict[str, dict[str, float]] = {}
    for r in rows:
        inst = labels[(str(r.get("Tipo")), str(r.get("Moneda") or ""))]
        bucket = str(r.get("Bucket"))
        matrix.setdefault(inst, {})[bucket] = matrix.setdefault(inst, {}).get(bucket, 0.0) + r["v"]
    if not matrix:
        return None

    buckets = _order_buckets({b for per in matrix.values() for b in per})
    instruments = _dcv_order_instruments(list(matrix))
    return HtmlTable(
        html=render_dcv_bucket_table(
            instruments, buckets, matrix, unit=dataset.unit, asof=_fmt_date(cut),
        ),
        dataset_id=dataset.id,
    )


def dcv_snapshot_stacked(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras apiladas del stock DCV al último corte.

    ``params``: ``x`` (categórica del eje X: ``"Tipo"`` o ``"Bucket"``),
    ``series`` (categórica del color: ``"Sector"`` o ``"Tipo"``) y ``sector``
    (filtra a un agente). Es la contraparte gráfica de las dos tablas de arriba:
    misma fuente, mismo corte, mismas etiquetas de instrumento."""
    x_col, s_col = params.get("x") or "Tipo", params.get("series") or "Sector"
    rows, cut = _dcv_rows(dataset, parquet_dir, params)
    if not rows:
        return None
    labels = _dcv_instrument_label(rows)

    def _label(row: dict, col: str) -> str:
        if col == "Tipo":
            return labels[(str(row.get("Tipo")), str(row.get("Moneda") or ""))]
        return str(row.get(col))

    agg: dict[str, dict[str, float]] = {}
    xs: list[str] = []
    for r in rows:
        xv, sv = _label(r, x_col), _label(r, s_col)
        if sv == "None" or xv == "None":
            continue
        if s_col == "Sector":
            sv = _DCV_AGENTS.get(sv, sv)
        agg.setdefault(sv, {})[xv] = agg.setdefault(sv, {}).get(xv, 0.0) + r["v"]
        if xv not in xs:
            xs.append(xv)
    if not agg:
        return None

    xcats = _order_buckets(xs) if x_col == "Bucket" else _dcv_order_instruments(xs)
    if s_col == "Sector":
        series_order = [a for a in _DCV_AGENTS.values() if a in agg]
    else:
        series_order = _dcv_order_instruments(list(agg))
    return _grouped(
        dataset.id, dataset.unit, agg, xcats, series_order,
        date_note=f"Corte: {_fmt_date(cut)}",
    )


# ── Rango histórico + dispersión x/y (NR: comparables GBI) ───────────────────
#
# Dos formas de gráfico NUEVAS (sin equivalente previo en el renderer): "range"
# (caja [mín,máx] + promedio + "hoy" por categoría, réplica de "Rendimiento
# monedas") y "scatter" (dispersión x/y etiquetada, réplica de "Retorno FX y
# tasas GBI"). Ver ``svg_chart._render_range_band`` / ``_render_scatter_labeled``.


def gbi_rendimiento_range(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Rango histórico (mín/máx), promedio y valor "hoy" por categoría, sobre un
    índice REBASADO a 100 al inicio de la ventana: ``100 * Valor / Valor_inicio``
    (réplica de "Rendimiento monedas" del tablero GBI — un parquet LARGO con
    fecha + categoría [p.ej. "País | Rating"] + nivel).

    A diferencia de ``_accumulate(mode="rebase")`` (que resta el nivel base, para
    VARIACIONES), acá se DIVIDE por el nivel base: el resultado es un índice
    (100 = inicio de la ventana), no una variación absoluta.

    params: ``category``/``value`` (si faltan, ``detect_roles``), ``order``
    (orden/selección de categorías — el tablero las ordena por calidad
    crediticia, no alfabético; se declara explícito, como el resto de bloques
    NR), ``window`` (default ``ytd``).
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
    by_cat = _aggregate_by_category(rows, roles.date_col, cat, val)
    if not by_cat:
        return None
    last = max(iso for s in by_cat.values() for iso, _ in s)
    start = _window_start(last, params.get("window", "ytd")) or min(iso for s in by_cat.values() for iso, _ in s)

    order = params.get("order") or sorted(by_cat)
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    means: dict[str, float] = {}
    hoys: dict[str, float] = {}
    for c in order:
        pts = [(iso, v) for iso, v in by_cat.get(c, []) if iso >= start]
        if not pts or pts[0][1] == 0:
            continue
        base = pts[0][1]
        rebased = [100.0 * v / base for _iso, v in pts]
        mins[c] = min(rebased)
        maxs[c] = max(rebased)
        means[c] = sum(rebased) / len(rebased)
        hoys[c] = rebased[-1]
    cats = [c for c in order if c in hoys]
    if not cats:
        return None
    series = [
        PlotSeries("Mínimo", [(c, mins[c]) for c in cats]),
        PlotSeries("Máximo", [(c, maxs[c]) for c in cats]),
        PlotSeries("Promedio", [(c, means[c]) for c in cats]),
        PlotSeries("Hoy", [(c, hoys[c]) for c in cats]),
    ]
    note = f"Índice base 100 = {_fmt_date(start)} · datos hasta {_fmt_date(last)}"
    plot = PlotData(dataset.id, "bar", "range", dataset.unit or "Índice (base 100)", series, date_note=note)
    return None if plot.is_empty() else plot


def gbi_rendimiento_range_sin_base(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Rango histórico (mín/máx), promedio y valor "hoy" por categoría, sobre un
    índice REBASADO a 100 al inicio de la ventana: ``100 * Valor / Valor_inicio``
    (réplica de "Rendimiento monedas" del tablero GBI — un parquet LARGO con
    fecha + categoría [p.ej. "País | Rating"] + nivel).

    A diferencia de ``_accumulate(mode="rebase")`` (que resta el nivel base, para
    VARIACIONES), acá se DIVIDE por el nivel base: el resultado es un índice
    (100 = inicio de la ventana), no una variación absoluta.

    params: ``category``/``value`` (si faltan, ``detect_roles``), ``order``
    (orden/selección de categorías — el tablero las ordena por calidad
    crediticia, no alfabético; se declara explícito, como el resto de bloques
    NR), ``window`` (default ``ytd``).
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
    by_cat = _aggregate_by_category(rows, roles.date_col, cat, val)
    if not by_cat:
        return None
    last = max(iso for s in by_cat.values() for iso, _ in s)
    start = _window_start(last, params.get("window", "ytd")) or min(iso for s in by_cat.values() for iso, _ in s)

    order = params.get("order") or sorted(by_cat)
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    means: dict[str, float] = {}
    hoys: dict[str, float] = {}
    for c in order:
        pts = [(iso, v) for iso, v in by_cat.get(c, []) if iso >= start]
        if not pts or pts[0][1] == 0:
            continue
        base = pts[0][1]
        rebased = [v for _iso, v in pts]
        mins[c] = min(rebased)
        maxs[c] = max(rebased)
        means[c] = sum(rebased) / len(rebased)
        hoys[c] = rebased[-1]
    cats = [c for c in order if c in hoys]
    if not cats:
        return None
    series = [
        PlotSeries("Mínimo", [(c, mins[c]) for c in cats]),
        PlotSeries("Máximo", [(c, maxs[c]) for c in cats]),
        PlotSeries("Promedio", [(c, means[c]) for c in cats]),
        PlotSeries("Hoy", [(c, hoys[c]) for c in cats]),
    ]
    note = f"{_fmt_date(start)} · datos hasta {_fmt_date(last)}"
    plot = PlotData(dataset.id, "bar", "range", dataset.unit or "", series, date_note=note)
    return None if plot.is_empty() else plot






def fx_tasas_scatter(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Retorno acumulado FX vs. tasa de mercado por país, snapshot al último dato:
    dispersión x=retorno FX (%), y=retorno tasas (%), un punto por país (réplica de
    "Retorno FX y tasas GBI" del tablero).

    Parquet ANCHO con columnas PREFIJADAS ``fx_{PAIS}`` / ``rates_{PAIS}`` (NIVELES,
    no retornos; país en MAYÚSCULA, p.ej. ``fx_USD``/``rates_USD`` — confirmado
    contra el catálogo real, ``chart_type="fx_retorno_scatter_interactive"``). El
    retorno de cada columna es la SUMA de sus variaciones % diarias desde el inicio
    de la ventana hasta el último dato — aproximación ADITIVA del retorno acumulado
    (no geométrica), réplica fiel del cálculo del tablero (``pct_change().cumsum()``),
    no la fórmula compuesta que usan otras transforms de rentabilidad de este módulo.

    params: ``fx_prefix``/``rate_prefix`` (default ``fx_``/``rates_``),
    ``negate_fx`` (default ``True``: el nivel FX es moneda-local-por-USD, así que
    negar el retorno da la convención "positivo = apreciación", igual que el
    tablero), ``window`` (default ``ytd``), ``highlight`` (código de país —
    cualquier capitalización, sin prefijo — a destacar como "hoy"/doméstico; va a
    ``plot.overlay``).
    """
    from banks_rag.infrastructure.sql import series_analytics as sa

    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    fx_pre = params.get("fx_prefix", "fx_")
    rate_pre = params.get("rate_prefix", "rates_")
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, *roles.value_cols])
    finally:
        con.close()

    def _cum_return(col: str) -> float | None:
        pts = sa.clean_series(rows, roles.date_col, col)
        if len(pts) < 2:
            return None
        start = _window_start(pts[-1][0], params.get("window", "ytd")) or pts[0][0]
        total = 0.0
        counted = False
        for i in range(1, len(pts)):
            iso, v = pts[i]
            prev = pts[i - 1][1]
            if iso < start or not prev:
                continue
            total += (v - prev) / prev * 100.0
            counted = True
        return total if counted else None

    negate_fx = params.get("negate_fx", True)
    fx_ret: dict[str, float] = {}
    rate_ret: dict[str, float] = {}
    last_all = ""
    for r in rows:
        d = r.get(roles.date_col)
        if d is not None:
            last_all = max(last_all, str(d))
    for col in roles.value_cols:
        if col.startswith(fx_pre):
            r = _cum_return(col)
            if r is not None:
                fx_ret[col[len(fx_pre):].lower()] = -r if negate_fx else r
        elif col.startswith(rate_pre):
            r = _cum_return(col)
            if r is not None:
                rate_ret[col[len(rate_pre):].lower()] = -r

    countries = sorted(set(fx_ret) & set(rate_ret))
    if not countries:
        return None
    highlight = str(params.get("highlight") or "").strip().lower()
    series = [PlotSeries(c.upper(), [(f"{fx_ret[c]:.6f}", rate_ret[c])]) for c in countries]
    overlay = tuple(c.upper() for c in countries if c == highlight)
    note = f"Retorno acumulado desde inicio de año · datos hasta {_fmt_date(last_all)}" if last_all else ""
    plot = PlotData(dataset.id, "point", "scatter", dataset.unit or "%", series, overlay=overlay, date_note=note)
    return None if plot.is_empty() else plot


# ── Flujos cambiarios (familia fx) ───────────────────────────────────────────
#
# El "Informe Flujos Cambiarios" del BCCh mira el MISMO día de mercado desde
# varios cortes: por sector, por agente offshore, por plazo del derivado y por
# banco informante del fixing. Estas transforms son las formas que ese informe
# usa y que no existían: fila-como-eje-X de un parquet ancho, doble categórica
# sumada en ventana, y dos tablas (resumen por sector, Δ por agente).

def _sum_window(
    rows: list[dict], date_col: str, *, days: int, last_iso: str | None = None,
) -> tuple[list[dict], str, str]:
    """Filas de los últimos ``days`` días naturales + ``(desde, hasta)`` ISO.

    ``days<=1`` deja SOLO el último día con dato (el informe compara "hoy" contra
    la acumulación de 5 días)."""
    isos = [str(r[date_col])[:10] for r in rows if r.get(date_col)]
    if not isos:
        return [], "", ""
    last = last_iso or max(isos)
    start = str(date.fromisoformat(last) - timedelta(days=max(0, days - 1)))
    sel = [r for r in rows if r.get(date_col) and start <= str(r[date_col])[:10] <= last]
    return sel, start, last


def _num(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def wide_row_stacked(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Parquet ANCHO y SIN fecha (un corte ya agregado): eje X = los valores de una
    columna categórica (las FILAS), una serie apilada por cada columna de valor.

    Es la forma de los tableros de fixing (``fixing_banca_sector``: una fila por
    banco informante, una columna por sector contraparte) y de los cortes por
    temporalidad (``fixing_por_fecha``). Réplica de los "Gráfico N°7 / N°8" del
    informe: barra apilada divergente por fila + Neto como punto.

    params: ``row`` (col categórica del eje X; default = la 1ª categórica),
    ``series`` (cols de valor apiladas; default = todas menos ``overlay``),
    ``overlay`` (cols que van como punto superpuesto, p.ej. ``["Neto"]``),
    ``labels`` (renombra las columnas al nombre del informe), ``exclude_rows``
    (filas a omitir, p.ej. ``["Total"]``), ``row_order``, ``top_n`` (deja las N
    filas de mayor |total|, conservando el orden original).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        row_col = params.get("row") or (roles.category_cols[0] if roles.category_cols else None)
        if not row_col or not roles.value_cols:
            return None
        overlay_cols = [c for c in (params.get("overlay") or []) if c in roles.value_cols]
        stacked = [c for c in (params.get("series") or roles.value_cols) if c in roles.value_cols]
        stacked = [c for c in stacked if c not in overlay_cols]
        if not stacked:
            return None
        rows = _read_rows(con, path, date_col=None, columns=[row_col, *stacked, *overlay_cols])
    finally:
        con.close()

    excluded = {str(v) for v in (params.get("exclude_rows") or [])}
    agg: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for r in rows:
        name = str(r.get(row_col) or "").strip()
        if not name or name in excluded:
            continue
        if name not in agg:
            agg[name] = {}
            order.append(name)
        for c in (*stacked, *overlay_cols):
            agg[name][c] = agg[name].get(c, 0.0) + _num(r.get(c))

    if params.get("row_order"):
        order = [c for c in params["row_order"] if c in agg]
    top_n = params.get("top_n")
    if top_n:
        keep = {c for c, _ in sorted(
            agg.items(), key=lambda kv: abs(sum(kv[1].get(s, 0.0) for s in stacked)), reverse=True,
        )[:int(top_n)]}
        order = [c for c in order if c in keep]
    order = order[:_MAX_PLOT_CATEGORIES]
    if not order:
        return None

    labels = dict(params.get("labels") or {})
    series_dict = {
        labels.get(c, c): {n: agg[n].get(c, 0.0) for n in order}
        for c in (*stacked, *overlay_cols)
    }
    return _grouped(dataset.id, dataset.unit, series_dict, order,
                    [labels.get(c, c) for c in (*stacked, *overlay_cols)],
                    overlay=tuple(labels.get(c, c) for c in overlay_cols))


def window_stacked_two_cat(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Parquet LARGO con DOS categóricas: suma de la última ventana con eje X = una
    categórica (``group``) y una serie APILADA por la otra (``series``).

    Réplica de "Suscripciones / Vencimientos netos derivados" (X = agente offshore,
    apilado por instrumento CCS/Forward/FX swap/Opciones + Neto como punto). El
    informe separa suscripciones de vencimientos con ``filter_col``/``filter_val``.

    params: ``group`` (eje X), ``series`` (color), ``value``, ``filter_col`` /
    ``filter_val`` (opcional), ``window_days`` (default 1 = el último día con dato),
    ``labels`` (renombra valores de ``series``), ``series_order``, ``group_order``,
    ``top_n`` (grupos con mayor |neto|), ``net`` (bool, default True → punto Neto),
    ``total_label`` (agrega una columna con la suma de todos los grupos, como el
    "Total" del informe).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    group, scol, val = params.get("group"), params.get("series"), params.get("value")
    if not group or not scol or not val:
        return None
    fcol, fval = params.get("filter_col"), params.get("filter_val")
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None:
            return None
        cols = [roles.date_col, group, scol, val] + ([fcol] if fcol else [])
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=cols)
    finally:
        con.close()
    if fcol and fval is not None:
        rows = [r for r in rows if str(r.get(fcol)) == str(fval)]
    sel, start, last = _sum_window(
        rows, roles.date_col, days=int(params.get("window_days", 1)),
        last_iso=params.get("weekly_asof"),
    )
    if not sel:
        return None

    labels = dict(params.get("labels") or {})
    excluded_groups = {str(v) for v in (params.get("exclude_groups") or [])}
    agg: dict[str, dict[str, float]] = {}
    groups: list[str] = []
    series_seen: list[str] = []
    for r in sel:
        g = str(r.get(group) or "").strip()
        s_raw = str(r.get(scol) or "").strip()
        if not g or not s_raw or g in excluded_groups:
            continue
        s = labels.get(s_raw, s_raw)
        if g not in agg:
            agg[g] = {}
            groups.append(g)
        if s not in series_seen:
            series_seen.append(s)
        agg[g][s] = agg[g].get(s, 0.0) + _num(r.get(val))

    if params.get("group_order"):
        groups = [g for g in params["group_order"] if g in agg]
    if params.get("top_n"):
        keep = {g for g, _ in sorted(
            agg.items(), key=lambda kv: abs(sum(kv[1].values())), reverse=True,
        )[:int(params["top_n"])]}
        groups = [g for g in groups if g in keep]
    groups = groups[:_MAX_PLOT_CATEGORIES - 1]  # deja lugar a la columna Total
    if not groups:
        return None
    if params.get("series_order"):
        series_seen = [s for s in params["series_order"] if s in series_seen] + \
                      [s for s in series_seen if s not in params["series_order"]]

    total_label = params.get("total_label")
    if total_label:
        agg[total_label] = {
            s: sum(agg[g].get(s, 0.0) for g in groups) for s in series_seen
        }
        groups = [*groups, total_label]

    series_dict = {s: {g: agg[g].get(s, 0.0) for g in groups} for s in series_seen}
    overlay: tuple[str, ...] = ()
    if params.get("net", True):
        series_dict["Neto"] = {g: sum(agg[g].get(s, 0.0) for s in series_seen) for g in groups}
        series_seen = [*series_seen, "Neto"]
        overlay = ("Neto",)
    span = f"{_fmt_date(start)} → {_fmt_date(last)}" if start != last else _fmt_date(last)
    return _grouped(dataset.id, dataset.unit, series_dict, groups, series_seen,
                    overlay=overlay, date_note=span)


def daily_wide_stacked(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Parquet ANCHO con fecha: últimos N días en el eje X, una serie apilada por
    columna de valor y Neto como punto.

    Réplica del "Gráfico N°9: Posición derivados" (por día: suscripciones arriba,
    vencimientos abajo, posición neta como punto). Las columnas de ``negate`` se
    invierten de signo antes de apilar, de modo que el ALTO NETO de la columna sea
    la variación de posición del día.

    params: ``include`` (cols en orden; default = todas), ``negate`` (cols que
    restan), ``last_n`` (días, default 10), ``net_label`` (default "Neto"),
    ``labels`` (renombra columnas).
    """
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        cols = [c for c in (params.get("include") or roles.value_cols) if c in roles.value_cols]
        if not cols:
            return None
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=[roles.date_col, *cols])
    finally:
        con.close()

    negate = {str(c) for c in (params.get("negate") or [])}
    by_date: dict[str, dict[str, float]] = {}
    for r in rows:
        d = r.get(roles.date_col)
        if not d:
            continue
        iso = str(d)[:10]
        slot = by_date.setdefault(iso, {})
        for c in cols:
            v = _num(r.get(c))
            slot[c] = slot.get(c, 0.0) + (-v if c in negate else v)

    n = int(params.get("last_n", 10))
    # ``from_start``: el parquet mira al FUTURO (perfil de vencimientos), donde lo
    # relevante son los primeros N días, no los últimos.
    dates = sorted(by_date)[:n] if params.get("from_start") else sorted(by_date)[-n:]
    if not dates:
        return None
    labels = dict(params.get("labels") or {})
    x_labels = [_fmt_date(d) for d in dates]
    series_dict = {
        labels.get(c, c): {_fmt_date(d): by_date[d].get(c, 0.0) for d in dates} for c in cols
    }
    order = [labels.get(c, c) for c in cols]
    overlay: tuple[str, ...] = ()
    # El Neto solo tiene sentido cuando hay columnas de signo opuesto (``negate``);
    # en un apilado de puros positivos duplicaría el alto de la barra.
    if params.get("net", True):
        net_label = params.get("net_label", "Neto")
        series_dict[net_label] = {
            _fmt_date(d): sum(by_date[d].get(c, 0.0) for c in cols) for d in dates
        }
        order.append(net_label)
        overlay = (net_label,)
    return _grouped(dataset.id, dataset.unit, series_dict, x_labels, order, overlay=overlay)


def fx_sector_flow_table(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Tabla RESUMEN GENERAL del informe de flujos cambiarios: una fila por sector,
    columnas = flujo del día y acumulado de 5 días para Spot, Derivados y su suma.

    El informe real abre además Spot en afecto/no-afecto y Derivados en NDF/resto;
    ``flujo_cambiario`` NO trae esas aperturas (solo ``Spot`` y ``Forward`` por
    sector), así que la tabla replica la ESTRUCTURA con las columnas disponibles.
    La apertura fina queda documentada como bloque faltante en el spec.

    params: ``spot`` / ``deriv`` (cols; default Spot/Forward), ``days`` (ventana
    larga, default 5), ``labels`` (sector del parquet → nombre del informe),
    ``order`` (orden de filas).
    """
    from .svg_chart import render_fx_summary_table

    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    spot_col = params.get("spot", "Spot")
    deriv_col = params.get("deriv", "Forward")
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        sector_col = params.get("sector") or (roles.category_cols[0] if roles.category_cols else None)
        if roles.date_col is None or not sector_col:
            return None
        rows = _read_series_rows(
            con, path, date_col=roles.date_col, columns=[roles.date_col, sector_col, spot_col, deriv_col],
        )
    finally:
        con.close()
    if not rows:
        return None

    asof = params.get("weekly_asof")
    day_rows, _, last = _sum_window(rows, roles.date_col, days=1, last_iso=asof)
    span_rows, start_n, _ = _sum_window(
        rows, roles.date_col, days=int(params.get("days", 5)), last_iso=asof,
    )

    labels = dict(params.get("labels") or {})

    def _tally(subset: list[dict]) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for r in subset:
            raw = str(r.get(sector_col) or "").strip()
            if not raw:
                continue
            name = labels.get(raw, raw)
            s, d = out.get(name, (0.0, 0.0))
            out[name] = (s + _num(r.get(spot_col)), d + _num(r.get(deriv_col)))
        return out

    day, span = _tally(day_rows), _tally(span_rows)
    names = list(params.get("order") or [])
    names = [n for n in names if n in day or n in span]
    names += sorted(n for n in set(day) | set(span) if n not in names)
    if not names:
        return None

    data = {
        n: (*day.get(n, (0.0, 0.0)), *span.get(n, (0.0, 0.0))) for n in names
    }
    html = render_fx_summary_table(
        names, data, unit=dataset.unit,
        day_label=_fmt_date(last),
        span_label=f"{_fmt_date(start_n)} → {_fmt_date(last)}",
    )
    return HtmlTable(html=html, dataset_id=dataset.id)


def fx_agent_delta_table(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Tabla "Posición derivados por agente" (Δ T-1 / Δ T-5 / Δ T-10 / Δ T-20):
    filas = agente offshore, columnas = variación NETA acumulada de cada ventana.

    El neto de un día es ``Suscripción - Vencimiento`` (parquet largo con columna de
    tipo) o la suma de la columna de valor si no hay tipo. Cada Δ T-N suma los N
    últimos días HÁBILES CON DATO (no días naturales): así el informe compara
    jornadas de mercado, como el correo real.

    params: ``agent`` (col de agente), ``value``, ``type_col`` / ``pos`` / ``neg``
    (opcionales), ``windows`` (lista de N, default ``[1, 5, 10, 20]``),
    ``exclude_agents`` (p.ej. ``["Total"]``), ``total_label``.
    """
    from .svg_chart import render_fx_delta_table

    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    agent = params.get("agent")
    val = params.get("value")
    if not agent or not val:
        return None
    tcol, pos, neg = params.get("type_col"), params.get("pos"), params.get("neg")
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None:
            return None
        cols = [roles.date_col, agent, val] + ([tcol] if tcol else [])
        rows = _read_series_rows(con, path, date_col=roles.date_col, columns=cols)
    finally:
        con.close()
    if not rows:
        return None

    excluded = {str(v) for v in (params.get("exclude_agents") or [])}
    # {agente: {fecha: neto}} — el signo lo fija type_col cuando existe.
    per_day: dict[str, dict[str, float]] = {}
    for r in rows:
        d = r.get(roles.date_col)
        name = str(r.get(agent) or "").strip()
        if not d or not name or name in excluded:
            continue
        v = _num(r.get(val))
        if tcol:
            t = str(r.get(tcol) or "").strip()
            if neg and t == neg:
                v = -v
            elif pos and t != pos:
                continue
        slot = per_day.setdefault(name, {})
        iso = str(d)[:10]
        slot[iso] = slot.get(iso, 0.0) + v

    all_dates = sorted({d for s in per_day.values() for d in s})
    if not all_dates:
        return None
    windows = [int(w) for w in (params.get("windows") or [1, 5, 10, 20])]
    cut = {w: set(all_dates[-w:]) for w in windows}

    agents = sorted(per_day, key=lambda a: abs(sum(per_day[a].values())), reverse=True)
    agents = agents[:_MAX_PLOT_CATEGORIES]
    agents.sort()
    data = {
        a: tuple(sum(v for d, v in per_day[a].items() if d in cut[w]) for w in windows)
        for a in agents
    }
    total_label = params.get("total_label", "Total")
    data[total_label] = tuple(
        sum(data[a][i] for a in agents) for i in range(len(windows))
    )
    html = render_fx_delta_table(
        agents, data, windows, unit=dataset.unit,
        total_label=total_label, asof=_fmt_date(all_dates[-1]),
    )
    return HtmlTable(html=html, dataset_id=dataset.id)


def allocation_wide_by_fund(
    dataset: ParquetDataset,
    parquet_dir: Path,
    params: dict,
) -> PlotData | None:
    """
    Allocation histórica de un fondo en un parquet ancho.

    Estructura esperada:
        Fecha, Fondo, RFN, RVN, RFI, RVI, OTROS, AUM

    Filtra Fondo y crea una serie temporal por cada clase de activo.
    Opcionalmente incluye AUM para graficarlo en el eje derecho.
    """
    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None

    fund = str(params.get("fund") or "").strip()
    columns = list(
        params.get("columns")
        or ["RFN", "RVN", "RFI", "RVI", "Otros"]
    )

    if params.get("include_aum", True) and "AUM" not in columns:
        columns.append("AUM")

    con = duckdb.connect()

    try:
        available_columns = {
            row[0]
            for row in con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet('{parquet_path.as_posix()}')
                """
            ).fetchall()
        }

        required = {"Fecha", "Fondo"}
        missing_required = required - available_columns

        if missing_required:
            log.warning(
                "Dataset %s no contiene las columnas requeridas: %s",
                dataset.id,
                sorted(missing_required),
            )
            return None

        value_columns = [
            column
            for column in columns
            if column in available_columns
        ]

        if not value_columns:
            log.warning(
                "Dataset %s no tiene columnas de allocation disponibles",
                dataset.id,
            )
            return None

        quoted_values = ", ".join(
            f'TRY_CAST("{column}" AS DOUBLE) AS "{column}"'
            for column in value_columns
        )


        conditions = ['TRY_CAST("Fecha" AS DATE) IS NOT NULL']
        #where_clause = ""
        query_params: list[str] = []

        if fund:
            conditions.append('TRIM(CAST("Fondo" AS VARCHAR)) = ?')
            query_params.append(fund)

        where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                TRY_CAST("Fecha" AS DATE) AS fecha,
                {quoted_values}
            FROM read_parquet('{parquet_path.as_posix()}')
            {where_clause}
            ORDER BY fecha
        """

        rows = con.execute(query, query_params).fetchall()

    finally:
        con.close()

    if not rows:
        log.warning(
            "Dataset %s no tiene datos para Fondo=%s",
            dataset.id,
            fund,
        )
        return None

    series: list[PlotSeries] = []

    for column_index, column in enumerate(value_columns, start=1):
        points = [
            (str(row[0]), float(row[column_index]))
            for row in rows
            if row[column_index] is not None
        ]

        if points:
            series.append(
                PlotSeries(
                    label=column,
                    points=_downsample(points),
                )
            )

    plot = PlotData(
        dataset.id,
        "dual_axis" if "AUM" in value_columns else "line",
        "timeseries",
        dataset.unit,
        series,
    )

    return None if plot.is_empty() else plot


# DCV especifico:
def _short_maturity_date_label(value: object) -> str:
    """
    Formatea Vencimiento para eje X:
    2026-08-19 00:00:00 -> 19-08-26
    """
    s = str(value)
    try:
        d = date.fromisoformat(s[:10])
        return f"{d.day:02d}-{d.month:02d}" #-{str(d.year)[2:]}
    except ValueError:
        return s[:10]

def dcv_maturities_three_months_by_type(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """
    Vencimientos próximos tres meses por Tipo.
    Eje X = fecha corta de vencimiento.
    Series = Tipo.
    Valor = suma Stock_USD.
    """
    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None

    con = duckdb.connect()
    try:
        rows = con.execute(f"""
            SELECT
                CAST(Vencimiento AS DATE) AS Vencimiento,
                Tipo,
                SUM(CAST(Stock_USD AS DOUBLE)) AS value
            FROM read_parquet('{parquet_path.as_posix()}')
            WHERE Stock_USD IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
        """).df().to_dict("records")
    finally:
        con.close()

    if not rows:
        return None

    tipo_order = ["PDBC", "DAP", "BB", "BE", "BTP", "Letras MdH", "Otros"]

    raw_dates = sorted({str(r["Vencimiento"])[:10] for r in rows})
    date_labels = {
        iso: _short_maturity_date_label(iso)
        for iso in raw_dates
    }

    cat_order = [date_labels[iso] for iso in raw_dates]

    series_dict: dict[str, dict[str, float]] = {}

    for r in rows:
        tipo = str(r["Tipo"])
        iso = str(r["Vencimiento"])[:10]
        label = date_labels[iso]
        value = float(r["value"] or 0.0)

        series_dict.setdefault(tipo, {})
        series_dict[tipo][label] = series_dict[tipo].get(label, 0.0) + value

    series_order = [
        t for t in tipo_order
        if t in series_dict
    ] + sorted(
        t for t in series_dict
        if t not in tipo_order
    )

    overlay = ("Neto",) if params.get("net") is True else ()

    return _grouped(
        dataset.id,
        dataset.unit,
        series_dict,
        cat_order,
        series_order,
        overlay=overlay,
        date_note="Vencimientos próximos tres meses"
    )




# ── Informe Cambiario AM (familia cambiarioam) ───────────────────────────────
#
# El informe original dibujaba con Plotly y calculaba sus derivados (bandas de
# Bollinger, percentiles S/R, base 100, spreads) en el mismo script que leía el
# Excel. Acá se separan: el parquet guarda el dato CRUDO por columna y estas
# transforms hacen el cálculo, para que el gráfico se recalcule solo cuando
# llegue una sesión nueva sin volver a tocar el origen.
#
# Todas leen parquets ANCHOS (Fecha + una columna por serie) generados por
# ``scripts/build_cambiario_parquets.py``.

# Percentiles que el informe usa como soportes y resistencias, y el color/estilo
# con que los dibuja el original. El orden importa: es el de la leyenda.
_SR_PERCENTILES = (10, 25, 50, 75, 90, 100)


def _fmt_num(value: float) -> str:
    """Formato numérico es-CL, el MISMO del renderer. Se delega en vez de
    reimplementarlo para que las tablas y los ejes no muestren dos formatos
    distintos del mismo número (import diferido: ``svg_chart`` importa de
    ``parquet_facts``, no de acá, pero el diferido evita cualquier ciclo futuro)."""
    from .svg_chart import _fmt_num as fmt

    return fmt(value)


def _cam_zero_base(params: dict) -> bool:
    """¿El eje Y de este bloque debe incluir el 0?

    En el informe cambiario NO por defecto: son precios, índices y tasas que
    nunca se acercan a cero (el CLP en 930, un base 100, un RSI entre 30 y 70) y
    anclarlos en 0 aplasta la serie contra el borde. El spec puede pedir
    ``zero_base: True`` en un bloque donde el cero sí sea la referencia."""
    return bool(params.get("zero_base", False))


def _cam_wide_rows(
    dataset: ParquetDataset, parquet_dir: Path, *, keep_time: bool = False,
) -> tuple[str, list[str], list[dict]] | None:
    """``(date_col, value_cols, rows)`` de un parquet ancho del informe cambiario.

    ``keep_time=True`` conserva la HORA de la marca temporal (gráficos intradía);
    por defecto la fecha se castea a DATE como en el resto del informe."""
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or not roles.value_cols:
            return None
        if not keep_time:
            rows = _read_series_rows(
                con, path, date_col=roles.date_col,
                columns=[roles.date_col, *roles.value_cols],
            )
        else:
            quoted = ", ".join(f'"{c}"' for c in roles.value_cols)
            raw = con.execute(
                f'SELECT strftime("{roles.date_col}", \'%Y-%m-%dT%H:%M\') AS "{roles.date_col}", '
                f'{quoted} FROM read_parquet(\'{path.as_posix()}\') '
                f'WHERE "{roles.date_col}" IS NOT NULL ORDER BY 1'
            ).fetchall()
            names = [roles.date_col, *roles.value_cols]
            rows = [dict(zip(names, r, strict=True)) for r in raw]
        return roles.date_col, list(roles.value_cols), rows
    finally:
        con.close()


def _cam_points(rows: list[dict], date_col: str, value_col: str) -> list[tuple[str, float]]:
    from banks_rag.infrastructure.sql import series_analytics as sa

    return sa.clean_series(rows, date_col, value_col)


def _cam_last_months(points: list[tuple[str, float]], months: int) -> list[tuple[str, float]]:
    """Últimos ``months`` meses de la serie. Los percentiles S/R y la base 100 se
    calculan sobre el régimen RECIENTE, no sobre toda la historia: el original
    filtra a 12 meses (``_last_year``) justo por eso."""
    if not points or months <= 0:
        return points
    last = date.fromisoformat(points[-1][0][:10])
    year, month = last.year, last.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(last.day, 28)
    start = date(year, month, day).isoformat()
    return [p for p in points if p[0][:10] >= start]


def _cam_flat(points: list[tuple[str, float]], level: float) -> list[tuple[str, float]]:
    """Línea horizontal al nivel ``level`` sobre el mismo dominio X de la serie.
    Es como el renderer dibuja un umbral (percentil, 70/30 del RSI) sin agregar
    una primitiva nueva: una serie de dos puntos, extremo a extremo."""
    if not points:
        return []
    return [(points[0][0], level), (points[-1][0], level)]


def _cam_percentile(sorted_values: list[float], pct: float) -> float:
    """Percentil por interpolación lineal, igual criterio que ``numpy.percentile``
    (el original usa numpy; acá no se importa numpy solo para esto)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def cam_lines(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Multi-línea de un parquet ancho del informe cambiario.

    Igual que ``wide_lines`` pero sin el tope de 6 series (los paneles de monedas
    llevan hasta 10) y con ``months`` para recortar la ventana visible, que es
    como el original acota los gráficos de puntas forward y tasas implícitas al
    último año. params: ``include``/``exclude``/``months``/``keep_time``,
    ``align_from`` (ver abajo)."""
    read = _cam_wide_rows(dataset, parquet_dir, keep_time=bool(params.get("keep_time")))
    if read is None:
        return None
    date_col, value_cols, rows = read
    exclude = set(params.get("exclude") or [])
    wanted = [c for c in (params.get("include") or value_cols) if c in value_cols and c not in exclude]
    months = int(params.get("months") or 0)

    raw: dict[str, list[tuple[str, float]]] = {}
    for col in wanted[:10]:
        pts = _cam_last_months(_cam_points(rows, date_col, col), months)
        if pts:
            raw[col] = pts
    if not raw:
        return None

    # ``align_from``: recorta TODAS las series al inicio de la MÁS TARDÍA entre
    # las nombradas. Sin esto, dos series con historia de distinto largo (el CLP
    # desde 2019, la posición de no residentes recién desde 2022) comparten eje
    # X pero la más corta deja un tramo vacío al principio mientras la más larga
    # sigue de fondo — el gráfico "compara" un período donde en realidad solo hay
    # una serie. El recorte es sobre el PLOT, no sobre el parquet: cada serie
    # conserva su historia completa en el dato, solo se dibuja desde que las dos
    # coinciden.
    align_from = [c for c in (params.get("align_from") or []) if c in raw]
    if align_from:
        start = max(raw[c][0][0] for c in align_from)
        raw = {c: [p for p in pts if p[0] >= start] for c, pts in raw.items()}
        raw = {c: pts for c, pts in raw.items() if pts}

    series = [PlotSeries(label=c, points=_downsample(pts)) for c, pts in raw.items()]
    if not series:
        return None
    note = ""
    if months:
        isos = [iso for s in series for iso, _ in s.points]
        note = f"Ventana: {_fmt_date(min(isos))} → {_fmt_date(max(isos))}"
    plot = PlotData(dataset.id, chart_family(dataset.chart_type), "timeseries", dataset.unit,
                    series, date_note=note, zero_base=_cam_zero_base(params))
    return None if plot.is_empty() else plot


# Color de resaltado por período de la media móvil — reutiliza acentos YA
# presentes en la paleta institucional del renderer (``svg_chart._PALETTE``,
# tabaco/rojo), no colores nuevos: MA50 marca el cruce dorado, MA200 el de la
# muerte; el resto de las medias (10/20/100, lo que traiga el Excel) queda muda.
_MA_HIGHLIGHT = {50: "#8a6d3b", 200: "#c8102e"}


def _cam_ma_period(col: str) -> int | None:
    """Extrae el período (días) del nombre de columna, mismo criterio que el
    dashboard original: el primer token compuesto solo por dígitos."""
    for token in col.split():
        if token.isdigit():
            return int(token)
    return None


def cam_moving_averages(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Cierre + sus medias móviles, con MA50/MA200 resaltadas (cruce dorado/de la
    muerte) y el resto en gris tenue de solo contexto — igual que el original.

    Sin esto (``cam_lines`` genérica) un gráfico con 5-6 medias móviles del mismo
    grosor y color cíclico no deja ver cuál es la que técnicamente importa.
    ``ma_cols`` se detectan por nombre (contienen "media m[o/ó]vil", tolerante a
    mojibake) en vez de venir fijas en el spec: el Excel real las trae con el
    nombre que tenga esa columna en el servidor, y el período se lee del propio
    nombre (``_cam_ma_period``), no de una lista hardcodeada.
    params: ``base`` (columna del cierre, default "CLP Cierre"), ``right``
    (columna del eje derecho, p.ej. el monto transado), ``months``."""
    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    base_col = params.get("base") or "CLP Cierre"
    right_col = params.get("right")
    ma_cols = [
        c for c in value_cols if c not in (base_col, right_col)
        and re.search(r"media\s*m[oó]vil", c, re.IGNORECASE)
    ]
    if base_col not in value_cols or not ma_cols:
        return None
    ma_cols.sort(key=lambda c: _cam_ma_period(c) if _cam_ma_period(c) is not None else 999)
    months = int(params.get("months") or 0)

    series = [PlotSeries(
        label=base_col, points=_downsample(_cam_last_months(_cam_points(rows, date_col, base_col), months)),
    )]
    emphasis: dict[str, str] = {}
    muted: list[str] = []
    for col in ma_cols:
        pts = _cam_last_months(_cam_points(rows, date_col, col), months)
        if not pts:
            continue
        series.append(PlotSeries(label=col, points=_downsample(pts)))
        period = _cam_ma_period(col)
        if period in _MA_HIGHLIGHT:
            emphasis[col] = _MA_HIGHLIGHT[period]
        else:
            muted.append(col)
    if right_col and right_col in value_cols:
        pts = _cam_last_months(_cam_points(rows, date_col, right_col), months)
        if pts:
            series.append(PlotSeries(label=right_col, points=_downsample(pts)))

    plot = PlotData(dataset.id, "line", "timeseries", dataset.unit, series,
                    emphasis=emphasis, muted=tuple(muted), zero_base=_cam_zero_base(params))
    return None if plot.is_empty() else plot


def cam_candlestick(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Vela OHLC del USD/CLP sobre la ventana visible (``months``, 3 en el original).

    Devuelve las cuatro series con los labels que ``_render_candlestick`` espera;
    el monto transado del parquet se deja fuera (vive en su propio gráfico)."""
    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    months = int(params.get("months") or 3)
    wanted = ["Apertura", "Máximo", "Mínimo", "Cierre"]
    if not all(c in value_cols for c in wanted):
        return None

    series: list[PlotSeries] = []
    for col in wanted:
        pts = _cam_last_months(_cam_points(rows, date_col, col), months)
        if pts:
            series.append(PlotSeries(label=col, points=pts))
    if len(series) < 4:
        return None
    isos = [iso for s in series for iso, _ in s.points]
    plot = PlotData(
        dataset.id, "line", "timeseries", dataset.unit, series,
        date_note=f"Últimos {months} meses: {_fmt_date(min(isos))} → {_fmt_date(max(isos))}",
        zero_base=_cam_zero_base(params),
    )
    return None if plot.is_empty() else plot


def cam_candle_table(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Tabla que acompaña a la vela: máximo y mínimo de la última sesión, más el
    soporte y la resistencia (P25 y P75 del cierre sobre la MISMA ventana de tres
    meses que dibuja el gráfico, para que tabla y vela no se contradigan)."""
    from .svg_chart import render_kv_table_html

    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, _value_cols, rows = read
    months = int(params.get("months") or 3)
    cierres = _cam_last_months(_cam_points(rows, date_col, "Cierre"), months)
    if not cierres:
        return None
    ordered = sorted(v for _d, v in cierres)
    soporte = _cam_percentile(ordered, 25)
    resistencia = _cam_percentile(ordered, 75)

    maximos = dict(_cam_points(rows, date_col, "Máximo"))
    minimos = dict(_cam_points(rows, date_col, "Mínimo"))
    last = next((d for d, _v in reversed(cierres) if d in maximos and d in minimos), None)
    if last is None:
        return None
    html = render_kv_table_html(
        ["Indicador", "Valor"],
        [
            ["Máximo del día", _fmt_num(maximos[last])],
            ["Mínimo del día", _fmt_num(minimos[last])],
            ["Soporte", _fmt_num(soporte)],
            ["Resistencia", _fmt_num(resistencia)],
        ],
        caption=f"{dataset.unit} · sesión {_fmt_date(last)} · S/R = P25 y P75 de {months} meses",
    )
    return HtmlTable(html=html, dataset_id=dataset.id)


def cam_bollinger(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Bandas de Bollinger: media móvil de ``period`` días ± ``mult`` desviaciones
    estándar, sobre la primera columna de valor del parquet."""
    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    col = params.get("value") or value_cols[0]
    period = int(params.get("period") or 20)
    mult = float(params.get("mult") or 2.0)
    months = int(params.get("months") or 0)
    pts = _cam_points(rows, date_col, col)
    if len(pts) < period:
        return None

    ma: list[tuple[str, float]] = []
    upper: list[tuple[str, float]] = []
    lower: list[tuple[str, float]] = []
    for i in range(period - 1, len(pts)):
        window = [v for _d, v in pts[i - period + 1:i + 1]]
        mean = sum(window) / period
        var = sum((v - mean) ** 2 for v in window) / (period - 1)
        sd = var ** 0.5
        iso = pts[i][0]
        ma.append((iso, mean))
        upper.append((iso, mean + mult * sd))
        lower.append((iso, mean - mult * sd))

    cut = _cam_last_months(ma, months)
    keep = {d for d, _v in cut} if months else None

    def _clip(seq: list[tuple[str, float]]) -> list[tuple[str, float]]:
        out = [p for p in seq if keep is None or p[0] in keep]
        return _downsample(out)

    series = [
        PlotSeries(label=f"Banda superior ({period}d)", points=_clip(upper)),
        PlotSeries(label=f"Media móvil {period}d", points=_clip(ma)),
        PlotSeries(label=f"Banda inferior ({period}d)", points=_clip(lower)),
        PlotSeries(label=col, points=_clip([p for p in pts if keep is None or p[0] in keep])),
    ]
    plot = PlotData(dataset.id, "line", "timeseries", dataset.unit, series,
                    zero_base=_cam_zero_base(params))
    return None if plot.is_empty() else plot


def _cam_sr_levels(
    dataset: ParquetDataset, parquet_dir: Path, params: dict,
) -> tuple[str, list[tuple[str, float]], list[tuple[int, float]]] | None:
    """``(label, serie recortada, [(percentil, nivel)])`` — cálculo COMPARTIDO por
    el gráfico S/R y su tabla, para que ambos citen exactamente los mismos niveles."""
    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    col = params.get("value") or value_cols[0]
    months = int(params.get("months") or 12)
    pts = _cam_last_months(_cam_points(rows, date_col, col), months)
    if not pts:
        return None
    ordered = sorted(v for _d, v in pts)
    levels = [(p, _cam_percentile(ordered, p)) for p in _SR_PERCENTILES]
    return col, pts, levels


def cam_sr_percentiles(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Serie + sus percentiles P10…P100 como líneas horizontales (soportes y
    resistencias del régimen de los últimos ``months`` meses, 12 por defecto)."""
    computed = _cam_sr_levels(dataset, parquet_dir, params)
    if computed is None:
        return None
    col, pts, levels = computed
    series = [PlotSeries(label=col, points=_downsample(pts))]
    series += [
        PlotSeries(label=f"P{p} · {_fmt_num(v)}", points=_cam_flat(pts, v))
        for p, v in levels
    ]
    plot = PlotData(
        dataset.id, "line", "timeseries", dataset.unit, series,
        date_note=f"Percentiles de {_fmt_date(pts[0][0])} → {_fmt_date(pts[-1][0])}",
        zero_base=_cam_zero_base(params),
    )
    return None if plot.is_empty() else plot


def cam_sr_table(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Tabla de los mismos percentiles del gráfico S/R (el original la pone justo
    debajo, para no depender solo de la leyenda para leer el nivel exacto)."""
    from .svg_chart import render_kv_table_html

    computed = _cam_sr_levels(dataset, parquet_dir, params)
    if computed is None:
        return None
    _col, pts, levels = computed
    html = render_kv_table_html(
        ["Percentil", "Nivel"],
        [[f"P{p}", _fmt_num(v)] for p, v in levels],
        caption=f"{dataset.unit} · {_fmt_date(pts[0][0])} → {_fmt_date(pts[-1][0])}",
    )
    return HtmlTable(html=html, dataset_id=dataset.id)


def cam_rsi_bands(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """RSI con sus umbrales de sobrecompra y sobreventa como líneas horizontales.
    params: ``levels`` (default 70 y 30), ``months``."""
    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    col = params.get("value") or value_cols[0]
    months = int(params.get("months") or 0)
    pts = _cam_last_months(_cam_points(rows, date_col, col), months)
    if not pts:
        return None
    levels = params.get("levels") or [70, 30]
    names = params.get("level_labels") or ["Sobrecompra", "Sobreventa"]
    series = [PlotSeries(label=col, points=_downsample(pts))]
    for i, level in enumerate(levels):
        label = names[i] if i < len(names) else f"Nivel {level}"
        series.append(PlotSeries(label=f"{label} · {level:g}", points=_cam_flat(pts, float(level))))
    plot = PlotData(dataset.id, "line", "timeseries", dataset.unit, series,
                    zero_base=_cam_zero_base(params))
    return None if plot.is_empty() else plot


def cam_base100(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Panel de monedas rebasado a 100 al inicio de la ventana visible.

    El original lo resuelve con botones (1M/3M/6M/1A/YTD/Todo) que recalculan la
    base; acá la ventana la fija el spec (``months``, 2 como en la vista por
    defecto) y el rebase se hace sobre el primer dato de CADA serie dentro de
    ella, así una moneda que empieza más tarde no arranca desalineada."""
    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    months = int(params.get("months") or 2)
    wanted = [c for c in (params.get("include") or value_cols) if c in value_cols]

    series: list[PlotSeries] = []
    for col in wanted[:10]:
        pts = _cam_last_months(_cam_points(rows, date_col, col), months)
        base = next((v for _d, v in pts if v), None)
        if not base:
            continue
        series.append(PlotSeries(
            label=col, points=_downsample([(d, v / base * 100.0) for d, v in pts]),
        ))
    if not series:
        return None
    isos = [iso for s in series for iso, _ in s.points]
    plot = PlotData(
        dataset.id, "line", "timeseries", "Índice base 100", series,
        date_note=f"Base 100 = {_fmt_date(min(isos))} · hasta {_fmt_date(max(isos))}",
        zero_base=_cam_zero_base(params),
    )
    return None if plot.is_empty() else plot


def cam_signed_bars(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Barras por categoría separadas en DOS series por el signo del valor.

    Réplica del ``color_discrete_map`` verde/rojo del original sin tocar el
    renderer: las variaciones positivas van en una serie y las negativas en otra,
    así cada barra queda del color de su signo y la leyenda lo explica.
    params: ``category``, ``value``, ``pos_label``/``neg_label``, ``ascending``."""
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        cat = params.get("category") or (roles.category_cols[0] if roles.category_cols else None)
        val = params.get("value") or (roles.value_cols[0] if roles.value_cols else None)
        if not cat or not val:
            return None
        rows = _read_rows(con, path, date_col=None, columns=[cat, val])
    finally:
        con.close()

    pairs: list[tuple[str, float]] = []
    for r in rows:
        name = r.get(cat)
        if name is None:
            continue
        try:
            pairs.append((str(name), float(r.get(val))))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return None
    pairs.sort(key=lambda kv: kv[1], reverse=not params.get("ascending"))

    pos_label = params.get("pos_label") or "Variación positiva"
    neg_label = params.get("neg_label") or "Variación negativa"
    cats = [c for c, _v in pairs]
    agg = {
        pos_label: {c: v for c, v in pairs if v >= 0},
        neg_label: {c: v for c, v in pairs if v < 0},
    }
    return _grouped(dataset.id, dataset.unit, agg, cats, [pos_label, neg_label])


def cam_histogram(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Histograma de un parquet SIN fecha cuyo eje X es NUMÉRICO (tramos de precio).

    ``detect_roles`` clasifica el tramo como columna de valor (es un número), así
    que ni ``snapshot_grouped`` ni ``latest_snapshot`` lo arman bien: acá el eje X
    se toma explícito y se formatea como etiqueta de categoría."""
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        x = params.get("x") or (roles.value_cols[0] if roles.value_cols else None)
        y = params.get("value") or next((c for c in roles.value_cols if c != x), None)
        if not x or not y:
            return None
        rows = _read_rows(con, path, date_col=None, columns=[x, y])
    finally:
        con.close()

    pairs: list[tuple[float, float]] = []
    for r in rows:
        try:
            pairs.append((float(r.get(x)), float(r.get(y))))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return None
    pairs.sort()
    label = params.get("series_label") or y
    cats = [_fmt_num(k) for k, _v in pairs]
    return _grouped(dataset.id, dataset.unit, {label: {_fmt_num(k): v for k, v in pairs}}, cats, [label])


def cam_contract_curve(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Curva de futuros: precio por fecha de vencimiento, con el nombre del
    contrato en el tooltip de cada punto."""
    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    col = params.get("value") or value_cols[0]
    pts = _cam_points(rows, date_col, col)
    if not pts:
        return None
    label = params.get("series_label") or dataset.name
    plot = PlotData(
        dataset.id, "line", "timeseries", dataset.unit,
        [PlotSeries(label=label, points=pts)],
        date_note=f"Vencimientos {_fmt_date(pts[0][0])} → {_fmt_date(pts[-1][0])}",
        zero_base=_cam_zero_base(params),
    )
    return None if plot.is_empty() else plot


def cam_fixing_stacked(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Fixing del ÚLTIMO día disponible: un grupo por banco informante, apilado
    por sector de la contraparte, con el total del agente superpuesto.

    Se ancla al último día CON DATO en vez de a "hoy" (como el original, que
    filtraba por ``Timestamp.today()`` y quedaba vacío en feriados)."""
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or len(roles.category_cols) < 2 or not roles.value_cols:
            return None
        agent = params.get("agent") or roles.category_cols[0]
        sector = params.get("sector") or next((c for c in roles.category_cols if c != agent), None)
        val = params.get("value") or roles.value_cols[0]
        if not sector:
            return None
        rows = _read_series_rows(
            con, path, date_col=roles.date_col,
            columns=[roles.date_col, agent, sector, val],
        )
    finally:
        con.close()

    dates = [str(r[roles.date_col]) for r in rows if r.get(roles.date_col)]
    if not dates:
        return None
    asof = params.get("weekly_asof") or max(dates)
    asof = max((d for d in dates if d <= asof), default=max(dates))

    agg: dict[str, dict[str, float]] = {}
    agents: list[str] = []
    sectors: list[str] = []
    for r in rows:
        if str(r.get(roles.date_col)) != asof:
            continue
        a, s = r.get(agent), r.get(sector)
        if a is None or s is None:
            continue
        try:
            v = float(r.get(val))
        except (TypeError, ValueError):
            continue
        agg.setdefault(str(s), {})[str(a)] = agg.setdefault(str(s), {}).get(str(a), 0.0) + v
        if str(a) not in agents:
            agents.append(str(a))
        if str(s) not in sectors:
            sectors.append(str(s))
    if not agg:
        return None

    # "Total" primero, como el eje X del original; el resto alfabético.
    total_first = params.get("total_first", "Total")
    agents.sort(key=lambda a: (a != total_first, a))
    sectors.sort()
    agg["Total agente"] = {a: sum(agg[s].get(a, 0.0) for s in sectors) for a in agents}
    return _grouped(
        dataset.id, dataset.unit, agg, agents, [*sectors, "Total agente"],
        overlay=("Total agente",), date_note=f"Fixing del {_fmt_date(asof)}",
    )


def cam_spread_expanding(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Dos tasas con su spread (en puntos base) y el promedio histórico acumulado
    del spread, para el gráfico de expectativas de TPM Chile vs Estados Unidos.

    params: ``minuend``/``subtrahend`` (columnas), ``scale`` (100 = a puntos base),
    ``spread_label``/``avg_label`` — que el spec pasa además como ``right_axis``."""
    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    a = params.get("minuend") or value_cols[0]
    b = params.get("subtrahend") or (value_cols[1] if len(value_cols) > 1 else None)
    if not b:
        return None
    scale = float(params.get("scale") or 100.0)
    pa, pb = dict(_cam_points(rows, date_col, a)), dict(_cam_points(rows, date_col, b))
    isos = sorted(set(pa) & set(pb))
    if not isos:
        return None

    spread_label = params.get("spread_label") or "Spread (pb)"
    avg_label = params.get("avg_label") or "Promedio histórico"
    spread: list[tuple[str, float]] = []
    avg: list[tuple[str, float]] = []
    running = 0.0
    for i, iso in enumerate(isos, start=1):
        value = (pa[iso] - pb[iso]) * scale
        running += value
        spread.append((iso, value))
        avg.append((iso, running / i))

    series = [
        PlotSeries(label=a, points=_downsample([(d, pa[d]) for d in isos])),
        PlotSeries(label=b, points=_downsample([(d, pb[d]) for d in isos])),
        PlotSeries(label=spread_label, points=_downsample(spread)),
        PlotSeries(label=avg_label, points=_downsample(avg)),
    ]
    plot = PlotData(dataset.id, "line", "timeseries", dataset.unit, series,
                    zero_base=_cam_zero_base(params))
    return None if plot.is_empty() else plot


def cam_market_table(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> HtmlTable | None:
    """Snapshot de mercado de la portada: último nivel, variación del día y de la
    semana de cada driver, y el signo de su impacto sobre el peso.

    ``mode`` por indicador: ``pct`` (variación porcentual), ``bp`` (diferencia en
    puntos base) o ``pts`` (diferencia en puntos). ``clp`` es el signo con que ese
    driver mueve al peso (+1 lo aprecia cuando sube, -1 lo deprecia, 0 neutral);
    el spec lo declara y acá solo se aplica."""
    from .svg_chart import _CELL_NEG, _CELL_POS, render_kv_table_html

    read = _cam_wide_rows(dataset, parquet_dir)
    if read is None:
        return None
    date_col, value_cols, rows = read
    indicators = params.get("indicators") or []
    if not indicators:
        return None

    def _series_for(spec: dict) -> list[tuple[str, float]]:
        """Serie del indicador: una columna, o la diferencia entre dos (SPC–OIS)."""
        if spec.get("minus"):
            pa = dict(_cam_points(rows, date_col, spec["column"]))
            pb = dict(_cam_points(rows, date_col, spec["minus"]))
            factor = float(spec.get("factor") or 1.0)
            return [(d, (pa[d] - pb[d]) * factor) for d in sorted(set(pa) & set(pb))]
        return _cam_points(rows, date_col, spec["column"])

    table_rows: list[list[str]] = []
    styles: list[str] = []
    asof = ""
    for spec in indicators:
        if spec.get("column") not in value_cols:
            continue
        pts = _series_for(spec)
        if not pts:
            continue
        asof = max(asof, pts[-1][0])
        last = pts[-1][1]
        dec = int(spec.get("decimals", 2))
        mode = spec.get("mode", "pct")

        def _delta(lag: int, _pts=pts, _last=last, _mode=mode) -> float | None:
            if len(_pts) <= lag:
                return None
            prev = _pts[-1 - lag][1]
            if _mode == "pct":
                return None if not prev else (_last / prev - 1) * 100
            return _last - prev

        day, week = _delta(1), _delta(5)
        suffix = {"pct": "%", "bp": " pb", "pts": " pts"}.get(mode, "")

        def _fmt_delta(v: float | None, _dec=dec, _sfx=suffix) -> str:
            return "—" if v is None else f"{v:+.{_dec}f}{_sfx}"

        direction = int(spec.get("clp", 0))
        if day is None or direction == 0:
            estado = "Neutral" if day is not None else "—"
            styles.append("")
        elif day * direction > 0:
            estado = "↑ Aprecia"
            styles.append(_CELL_POS)
        else:
            estado = "↓ Deprecia"
            styles.append(_CELL_NEG)
        table_rows.append([
            f"{spec.get('label') or spec['column']} · {spec.get('unit', '')}".strip(" ·"),
            f"{last:,.{dec}f}".replace(",", "."),
            _fmt_delta(day), _fmt_delta(week), estado,
        ])

    if not table_rows:
        return None
    html = render_kv_table_html(
        ["Indicador", "Último", "Var. día", "Var. semana", "Impacto CLP"],
        table_rows, row_styles=styles, max_width=620,
        caption=f"Cierre del {_fmt_date(asof)} · el impacto compara la variación del día "
                "con el signo con que ese driver mueve al peso",
    )
    return HtmlTable(html=html, dataset_id=dataset.id)


def cam_gamma_heatmap(dataset: ParquetDataset, parquet_dir: Path, params: dict) -> PlotData | None:
    """Mapa de calor de la gamma proxy: filas = strike, columnas = vencimiento.

    Una ``PlotSeries`` POR STRIKE (fila), con ``points = [(vencimiento, gamma)]``
    — mismo ``PlotData`` que cualquier otro gráfico, con ``kind="heatmap"`` para
    que ``svg_chart._render_heatmap`` la lea como matriz en vez de curva. Al ser
    SVG con el mismo ``viewBox``/``width="100%"`` que el resto de los gráficos,
    se auto-ajusta al ancho de su tarjeta y puede compartir fila con "Gamma
    Proxy" — una tabla HTML de celdas fijas no podía sin scroll horizontal."""
    path = dataset.parquet_path(parquet_dir)
    if not path.exists():
        return None
    con = duckdb.connect()
    try:
        roles = detect_roles(path, con)
        if roles.date_col is None or len(roles.value_cols) < 2:
            return None
        row_col = params.get("row") or roles.value_cols[0]
        val_col = params.get("value") or next((c for c in roles.value_cols if c != row_col), None)
        if not val_col:
            return None
        rows = _read_series_rows(
            con, path, date_col=roles.date_col,
            columns=[roles.date_col, row_col, val_col],
        )
    finally:
        con.close()

    by_row: dict[float, list[tuple[str, float]]] = {}
    for r in rows:
        d, k, v = r.get(roles.date_col), r.get(row_col), r.get(val_col)
        if d is None or k is None or v is None:
            continue
        try:
            row_num, value = float(k), float(v)
        except (TypeError, ValueError):
            continue
        by_row.setdefault(row_num, []).append((_fmt_date(str(d)), value))
    if not by_row:
        return None

    # Strike descendente (el mapa se lee como un eje Y: el más alto arriba).
    series = [PlotSeries(label=_fmt_num(n), points=by_row[n]) for n in sorted(by_row, reverse=True)]
    plot = PlotData(dataset.id, "heatmap", "heatmap", dataset.unit, series)
    return None if plot.is_empty() else plot


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
    "monthly_bars_by_cat": monthly_bars_by_cat,
    "ytd_grouped_by_cat": ytd_grouped_by_cat,
    "wide_lines": wide_lines,
    "window_grouped": window_grouped,
    "wide_window_bars": wide_window_bars,
    "wide_monthly_bars": wide_monthly_bars,
    "window_stacked_by_cat": window_stacked_by_cat,
    "window_stacked_diff_by_cat": window_stacked_diff_by_cat,
    "window_accum_by_cat": window_accum_by_cat,
    "window_accum_stacked_by_cat": window_accum_stacked_by_cat,
    "window_grouped_long": window_grouped_long,
    "window_pivot_grouped": window_pivot_grouped,
    "snapshot_grouped": snapshot_grouped,
    "snapshot_stacked": snapshot_stacked,
    "latest_snapshot": latest_snapshot,
    # Flujos cambiarios (familia fx).
    "wide_row_stacked": wide_row_stacked,
    "window_stacked_two_cat": window_stacked_two_cat,
    "daily_wide_stacked": daily_wide_stacked,
    # Tablas con color condicional (heatmap DCV): implementadas.
    "dcv_cut_dates": dcv_cut_dates,
    "dcv_heatmap": dcv_heatmap,
    # Informe DCV (familia dcv): tablas por agente / tramo + su gráfico apilado.
    "dcv_portfolio_table": dcv_portfolio_table,
    "dcv_duration_scatter": dcv_duration_scatter,
    "dcv_upcoming_maturities_table": dcv_upcoming_maturities_table,
    "dcv_bucket_table": dcv_bucket_table,
    "dcv_snapshot_stacked": dcv_snapshot_stacked,
    "fx_sector_flow_table": fx_sector_flow_table,
    "fx_agent_delta_table": fx_agent_delta_table,
    # Rango histórico + dispersión x/y (NR: comparables GBI).
    "gbi_rendimiento_range": gbi_rendimiento_range,
    "gbi_rendimiento_range_sin_base": gbi_rendimiento_range_sin_base,
    "fx_tasas_scatter": fx_tasas_scatter,
    "allocation_wide_by_fund":allocation_wide_by_fund,
    "wide_daily_diff_ytd": wide_daily_diff_ytd,
    "wide_monthly_diff_bars": wide_monthly_diff_bars,
    #dcv
    "dcv_maturities_three_months_by_type": dcv_maturities_three_months_by_type,
    # Informe Cambiario AM (familia cambiarioam): el parquet trae el dato crudo y
    # la transform calcula lo derivado (Bollinger, percentiles S/R, base 100, spreads).
    "cam_lines": cam_lines,
    "cam_moving_averages": cam_moving_averages,
    "cam_candlestick": cam_candlestick,
    "cam_candle_table": cam_candle_table,
    "cam_bollinger": cam_bollinger,
    "cam_sr_percentiles": cam_sr_percentiles,
    "cam_sr_table": cam_sr_table,
    "cam_rsi_bands": cam_rsi_bands,
    "cam_base100": cam_base100,
    "cam_signed_bars": cam_signed_bars,
    "cam_histogram": cam_histogram,
    "cam_contract_curve": cam_contract_curve,
    "cam_fixing_stacked": cam_fixing_stacked,
    "cam_spread_expanding": cam_spread_expanding,
    "cam_market_table": cam_market_table,
    "cam_gamma_heatmap": cam_gamma_heatmap,
}


def get_transform(name: str | None) -> Transform | None:
    """Resuelve el nombre a su función; ``None`` si no está implementada todavía
    (o no existe). El builder distingue ambos casos por presencia en el registro."""
    return _REGISTRY.get(name or "")


def is_known(name: str | None) -> bool:
    """True si el nombre está REGISTRADO (implementado o declarado pendiente)."""
    return (name or "") in _REGISTRY
