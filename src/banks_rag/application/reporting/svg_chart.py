"""Renderer de gráficos a SVG estático — Python puro, SIN dependencias.

El gráfico se incrusta inline en el HTML del informe: es texto determinista, se
imprime nativo (sin JavaScript que deba ejecutarse), pesa kilobytes, se abre en
cualquier máquina y se inspecciona como cualquier SVG. Es la decisión análoga a
la que ya tomó el proyecto con el informe (Python determinista le ganó al
runtime inteligente): nada de Vega-Lite vendoreado ni matplotlib.

Render según el ``kind`` de ``PlotData`` y el tipo objetivo (``chart``):
    timeseries → línea (``line``) o área apilada (``area``/``stacked_area``)
    grouped    → barras verticales agrupadas (``grouped_bar``) o apiladas (``stacked_bar``)
    snapshot   → composición horizontal (``composition``) o torta (``pie``)

``render_plot_svg(plot, chart=...)`` dibuja el tipo pedido si la forma del dato lo
soporta; si no, cae a la marca natural del ``kind`` (línea/composición). El caller
usa ``renders_natively`` para saber si fue el tipo pedido o un fallback (vista
preliminar). ``family == "table"`` devuelve ``None`` → mini-tabla HTML.
"""

from __future__ import annotations

import html
import math
from datetime import date

from .parquet_facts import PlotData, PlotSeries

# Paleta: azules del informe + acentos distinguibles, segura en impresión B/N
# (varían en luminancia, no solo en tono).
_PALETTE = ["#0b3766", "#c8102e", "#0a8a5f", "#e08a00", "#6a3d9a", "#1f9bcf"]

_W = 760
_H = 320
_MARGIN = {"top": 30, "right": 18, "bottom": 70, "left": 76}


# ── Formato es-CL ────────────────────────────────────────────────────────────

def _fmt_num(value: float) -> str:
    """Número en formato chileno: miles con punto, decimales con coma.

    Decimales adaptativos: enteros grandes sin decimales, magnitudes pequeñas
    con 2 — un eje no necesita 6 cifras."""
    av = abs(value)
    decimals = 0 if av >= 1000 else (1 if av >= 10 else 2)
    s = f"{value:,.{decimals}f}"  # en-US: 1,234.5
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _fmt_date(iso: str) -> str:
    """``YYYY-MM-DD`` → ``DD-MM-YY`` (formato chileno compacto para ticks)."""
    try:
        d = date.fromisoformat(iso[:10])
    except ValueError:
        return html.escape(iso[:10])
    return f"{d.day:02d}-{d.month:02d}-{str(d.year)[2:]}"


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


# ── Geometría ────────────────────────────────────────────────────────────────

def _date_ord(iso: str) -> int | None:
    try:
        return date.fromisoformat(iso[:10]).toordinal()
    except ValueError:
        return None


# ── Ejes "lindos": números redondos y base 0 ────────────────────────────────
#
# El jefe pidió ejes con valores HABITUALES (0, 500, 1.000, 1.500…), no la marca
# del mínimo/máximo crudo, y con base 0. ``_nice_ticks`` reemplaza al viejo
# ``[lo, medio, hi]``: redondea el paso a 1/2/2.5/5x10^k y, salvo que se pida lo
# contrario, mete el 0 en el dominio para que el eje arranque (o cruce) en 0.

def _nice_num(x: float, *, round_: bool) -> float:
    """Número 'lindo' cercano a ``x`` (algoritmo clásico 1/2/5x10^k)."""
    if x == 0:
        return 0.0
    exp = math.floor(math.log10(abs(x)))
    frac = abs(x) / (10.0 ** exp)
    if round_:
        nf = 1.0 if frac < 1.5 else (2.0 if frac < 3.0 else (5.0 if frac < 7.0 else 10.0))
    else:
        nf = 1.0 if frac <= 1.0 else (2.0 if frac <= 2.0 else (5.0 if frac <= 5.0 else 10.0))
    return nf * (10.0 ** exp)


def _nice_ticks(
    lo: float, hi: float, *, n: int = 5, include_zero: bool = True,
) -> tuple[list[float], float, float]:
    """``(ticks, axis_lo, axis_hi)`` con pasos redondos. ``include_zero`` mete el
    0 en el dominio (base de eje 0). Devuelve un dominio ya redondeado a los
    ticks, para que el caller lo use como rango del eje."""
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    step = _nice_num((hi - lo) / max(1, n - 1), round_=True) or 1.0
    axis_lo = math.floor(lo / step) * step
    axis_hi = math.ceil(hi / step) * step
    ticks: list[float] = []
    v = axis_lo
    while v <= axis_hi + step * 0.5:
        ticks.append(round(v, 10))
        v += step
    return ticks, axis_lo, axis_hi


# ── Tooltips nativos (hover) ─────────────────────────────────────────────────
#
# SVG ``<title>`` da un tooltip del navegador al pasar el mouse, sin JavaScript
# (coherente con el resto del informe: texto determinista, se abre offline). En
# barras/torta/composición se cuelga del propio shape; en líneas/áreas se agregan
# puntos-objetivo transparentes (subsampleados) sobre la curva.

_MAX_TIP_PTS = 40  # puntos-objetivo de hover por serie (controla el peso del SVG)


def _tip(text: str) -> str:
    return f"<title>{_esc(text)}</title>"


def _tip_series(label: str, iso: str, v: float, unit: str) -> str:
    u = f" {unit}" if unit else ""
    return f"{label} · {_fmt_date(iso)}: {_fmt_num(v)}{u}"


def _tip_cat(label: str, cat: str, v: float, unit: str) -> str:
    u = f" {unit}" if unit else ""
    sep = " · " if label else ""
    return f"{label}{sep}{cat}: {_fmt_num(v)}{u}"


def _subsample(points: list, max_n: int = _MAX_TIP_PTS) -> list[int]:
    """Índices (incluyendo extremos) de hasta ``max_n`` puntos equiespaciados."""
    n = len(points)
    if n <= max_n:
        return list(range(n))
    step = (n - 1) / (max_n - 1)
    return sorted({round(i * step) for i in range(max_n)} | {0, n - 1})


# ── Render principal ─────────────────────────────────────────────────────────

# Tipo objetivo → ``kind`` de PlotData que lo dibuja de forma NATIVA. Si el dato
# llega con otro kind, el render cae a la marca natural (preliminar).
_CHART_NATIVE_KIND: dict[str, str] = {
    "line": "timeseries",
    "area": "timeseries",
    "stacked_area": "timeseries",
    "grouped_bar": "grouped",
    "stacked_bar": "grouped",
    "bar_time": "grouped",
    "composition": "snapshot",
    "pie": "snapshot",
}


def renders_natively(plot_kind: str, chart: str | None) -> bool:
    """True si ``chart`` se dibuja de forma nativa con un PlotData de ese kind
    (no es un fallback/vista preliminar)."""
    return _CHART_NATIVE_KIND.get((chart or "").strip()) == plot_kind


def render_plot_svg(
    plot: PlotData, *, chart: str | None = None, width: int = _W, height: int = _H,
) -> str | None:
    """``PlotData`` → SVG inline (str) del tipo ``chart`` (o el natural del kind si
    ``chart`` no aplica). ``None`` si ``family == 'table'`` (→ mini-tabla)."""
    if plot.is_empty() or plot.family == "table":
        return None
    target = (chart or "").strip()
    if plot.kind == "grouped":
        return _render_grouped_bars(plot, width, height, stacked=(target == "stacked_bar"))
    if plot.kind == "snapshot":
        if target == "pie":
            return _render_pie(plot, width, height)
        return _render_snapshot(plot, width, height)
    # timeseries
    if target in ("area", "stacked_area"):
        return _render_stacked_area(plot, width, height)
    return _render_timeseries(plot, width, height)


def _svg_open(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg class="report-chart" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="Arial, Helvetica, sans-serif">',
        f"<title>{_esc(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]


def _render_timeseries(plot: PlotData, width: int, height: int) -> str:
    left, right = _MARGIN["left"], _MARGIN["right"]
    top, bottom = _MARGIN["top"], _MARGIN["bottom"]
    plot_w = width - left - right
    plot_h = height - top - bottom

    # Dominios: X por fecha (ordinal), Y por valor, sobre TODAS las series.
    xs: list[int] = []
    ys: list[float] = []
    parsed: list[tuple[PlotSeries, list[tuple[int, float]]]] = []
    iso_by_ord: dict[int, str] = {}
    for s in plot.series:
        pts: list[tuple[int, float]] = []
        for iso, v in s.points:
            o = _date_ord(iso)
            if o is None:
                continue
            pts.append((o, v))
            iso_by_ord.setdefault(o, iso)
        if pts:
            parsed.append((s, pts))
            xs.extend(o for o, _ in pts)
            ys.extend(v for _, v in pts)
    if not xs:
        return _no_axis_message(plot, width, height)

    xmin, xmax = min(xs), max(xs)
    ticks, ymin, ymax = _nice_ticks(min(ys), max(ys))
    if ymax == ymin:
        ymax = ymin + 1.0

    def px(o: int) -> float:
        return left + (plot_w / 2 if xmax == xmin else plot_w * (o - xmin) / (xmax - xmin))

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    out = _svg_open(width, height, f"{plot.dataset_id} — serie temporal")

    # Rejilla + eje Y (ticks redondos; resalta la línea del 0 si el eje lo cruza).
    for tick in ticks:
        y = py(tick)
        emph = abs(tick) < 1e-9 and ymin < 0 < ymax
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{"#999" if emph else "#e6e6e6"}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')
    # Ejes.
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')

    # Ticks de fecha (hasta 5, equiespaciados sobre el dominio observado).
    tick_ords = sorted(set(xs))
    n_ticks = min(5, len(tick_ords))
    if n_ticks >= 1:
        step = (len(tick_ords) - 1) / max(1, n_ticks - 1)
        chosen = sorted({tick_ords[round(i * step)] for i in range(n_ticks)})
        for j, o in enumerate(chosen):
            x = px(o)
            label = _fmt_date(iso_by_ord.get(o, date.fromordinal(o).isoformat()))
            # Anclar extremos hacia adentro para que la etiqueta no se salga del viewBox.
            anchor = "start" if j == 0 else ("end" if j == len(chosen) - 1 else "middle")
            out.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#999" stroke-width="1"/>')
            out.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-size="11" fill="#555" text-anchor="{anchor}">{_esc(label)}</text>')

    # Series + puntos-objetivo de hover (tooltip nativo con fecha y valor).
    legend: list[str] = []
    for i, (s, pts) in enumerate(parsed):
        color = _PALETTE[i % len(_PALETTE)]
        coords = " ".join(f"{px(o):.1f},{py(v):.1f}" for o, v in pts)
        if len(pts) == 1:
            o, v = pts[0]
            out.append(f'<circle cx="{px(o):.1f}" cy="{py(v):.1f}" r="3" fill="{color}"/>')
        else:
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        for k in _subsample(pts):
            o, v = pts[k]
            iso = iso_by_ord.get(o) or date.fromordinal(o).isoformat()
            out.append(
                f'<circle cx="{px(o):.1f}" cy="{py(v):.1f}" r="6" fill="transparent" '
                f'pointer-events="all">{_tip(_tip_series(s.label, iso, v, plot.unit))}</circle>'
            )
        legend.append((s.label, color))

    out.append(_legend_row(legend, left, height - 30, plot_w))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _render_snapshot(plot: PlotData, width: int, height: int) -> str:
    """Barras horizontales de composición: una barra por categoría, ordenadas."""
    points = plot.series[0].points if plot.series else []
    points = [(c, v) for c, v in points]
    if not points:
        return _no_axis_message(plot, width, height)

    total = sum(abs(v) for _, v in points) or 1.0
    n = len(points)
    row_h = 26
    top = _MARGIN["top"]
    left = 150
    right = 150
    bar_w_max = width - left - right
    height = top + n * row_h + 16
    vmax = max(abs(v) for _, v in points) or 1.0

    out = _svg_open(width, height, f"{plot.dataset_id} — composición")
    for i, (cat, v) in enumerate(points):
        y = top + i * row_h
        color = _PALETTE[i % len(_PALETTE)]
        bar_w = bar_w_max * (abs(v) / vmax)
        share = 100.0 * abs(v) / total
        unit_sfx = f" {plot.unit}" if plot.unit else ""
        tip = f"{cat}: {_fmt_num(v)}{unit_sfx} ({_fmt_num(share)}%)"
        out.append(f'<text x="{left - 8}" y="{y + 15}" font-size="12" fill="#333" text-anchor="end">{_esc(cat)}</text>')
        out.append(f'<rect x="{left}" y="{y + 3}" width="{bar_w:.1f}" height="{row_h - 10}" fill="{color}">{_tip(tip)}</rect>')
        out.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 15}" font-size="11" fill="#555">{_esc(_fmt_num(v))} ({_esc(_fmt_num(share))}%)</text>')
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _render_grouped_bars(plot: PlotData, width: int, height: int, *, stacked: bool) -> str:
    """Barras verticales agrupadas o apiladas. Categorías (eje X) compartidas por
    todas las series; cada serie es un color. Maneja valores negativos (bajo 0)."""
    series = plot.series
    cats: list[str] = []
    seen: set[str] = set()
    for s in series:
        for c, _v in s.points:
            if c not in seen:
                seen.add(c)
                cats.append(c)
    if not cats:
        return _no_axis_message(plot, width, height)
    lut = [{c: v for c, v in s.points} for s in series]

    long_labels = any(len(c) > 9 for c in cats)
    left, right = 64, 18
    top = 30
    bottom = 90 if long_labels else 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    if stacked:
        pos = [sum(max(0.0, lut[i].get(c, 0.0)) for i in range(len(series))) for c in cats]
        neg = [sum(min(0.0, lut[i].get(c, 0.0)) for i in range(len(series))) for c in cats]
        dmax, dmin = max([0.0, *pos]), min([0.0, *neg])
    else:
        allv = [lut[i].get(c, 0.0) for i in range(len(series)) for c in cats]
        dmax, dmin = max([0.0, *allv]), min([0.0, *allv])
    if dmax == dmin:
        dmax += 1.0
    ticks, ymin, ymax = _nice_ticks(dmin, dmax)  # incluye 0 → base de eje 0

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    group_w = plot_w / len(cats)
    out = _svg_open(width, height, f"{plot.dataset_id} — barras")

    for tick in ticks:
        y = py(tick)
        emph = abs(tick) < 1e-9
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{"#999" if emph else "#e6e6e6"}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')

    for ci, c in enumerate(cats):
        gx = left + ci * group_w
        if stacked:
            bw = group_w * 0.55
            x = gx + (group_w - bw) / 2
            pos_acc = neg_acc = 0.0
            for si in range(len(series)):
                v = lut[si].get(c, 0.0)
                if v >= 0:
                    y_top, y_bot = py(pos_acc + v), py(pos_acc)
                    pos_acc += v
                else:
                    y_top, y_bot = py(neg_acc), py(neg_acc + v)
                    neg_acc += v
                h = abs(y_bot - y_top)
                if h > 0.2:
                    tip = _tip(_tip_cat(series[si].label, c, v, plot.unit))
                    out.append(f'<rect x="{x:.1f}" y="{min(y_top, y_bot):.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{_PALETTE[si % len(_PALETTE)]}">{tip}</rect>')
        else:
            inner = group_w * 0.8
            bw = inner / len(series)
            x0 = gx + (group_w - inner) / 2
            for si in range(len(series)):
                v = lut[si].get(c, 0.0)
                y_top, y_bot = py(max(0.0, v)), py(min(0.0, v))
                h = abs(y_bot - y_top)
                if h > 0.2:
                    tip = _tip(_tip_cat(series[si].label, c, v, plot.unit))
                    out.append(f'<rect x="{x0 + si * bw:.1f}" y="{min(y_top, y_bot):.1f}" width="{bw * 0.86:.1f}" height="{h:.1f}" fill="{_PALETTE[si % len(_PALETTE)]}">{tip}</rect>')
        cx = left + ci * group_w + group_w / 2
        if long_labels:
            cy = top + plot_h + 12
            out.append(
                f'<text transform="translate({cx:.1f},{cy:.1f}) rotate(-45)" '
                f'font-size="10" fill="#555" text-anchor="end">{_esc(c)}</text>'
            )
        else:
            out.append(
                f'<text x="{cx:.1f}" y="{top + plot_h + 16}" '
                f'font-size="10" fill="#555" text-anchor="middle">{_esc(c)}</text>'
            )

    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    legend_y = height - (20 if long_labels else 26)
    out.append(_legend_row([(s.label, _PALETTE[i % len(_PALETTE)]) for i, s in enumerate(series)], left, legend_y, plot_w))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _render_stacked_area(plot: PlotData, width: int, height: int) -> str:
    """Área apilada: series temporales sumadas verticalmente (positivas). Para
    allocation/composición en el tiempo. Negativos se recortan a 0 (el apilado de
    áreas asume aportes no negativos)."""
    series = plot.series
    dates = sorted({iso for s in series for iso, _ in s.points if _date_ord(iso) is not None})
    if not dates:
        return _no_axis_message(plot, width, height)
    ordv = [_date_ord(d) for d in dates]
    lut = [dict(s.points) for s in series]

    left, right = 76, 18
    top, bottom = 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    totals = [sum(max(0.0, lut[i].get(d, 0.0)) for i in range(len(series))) for d in dates]
    ticks, _ylo, ymax = _nice_ticks(0.0, max(totals) or 1.0)
    ymax = ymax or 1.0
    xmin, xmax = min(ordv), max(ordv)

    def px(o: int) -> float:
        return left + (plot_w / 2 if xmax == xmin else plot_w * (o - xmin) / (xmax - xmin))

    def py(v: float) -> float:
        return top + plot_h * (1 - v / ymax)

    out = _svg_open(width, height, f"{plot.dataset_id} — área apilada")
    for tick in ticks:
        y = py(tick)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')

    tip_idx = _subsample(dates)
    cum = [0.0] * len(dates)
    for si in range(len(series)):
        upper = [cum[k] + max(0.0, lut[si].get(dates[k], 0.0)) for k in range(len(dates))]
        up = " ".join(f"{px(ordv[k]):.1f},{py(upper[k]):.1f}" for k in range(len(dates)))
        dn = " ".join(f"{px(ordv[k]):.1f},{py(cum[k]):.1f}" for k in reversed(range(len(dates))))
        out.append(f'<polygon points="{up} {dn}" fill="{_PALETTE[si % len(_PALETTE)]}" fill-opacity="0.85" stroke="none"/>')
        # Punto-objetivo de hover en el centro de cada banda (valor de la serie en esa fecha).
        for k in tip_idx:
            band_v = max(0.0, lut[si].get(dates[k], 0.0))
            mid_y = py((cum[k] + upper[k]) / 2)
            out.append(
                f'<circle cx="{px(ordv[k]):.1f}" cy="{mid_y:.1f}" r="6" fill="transparent" '
                f'pointer-events="all">{_tip(_tip_series(series[si].label, dates[k], band_v, plot.unit))}</circle>'
            )
        cum = upper

    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    # ticks de fecha (hasta 5).
    n = min(5, len(dates))
    if n >= 1:
        step = (len(dates) - 1) / max(1, n - 1)
        for j in range(n):
            k = round(j * step)
            x = px(ordv[k])
            anchor = "start" if j == 0 else ("end" if j == n - 1 else "middle")
            out.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-size="11" fill="#555" text-anchor="{anchor}">{_esc(_fmt_date(dates[k]))}</text>')
    out.append(_legend_row([(s.label, _PALETTE[i % len(_PALETTE)]) for i, s in enumerate(series)], left, height - 26, plot_w))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _render_pie(plot: PlotData, width: int, height: int) -> str:
    """Torta de composición (snapshot). Una porción por categoría; leyenda con
    valor y porcentaje. Usa |valor| (las porciones no admiten negativos)."""
    points = [(c, abs(v)) for c, v in (plot.series[0].points if plot.series else []) if v]
    if not points:
        return _no_axis_message(plot, width, height)
    total = sum(v for _, v in points) or 1.0
    cx, cy, r = 170, 150, 120
    height = 300

    out = _svg_open(width, height, f"{plot.dataset_id} — torta")
    unit_sfx = f" {plot.unit}" if plot.unit else ""
    angle = -math.pi / 2  # arranca arriba
    for i, (cat, v) in enumerate(points):
        frac = v / total
        a2 = angle + frac * 2 * math.pi
        x1, y1 = cx + r * math.cos(angle), cy + r * math.sin(angle)
        x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
        large = 1 if frac > 0.5 else 0
        tip = _tip(f"{cat}: {_fmt_num(v)}{unit_sfx} ({_fmt_num(100.0 * frac)}%)")
        color = _PALETTE[i % len(_PALETTE)]
        if frac >= 0.999:  # una sola categoría → círculo completo
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}">{tip}</circle>')
        else:
            out.append(f'<path d="M {cx} {cy} L {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f} Z" fill="{color}">{tip}</path>')
        angle = a2
    # leyenda a la derecha
    lx, ly = cx + r + 40, 40
    for i, (cat, v) in enumerate(points):
        y = ly + i * 22
        pct = 100.0 * v / total
        out.append(f'<rect x="{lx}" y="{y - 9}" width="11" height="11" fill="{_PALETTE[i % len(_PALETTE)]}"/>')
        out.append(f'<text x="{lx + 17}" y="{y}" font-size="12" fill="#333">{_esc(cat)}: {_esc(_fmt_num(v))} ({_esc(_fmt_num(pct))}%)</text>')
    if plot.unit:
        out.append(f'<text x="20" y="20" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _legend_row(items: list[tuple[str, str]], x0: int, y: float, max_w: int) -> str:
    """Fila de swatches + etiquetas; envuelve a una segunda línea si no caben."""
    parts: list[str] = []
    x = x0
    line = 0
    for label, color in items:
        label_s = label if len(label) <= 22 else label[:21] + "…"
        w = 16 + len(label_s) * 6.2 + 14
        if x + w > x0 + max_w and x > x0:
            line += 1
            x = x0
        yy = y + line * 16
        parts.append(f'<rect x="{x:.1f}" y="{yy - 8:.1f}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{x + 14:.1f}" y="{yy:.1f}" font-size="11" fill="#444">{_esc(label_s)}</text>')
        x += w
    return "".join(parts)


def _no_axis_message(plot: PlotData, width: int, height: int) -> str:
    out = _svg_open(width, 60, f"{plot.dataset_id} — sin serie graficable")
    out.append(f'<text x="{width / 2:.0f}" y="34" font-size="12" fill="#999" text-anchor="middle">Sin serie graficable.</text>')
    out.append("</svg>")
    return "\n".join(out)


# ── Fallback: mini-tabla HTML para familias no graficables (table) ───────────

def render_mini_table_html(plot: PlotData, *, max_rows: int = 8) -> str:
    """Mini-tabla de los últimos puntos / categorías — fallback para familias
    que no se grafican como serie (``table``) o como respaldo verificable."""
    rows: list[str] = []
    if plot.kind == "snapshot" and plot.series:
        for cat, v in plot.series[0].points[:max_rows]:
            rows.append(f"<tr><td>{_esc(cat)}</td><td style='text-align:right'>{_esc(_fmt_num(v))}</td></tr>")
        header = "<tr><th>Categoría</th><th style='text-align:right'>Valor</th></tr>"
    else:
        # Serie(s) temporal(es): últimas filas por fecha de la primera serie.
        s = plot.series[0] if plot.series else PlotSeries("", [])
        for iso, v in s.points[-max_rows:]:
            rows.append(f"<tr><td>{_esc(_fmt_date(iso))}</td><td style='text-align:right'>{_esc(_fmt_num(v))}</td></tr>")
        header = f"<tr><th>Fecha</th><th style='text-align:right'>{_esc(s.label)}</th></tr>"
    return (
        '<table class="chart-fallback" style="border-collapse:collapse;font-size:12px;margin-top:6px">'
        f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"
    )


# ── Tablas HTML con color condicional (heatmap DCV) ──────────────────────────

_CELL_POS = "background:#d4edda;color:#155724"
_CELL_NEG = "background:#f8d7da;color:#721c24"


def _delta_style(v: float | None) -> str:
    if v is None:
        return ""
    return _CELL_POS if v > 0 else (_CELL_NEG if v < 0 else "")


def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "&#8212;"
    sign = "+" if v > 0 else ""
    return f"{sign}{_fmt_num(v)}"


def render_dcv_cut_table(
    tipos: list[str],
    dates: tuple[str, str, str],
    data: dict[str, tuple[float | None, float | None, float | None]],
    unit: str = "US$ Mill.",
) -> str:
    """Tabla de fechas de corte DCV (T, T-5, T-20): filas=instrumento,
    columnas=(T, T-5, T-20, Delta T-5, Delta T-20). Deltas coloreados verde/rojo."""
    t_iso, t5_iso, t20_iso = dates
    th = "text-align:right;padding:6px 8px;background:#4a5a72;color:#fff;white-space:nowrap;font-size:12px"
    thl = "text-align:left;padding:6px 8px;background:#4a5a72;color:#fff;font-size:12px"
    td = "text-align:right;padding:5px 8px;border-bottom:1px solid #eee;font-size:12px"
    tdl = "text-align:left;padding:5px 8px;border-bottom:1px solid #eee;font-size:12px"

    head = (
        f'<thead><tr>'
        f'<th style="{thl}">Instrumento</th>'
        f'<th style="{th}">T&nbsp;({_fmt_date(t_iso)})</th>'
        f'<th style="{th}">T-5&nbsp;({_fmt_date(t5_iso)})</th>'
        f'<th style="{th}">T-20&nbsp;({_fmt_date(t20_iso)})</th>'
        f'<th style="{th}">Δ T-5</th>'
        f'<th style="{th}">Δ T-20</th>'
        f'</tr></thead>'
    )
    body_rows = []
    for tipo in tipos:
        vt, vt5, vt20 = data.get(tipo, (None, None, None))
        d5 = (vt - vt5) if vt is not None and vt5 is not None else None
        d20 = (vt - vt20) if vt is not None and vt20 is not None else None
        d5_sty = f"{td};{_delta_style(d5)}" if d5 else td
        d20_sty = f"{td};{_delta_style(d20)}" if d20 else td
        body_rows.append(
            f'<tr>'
            f'<td style="{tdl}">{_esc(tipo)}</td>'
            f'<td style="{td}">{_fmt_num(vt) if vt is not None else "&#8212;"}</td>'
            f'<td style="{td}">{_fmt_num(vt5) if vt5 is not None else "&#8212;"}</td>'
            f'<td style="{td}">{_fmt_num(vt20) if vt20 is not None else "&#8212;"}</td>'
            f'<td style="{d5_sty}">{_fmt_delta(d5)}</td>'
            f'<td style="{d20_sty}">{_fmt_delta(d20)}</td>'
            f'</tr>'
        )
    unit_note = f'<div style="font-size:11px;color:#777;margin:2px 0 0">{_esc(unit)}</div>' if unit else ""
    return (
        '<div style="overflow-x:auto;max-width:760px;margin:6px auto">'
        '<table style="width:100%;border-collapse:collapse">'
        + head + "<tbody>" + "".join(body_rows) + "</tbody></table>"
        + unit_note + "</div>"
    )


def render_dcv_heatmap_tables(
    tipos: list[str],
    buckets: list[str],
    delta5: dict[str, dict[str, float | None]],
    delta20: dict[str, dict[str, float | None]],
    unit: str = "US$ Mill.",
) -> str:
    """Dos matrices heatmap (Delta T-5 / Delta T-20): filas=instrumento,
    cols=plazo, celdas coloreadas verde (positivo) / rojo (negativo)."""

    def _matrix(label: str, matrix: dict[str, dict[str, float | None]]) -> str:
        th = "text-align:right;padding:5px 7px;background:#4a5a72;color:#fff;font-size:11px;white-space:nowrap"
        thl = "text-align:left;padding:5px 7px;background:#4a5a72;color:#fff;font-size:11px"
        tdl = "text-align:left;padding:4px 7px;border-bottom:1px solid #eee;font-size:11px"
        tdr = "text-align:right;padding:4px 7px;border-bottom:1px solid #eee;font-size:11px"
        head = (
            f'<div style="font-weight:700;font-size:12px;color:#0b3766;margin:8px 0 4px">{_esc(label)}</div>'
            f'<table style="width:100%;border-collapse:collapse;margin-bottom:12px">'
            f'<thead><tr><th style="{thl}">Instrumento</th>'
            + "".join(f'<th style="{th}">{_esc(b)}</th>' for b in buckets)
            + "</tr></thead>"
        )
        body_rows = []
        for tipo in tipos:
            cells = ""
            for b in buckets:
                v = matrix.get(tipo, {}).get(b)
                sty = f"{tdr};{_delta_style(v)}" if v else tdr
                cells += f'<td style="{sty}">{_fmt_delta(v)}</td>'
            body_rows.append(f'<tr><td style="{tdl}">{_esc(tipo)}</td>{cells}</tr>')
        return head + "<tbody>" + "".join(body_rows) + "</tbody></table>"

    unit_note = f'<div style="font-size:11px;color:#777;margin:4px 0 0">{_esc(unit)}</div>' if unit else ""
    return (
        '<div style="overflow-x:auto;max-width:760px;margin:6px auto">'
        + _matrix("Δ T-5", delta5)
        + _matrix("Δ T-20", delta20)
        + unit_note + "</div>"
    )
