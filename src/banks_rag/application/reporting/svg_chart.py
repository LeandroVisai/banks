"""Renderer de gráficos a SVG estático — Python puro, SIN dependencias.

El gráfico se incrusta inline en el HTML del informe: es texto determinista, se
imprime nativo (sin JavaScript que deba ejecutarse), pesa kilobytes, se abre en
cualquier máquina y se inspecciona como cualquier SVG. Es la decisión análoga a
la que ya tomó el proyecto con el informe (Python determinista le ganó al
runtime inteligente): nada de Vega-Lite vendoreado ni matplotlib.

Dibuja dos formas a partir de ``PlotData`` (``parquet_facts.compute_series``):
    timeseries → multi-línea (eje X temporal, una línea por serie/categoría)
    snapshot   → barras horizontales de composición (eje X categórico)

Las familias de barra del catálogo (grouped_bar/stacked_bar) se grafican como
multi-línea cuando hay muchas observaciones (igual que el invariante de
``chart_types``: barras con mucha data degradan a línea). ``family == "table"``
devuelve ``None`` → el caller cae a una mini-tabla HTML.
"""

from __future__ import annotations

import html
from datetime import date

from .parquet_facts import PlotData, PlotSeries

# Paleta: azules del informe + acentos distinguibles, segura en impresión B/N
# (varían en luminancia, no solo en tono).
_PALETTE = ["#0b3766", "#c8102e", "#0a8a5f", "#e08a00", "#6a3d9a", "#1f9bcf"]

_W = 760
_H = 280
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


def _y_ticks(lo: float, hi: float) -> list[float]:
    if hi <= lo:
        return [lo]
    return [lo, lo + (hi - lo) / 2, hi]


# ── Render principal ─────────────────────────────────────────────────────────

def render_plot_svg(plot: PlotData, *, width: int = _W, height: int = _H) -> str | None:
    """``PlotData`` → SVG inline (str), o ``None`` si la familia no es graficable
    como serie (``table``) y el caller debe usar una mini-tabla."""
    if plot.is_empty() or plot.family == "table":
        return None
    if plot.kind == "snapshot":
        return _render_snapshot(plot, width, height)
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
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax, ymin = ymax + 1, ymin - 1
    pad = (ymax - ymin) * 0.06
    ymin, ymax = ymin - pad, ymax + pad

    def px(o: int) -> float:
        return left + (plot_w / 2 if xmax == xmin else plot_w * (o - xmin) / (xmax - xmin))

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    out = _svg_open(width, height, f"{plot.dataset_id} — serie temporal")

    # Rejilla + eje Y.
    for tick in _y_ticks(ymin + pad, ymax - pad):
        y = py(tick)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
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

    # Series.
    legend: list[str] = []
    for i, (s, pts) in enumerate(parsed):
        color = _PALETTE[i % len(_PALETTE)]
        coords = " ".join(f"{px(o):.1f},{py(v):.1f}" for o, v in pts)
        if len(pts) == 1:
            o, v = pts[0]
            out.append(f'<circle cx="{px(o):.1f}" cy="{py(v):.1f}" r="3" fill="{color}"/>')
        else:
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.8"/>')
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
        out.append(f'<text x="{left - 8}" y="{y + 15}" font-size="12" fill="#333" text-anchor="end">{_esc(cat)}</text>')
        out.append(f'<rect x="{left}" y="{y + 3}" width="{bar_w:.1f}" height="{row_h - 10}" fill="{color}"/>')
        out.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 15}" font-size="11" fill="#555">{_esc(_fmt_num(v))} ({_esc(_fmt_num(share))}%)</text>')
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
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
