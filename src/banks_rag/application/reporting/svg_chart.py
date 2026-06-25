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
import json as _json
import math
from datetime import date

from .parquet_facts import PlotData, PlotSeries

# Paleta: azules del informe + acentos distinguibles, segura en impresión B/N
# (varían en luminancia, no solo en tono). Los 6 primeros son los del informe; los
# extra (7-10) solo se usan en gráficos con muchas series (p.ej. rentabilidad
# acumulada con TODOS los fondos) — los demás gráficos topan en 6 y no cambian.
_PALETTE = ["#0b3766", "#c8102e", "#0a8a5f", "#e08a00", "#6a3d9a", "#1f9bcf",
            "#8a6d3b", "#e5559e", "#5c5c5c", "#17807e"]

# Color de la serie superpuesta "Neto"/"Total" (rojo del informe BCCh).
_OVERLAY_COLOR = "#c8102e"

# Paleta para las series APILADAS cuando hay overlay: sin rojo (reservado para el
# Neto), tonos del informe (azul/tabaco/pizarra/verde/ámbar/morado).
_STACK_PALETTE = ["#1f6fb2", "#b08d57", "#5a6b7b", "#0a8a5f", "#e08a00", "#6a3d9a"]


def _split_overlay(plot: PlotData) -> tuple[list[PlotSeries], list[PlotSeries]]:
    """``(series apiladas, series superpuestas)`` según ``plot.overlay``."""
    labels = set(plot.overlay or ())
    if not labels:
        return list(plot.series), []
    stack = [s for s in plot.series if s.label not in labels]
    over = [s for s in plot.series if s.label in labels]
    return stack, over

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


def _val_unit(v: float, unit: str) -> str:
    return f"{_fmt_num(v)} {unit}" if unit else _fmt_num(v)


def _build_pts_json(
    parsed: "list[tuple[PlotSeries, list[tuple[int, float]]]]",
    px_fn: "Callable[[int], float]",
    iso_by_ord: "dict[int, str]",
    unit: str,
    *,
    py_fn: "Callable[[float], float] | None" = None,
    colors: list[str] | None = None,
) -> str:
    """JSON compacto para el hover unificado (crosshair): lista ordenada de puntos
    por fecha. Cada entrada: ``{k, x, vals:[{s, c, v, i[, y]}]}``.
    ``i`` es el índice de la serie para filtrar items de leyenda ocultos.
    ``y`` (presente si ``py_fn`` se pasa) es la coordenada Y SVG del valor —
    el JS la usa para destacar la serie visualmente más cercana al cursor.
    ``colors`` overridea la paleta por índice (p.ej. para la serie superpuesta)."""
    by_date: dict[str, dict] = {}
    for i, (s, pts) in enumerate(parsed):
        color = colors[i] if colors and i < len(colors) else _PALETTE[i % len(_PALETTE)]
        for o, v in pts:
            iso = iso_by_ord.get(o) or date.fromordinal(o).isoformat()
            k = _fmt_date(iso)
            x = px_fn(o)
            if k not in by_date:
                by_date[k] = {"k": k, "x": round(x, 1), "vals": []}
            entry: dict = {"s": s.label, "c": color, "v": _val_unit(v, unit), "i": i}
            if py_fn is not None:
                entry["y"] = round(py_fn(v), 1)
            by_date[k]["vals"].append(entry)
    return _json.dumps(
        sorted(by_date.values(), key=lambda d: d["x"]),
        ensure_ascii=False, separators=(",", ":"),
    )


def _tip_attrs(color: str, *, s: str = "", k: str = "", v: str = "") -> str:
    """Atributos ``data-*`` para el tooltip interactivo (estilo Plotly) que arma el
    JS del informe: ``data-c`` (swatch de color), ``data-s`` (serie), ``data-k``
    (clave: fecha o categoría) y ``data-v`` (valor con unidad). Reemplazan al
    ``<title>`` nativo (tooltip pobre del navegador) y dan el marcador on-hover."""
    out = f' class="tip-pt" data-c="{_esc(color)}"'
    if s:
        out += f' data-s="{_esc(s)}"'
    if k:
        out += f' data-k="{_esc(k)}"'
    if v:
        out += f' data-v="{_esc(v)}"'
    return out


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
    "dual_axis": "timeseries",
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
    right_axis: list[str] | None = None, right_unit: str = "",
) -> str | None:
    """``PlotData`` → SVG inline (str) del tipo ``chart`` (o el natural del kind si
    ``chart`` no aplica). ``None`` si ``family == 'table'`` (→ mini-tabla).

    ``right_axis`` (labels de series) + ``right_unit`` activan el doble eje Y para
    ``chart='dual_axis'`` (esas series van contra un eje derecho independiente)."""
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
    if target == "dual_axis":
        return _render_dual_axis(plot, width, height, right_axis, right_unit)
    if target in ("area", "stacked_area"):
        return _render_stacked_area(plot, width, height)
    return _render_timeseries(plot, width, height)


def _svg_open(width: int, height: int, title: str, extra_attrs: str = "") -> list[str]:
    # aria-label (no <title>): describe el gráfico para lectores de pantalla SIN
    # disparar el tooltip nativo del navegador al pasar el mouse por zonas vacías
    # (el tooltip lindo lo hace el JS sobre los puntos data-*).
    return [
        f'<svg class="report-chart" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="{_esc(title)}" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="Arial, Helvetica, sans-serif"'
        f'{extra_attrs}>',
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

    pts_json = _build_pts_json(parsed, px, iso_by_ord, plot.unit, py_fn=py)
    out = _svg_open(width, height, f"{plot.dataset_id} — serie temporal",
                    f' data-pts="{_esc(pts_json)}"')

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

    # Guía vertical del crosshair (oculta; el JS la posiciona en el hover).
    out.append(
        f'<line class="x-guide" x1="{left:.1f}" y1="{top}" x2="{left:.1f}" y2="{top + plot_h}" '
        f'stroke="#aaa" stroke-width="1" stroke-dasharray="4,3" display="none" pointer-events="none"/>'
    )

    # Series + puntos-objetivo (data-si para el toggle de leyenda; pointer-events="none"
    # porque el overlay encima captura todos los eventos del mouse).
    legend: list[tuple[str, str]] = []
    for i, (s, pts) in enumerate(parsed):
        color = _PALETTE[i % len(_PALETTE)]
        if len(pts) == 1:
            o, v = pts[0]
            iso = iso_by_ord.get(o) or date.fromordinal(o).isoformat()
            out.append(f'<circle cx="{px(o):.1f}" cy="{py(v):.1f}" r="3" fill="{color}" data-si="{i}"/>')
        else:
            coords = " ".join(f"{px(o):.1f},{py(v):.1f}" for o, v in pts)
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.8" data-si="{i}"/>')
        for k in _subsample(pts):
            o, v = pts[k]
            iso = iso_by_ord.get(o) or date.fromordinal(o).isoformat()
            out.append(
                f'<circle cx="{px(o):.1f}" cy="{py(v):.1f}" r="4.5" fill="{color}" '
                f'fill-opacity="0" pointer-events="none" data-si="{i}"'
                f'{_tip_attrs(color, s=s.label, k=_fmt_date(iso), v=_val_unit(v, plot.unit))}/>'
            )
        legend.append((s.label, color))

    # Overlay transparente encima de las series: captura los eventos del mouse para
    # el crosshair + tooltip unificado (el JS busca la fecha más cercana en data-pts).
    out.append(
        f'<rect class="hover-overlay" x="{left}" y="{top}" '
        f'width="{plot_w}" height="{plot_h}" fill="transparent" stroke="none"/>'
    )

    out.append(_legend_row(legend, left, height - 30, plot_w, interactive=True))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _render_dual_axis(
    plot: PlotData, width: int, height: int,
    right_labels: list[str] | None, right_unit: str = "",
) -> str:
    """Serie temporal con DOBLE eje Y: las series cuyo label esté en
    ``right_labels`` se escalan contra un eje derecho independiente (y se dibujan
    como área tenue, estilo "AUM" del informe BCCh); el resto va contra el eje
    izquierdo como líneas. X compartido. Replica los gráficos con eje secundario
    sin separar el gráfico en dos.

    Si no hay ninguna serie para el eje derecho, cae al render de línea normal."""
    right_set = set(right_labels or [])
    has_right = any(s.label in right_set for s in plot.series)
    if not has_right:
        return _render_timeseries(plot, width, height)

    left, right = 76, 66
    top, bottom = 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    # Parseo común (ord, valor) por serie, conservando el índice original (color).
    parsed: list[tuple[int, PlotSeries, list[tuple[int, float]], bool]] = []
    xs: list[int] = []
    iso_by_ord: dict[int, str] = {}
    lys: list[float] = []
    rys: list[float] = []
    for i, s in enumerate(plot.series):
        pts: list[tuple[int, float]] = []
        for iso, v in s.points:
            o = _date_ord(iso)
            if o is None:
                continue
            pts.append((o, v))
            iso_by_ord.setdefault(o, iso)
        if not pts:
            continue
        is_right = s.label in right_set
        parsed.append((i, s, pts, is_right))
        xs.extend(o for o, _ in pts)
        (rys if is_right else lys).extend(v for _, v in pts)
    if not xs or not lys or not rys:
        return _render_timeseries(plot, width, height)

    xmin, xmax = min(xs), max(xs)
    lticks, llo, lhi = _nice_ticks(min(lys), max(lys))
    rticks, rlo, rhi = _nice_ticks(min(rys), max(rys))

    def px(o: int) -> float:
        return left + (plot_w / 2 if xmax == xmin else plot_w * (o - xmin) / (xmax - xmin))

    def ply(v: float) -> float:
        return top + plot_h * (1 - (v - llo) / (lhi - llo or 1.0))

    def pry(v: float) -> float:
        return top + plot_h * (1 - (v - rlo) / (rhi - rlo or 1.0))

    # JSON del hover unificado: cada serie con su propio eje (y por serie).
    by_date: dict[str, dict] = {}
    for idx, s, pts, is_right in parsed:
        color = "#b9c0cc" if is_right else _PALETTE[idx % len(_PALETTE)]
        unit = right_unit if is_right else plot.unit
        yfn = pry if is_right else ply
        for o, v in pts:
            iso = iso_by_ord.get(o) or date.fromordinal(o).isoformat()
            k = _fmt_date(iso)
            ent = by_date.setdefault(k, {"k": k, "x": round(px(o), 1), "vals": []})
            ent["vals"].append({"s": s.label, "c": color, "v": _val_unit(v, unit),
                                "i": idx, "y": round(yfn(v), 1)})
    pts_json = _json.dumps(sorted(by_date.values(), key=lambda d: d["x"]),
                           ensure_ascii=False, separators=(",", ":"))
    out = _svg_open(width, height, f"{plot.dataset_id} — doble eje", f' data-pts="{_esc(pts_json)}"')

    # Rejilla + eje izquierdo (ticks redondos).
    for tick in lticks:
        y = ply(tick)
        emph = abs(tick) < 1e-9 and llo < 0 < lhi
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{"#999" if emph else "#e6e6e6"}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')
    # Eje derecho (ticks en gris, alineados a la derecha del área).
    for tick in rticks:
        y = pry(tick)
        out.append(f'<text x="{left + plot_w + 8}" y="{y + 4:.1f}" font-size="11" fill="#8a93a3" text-anchor="start">{_esc(_fmt_num(tick))}</text>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    out.append(f'<line x1="{left + plot_w}" y1="{top}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#c8ccd4" stroke-width="1"/>')
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')

    # Ticks de fecha (hasta 5).
    tick_ords = sorted(set(xs))
    n_ticks = min(5, len(tick_ords))
    if n_ticks >= 1:
        step = (len(tick_ords) - 1) / max(1, n_ticks - 1)
        chosen = sorted({tick_ords[round(i * step)] for i in range(n_ticks)})
        for j, o in enumerate(chosen):
            x = px(o)
            label = _fmt_date(iso_by_ord.get(o, date.fromordinal(o).isoformat()))
            anchor = "start" if j == 0 else ("end" if j == len(chosen) - 1 else "middle")
            out.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-size="11" fill="#555" text-anchor="{anchor}">{_esc(label)}</text>')

    out.append(
        f'<line class="x-guide" x1="{left:.1f}" y1="{top}" x2="{left:.1f}" y2="{top + plot_h}" '
        f'stroke="#aaa" stroke-width="1" stroke-dasharray="4,3" display="none" pointer-events="none"/>'
    )

    legend: list[tuple[int, str, str]] = []
    # Primero las áreas del eje derecho (al fondo), luego las líneas del izquierdo.
    for idx, s, pts, is_right in sorted(parsed, key=lambda t: not t[3]):
        if is_right:
            base = top + plot_h
            up = " ".join(f"{px(o):.1f},{pry(v):.1f}" for o, v in pts)
            dn = f"{px(pts[-1][0]):.1f},{base:.1f} {px(pts[0][0]):.1f},{base:.1f}"
            out.append(f'<polygon points="{up} {dn}" fill="#c8ccd4" fill-opacity="0.55" stroke="none" data-si="{idx}"/>')
            legend.append((idx, f"{s.label} (eje der.)", "#b9c0cc"))
        else:
            color = _PALETTE[idx % len(_PALETTE)]
            coords = " ".join(f"{px(o):.1f},{ply(v):.1f}" for o, v in pts)
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.8" data-si="{idx}"/>')
            legend.append((idx, s.label, color))

    out.append(
        f'<rect class="hover-overlay" x="{left}" y="{top}" '
        f'width="{plot_w}" height="{plot_h}" fill="transparent" stroke="none"/>'
    )
    legend.sort(key=lambda t: t[0])
    out.append(_legend_row([(lbl, col) for _i, lbl, col in legend], left, height - 30, plot_w, interactive=True))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    if right_unit:
        out.append(f'<text x="{left + plot_w}" y="{top - 12}" font-size="11" fill="#8a93a3" text-anchor="end">{_esc(right_unit)}</text>')
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
        tip_v = f"{_val_unit(v, plot.unit)} ({_fmt_num(share)}%)"
        out.append(f'<text x="{left - 8}" y="{y + 15}" font-size="12" fill="#333" text-anchor="end">{_esc(cat)}</text>')
        out.append(f'<rect x="{left}" y="{y + 3}" width="{bar_w:.1f}" height="{row_h - 10}" fill="{color}"{_tip_attrs(color, s=cat, v=tip_v)}/>')
        out.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 15}" font-size="11" fill="#555">{_esc(_fmt_num(v))} ({_esc(_fmt_num(share))}%)</text>')
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _render_grouped_bars(plot: PlotData, width: int, height: int, *, stacked: bool) -> str:
    """Barras verticales agrupadas o apiladas. Categorías (eje X) compartidas por
    todas las series; cada serie es un color. Maneja valores negativos (bajo 0).
    Las series en ``plot.overlay`` ("Neto"/"Total") no se barran: se dibujan como
    un punto por categoría sobre las barras."""
    series, overlays = _split_overlay(plot)
    pal = _STACK_PALETTE if overlays else _PALETTE
    cats: list[str] = []
    seen: set[str] = set()
    for s in plot.series:
        for c, _v in s.points:
            if c not in seen:
                seen.add(c)
                cats.append(c)
    if not cats or not series:
        return _no_axis_message(plot, width, height)
    lut = [{c: v for c, v in s.points} for s in series]
    ov_lut = [{c: v for c, v in s.points} for s in overlays]

    long_labels = any(len(c) > 9 for c in cats)
    left, right = 64, 18
    top = 30
    bottom = 90 if long_labels else 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    ov_all = [ov_lut[j].get(c, 0.0) for j in range(len(overlays)) for c in cats]
    if stacked:
        pos = [sum(max(0.0, lut[i].get(c, 0.0)) for i in range(len(series))) for c in cats]
        neg = [sum(min(0.0, lut[i].get(c, 0.0)) for i in range(len(series))) for c in cats]
        dmax, dmin = max([0.0, *pos, *ov_all]), min([0.0, *neg, *ov_all])
    else:
        allv = [lut[i].get(c, 0.0) for i in range(len(series)) for c in cats]
        dmax, dmin = max([0.0, *allv, *ov_all]), min([0.0, *allv, *ov_all])
    if dmax == dmin:
        dmax += 1.0
    ticks, ymin, ymax = _nice_ticks(dmin, dmax)  # incluye 0 → base de eje 0

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    group_w = plot_w / len(cats)
    # data-stack: igual que el área apilada, datos crudos por serie + eje fijo para
    # que el JS re-apile las barras visibles al ocultar una serie (solo apiladas;
    # las agrupadas no se re-apilan). Las rects llevan data-ci para mapear su
    # categoría aunque alguna barra ~0 no se haya dibujado.
    extra_attrs = ""
    if stacked:
        stack_json = _json.dumps(
            {
                "k": "bar",
                "top": top, "ph": plot_h, "ymin": round(ymin, 6), "ymax": round(ymax, 6),
                "s": [
                    {"i": si, "v": [round(lut[si].get(c, 0.0), 6) for c in cats]}
                    for si in range(len(series))
                ],
            },
            ensure_ascii=False, separators=(",", ":"),
        )
        extra_attrs = f' data-stack="{_esc(stack_json)}"'
    out = _svg_open(width, height, f"{plot.dataset_id} — barras", extra_attrs)

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
                    col = pal[si % len(pal)]
                    attrs = _tip_attrs(col, s=series[si].label, k=c, v=_val_unit(v, plot.unit))
                    out.append(f'<rect x="{x:.1f}" y="{min(y_top, y_bot):.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{col}" data-si="{si}" data-ci="{ci}"{attrs}/>')
        else:
            inner = group_w * 0.8
            bw = inner / len(series)
            x0 = gx + (group_w - inner) / 2
            for si in range(len(series)):
                v = lut[si].get(c, 0.0)
                y_top, y_bot = py(max(0.0, v)), py(min(0.0, v))
                h = abs(y_bot - y_top)
                if h > 0.2:
                    col = pal[si % len(pal)]
                    attrs = _tip_attrs(col, s=series[si].label, k=c, v=_val_unit(v, plot.unit))
                    out.append(f'<rect x="{x0 + si * bw:.1f}" y="{min(y_top, y_bot):.1f}" width="{bw * 0.86:.1f}" height="{h:.1f}" fill="{col}" data-si="{si}"{attrs}/>')
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

    # Punto(s) "Neto"/"Total" sobre cada categoría (no se apilan).
    for oj, s in enumerate(overlays):
        si = len(series) + oj
        for ci, c in enumerate(cats):
            if c not in ov_lut[oj]:
                continue
            v = ov_lut[oj][c]
            cx = left + ci * group_w + group_w / 2
            attrs = _tip_attrs(_OVERLAY_COLOR, s=s.label, k=c, v=_val_unit(v, plot.unit))
            out.append(
                f'<circle cx="{cx:.1f}" cy="{py(v):.1f}" r="4" fill="{_OVERLAY_COLOR}" '
                f'stroke="#fff" stroke-width="1" data-si="{si}"{attrs}/>'
            )

    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    legend_y = height - (20 if long_labels else 26)
    legend = [(s.label, pal[i % len(pal)]) for i, s in enumerate(series)]
    legend += [(s.label, _OVERLAY_COLOR) for s in overlays]
    out.append(_legend_row(legend, left, legend_y, plot_w, interactive=True))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _render_stacked_area(plot: PlotData, width: int, height: int) -> str:
    """Área apilada DIVERGENTE: contribuciones positivas se apilan sobre 0 y las
    negativas bajo 0 (réplica de "Pagan/Reciben Fija" del BCCh). Si todas las
    series son ≥0 se comporta como un apilado normal. Las series en
    ``plot.overlay`` (p.ej. "Neto") NO se apilan: se dibujan como una línea
    superpuesta sobre el área."""
    series, overlays = _split_overlay(plot)
    pal = _STACK_PALETTE if overlays else _PALETTE
    dates = sorted({iso for s in plot.series for iso, _ in s.points if _date_ord(iso) is not None})
    if not dates:
        return _no_axis_message(plot, width, height)
    ordv = [_date_ord(d) for d in dates]
    lut = [dict(s.points) for s in series]
    ov_lut = [dict(s.points) for s in overlays]

    left, right = 76, 18
    top, bottom = 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    pos_tot = [sum(max(0.0, lut[i].get(d, 0.0)) for i in range(len(series))) for d in dates]
    neg_tot = [sum(min(0.0, lut[i].get(d, 0.0)) for i in range(len(series))) for d in dates]
    ov_all = [ov_lut[j].get(d, 0.0) for j in range(len(overlays)) for d in dates]
    hi = max([0.0, *pos_tot, *ov_all]) or 1.0
    lo = min([0.0, *neg_tot, *ov_all])
    ticks, ymin, ymax = _nice_ticks(lo, hi)
    if ymax == ymin:
        ymax = ymin + 1.0
    xmin, xmax = min(ordv), max(ordv)

    def px(o: int) -> float:
        return left + (plot_w / 2 if xmax == xmin else plot_w * (o - xmin) / (xmax - xmin))

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    # JSON para hover unificado: valores INDIVIDUALES de cada serie (no acumulados),
    # incluida la(s) superpuesta(s) con su color rojo.
    ord_to_date = {ordv[k]: dates[k] for k in range(len(dates)) if ordv[k] is not None}
    parsed_pts = [
        (s, [(ordv[k], (lut + ov_lut)[si].get(dates[k], 0.0))
             for k in range(len(dates)) if ordv[k] is not None])
        for si, s in enumerate(series + overlays)
    ]
    colors = [pal[i % len(pal)] for i in range(len(series))] + [_OVERLAY_COLOR] * len(overlays)
    pts_json = _build_pts_json(parsed_pts, px, ord_to_date, plot.unit, colors=colors)
    # data-stack: datos crudos por serie APILADA (sin acumular) + geometría del
    # eje fijo, para que el JS de la leyenda re-apile en el cliente al ocultar una
    # serie (las visibles se completan hacia la base; ver _TIP_JS → restack).
    stack_json = _json.dumps(
        {
            "k": "area",
            "x": [round(px(ordv[k]), 1) for k in range(len(dates))],
            "top": top, "ph": plot_h, "ymin": round(ymin, 6), "ymax": round(ymax, 6),
            "s": [
                {"i": si, "v": [round(lut[si].get(dates[k], 0.0), 6) for k in range(len(dates))]}
                for si in range(len(series))
            ],
        },
        ensure_ascii=False, separators=(",", ":"),
    )
    out = _svg_open(width, height, f"{plot.dataset_id} — área apilada",
                    f' data-pts="{_esc(pts_json)}" data-stack="{_esc(stack_json)}"')

    for tick in ticks:
        y = py(tick)
        emph = abs(tick) < 1e-9 and ymin < 0 < ymax
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{"#999" if emph else "#e6e6e6"}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')

    # Guía vertical del crosshair.
    out.append(
        f'<line class="x-guide" x1="{left:.1f}" y1="{top}" x2="{left:.1f}" y2="{top + plot_h}" '
        f'stroke="#aaa" stroke-width="1" stroke-dasharray="4,3" display="none" pointer-events="none"/>'
    )

    tip_idx = _subsample(dates)
    pos_cum = [0.0] * len(dates)
    neg_cum = [0.0] * len(dates)
    for si in range(len(series)):
        lower: list[float] = []
        upper: list[float] = []
        for k in range(len(dates)):
            v = lut[si].get(dates[k], 0.0)
            if v >= 0:
                lo_e, hi_e = pos_cum[k], pos_cum[k] + v
                pos_cum[k] = hi_e
            else:
                hi_e, lo_e = neg_cum[k], neg_cum[k] + v
                neg_cum[k] = lo_e
            lower.append(lo_e)
            upper.append(hi_e)
        up = " ".join(f"{px(ordv[k]):.1f},{py(upper[k]):.1f}" for k in range(len(dates)))
        dn = " ".join(f"{px(ordv[k]):.1f},{py(lower[k]):.1f}" for k in reversed(range(len(dates))))
        col = pal[si % len(pal)]
        out.append(f'<polygon points="{up} {dn}" fill="{col}" fill-opacity="0.85" stroke="none" data-si="{si}"/>')
        for k in tip_idx:
            band_v = lut[si].get(dates[k], 0.0)
            mid_y = py((lower[k] + upper[k]) / 2)
            out.append(
                f'<circle cx="{px(ordv[k]):.1f}" cy="{mid_y:.1f}" r="4.5" fill="{col}" '
                f'fill-opacity="0" pointer-events="none" data-si="{si}"'
                f'{_tip_attrs(col, s=series[si].label, k=_fmt_date(dates[k]), v=_val_unit(band_v, plot.unit))}/>'
            )

    # Eje X en 0 (o en la base si no cruza 0).
    y0 = py(0.0) if ymin <= 0 <= ymax else top + plot_h
    out.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{left + plot_w}" y2="{y0:.1f}" stroke="#999" stroke-width="1"/>')

    # Serie(s) superpuesta(s) "Neto": línea sobre el área.
    for oj, s in enumerate(overlays):
        pts = [(ordv[k], ov_lut[oj].get(dates[k], 0.0)) for k in range(len(dates))]
        poly = " ".join(f"{px(o):.1f},{py(v):.1f}" for o, v in pts)
        si = len(series) + oj
        out.append(f'<polyline points="{poly}" fill="none" stroke="{_OVERLAY_COLOR}" stroke-width="2.2" data-si="{si}"/>')
        for k in tip_idx:
            v = ov_lut[oj].get(dates[k], 0.0)
            out.append(
                f'<circle cx="{px(ordv[k]):.1f}" cy="{py(v):.1f}" r="4.5" fill="{_OVERLAY_COLOR}" '
                f'fill-opacity="0" pointer-events="none" data-si="{si}"'
                f'{_tip_attrs(_OVERLAY_COLOR, s=s.label, k=_fmt_date(dates[k]), v=_val_unit(v, plot.unit))}/>'
            )

    # ticks de fecha (hasta 5).
    n = min(5, len(dates))
    if n >= 1:
        step = (len(dates) - 1) / max(1, n - 1)
        for j in range(n):
            k = round(j * step)
            x = px(ordv[k])
            anchor = "start" if j == 0 else ("end" if j == n - 1 else "middle")
            out.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-size="11" fill="#555" text-anchor="{anchor}">{_esc(_fmt_date(dates[k]))}</text>')

    # Overlay encima de todo: captura el mouse para el crosshair unificado.
    out.append(
        f'<rect class="hover-overlay" x="{left}" y="{top}" '
        f'width="{plot_w}" height="{plot_h}" fill="transparent" stroke="none"/>'
    )
    legend = [(s.label, pal[i % len(pal)]) for i, s in enumerate(series)]
    legend += [(s.label, _OVERLAY_COLOR) for s in overlays]
    out.append(_legend_row(legend, left, height - 26, plot_w, interactive=True))
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
    angle = -math.pi / 2  # arranca arriba
    for i, (cat, v) in enumerate(points):
        frac = v / total
        a2 = angle + frac * 2 * math.pi
        x1, y1 = cx + r * math.cos(angle), cy + r * math.sin(angle)
        x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
        large = 1 if frac > 0.5 else 0
        color = _PALETTE[i % len(_PALETTE)]
        attrs = _tip_attrs(color, s=cat, v=f"{_val_unit(v, plot.unit)} ({_fmt_num(100.0 * frac)}%)")
        if frac >= 0.999:  # una sola categoría → círculo completo
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"{attrs}/>')
        else:
            out.append(f'<path d="M {cx} {cy} L {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f} Z" fill="{color}"{attrs}/>')
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


def _legend_row(items: list[tuple[str, str]], x0: int, y: float, max_w: int, *, interactive: bool = False) -> str:
    """Fila de swatches + etiquetas; envuelve a una segunda línea si no caben.

    ``interactive=True`` envuelve cada ítem en ``<g class="lg-item" data-idx="i">``
    para el toggle click-to-hide de series (el JS escucha los clics en el grupo)."""
    parts: list[str] = []
    x = x0
    line = 0
    for idx, (label, color) in enumerate(items):
        label_s = label if len(label) <= 22 else label[:21] + "…"
        w = 16 + len(label_s) * 6.2 + 14
        if x + w > x0 + max_w and x > x0:
            line += 1
            x = x0
        yy = y + line * 16
        if interactive:
            parts.append(f'<g class="lg-item" data-idx="{idx}" style="cursor:pointer">')
            # Área transparente más ancha que swatch+texto para facilitar el click.
            parts.append(f'<rect x="{x - 2:.1f}" y="{yy - 10:.1f}" width="{w:.0f}" height="18" fill="transparent" stroke="none"/>')
        parts.append(f'<rect x="{x:.1f}" y="{yy - 8:.1f}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{x + 14:.1f}" y="{yy:.1f}" font-size="11" fill="#444">{_esc(label_s)}</text>')
        if interactive:
            parts.append('</g>')
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
    """Tabla de fechas de corte DCV (T, T-7, T-30): filas=instrumento,
    columnas=(T, T-7, T-30, Δ T-7, Δ T-30). Deltas (variación a 1 semana / 1 mes)
    coloreados verde/rojo."""
    t_iso, t7_iso, t30_iso = dates
    th = "text-align:right;padding:6px 8px;background:#4a5a72;color:#fff;white-space:nowrap;font-size:12px"
    thl = "text-align:left;padding:6px 8px;background:#4a5a72;color:#fff;font-size:12px"
    td = "text-align:right;padding:5px 8px;border-bottom:1px solid #eee;font-size:12px"
    tdl = "text-align:left;padding:5px 8px;border-bottom:1px solid #eee;font-size:12px"

    head = (
        f'<thead><tr>'
        f'<th style="{thl}">Instrumento</th>'
        f'<th style="{th}">T&nbsp;({_fmt_date(t_iso)})</th>'
        f'<th style="{th}">T-7&nbsp;({_fmt_date(t7_iso)})</th>'
        f'<th style="{th}">T-30&nbsp;({_fmt_date(t30_iso)})</th>'
        f'<th style="{th}">Δ T-7</th>'
        f'<th style="{th}">Δ T-30</th>'
        f'</tr></thead>'
    )
    body_rows = []
    for tipo in tipos:
        vt, vt7, vt30 = data.get(tipo, (None, None, None))
        d7 = (vt - vt7) if vt is not None and vt7 is not None else None
        d30 = (vt - vt30) if vt is not None and vt30 is not None else None
        d7_sty = f"{td};{_delta_style(d7)}" if d7 else td
        d30_sty = f"{td};{_delta_style(d30)}" if d30 else td
        body_rows.append(
            f'<tr>'
            f'<td style="{tdl}">{_esc(tipo)}</td>'
            f'<td style="{td}">{_fmt_num(vt) if vt is not None else "&#8212;"}</td>'
            f'<td style="{td}">{_fmt_num(vt7) if vt7 is not None else "&#8212;"}</td>'
            f'<td style="{td}">{_fmt_num(vt30) if vt30 is not None else "&#8212;"}</td>'
            f'<td style="{d7_sty}">{_fmt_delta(d7)}</td>'
            f'<td style="{d30_sty}">{_fmt_delta(d30)}</td>'
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
    delta7: dict[str, dict[str, float | None]],
    delta30: dict[str, dict[str, float | None]],
    unit: str = "US$ Mill.",
) -> str:
    """Dos matrices heatmap (Delta T-7 / Delta T-30, variación a 1 semana / 1 mes):
    filas=instrumento, cols=plazo, celdas coloreadas verde (positivo) / rojo
    (negativo)."""

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
        + _matrix("Δ T-7", delta7)
        + _matrix("Δ T-30", delta30)
        + unit_note + "</div>"
    )
