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
    range      → caja de rango [mín,máx] + tick de promedio + punto "hoy" por
                 categoría (``hist_range``; réplica de "Rendimiento monedas" GBI)
    scatter    → dispersión x/y etiquetada por punto, con cruce en 0 (``point``;
                 réplica de "Retorno FX y tasas GBI")

``render_plot_svg(plot, chart=...)`` dibuja el tipo pedido si la forma del dato lo
soporta; si no, cae a la marca natural del ``kind`` (línea/composición). El caller
usa ``renders_natively`` para saber si fue el tipo pedido o un fallback (vista
preliminar). ``family == "table"`` devuelve ``None`` → mini-tabla HTML.
"""

from __future__ import annotations

import html
import itertools
import json as _json
import math
import re
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

# Área del eje IZQUIERDO en dual_axis (``left_style="area"``): celeste tenue,
# distinguible del gris del área derecha (``#b9c0cc``) — así no se confunden
# cuando el gráfico tiene área de los dos lados a la vez.
_LEFT_AREA_COLOR = "#a9c4e0"

# Paleta para las series APILADAS cuando hay overlay: sin rojo (reservado para el
# Neto), tonos del informe (azul/tabaco/pizarra/verde/ámbar/morado).
#
# Los 6 primeros son los colores canónicos del informe; los 6 siguientes son tonos
# claros/oscuros de esos mismos para que un apilado de hasta 12 series (el fixing
# abre por 9 sectores contraparte) no CICLE la paleta — con 6 colores, la serie 7
# salía del mismo azul que la 1 y la leyenda quedaba ambigua.
_STACK_PALETTE = ["#1f6fb2", "#b08d57", "#5a6b7b", "#0a8a5f", "#e08a00", "#6a3d9a",
                  "#7fb2e5", "#6b5327", "#2c3a47", "#5fcfa4", "#a35c00", "#b08fd6"]


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

# viewBox de un bloque que ocupa la FILA COMPLETA de la grilla (layout "grid":
# ver ``curated_report._wide_block_ids``). Los márgenes/tamaños de fuente del
# renderer son PÍXELES ABSOLUTOS del viewBox, no relativos — así que si un
# gráfico ancho siguiera usando el viewBox normal (_W x _H), el navegador lo
# ESTIRA al doblar su ancho en pantalla y todo (ejes, leyenda, título, grosor de
# línea) se ve "con zoom". Doblar el viewBox mantiene el factor de escala
# CSS≈constante entre un bloque normal y uno ancho, así el tamaño VISUAL de esos
# elementos es el mismo en los dos — solo cambia cuánto se ve del eje X.
_W_WIDE = _W * 2

# Un bloque "wide" NO se estira a 2x en pantalla: vive en una tarjeta
# ``card-wide`` de ``.cards-grid`` (grilla de 2 columnas, ver
# ``curated_report._is_grid_section``), cuyo ancho real es MENOR que el viewBox
# doblado (1520) — un bloque normal, en cambio, vive fuera de tarjeta y su
# viewBox (_W=760) se recorta 1:1 (``.report-chart {max-width:760px}``), escala
# real 1. Con el mismo font-size ABSOLUTO en los dos, el wide se ve más chico.
# ``_WIDE_FONT_SCALE`` agranda un POCO esos font-size para que ejes y leyenda
# no se vean chicos frente al resto de los gráficos del informe (p.ej. dcv:
# "Todos los instrumentos — distribución por tramo de plazo", que comparte
# tarjeta con su tabla) — a propósito MODESTO (no 1:1 con el bloque normal):
# compensar el 100% del achique se termina viendo más GRANDE que el resto.
_WIDE_FONT_SCALE = 1.12


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
    """``YYYY-MM-DD`` → ``DD-MM-YY`` (formato chileno compacto para ticks).

    Si la marca trae HORA distinta de medianoche (serie intradía) el tick la
    muestra: ``DD-MM HH:MM``. Sin esto, un gráfico de una sola jornada repetía la
    misma fecha en los cinco ticks y no se leía a qué hora ocurrió nada.

    Medianoche EXACTA se trata como fecha sin hora a propósito: un parquet diario
    con la fecha guardada como TIMESTAMP llega como ``...T00:00:00`` y no tiene
    sentido rotular cada día con un "00:00"."""
    try:
        d = date.fromisoformat(iso[:10])
    except ValueError:
        return html.escape(iso[:10])
    hh, mm = iso[11:13], iso[14:16]
    if hh.isdigit() and mm.isdigit() and (hh, mm) != ("00", "00"):
        return f"{d.day:02d}-{d.month:02d} {hh}:{mm}"
    return f"{d.day:02d}-{d.month:02d}-{str(d.year)[2:]}"


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


# ── Geometría ────────────────────────────────────────────────────────────────

def _date_ord(iso: str) -> float | None:
    """Fecha ISO → ordinal para el eje X. Devuelve un FLOAT porque el eje admite
    resolución intradía: ``2026-08-07T11:30`` cae a mitad de camino entre el 7 y
    el 8. Sin esto, un gráfico intradía (CLP/DXY/cobre minuto a minuto) apilaba
    todos los puntos de un día sobre la misma abscisa y salía como una barra
    vertical. Las series diarias siguen dando un ordinal entero, igual que antes."""
    try:
        d = date.fromisoformat(iso[:10]).toordinal()
    except ValueError:
        return None
    hh, mm = iso[11:13], iso[14:16]
    if hh.isdigit() and mm.isdigit():
        return d + (int(hh) * 60 + int(mm)) / 1440.0
    return float(d)


def _ord_iso(o: float) -> str:
    """Ordinal (posiblemente fraccional) → fecha ISO. Solo se usa como respaldo
    cuando el ordinal no está en ``iso_by_ord``; la parte intradía se descarta."""
    return date.fromordinal(int(o)).isoformat()


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

# Ancho aproximado de un carácter a font-size 10 (la tipografía de las etiquetas
# de categoría). Sirve para decidir si una etiqueta CABE horizontal o hay que
# rotarla; es una estimación, no hace falta medir la fuente real.
_CAT_LABEL_CHAR_W = 5.6


def _val_unit(v: float, unit: str) -> str:
    return f"{_fmt_num(v)} {unit}" if unit else _fmt_num(v)


def _build_pts_json(
    parsed: "list[tuple[PlotSeries, list[tuple[float, float]]]]",
    px_fn: "Callable[[float], float]",
    iso_by_ord: "dict[float, str]",
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
            iso = iso_by_ord.get(o) or _ord_iso(o)
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
# "hist_range"/"point" son los ``chart_type`` canónicos del catálogo (ver
# ``domain/agent/chart_types.py``: hist_range→bar, scatter→point) reusados tal
# cual como ``block.chart`` del informe curado, sin inventar vocabulario nuevo.
_CHART_NATIVE_KIND: dict[str, str | tuple[str, ...]] = {
    "line": "timeseries",
    "area": "timeseries",
    "stacked_area": "timeseries",
    "dual_axis": "timeseries",
    "candlestick": "timeseries",
    "grouped_bar": "grouped",
    "stacked_bar": "grouped",
    "bar_time": "grouped",
    "composition": "snapshot",
    "pie": "snapshot",
    "hist_range": "range",
    # "point" cubre DOS formas: dispersión x/y numérica (GBI, kind='scatter') y
    # dispersión CATEGÓRICA por grupo (Duración agentes IIF/RF del DCV,
    # kind='grouped', ver _render_grouped_dots) — ambas se dibujan como puntos.
    "point": ("scatter", "grouped"),
    # Curva sobre eje CATEGÓRICO (plazo), no temporal: dos cortes de la misma curva
    # ("Hoy" vs "t-5") del informe de renta fija. "spc_curve" es el nombre canónico
    # del catálogo para esos datasets; "curve" el alias corto para el spec.
    "curve": "grouped",
    "spc_curve": "grouped",
    "heatmap_table": "heatmap",
    "treemap": "hierarchy",
}


def renders_natively(plot_kind: str, chart: str | None) -> bool:
    """True si ``chart`` se dibuja de forma nativa con un PlotData de ese kind
    (no es un fallback/vista preliminar)."""
    native = _CHART_NATIVE_KIND.get((chart or "").strip())
    return plot_kind in native if isinstance(native, tuple) else native == plot_kind


def render_plot_svg(
    plot: PlotData, *, chart: str | None = None, width: int = _W, height: int = _H,
    right_axis: list[str] | None = None, right_unit: str = "", right_style: str = "",
    right_invert: bool = False, left_style: str = "",
    x_label: str = "", y_label: str = "",
) -> str | None:
    """``PlotData`` → SVG inline (str) del tipo ``chart`` (o el natural del kind si
    ``chart`` no aplica). ``None`` si ``family == 'table'`` (→ mini-tabla).

    ``right_axis`` (labels de series) + ``right_unit`` activan el doble eje Y para
    ``chart='dual_axis'`` (esas series van contra un eje derecho independiente);
    ``right_style``/``left_style`` eligen cómo se dibuja cada lado (``"line"`` por
    defecto en ambos, salvo el derecho que por compatibilidad sigue siendo
    ``"area"`` si no se pasa nada — ver ``_render_dual_axis``). ``right_invert``
    da vuelta el eje derecho (0 arriba/abajo según el signo de los datos): réplica
    de series como "Posición cambiaria (eje inv.)" del tablero, donde la serie es
    negativa y se lee invertida para que se mueva visualmente CON la otra serie en
    vez de en espejo. ``x_label``/``y_label`` rotulan los ejes del scatter
    (``kind='scatter'``); vacío → usa ``plot.unit`` en ambos ejes."""
    if plot.is_empty() or plot.family == "table":
        return None
    target = (chart or "").strip()
    if plot.kind == "grouped":
        if target == "point":
            return _render_grouped_dots(plot, width, height)
        if target in ("curve", "spc_curve"):
            return _render_grouped_lines(plot, width, height)
        return _render_grouped_bars(plot, width, height, stacked=(target == "stacked_bar"))
    if plot.kind == "snapshot":
        if target == "pie":
            return _render_pie(plot, width, height)
        return _render_snapshot(plot, width, height)
    if plot.kind == "range":
        return _render_range_band(plot, width, height)
    if plot.kind == "scatter":
        return _render_scatter_labeled(plot, width, height, x_label=x_label, y_label=y_label)
    if plot.kind == "heatmap":
        return _render_heatmap(plot, width, height)
    if plot.kind == "hierarchy":
        return _render_treemap(plot, width, height)
    # timeseries
    if target == "candlestick":
        return _render_candlestick(plot, width, height)
    if target == "dual_axis":
        return _render_dual_axis(plot, width, height, right_axis, right_unit, right_style,
                                 right_invert=right_invert, left_style=left_style)
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


def _series_style(plot: PlotData, idx: int, label: str) -> tuple[str, float]:
    """``(color, stroke_width)`` de una serie del EJE IZQUIERDO: ``plot.emphasis``
    fuerza un color puntual (p.ej. MA50 en oro), ``plot.muted`` la deja fina y
    gris; sin ninguno de los dos, la paleta cíclica de siempre por índice."""
    if label in plot.emphasis:
        return plot.emphasis[label], 2.4
    if label in plot.muted:
        return "#9aa2b1", 0.9
    return _PALETTE[idx % len(_PALETTE)], 1.8


def _render_timeseries(plot: PlotData, width: int, height: int) -> str:
    left, right = _MARGIN["left"], _MARGIN["right"]
    top, bottom = _MARGIN["top"], _MARGIN["bottom"]
    plot_w = width - left - right
    plot_h = height - top - bottom

    # Dominios: X por fecha (ordinal), Y por valor, sobre TODAS las series.
    xs: list[float] = []
    ys: list[float] = []
    parsed: list[tuple[PlotSeries, list[tuple[float, float]]]] = []
    iso_by_ord: dict[float, str] = {}
    for s in plot.series:
        pts: list[tuple[float, float]] = []
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
    ticks, ymin, ymax = _nice_ticks(min(ys), max(ys), include_zero=plot.zero_base)
    if ymax == ymin:
        ymax = ymin + 1.0

    def px(o: float) -> float:
        return left + (plot_w / 2 if xmax == xmin else plot_w * (o - xmin) / (xmax - xmin))

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    styles = [_series_style(plot, i, s.label) for i, (s, _pts) in enumerate(parsed)]
    pts_json = _build_pts_json(parsed, px, iso_by_ord, plot.unit, py_fn=py,
                               colors=[c for c, _w in styles])
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
            label = _fmt_date(iso_by_ord.get(o, _ord_iso(o)))
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
        color, width = styles[i]
        if len(pts) == 1:
            o, v = pts[0]
            iso = iso_by_ord.get(o) or _ord_iso(o)
            out.append(f'<circle cx="{px(o):.1f}" cy="{py(v):.1f}" r="3" fill="{color}" data-si="{i}"/>')
        else:
            coords = " ".join(f"{px(o):.1f},{py(v):.1f}" for o, v in pts)
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" data-si="{i}"/>')
        for k in _subsample(pts):
            o, v = pts[k]
            iso = iso_by_ord.get(o) or _ord_iso(o)
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


# ── Candlestick (OHLC) ───────────────────────────────────────────────────────
#
# Etiquetas EXACTAS que identifican las cuatro patas de la vela dentro de un
# ``PlotData`` de kind "timeseries". Es la misma convención que ya usa el rango
# histórico ("Mínimo"/"Máximo"/"Promedio"/"Hoy"): el dataclass no cambia, la
# semántica va en el label, y cualquier serie extra (p.ej. el monto transado) se
# ignora en vez de romper el gráfico.
_OHLC_LABELS = ("Apertura", "Máximo", "Mínimo", "Cierre")

_CANDLE_UP = "#0a8a5f"    # cierre >= apertura
_CANDLE_DOWN = "#c8102e"  # cierre <  apertura


def _render_candlestick(plot: PlotData, width: int, height: int) -> str:
    """Vela japonesa diaria: mecha entre mínimo y máximo, cuerpo entre apertura y
    cierre, verde si la sesión cerró al alza y roja si cerró a la baja.

    Requiere las cuatro series ``_OHLC_LABELS``; si falta alguna cae al render de
    línea (la serie igual se ve, marcada como vista preliminar por el caller)."""
    by_label = {s.label: dict(s.points) for s in plot.series}
    if not all(lab in by_label for lab in _OHLC_LABELS):
        return _render_timeseries(plot, width, height)
    op, hi, lo, cl = (by_label[lab] for lab in _OHLC_LABELS)

    # Solo los días con las cuatro patas: una vela a medias no se puede dibujar.
    isos = sorted(d for d in cl if d in op and d in hi and d in lo and _date_ord(d) is not None)
    if not isos:
        return _no_axis_message(plot, width, height)

    left, right = 76, 18
    top, bottom = 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    ords = [_date_ord(d) for d in isos]
    xmin, xmax = min(ords), max(ords)
    # Base 0 forzada a False: el dominio útil de un tipo de cambio es su rango, no
    # el 0 (una vela sobre un eje que arranca en 0 queda plana e ilegible).
    ticks, ymin, ymax = _nice_ticks(min(lo[d] for d in isos), max(hi[d] for d in isos),
                                    include_zero=False)
    if ymax == ymin:
        ymax = ymin + 1.0

    def px(o: float) -> float:
        return left + (plot_w / 2 if xmax == xmin else plot_w * (o - xmin) / (xmax - xmin))

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    # Ancho del cuerpo: el paso entre velas menos un canal, acotado para que ni
    # tres velas salgan como bloques ni doscientas se toquen entre sí.
    step = plot_w / max(1, len(isos) - 1) if len(isos) > 1 else plot_w
    body_w = max(1.5, min(11.0, step * 0.62))

    out = _svg_open(width, height, f"{plot.dataset_id} — velas OHLC")
    for tick in ticks:
        y = py(tick)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')
    if plot.unit:
        out.append(f'<text x="{left - 8}" y="{top - 10}" font-size="11" fill="#777" text-anchor="end">{_esc(plot.unit)}</text>')

    for d, o in zip(isos, ords, strict=True):
        x = px(o)
        up = cl[d] >= op[d]
        color = _CANDLE_UP if up else _CANDLE_DOWN
        y_hi, y_lo = py(hi[d]), py(lo[d])
        y_top, y_bot = py(max(op[d], cl[d])), py(min(op[d], cl[d]))
        tip = (f"A {_fmt_num(op[d])} · M {_fmt_num(hi[d])} · "
               f"m {_fmt_num(lo[d])} · C {_fmt_num(cl[d])}")
        attrs = _tip_attrs(color, s="USD/CLP", k=_fmt_date(d), v=tip)
        out.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1"/>')
        # Cuerpo: alto mínimo de 1px para que un día sin recorrido (apertura =
        # cierre) siga siendo visible como el doji que es.
        out.append(
            f'<rect x="{x - body_w / 2:.1f}" y="{y_top:.1f}" width="{body_w:.1f}" '
            f'height="{max(1.0, y_bot - y_top):.1f}" fill="{color}"{attrs}/>'
        )

    # Ticks de fecha (hasta 5, equiespaciados sobre las velas dibujadas).
    n_ticks = min(5, len(isos))
    step_t = (len(isos) - 1) / max(1, n_ticks - 1)
    chosen = sorted({round(i * step_t) for i in range(n_ticks)})
    for j, k in enumerate(chosen):
        x = px(ords[k])
        anchor = "start" if j == 0 else ("end" if j == len(chosen) - 1 else "middle")
        out.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-size="11" fill="#555" text-anchor="{anchor}">{_esc(_fmt_date(isos[k]))}</text>')

    out.append(_legend_row([("Cierre al alza", _CANDLE_UP), ("Cierre a la baja", _CANDLE_DOWN)],
                           left, top + plot_h + 40, plot_w))
    out.append("</svg>")
    return "\n".join(out)


def _render_dual_axis(
    plot: PlotData, width: int, height: int,
    right_labels: list[str] | None, right_unit: str = "", right_style: str = "",
    *, right_invert: bool = False, left_style: str = "",
) -> str:
    """Serie temporal con DOBLE eje Y: las series cuyo label esté en
    ``right_labels`` se escalan contra un eje derecho independiente; el resto va
    contra el eje izquierdo. X compartido. Replica los gráficos con eje
    secundario sin separar el gráfico en dos.

    ``right_style``/``left_style`` deciden cómo se dibuja cada lado:

    - ``"area"`` → área tenue (gris a la derecha, celeste a la izquierda), estilo
      "AUM" del informe BCCh. Es lo correcto cuando esa serie es un VOLUMEN de
      fondo (monto transado, inventarios, AUM) que contextualiza a la principal
      sin competir con ella.
    - ``"line"`` → línea con su color de paleta. Es lo correcto cuando las dos
      series son magnitudes COMPARABLES que se leen a la par (CLP contra cobre,
      contra DXY, contra su propia posición offshore): pintar una de área sugiere
      una jerarquía que no existe.

    El DEFAULT de ``right_style`` sigue siendo ``"area"`` (compatibilidad con los
    bloques existentes); el de ``left_style`` es ``"line"`` (el eje izquierdo
    nunca se pintó de área hasta ahora).

    ``right_invert`` da vuelta el eje derecho (el valor más negativo queda ARRIBA
    en vez de abajo): réplica de "Posición cambiaria (eje inv. | eje der.)" del
    tablero — una serie negativa que se lee invertida para que se mueva
    visualmente CON la otra serie en vez de en espejo. Los ticks y el hover usan
    la MISMA proyección, así que quedan consistentes solos.

    Si no hay ninguna serie para el eje derecho, cae al render de línea normal."""
    right_as_line = (right_style or "area").strip().lower() == "line"
    left_as_area = (left_style or "line").strip().lower() == "area"
    right_set = set(right_labels or [])
    has_right = any(s.label in right_set for s in plot.series)
    if not has_right:
        return _render_timeseries(plot, width, height)

    left, right = 76, 66
    top, bottom = 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    # Parseo común (ord, valor) por serie, conservando el índice original (color).
    parsed: list[tuple[int, PlotSeries, list[tuple[float, float]], bool]] = []
    xs: list[float] = []
    iso_by_ord: dict[float, str] = {}
    lys: list[float] = []
    rys: list[float] = []
    for i, s in enumerate(plot.series):
        pts: list[tuple[float, float]] = []
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
    lticks, llo, lhi = _nice_ticks(min(lys), max(lys), include_zero=plot.zero_base)
    rticks, rlo, rhi = _nice_ticks(min(rys), max(rys), include_zero=plot.zero_base)

    def px(o: float) -> float:
        return left + (plot_w / 2 if xmax == xmin else plot_w * (o - xmin) / (xmax - xmin))

    def ply(v: float) -> float:
        return top + plot_h * (1 - (v - llo) / (lhi - llo or 1.0))

    def pry(v: float) -> float:
        frac = (v - rlo) / (rhi - rlo or 1.0)
        return top + plot_h * (frac if right_invert else (1 - frac))

    # JSON del hover unificado: cada serie con su propio eje (y por serie).
    by_date: dict[str, dict] = {}
    for idx, s, pts, is_right in parsed:
        if is_right and not right_as_line:
            color = "#b9c0cc"
        elif not is_right and left_as_area:
            color = _LEFT_AREA_COLOR
        else:
            color = _series_style(plot, idx, s.label)[0]
        unit = right_unit if is_right else plot.unit
        yfn = pry if is_right else ply
        for o, v in pts:
            iso = iso_by_ord.get(o) or _ord_iso(o)
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
            label = _fmt_date(iso_by_ord.get(o, _ord_iso(o)))
            anchor = "start" if j == 0 else ("end" if j == len(chosen) - 1 else "middle")
            out.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-size="11" fill="#555" text-anchor="{anchor}">{_esc(label)}</text>')

    out.append(
        f'<line class="x-guide" x1="{left:.1f}" y1="{top}" x2="{left:.1f}" y2="{top + plot_h}" '
        f'stroke="#aaa" stroke-width="1" stroke-dasharray="4,3" display="none" pointer-events="none"/>'
    )

    # Sufijo corto: la leyenda trunca a 22 caracteres (_legend_row) y "(eje inv. |
    # eje der.)" del tablero real no entra ni con series de nombre corto.
    right_suffix = " (eje inv.)" if right_invert else " (eje der.)"
    legend: list[tuple[int, str, str]] = []
    # Primero las áreas (al fondo), luego las líneas — de cualquiera de los dos ejes.
    def _is_area(t: tuple) -> bool:
        _idx, _s, _pts, is_r = t
        return (is_r and not right_as_line) or (not is_r and left_as_area)

    for idx, s, pts, is_right in sorted(parsed, key=lambda t: not _is_area(t)):
        if is_right and not right_as_line:
            base = top + plot_h
            up = " ".join(f"{px(o):.1f},{pry(v):.1f}" for o, v in pts)
            dn = f"{px(pts[-1][0]):.1f},{base:.1f} {px(pts[0][0]):.1f},{base:.1f}"
            out.append(f'<polygon points="{up} {dn}" fill="#c8ccd4" fill-opacity="0.55" stroke="none" data-si="{idx}"/>')
            legend.append((idx, f"{s.label}{right_suffix}", "#b9c0cc"))
        elif is_right:
            color, stroke_w = _series_style(plot, idx, s.label)
            coords = " ".join(f"{px(o):.1f},{pry(v):.1f}" for o, v in pts)
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{stroke_w}" data-si="{idx}"/>')
            legend.append((idx, f"{s.label}{right_suffix}", color))
        elif left_as_area:
            base = top + plot_h
            up = " ".join(f"{px(o):.1f},{ply(v):.1f}" for o, v in pts)
            dn = f"{px(pts[-1][0]):.1f},{base:.1f} {px(pts[0][0]):.1f},{base:.1f}"
            out.append(f'<polygon points="{up} {dn}" fill="{_LEFT_AREA_COLOR}" fill-opacity="0.55" stroke="none" data-si="{idx}"/>')
            legend.append((idx, s.label, _LEFT_AREA_COLOR))
        else:
            color, width = _series_style(plot, idx, s.label)
            coords = " ".join(f"{px(o):.1f},{ply(v):.1f}" for o, v in pts)
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" data-si="{idx}"/>')
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

    left, right = 64, 18
    top = 30
    plot_w = width - left - right
    # Fonts más grandes en bloques "wide" (card-wide de la grilla): compensa que
    # ese viewBox se estira a una tarjeta más angosta que el doble, ver
    # ``_WIDE_FONT_SCALE`` — así ejes y leyenda no se ven más chicos que en el
    # resto de los gráficos del informe.
    fscale = _WIDE_FONT_SCALE if width >= _W_WIDE else 1.0
    fs_axis = round(11 * fscale, 1)
    fs_cat = round(10 * fscale, 1)
    fs_unit = round(11 * fscale, 1)
    fs_legend = round(11 * fscale, 1)
    # Rotar la etiqueta si es larga O si NO CABE en el ancho de su categoría. Lo
    # segundo importa cuando hay muchas barras de nombre corto (37 monedas): sin
    # el chequeo de ancho ninguna superaba los 9 caracteres, no se rotaba, y las
    # etiquetas se pisaban unas con otras hasta ser ilegibles.
    max_len = max(len(c) for c in cats)
    fits = max_len * _CAT_LABEL_CHAR_W <= plot_w / len(cats) - 2
    long_labels = max_len > 9 or not fits
    bottom = 90 if long_labels else 80
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
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="{fs_axis}" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')

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
                f'font-size="{fs_cat}" fill="#555" text-anchor="end">{_esc(c)}</text>'
            )
        else:
            out.append(
                f'<text x="{cx:.1f}" y="{top + plot_h + 16}" '
                f'font-size="{fs_cat}" fill="#555" text-anchor="middle">{_esc(c)}</text>'
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
    out.append(_legend_row(legend, left, legend_y, plot_w, interactive=True, font_size=fs_legend))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="{fs_unit}" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _render_grouped_lines(plot: PlotData, width: int, height: int) -> str:
    """Curva sobre eje CATEGÓRICO: una línea por serie, un punto por categoría.

    Es la forma de las curvas del informe de renta fija (Curva BTP, Curva BTU,
    Breakeven, Swap Spread): el eje X no es el tiempo sino el PLAZO (2Y, 5Y, 10Y…)
    y se comparan dos cortes ("Hoy" contra "t-5"). Ni ``_render_timeseries`` (eje X
    de fechas) ni ``_render_grouped_bars`` (barras) sirven para eso.

    El eje Y NO arranca en 0: son niveles de tasa que se mueven en décimas, y
    forzar el 0 aplasta la curva contra el borde (misma razón por la que
    ``PlotData.zero_base`` existe).
    """
    series = [s for s in plot.series if s.label not in (plot.overlay or ())]
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

    left, right, top = 64, 18, 30
    plot_w = width - left - right
    fscale = _WIDE_FONT_SCALE if width >= _W_WIDE else 1.0
    fs_axis = round(11 * fscale, 1)
    fs_cat = round(10 * fscale, 1)
    fs_val = round(10 * fscale, 1)
    fs_unit = round(11 * fscale, 1)
    fs_legend = round(11 * fscale, 1)
    max_len = max(len(c) for c in cats)
    fits = max_len * _CAT_LABEL_CHAR_W <= plot_w / max(len(cats), 1) - 2
    long_labels = max_len > 9 or not fits
    bottom = 90 if long_labels else 80
    plot_h = height - top - bottom

    allv = [v for d in lut for v in d.values()]
    dmin, dmax = min(allv), max(allv)
    if dmax == dmin:
        dmax += 1.0
    ticks, ymin, ymax = _nice_ticks(dmin, dmax, include_zero=plot.zero_base)

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    # Los extremos van sobre la primera y la última categoría (no centrados en una
    # "banda" como las barras): una curva se lee de punta a punta.
    step = plot_w / max(len(cats) - 1, 1)

    def px(ci: int) -> float:
        return left + (ci * step if len(cats) > 1 else plot_w / 2)

    out = _svg_open(width, height, f"{plot.dataset_id} — curva")
    for tick in ticks:
        y = py(tick)
        emph = abs(tick) < 1e-9
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                   f'stroke="{"#999" if emph else "#e6e6e6"}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="{fs_axis}" fill="#555" '
                   f'text-anchor="end">{_esc(_fmt_num(tick))}</text>')

    pal = _PALETTE
    for si, s in enumerate(series):
        color = pal[si % len(pal)]
        pts = [(ci, lut[si][c]) for ci, c in enumerate(cats) if c in lut[si]]
        if len(pts) > 1:
            coords = " ".join(f"{px(ci):.1f},{py(v):.1f}" for ci, v in pts)
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                       f'stroke-width="2" data-si="{si}"/>')
        for ci, v in pts:
            attrs = _tip_attrs(color, s=s.label, k=cats[ci], v=_val_unit(v, plot.unit))
            out.append(f'<circle cx="{px(ci):.1f}" cy="{py(v):.1f}" r="3.2" fill="{color}" '
                       f'data-si="{si}"{attrs}/>')
        # El valor rotulado sobre la PRIMERA serie, como el correo real (la curva de
        # hoy lleva su nivel escrito; la de comparación va limpia para no amontonar).
        # Los extremos se anclan al borde en vez de centrarse: una etiqueta centrada
        # sobre el primer o el último plazo se sale del gráfico y queda cortada.
        if si == 0:
            for ci, v in pts:
                if ci == 0:
                    x, anchor = px(ci) + 2, "start"
                elif ci == len(cats) - 1:
                    x, anchor = px(ci) - 2, "end"
                else:
                    x, anchor = px(ci), "middle"
                y = max(top + fs_val, py(v) - 8)   # nunca por encima del área de dibujo
                out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs_val}" '
                           f'fill="#333" text-anchor="{anchor}">{_esc(_fmt_num(v))}</text>')

    for ci, c in enumerate(cats):
        cx = px(ci)
        if long_labels:
            out.append(f'<text transform="translate({cx:.1f},{top + plot_h + 12:.1f}) rotate(-45)" '
                       f'font-size="{fs_cat}" fill="#555" text-anchor="end">{_esc(c)}</text>')
        else:
            out.append(f'<text x="{cx:.1f}" y="{top + plot_h + 16}" font-size="{fs_cat}" '
                       f'fill="#555" text-anchor="middle">{_esc(c)}</text>')

    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    legend_y = height - (20 if long_labels else 26)
    legend = [(s.label, pal[i % len(pal)]) for i, s in enumerate(series)]
    out.append(_legend_row(legend, left, legend_y, plot_w, interactive=True, font_size=fs_legend))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="{fs_unit}" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "".join(out)


def _render_grouped_dots(plot: PlotData, width: int, height: int) -> str:
    """Dispersión CATEGÓRICA: eje X = categorías compartidas (no numérico), eje Y
    = valor, un color por serie — varios puntos por categoría (uno por serie que
    tenga dato ahí), SIN barras. Réplica de "Duración agentes IIF/RF" del informe
    DCV: instrumento en X, duración en Y, un punto por agente que sostiene ese
    instrumento (no todos lo hacen, a diferencia de las barras agrupadas donde
    toda categoría suma sus series). Comparte el layout de eje/categorías con
    ``_render_grouped_bars`` pero sin apilado ni negativos."""
    cats: list[str] = []
    seen: set[str] = set()
    for s in plot.series:
        for c, _v in s.points:
            if c not in seen:
                seen.add(c)
                cats.append(c)
    if not cats or not plot.series:
        return _no_axis_message(plot, width, height)
    lut = [{c: v for c, v in s.points} for s in plot.series]

    left, right = 64, 18
    top = 30
    plot_w = width - left - right
    max_len = max(len(c) for c in cats)
    fits = max_len * _CAT_LABEL_CHAR_W <= plot_w / len(cats) - 2
    long_labels = max_len > 9 or not fits
    bottom = 90 if long_labels else 80
    plot_h = height - top - bottom

    allv = [lut[i][c] for i in range(len(plot.series)) for c in cats if c in lut[i]]
    if not allv:
        return _no_axis_message(plot, width, height)
    dmax, dmin = max([0.0, *allv]), min([0.0, *allv])
    if dmax == dmin:
        dmax += 1.0
    ticks, ymin, ymax = _nice_ticks(dmin, dmax)

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    group_w = plot_w / len(cats)
    out = _svg_open(width, height, f"{plot.dataset_id} — dispersión categórica")

    for tick in ticks:
        y = py(tick)
        emph = abs(tick) < 1e-9
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{"#999" if emph else "#e6e6e6"}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')

    n = len(plot.series)
    for ci, c in enumerate(cats):
        gx = left + ci * group_w
        inner = group_w * 0.7
        step = inner / max(1, n - 1) if n > 1 else 0.0
        x0 = gx + (group_w - inner) / 2 if n > 1 else gx + group_w / 2
        for si, s in enumerate(plot.series):
            v = lut[si].get(c)
            if v is None:
                continue
            cx = x0 + si * step
            col = _PALETTE[si % len(_PALETTE)]
            attrs = _tip_attrs(col, s=s.label, k=c, v=_val_unit(v, plot.unit))
            out.append(f'<circle cx="{cx:.1f}" cy="{py(v):.1f}" r="4" fill="{col}" stroke="#fff" stroke-width="1" data-si="{si}"{attrs}/>')
        cx_lbl = left + ci * group_w + group_w / 2
        if long_labels:
            out.append(
                f'<text transform="translate({cx_lbl:.1f},{top + plot_h + 12:.1f}) rotate(-45)" '
                f'font-size="10" fill="#555" text-anchor="end">{_esc(c)}</text>'
            )
        else:
            out.append(
                f'<text x="{cx_lbl:.1f}" y="{top + plot_h + 16}" '
                f'font-size="10" fill="#555" text-anchor="middle">{_esc(c)}</text>'
            )

    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')

    legend_y = height - (20 if long_labels else 26)
    legend = [(s.label, _PALETTE[i % len(_PALETTE)]) for i, s in enumerate(plot.series)]
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

    def px(o: float) -> float:
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


_TREEMAP_HEADER_H = 18.0
_TREEMAP_HEADER_FILL = "#1f2c3d"
_TREEMAP_GAP = 1.5  # separación entre rectángulos hermanos (blanco de fondo)


def _fit_label(text: str, box_w: float, font_px: float) -> str | None:
    """``text`` si entra en ``box_w`` a ``font_px``, truncado con "…", o ``None``
    si ni una letra entra. Estimación (no medición real: SVG no la da sin
    layout del navegador) — 0.56×``font_px`` por carácter, Arial/Helvetica
    minúscula-mayúscula mixta, harto probada en las demás tarjetas del informe."""
    max_chars = int(box_w / (font_px * 0.56))
    if max_chars < 2:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "…"


def _render_treemap(plot: PlotData, width: int, height: int) -> str:
    """Treemap de dos niveles: exterior = ``series[i].label`` (p.ej. división),
    interior = sus ``points`` (p.ej. grupo de productos). Área ∝ tamaño
    (``points`` value, p.ej. ponderación); color = ``plot.node_color`` en escala
    DIVERGENTE centrada en 0 (p.ej. variación mensual) — réplica en SVG puro del
    ``px.treemap(..., color_continuous_scale='RdBu_r')`` del informe IPC
    original, sin llegar al nivel de producto individual (ver nota del spec).

    Layout: ``_squarify`` reparte el lienzo entre los grupos EXTERIORES; cada
    uno lleva una franja de título (fondo oscuro) y, debajo, sus rectángulos
    INTERIORES —a su vez squarified— coloreados por ``node_color``. Un exterior
    cuya franja no entra (rectángulo muy chico) se dibuja sin título: el dato
    sigue en el tooltip.
    """
    height = max(height, 420)  # 320 (default _H) deja los grupos ilegibles
    outer = [(s.label, sum(v for _c, v in s.points)) for s in plot.series if s.points]
    if not outer:
        return _no_axis_message(plot, width, height)

    # Cap simétrico de la escala de color: el máximo |valor| observado, con un
    # piso de 0.5 para que un mes sin ninguna variación relevante no infle el
    # rango a algo minúsculo y sature todo el mapa en rojo/azul intensos.
    all_colors = [c for cs in plot.node_color.values() for c in cs]
    cap = max([abs(c) for c in all_colors] + [0.5]) if all_colors else 0.5

    top, bottom = 10, 34  # `bottom`: leyenda de la escala divergente
    canvas_h = height - top - bottom
    outer_rects = _squarify([v for _l, v in outer], 0.0, float(top), float(width), float(canvas_h))

    out = _svg_open(width, height, f"{plot.dataset_id} — treemap")
    for (label, _total), (ox, oy, ow, oh) in zip(outer, outer_rects):
        series = next(s for s in plot.series if s.label == label)
        colors = plot.node_color.get(label, [])
        has_header = oh > _TREEMAP_HEADER_H + 24 and ow > 30
        header_h = _TREEMAP_HEADER_H if has_header else 0.0
        ix, iy = ox + _TREEMAP_GAP, oy + header_h + (_TREEMAP_GAP if has_header else 0.0)
        iw = max(0.0, ow - 2 * _TREEMAP_GAP)
        ih = max(0.0, oh - header_h - (2 * _TREEMAP_GAP if has_header else _TREEMAP_GAP))

        inner_rects = _squarify([v for _c, v in series.points], ix, iy, iw, ih)
        for idx, ((leaf, size), (lx, ly, lw, lh)) in enumerate(zip(series.points, inner_rects)):
            if lw <= 0 or lh <= 0:
                continue
            cval = colors[idx] if idx < len(colors) else 0.0
            r, g, b = _diverging_color(cval / cap if cap else 0.0)
            fill = f"rgb({r},{g},{b})"
            ink = "#fff" if (r * 0.299 + g * 0.587 + b * 0.114) < 140 else "#1c1c1c"
            tip_v = f"{_val_unit(size, plot.unit)}"
            if plot.node_color_unit:
                tip_v += f" · {_val_unit(cval, plot.node_color_unit)}"
            attrs = _tip_attrs(fill, s=label, k=leaf, v=tip_v)
            out.append(
                f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{max(0.0, lw - _TREEMAP_GAP):.1f}" '
                f'height="{max(0.0, lh - _TREEMAP_GAP):.1f}" fill="{fill}" stroke="#fff" '
                f'stroke-width="0.5"{attrs}/>'
            )
            fs = 10.5
            if lw > 46 and lh > 16:
                lbl = _fit_label(leaf, lw - 8, fs)
                if lbl:
                    out.append(
                        f'<text x="{lx + lw / 2:.1f}" y="{ly + lh / 2 + 3.5:.1f}" font-size="{fs}" '
                        f'fill="{ink}" text-anchor="middle" pointer-events="none">{_esc(lbl)}</text>'
                    )

        if has_header:
            out.append(
                f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{max(0.0, ow - _TREEMAP_GAP):.1f}" '
                f'height="{_TREEMAP_HEADER_H:.1f}" fill="{_TREEMAP_HEADER_FILL}"/>'
            )
            hdr = _fit_label(label, ow - 10, 11)
            if hdr:
                out.append(
                    f'<text x="{ox + 5:.1f}" y="{oy + _TREEMAP_HEADER_H - 5:.1f}" font-size="11" '
                    f'font-weight="700" fill="#fff" pointer-events="none">{_esc(hdr)}</text>'
                )

    # Leyenda: escala divergente horizontal, -cap .. 0 .. +cap.
    leg_x, leg_y, leg_w, leg_h = 12.0, height - bottom + 14, min(220.0, width * 0.3), 10.0
    steps = 24
    for i in range(steps):
        frac = -1.0 + 2.0 * i / (steps - 1)
        r, g, b = _diverging_color(frac)
        seg_w = leg_w / steps
        out.append(f'<rect x="{leg_x + i * seg_w:.1f}" y="{leg_y:.1f}" width="{seg_w + 0.5:.1f}" '
                  f'height="{leg_h:.1f}" fill="rgb({r},{g},{b})"/>')
    unidad = plot.node_color_unit or plot.unit
    for frac, texto in ((-1.0, f"-{_fmt_num(cap)}{unidad}"), (0.0, "0"), (1.0, f"+{_fmt_num(cap)}{unidad}")):
        tx = leg_x + (frac + 1.0) / 2.0 * leg_w
        anchor = "start" if frac < -0.5 else ("end" if frac > 0.5 else "middle")
        out.append(f'<text x="{tx:.1f}" y="{leg_y + leg_h + 12:.1f}" font-size="9.5" fill="#666" '
                  f'text-anchor="{anchor}">{_esc(texto)}</text>')
    out.append(f'<text x="{leg_x + leg_w + 12:.1f}" y="{leg_y + leg_h - 1:.1f}" font-size="9.5" fill="#999">'
              f'área = {_esc(plot.unit or "tamaño")}</text>')
    out.append("</svg>")
    return "\n".join(out)


_RANGE_BOX_FILL = "#cfe0ef"
_RANGE_BOX_STROKE = "#7ea3c9"
_RANGE_MEAN_COLOR = "#8a2b2b"
_RANGE_HOY_COLOR = "#0b3766"


def _render_range_band(plot: PlotData, width: int, height: int) -> str:
    """Caja de rango [Mínimo,Máximo] por categoría + tick de Promedio + punto "Hoy"
    con su valor etiquetado (réplica de "Rendimiento monedas" del tablero GBI:
    índice rebasado a 100, con su rango histórico de la ventana). Requiere series
    con esas 4 etiquetas EXACTAS (las arma la transform, p.ej. ``gbi_rendimiento_range``);
    si falta Mínimo/Máximo cae al mensaje "sin serie"."""
    by_label = {s.label: dict(s.points) for s in plot.series}
    lo_s, hi_s = by_label.get("Mínimo"), by_label.get("Máximo")
    if not lo_s or not hi_s:
        return _no_axis_message(plot, width, height)
    mean_s = by_label.get("Promedio", {})
    hoy_s = by_label.get("Hoy", {})
    # Orden de categorías: el de la propia serie "Mínimo" (la transform ya la deja
    # en el orden pedido por ``params['order']``, igual que _render_grouped_bars).
    cats = [c for s in plot.series if s.label == "Mínimo" for c, _v in s.points]
    if not cats:
        return _no_axis_message(plot, width, height)

    long_labels = any(len(c) > 9 for c in cats)
    left, right = 64, 18
    top = 40  # espacio extra: la etiqueta del valor "Hoy" se dibuja SOBRE el punto
    bottom = 90 if long_labels else 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    allv = [lo_s[c] for c in cats] + [hi_s[c] for c in cats]
    allv += [mean_s[c] for c in cats if c in mean_s] + [hoy_s[c] for c in cats if c in hoy_s]
    ticks, ymin, ymax = _nice_ticks(min(allv), max(allv), include_zero=False)

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / (ymax - ymin))

    group_w = plot_w / len(cats)
    out = _svg_open(width, height, f"{plot.dataset_id} — rango histórico")

    for tick in ticks:
        y = py(tick)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')

    for ci, c in enumerate(cats):
        gx = left + ci * group_w
        bw = group_w * 0.55
        x = gx + (group_w - bw) / 2
        y_top, y_bot = py(hi_s[c]), py(lo_s[c])
        attrs = _tip_attrs(_RANGE_BOX_STROKE, s="Rango histórico", k=c,
                           v=f"{_val_unit(lo_s[c], plot.unit)} - {_val_unit(hi_s[c], plot.unit)}")
        out.append(
            f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bw:.1f}" height="{max(1.0, y_bot - y_top):.1f}" '
            f'fill="{_RANGE_BOX_FILL}" stroke="{_RANGE_BOX_STROKE}" stroke-width="1" data-si="0"{attrs}/>'
        )
        if c in mean_s:
            ym = py(mean_s[c])
            attrs = _tip_attrs(_RANGE_MEAN_COLOR, s="Promedio", k=c, v=_val_unit(mean_s[c], plot.unit))
            out.append(f'<line x1="{x:.1f}" y1="{ym:.1f}" x2="{x + bw:.1f}" y2="{ym:.1f}" stroke="{_RANGE_MEAN_COLOR}" stroke-width="2.4" data-si="1"{attrs}/>')
        if c in hoy_s:
            v = hoy_s[c]
            yh = py(v)
            cx = gx + group_w / 2
            attrs = _tip_attrs(_RANGE_HOY_COLOR, s="Hoy", k=c, v=_val_unit(v, plot.unit))
            out.append(f'<circle cx="{cx:.1f}" cy="{yh:.1f}" r="4.5" fill="{_RANGE_HOY_COLOR}" stroke="#fff" stroke-width="1" data-si="2"{attrs}/>')
            label_y = yh - 9 if yh - top > 14 else yh + 16
            out.append(f'<text x="{cx:.1f}" y="{label_y:.1f}" font-size="11" fill="{_RANGE_HOY_COLOR}" font-weight="700" text-anchor="middle">{_esc(_fmt_num(v))}</text>')
        cx = gx + group_w / 2
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
    legend = [("Rango histórico", _RANGE_BOX_FILL), ("Promedio", _RANGE_MEAN_COLOR), ("Hoy", _RANGE_HOY_COLOR)]
    out.append(_legend_row(legend, left, legend_y, plot_w))
    if plot.unit:
        out.append(f'<text x="{left}" y="{top - 12}" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


_SCATTER_POINT_COLOR = "#1f9bcf"
_SCATTER_HIGHLIGHT_COLOR = "#0b3766"


def _render_scatter_labeled(plot: PlotData, width: int, height: int, *, x_label: str = "", y_label: str = "") -> str:
    """Dispersión x/y etiquetada por punto, con cruce en 0 (réplica de "Retorno FX
    y tasas GBI"): cada ``PlotSeries`` es UN punto — ``label`` = etiqueta de texto,
    ``points[0] = (str(x), y)`` (convención de ``kind='scatter'``, ver
    ``parquet_facts.PlotSeries``). Las etiquetas en ``plot.overlay`` se dibujan
    destacadas (relleno), el resto huecas."""
    pts: list[tuple[str, float, float]] = []
    for s in plot.series:
        if not s.points:
            continue
        x_raw, y = s.points[0]
        try:
            x = float(x_raw)
        except (TypeError, ValueError):
            continue
        pts.append((s.label, x, y))
    if not pts:
        return _no_axis_message(plot, width, height)

    left, right = 60, 40
    top, bottom = 30, 56
    plot_w = width - left - right
    plot_h = height - top - bottom

    xticks, xmin, xmax = _nice_ticks(min(p[1] for p in pts), max(p[1] for p in pts), n=6)
    yticks, ymin, ymax = _nice_ticks(min(p[2] for p in pts), max(p[2] for p in pts), n=5)

    def px(v: float) -> float:
        return left + plot_w * (v - xmin) / ((xmax - xmin) or 1.0)

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - ymin) / ((ymax - ymin) or 1.0))

    out = _svg_open(width, height, f"{plot.dataset_id} — dispersión")

    for tick in yticks:
        y = py(tick)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{_esc(_fmt_num(tick))}</text>')
    for tick in xticks:
        x = px(tick)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f4f4f4" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top + plot_h + 16}" font-size="10" fill="#555" text-anchor="middle">{_esc(_fmt_num(tick))}</text>')

    # Cruce en 0 (cuadrantes): guías punteadas si el 0 cae dentro del dominio.
    if ymin <= 0 <= ymax:
        y0 = py(0.0)
        out.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{left + plot_w}" y2="{y0:.1f}" stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>')
    if xmin <= 0 <= xmax:
        x0 = px(0.0)
        out.append(f'<line x1="{x0:.1f}" y1="{top}" x2="{x0:.1f}" y2="{top + plot_h}" stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#999" stroke-width="1"/>')

    highlight = set(plot.overlay or ())
    for label, x, y in pts:
        cx, cy = px(x), py(y)
        is_hi = label in highlight
        color = _SCATTER_HIGHLIGHT_COLOR if is_hi else _SCATTER_POINT_COLOR
        fill = color if is_hi else "white"
        attrs = _tip_attrs(color, s=label, v=f"x={_val_unit(x, plot.unit)}  y={_val_unit(y, plot.unit)}")
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{fill}" stroke="{color}" stroke-width="1.6"{attrs}/>')
        out.append(f'<text x="{cx + 7:.1f}" y="{cy - 6:.1f}" font-size="10" fill="#333">{_esc(label)}</text>')

    xl = x_label or plot.unit
    yl = y_label or plot.unit
    if xl:
        out.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" font-size="11" fill="#777" text-anchor="middle">{_esc(xl)}</text>')
    if yl:
        out.append(
            f'<text x="14" y="{top + plot_h / 2:.1f}" font-size="11" fill="#777" text-anchor="middle" '
            f'transform="rotate(-90,14,{top + plot_h / 2:.1f})">{_esc(yl)}</text>'
        )
    if highlight:
        out.append(_legend_row([("Comparables", _SCATTER_POINT_COLOR), ("Hoy", _SCATTER_HIGHLIGHT_COLOR)], left, height - 8, plot_w))
    out.append("</svg>")
    return "\n".join(out)


def _legend_row(
    items: list[tuple[str, str]], x0: int, y: float, max_w: int, *,
    interactive: bool = False, font_size: float = 11,
) -> str:
    """Fila de swatches + etiquetas; envuelve a una segunda línea si no caben.

    ``interactive=True`` envuelve cada ítem en ``<g class="lg-item" data-idx="i">``
    para el toggle click-to-hide de series (el JS escucha los clics en el grupo).
    ``font_size`` escala TODA la fila (swatch, separaciones, alto de línea) en el
    mismo factor que el texto — lo usa ``_render_grouped_bars`` para agrandar la
    leyenda en bloques ``wide`` (ver ``_WIDE_FONT_SCALE``), sin desarmar el layout."""
    k = font_size / 11
    swatch, step, dx = 10 * k, 16 * k, 14 * k
    parts: list[str] = []
    x = x0
    line = 0
    for idx, (label, color) in enumerate(items):
        label_s = label if len(label) <= 22 else label[:21] + "…"
        w = 16 * k + len(label_s) * 6.2 * k + 14 * k
        if x + w > x0 + max_w and x > x0:
            line += 1
            x = x0
        yy = y + line * step
        if interactive:
            parts.append(f'<g class="lg-item" data-idx="{idx}" style="cursor:pointer">')
            # Área transparente más ancha que swatch+texto para facilitar el click.
            parts.append(f'<rect x="{x - 2:.1f}" y="{yy - 10 * k:.1f}" width="{w:.0f}" height="{18 * k:.0f}" fill="transparent" stroke="none"/>')
        parts.append(f'<rect x="{x:.1f}" y="{yy - swatch * 0.8:.1f}" width="{swatch:.1f}" height="{swatch:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + dx:.1f}" y="{yy:.1f}" font-size="{font_size:.1f}" fill="#444">{_esc(label_s)}</text>')
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
    label7: str = "Δ T-7",
    label30: str = "Δ T-30",
) -> str:
    """Dos matrices heatmap (Delta T-7 / Delta T-30, variación a 1 semana / 1 mes):
    filas=instrumento, cols=plazo, celdas coloreadas verde (positivo) / rojo
    (negativo). ``label7``/``label30`` rotulan cada matriz (incluyen el span de
    fechas que considera el delta)."""

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
        + _matrix(label7, delta7)
        + _matrix(label30, delta30)
        + unit_note + "</div>"
    )


# ── Tablas del informe de Flujos Cambiarios ──────────────────────────────────
#
# El correo real colorea la celda cuando el flujo es GRANDE en términos absolutos
# (verde = venta de dólares / flujo negativo, rojo = compra), no cuando es
# simplemente distinto de cero: con ~13 sectores por 6 columnas, colorear todo deja
# la tabla ilegible. Se resalta el decil superior por |monto| de cada columna.

_HL_THRESHOLD = 0.80  # percentil de |monto| sobre el que se pinta la celda


def _fx_cut(values: list[float]) -> float:
    """Umbral de resaltado: |monto| del percentil ``_HL_THRESHOLD`` de la columna.
    Devuelve ``inf`` si no hay datos (nada se resalta)."""
    mags = sorted(abs(v) for v in values if v)
    if not mags:
        return float("inf")
    return mags[min(len(mags) - 1, int(len(mags) * _HL_THRESHOLD))]


def _fx_cell(v: float, cut: float, *, bold: bool = False) -> str:
    tdr = "text-align:right;padding:4px 8px;border-bottom:1px solid #eee;font-size:12px;white-space:nowrap"
    if bold:
        tdr += ";font-weight:700"
    if v and abs(v) >= cut:
        tdr += ";" + (_CELL_NEG if v > 0 else _CELL_POS)
    return f'<td style="{tdr}">{_fmt_num(v)}</td>'


def render_fx_summary_table(
    sectors: list[str],
    data: dict[str, tuple[float, float, float, float, float, float, float, float]],
    *,
    unit: str = "US$ Mill.",
    split_before: str = "BANCOS",
) -> str:
    """RESUMEN GENERAL: Spot (No afecto/Afecto) y Derivados (NDF/Resto) por sector,
    día y acumulado de 5 días, con su columna de consolidación cada uno.

    ``data[sector] = (no_afecto, afecto, ndf, resto,
                       no_afecto_5d, afecto_5d, ndf_5d, resto_5d)``; las columnas
    "Spot y derivados" (día y 5 días) se calculan acá. ``split_before`` es la fila
    (p.ej. "BANCOS") que va después de una línea punteada, como en el correo."""
    th = "text-align:right;padding:6px 8px;background:#4a5a72;color:#fff;font-size:11px;white-space:nowrap"
    thl = "text-align:left;padding:6px 8px;background:#4a5a72;color:#fff;font-size:11px"
    thg = "text-align:center;padding:5px 8px;background:#0b3766;color:#fff;font-size:11px;white-space:nowrap"
    tdl = "text-align:left;padding:4px 8px;border-bottom:1px solid #eee;font-size:12px;font-weight:700"
    tdl_split = tdl + ";border-top:2px dashed #4a5a72"

    def _expand(d: tuple[float, ...]) -> tuple[float, ...]:
        no_af, af, ndf, resto, no_af5, af5, ndf5, resto5 = d
        return (no_af, af, ndf, resto, no_af + af + ndf + resto,
                no_af5, af5, ndf5, resto5, no_af5 + af5 + ndf5 + resto5)

    rows_num = {s: _expand(d) for s, d in data.items()}
    bold_cols = {4, 9}
    cuts = [_fx_cut([rows_num[s][i] for s in sectors]) for i in range(10)]

    body = ""
    for s in sectors:
        vals = rows_num[s]
        cells = "".join(_fx_cell(v, cuts[i], bold=i in bold_cols) for i, v in enumerate(vals))
        label_style = tdl_split if s == split_before else tdl
        body += f'<tr><td style="{label_style}">{_esc(s)}</td>{cells}</tr>'

    return (
        '<div style="overflow-x:auto;max-width:1100px;margin:6px auto">'
        '<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr><th style="{thl}" rowspan="2">Sector</th>'
        f'<th style="{thg}" colspan="2">Spot</th><th style="{thg}" colspan="2">Derivados</th>'
        f'<th style="{thg}">Spot y derivados</th>'
        f'<th style="{thg}" colspan="2">Spot 5 días</th><th style="{thg}" colspan="2">Derivados 5 días</th>'
        f'<th style="{thg}">Spot y derivados 5 días</th></tr>'
        f'<tr><th style="{th}">No afecto</th><th style="{th}">Afecto</th>'
        f'<th style="{th}">NDF</th><th style="{th}">Resto</th><th style="{th}"></th>'
        f'<th style="{th}">No afecto</th><th style="{th}">Afecto</th>'
        f'<th style="{th}">NDF</th><th style="{th}">Resto</th><th style="{th}"></th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
        f'<div style="font-size:11px;color:#777;margin:4px 0 0">{_esc(unit)} · '
        " </div></div>"
    )


def render_fx_delta_table(
    agents: list[str],
    data: dict[str, tuple[float, ...]],
    windows: list[int],
    *,
    unit: str = "US$ Mill.",
    total_label: str = "Total",
    asof: str = "",
    title: str = "",
    note: str = "variación neta acumulada de las últimas N jornadas con dato",
    data_spot: dict[str, tuple[float, ...]] | None = None,
    spot_note: str = "monto acumulado de las últimas N jornadas con dato",
) -> str:
    """Posición por agente: filas = agente, columnas = Δ T-N. Si ``data_spot`` viene
    dado, Spot y Derivados salen en UNA sola tabla (grupos de columnas, igual que
    ``render_fx_summary_table``) en vez de dos tablas apiladas."""
    th = "text-align:right;padding:6px 8px;background:#4a5a72;color:#fff;font-size:11px;white-space:nowrap"
    thl = "text-align:left;padding:6px 8px;background:#4a5a72;color:#fff;font-size:11px"
    thg = "text-align:center;padding:5px 8px;background:#0b3766;color:#fff;font-size:11px;white-space:nowrap"
    tdl = "text-align:left;padding:4px 8px;border-bottom:1px solid #eee;font-size:12px;font-weight:700"
    tdt = ("text-align:right;padding:6px 8px;font-size:12px;font-weight:700;"
           "background:#eef3f9;border-top:2px solid #4a5a72;white-space:nowrap")
    n = len(windows)
    cuts = [_fx_cut([data[a][i] for a in agents]) for i in range(n)]
    asof_note = f" · corte {_esc(asof)}" if asof else ""

    if data_spot is not None:
        cuts_spot = [_fx_cut([data_spot[a][i] for a in agents]) for i in range(n)]
        body = ""
        for a in agents:
            spot_cells = "".join(_fx_cell(v, cuts_spot[i]) for i, v in enumerate(data_spot[a]))
            deriv_cells = "".join(_fx_cell(v, cuts[i]) for i, v in enumerate(data[a]))
            body += f'<tr><td style="{tdl}">{_esc(a)}</td>{spot_cells}{deriv_cells}</tr>'
        body += (
            f'<tr><td style="{tdl};background:#eef3f9;border-top:2px solid #4a5a72">{_esc(total_label)}</td>'
            + "".join(f'<td style="{tdt}">{_fmt_num(v)}</td>' for v in data_spot.get(total_label, ()))
            + "".join(f'<td style="{tdt}">{_fmt_num(v)}</td>' for v in data.get(total_label, ()))
            + "</tr>"
        )
        deriv_label = title or "Derivados"
        return (
            '<div style="overflow-x:auto;max-width:1300px;margin:6px auto">'
            '<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr><th style="{thl}" rowspan="2">Agente</th>'
            f'<th style="{thg}" colspan="{n}">Spot</th>'
            f'<th style="{thg}" colspan="{n}">{_esc(deriv_label)}</th></tr>'
            '<tr>' + ("".join(f'<th style="{th}">&#916; T-{w}</th>' for w in windows) * 2) + '</tr></thead>'
            f"<tbody>{body}</tbody></table>"
            f'<div style="font-size:11px;color:#777;margin:4px 0 0">{_esc(unit)} · '
            f"Spot: {spot_note} · {deriv_label}: {note}{asof_note}</div></div>"
        )

    # ── sin Spot: tabla única (comportamiento de siempre) ──
    body = ""
    for a in agents:
        cells = "".join(_fx_cell(v, cuts[i]) for i, v in enumerate(data[a]))
        body += f'<tr><td style="{tdl}">{_esc(a)}</td>{cells}</tr>'
    body += (
        f'<tr><td style="{tdl};background:#eef3f9;border-top:2px solid #4a5a72">{_esc(total_label)}</td>'
        + "".join(f'<td style="{tdt}">{_fmt_num(v)}</td>' for v in data.get(total_label, ()))
        + "</tr>"
    )
    title_html = (
        f'<div style="font-size:12px;font-weight:700;margin:10px 0 2px">{_esc(title)}</div>' if title else ""
    )
    return (
        f'{title_html}'
        '<div style="overflow-x:auto;max-width:760px;margin:6px auto">'
        '<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr><th style="{thl}">Agente</th>'
        + "".join(f'<th style="{th}">&#916; T-{w}</th>' for w in windows)
        + "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f'<div style="font-size:11px;color:#777;margin:4px 0 0">{_esc(unit)} · '
        f"{note}{asof_note}</div></div>"
    )

# ── Tablas del informe DCV (Stocks Depósito Central de Valores) ──────────────
#
# El correo real pinta los montos con una escala CONTINUA verde/blanco: cuanto
# mayor el monto dentro de su columna, más saturada la celda. No es un semáforo
# de signo (los stocks son siempre ≥ 0) sino un mapa de calor de concentración,
# así se lee de un vistazo dónde está cargado cada agente. Las filas/columnas de
# TOTAL quedan fuera de la escala (si entraran, dominarían y el resto saldría
# blanco) — el original anota justamente "sin considerar montos totales".

_DCV_GREEN = (99, 190, 123)  # tono base de la escala verde del correo


def _dcv_shade(value: float | None, vmax: float) -> str:
    """Fondo verde proporcional a ``value/vmax`` (0 → blanco, vmax → verde pleno).

    Se usa raíz cuadrada para que los montos medianos no se vean casi blancos:
    en estas tablas unas pocas celdas concentran el stock y una escala lineal
    dejaría ilegible al resto."""
    if not value or value <= 0 or vmax <= 0:
        return ""
    frac = min(1.0, (value / vmax) ** 0.5)
    r, g, b = _DCV_GREEN
    # Mezcla contra blanco en vez de usar alpha: los clientes de correo (Outlook)
    # no interpolan rgba() sobre el fondo de la celda de forma confiable.
    mix = tuple(round(255 - (255 - c) * frac) for c in (r, g, b))
    color = "#155724" if frac > 0.55 else "#1c1c1c"
    return f"background:rgb({mix[0]},{mix[1]},{mix[2]});color:{color}"


def _dcv_num(v: float | None) -> str:
    """Monto de la tabla DCV: ``—`` cuando no hay dato y ``-`` cuando es cero
    (igual que el original, que deja el guion para las celdas sin tenencia)."""
    if v is None:
        return "&#8212;"
    if abs(v) < 0.5:
        return "-"
    return _fmt_num(v)


def _dcv_dur(v: float | None) -> str:
    """Duración de la tabla DCV: SIEMPRE 2 decimales (coma chilena), a
    diferencia de ``_fmt_num`` (decimales adaptativos según magnitud) — el
    original muestra ``51,25``/``87,46`` con 2 decimales aunque el valor sea
    grande. ``—`` cuando no hay dato (instrumento sin duración calculable)."""
    if v is None:
        return "&#8212;"
    s = f"{v:,.2f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _weighted_dur(pairs: list[tuple[float, float | None]]) -> float | None:
    """Duración ponderada por Monto sobre pares ``(monto, duracion)``: la
    duración NO es sumable como el monto, así que el total de una fila/columna
    es un promedio ponderado, no una suma. ``None`` si no hay ningún par con
    ambos datos (monto>0 y duración conocida)."""
    num = den = 0.0
    for m, d in pairs:
        if d is None or not m:
            continue
        num += m * d
        den += m
    return (num / den) if den else None


def render_dcv_portfolio_table(
    instruments: list[str],
    agents: list[str],
    monto: dict[str, dict[str, float]],
    *,
    unit: str = "US$ Mill.",
    asof: str = "",
    total_label: str = "Total",
    duracion: dict[str, dict[str, float]] | None = None,
) -> str:
    """Portafolio por agente: filas = instrumento, y por cada agente dos o tres
    columnas (Monto, Duración si hay dato, y % del portafolio de ESE agente),
    más una columna de total.

    ``monto[agente][instrumento]`` en la unidad del dataset. El ``%`` se calcula
    acá sobre el total de la columna, así porcentaje y monto nunca se contradicen.
    ``duracion[agente][instrumento]`` es OPCIONAL (``None`` → sin columna de
    duración, comportamiento histórico intacto): sale de parquets aparte (ver
    ``dcv_spec``). La duración de la fila/columna Total es un PROMEDIO ponderado
    por Monto (``_weighted_dur``), no una suma — duración no es aditiva."""
    th = "text-align:right;padding:5px 6px;background:#4a5a72;color:#fff;font-size:11px;white-space:nowrap"
    thl = "text-align:left;padding:5px 7px;background:#4a5a72;color:#fff;font-size:11px"
    thg = "text-align:center;padding:4px 6px;background:#0b3766;color:#fff;font-size:11px;white-space:nowrap"
    tdl = "text-align:left;padding:4px 7px;border-bottom:1px solid #eee;font-size:11px;font-weight:700"
    tdr = "text-align:right;padding:4px 6px;border-bottom:1px solid #eee;font-size:11px;white-space:nowrap"

    cols = [*agents, total_label]
    totals = {a: sum(monto.get(a, {}).values()) for a in agents}
    totals[total_label] = sum(totals.values())
    for inst in instruments:
        monto.setdefault(total_label, {})[inst] = sum(monto.get(a, {}).get(inst, 0.0) for a in agents)

    has_dur = bool(duracion)
    # Duración del Total por INSTRUMENTO (columna Total): ponderada por el Monto
    # de cada agente en ESE instrumento.
    dur_col_total: dict[str, float | None] = {}
    # Duración del Total por AGENTE/columna (fila Total): ponderada por el Monto
    # de cada instrumento en ESA columna.
    dur_row_total: dict[str, float | None] = {}
    if has_dur:
        for inst in instruments:
            dur_col_total[inst] = _weighted_dur(
                [(monto.get(a, {}).get(inst, 0.0), duracion.get(a, {}).get(inst)) for a in agents]
            )
        for a in agents:
            dur_row_total[a] = _weighted_dur(
                [(monto.get(a, {}).get(i, 0.0), duracion.get(a, {}).get(i)) for i in instruments]
            )
        # Esquina Total x Total: ponderada por el Monto total de cada instrumento,
        # sobre la duración YA promediada por instrumento (dur_col_total).
        dur_row_total[total_label] = _weighted_dur(
            [(monto.get(total_label, {}).get(i, 0.0), dur_col_total.get(i)) for i in instruments]
        )

    # Escala de color por COLUMNA: cada agente se lee contra su propio máximo.
    vmax = {c: max((monto.get(c, {}).get(i, 0.0) for i in instruments), default=0.0) for c in cols}

    def _dur_cell(c: str, inst: str) -> str:
        if not has_dur:
            return ""
        d = dur_col_total.get(inst) if c == total_label else duracion.get(c, {}).get(inst)
        return f'<td style="{tdr}">{_dcv_dur(d)}</td>'

    body = ""
    for inst in instruments:
        cells = ""
        for c in cols:
            v = monto.get(c, {}).get(inst)
            share = (v / totals[c] * 100) if v and totals.get(c) else 0.0
            cells += (
                f'<td style="{tdr};{_dcv_shade(v, vmax[c])}">{_dcv_num(v)}</td>'
                f'{_dur_cell(c, inst)}'
                f'<td style="{tdr};color:#666">{f"{share:.0f}%" if share >= 0.5 else "-"}</td>'
            )
        body += f'<tr><td style="{tdl}">{_esc(inst)}</td>{cells}</tr>'

    tdt = ("text-align:right;padding:5px 6px;font-size:11px;font-weight:700;"
           "background:#eef3f9;border-top:2px solid #4a5a72;white-space:nowrap")
    total_cells = "".join(
        f'<td style="{tdt}">{_dcv_num(totals[c])}</td>'
        + (f'<td style="{tdt}">{_dcv_dur(dur_row_total.get(c))}</td>' if has_dur else "")
        + f'<td style="{tdt}">-</td>'
        for c in cols
    )
    body += (
        f'<tr><td style="{tdl};background:#eef3f9;border-top:2px solid #4a5a72">{_esc(total_label)}</td>'
        + total_cells + "</tr>"
    )

    ncols = 3 if has_dur else 2
    asof_note = f" · corte {_esc(asof)}" if asof else ""
    # Esta tabla trae muchas más columnas que el resto (Monto/Dur./% por cada
    # agente): el tope de 760px + overflow-x:auto del resto de las tablas del
    # informe la dejaba con una barra de scroll horizontal y la tapa cortada.
    # Acá se libera el ancho (sin tope ni scroll) para que la fila se vea
    # completa, igual que el resto del informe se estira a la fila entera.
    return (
        '<div style="overflow-x:visible;max-width:100%;margin:6px auto">'
        '<table style="width:auto;min-width:100%;border-collapse:collapse">'
        f'<thead><tr><th style="{thl}" rowspan="2">Instrumento</th>'
        + "".join(f'<th style="{thg}" colspan="{ncols}">{_esc(c)}</th>' for c in cols)
        + "</tr><tr>"
        + "".join(
            f'<th style="{th}">Monto</th>' + (f'<th style="{th}">Dur.</th>' if has_dur else "")
            + f'<th style="{th}">% port.</th>'
            for _ in cols
        )
        + "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f'<div style="font-size:11px;color:#777;margin:4px 0 0">{_esc(unit)}{asof_note} · '
        "el % es sobre el portafolio de cada agente"
        + ("; la duración total es el promedio ponderado por monto" if has_dur else "")
        + "</div></div>"
    )


def render_dcv_bucket_table(
    instruments: list[str],
    buckets: list[str],
    matrix: dict[str, dict[str, float]],
    *,
    unit: str = "US$ Mill.",
    asof: str = "",
    total_label: str = "Total",
) -> str:
    """Distribución por tramo de plazo: filas = instrumento, columnas = Total +
    un tramo por columna. ``matrix[instrumento][bucket]``.

    El color escala sobre las celdas de TRAMO únicamente (la columna Total y la
    fila Total quedan fuera, como en el correo original)."""
    th = "text-align:right;padding:5px 7px;background:#4a5a72;color:#fff;font-size:11px;white-space:nowrap"
    thl = "text-align:left;padding:5px 7px;background:#4a5a72;color:#fff;font-size:11px"
    tdl = "text-align:left;padding:4px 7px;border-bottom:1px solid #eee;font-size:11px;font-weight:700"
    tdr = "text-align:right;padding:4px 7px;border-bottom:1px solid #eee;font-size:11px;white-space:nowrap"
    tdtot = f"{tdr};font-weight:700;background:#f6f8fb"

    vmax = max(
        (matrix.get(i, {}).get(b, 0.0) for i in instruments for b in buckets), default=0.0,
    )
    body = ""
    for inst in instruments:
        row = matrix.get(inst, {})
        row_total = sum(row.get(b, 0.0) for b in buckets)
        cells = "".join(
            f'<td style="{tdr};{_dcv_shade(row.get(b), vmax)}">{_dcv_num(row.get(b))}</td>'
            for b in buckets
        )
        body += (
            f'<tr><td style="{tdl}">{_esc(inst)}</td>'
            f'<td style="{tdtot}">{_dcv_num(row_total)}</td>{cells}</tr>'
        )

    col_totals = [sum(matrix.get(i, {}).get(b, 0.0) for i in instruments) for b in buckets]
    tdt = ("text-align:right;padding:5px 7px;font-size:11px;font-weight:700;"
           "background:#eef3f9;border-top:2px solid #4a5a72;white-space:nowrap")
    body += (
        f'<tr><td style="{tdl};background:#eef3f9;border-top:2px solid #4a5a72">{_esc(total_label)}</td>'
        f'<td style="{tdt}">{_dcv_num(sum(col_totals))}</td>'
        + "".join(f'<td style="{tdt}">{_dcv_num(v)}</td>' for v in col_totals)
        + "</tr>"
    )

    asof_note = f" · corte {_esc(asof)}" if asof else ""
    return (
        '<div style="overflow-x:auto;max-width:760px;margin:6px auto">'
        '<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr><th style="{thl}">Instrumento</th>'
        f'<th style="{th}">{_esc(total_label)}</th>'
        + "".join(f'<th style="{th}">{_esc(b)}</th>' for b in buckets)
        + "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f'<div style="font-size:11px;color:#777;margin:4px 0 0">{_esc(unit)}{asof_note} · '
        "escala de color sin considerar montos totales</div></div>"
    )


def render_dcv_maturities_grid(
    agents: list[str],
    columns: list[str],
    data: dict[str, dict[str, dict[str, float]]],
    *,
    unit: str = "US$ Mill.",
    rows: tuple[str, ...] = ("PDBC", "DAP $", "DAP UF", "DAP USD", "RF"),
    total_label: str = "Total",
) -> str:
    """"Próximos Vencimientos": una mini-tabla POR AGENTE (``agents``, ya en
    orden — normalmente "Totales" + los 7 agentes reales), columnas comunes
    (``columns``, ej. T/T+1/Acum 5d./Mes) y filas de instrumento fijas
    (``rows``) + Total. ``data[agente][columna][instrumento]``.

    Se acomodan en una grilla CSS de 4 columnas (como el correo, que las arma en
    2 filas de 5+3): a diferencia de ``render_dcv_portfolio_table``/
    ``render_dcv_bucket_table`` (una tabla ancha con scroll horizontal), acá son
    8 tablas angostas — envolver es más legible que un scroll gigante.

    Cada mini-tabla es un ``<table>`` INDEPENDIENTE, así que sin más el ancho de
    columnas lo decide cada una según su propio contenido (``table-layout:auto``)
    y quedan descuadradas entre sí (ej. "DAP UF" envuelve en dos líneas en una
    tarjeta y no en la de al lado). Acá se fuerza ``table-layout:fixed`` con los
    MISMOS anchos porcentuales (columna instrumento + una por columna común) en
    las 8 tablas, más ``white-space:nowrap`` en la columna instrumento: mismas
    proporciones en toda tarjeta → los límites quedan alineados entre filas."""
    th = "text-align:right;padding:3px 6px;background:#4a5a72;color:#fff;font-size:10px;white-space:nowrap"
    thl = "padding:3px 6px;background:#4a5a72;color:#fff;font-size:10px;white-space:nowrap"
    tdl = ("text-align:left;padding:2px 6px;border-bottom:1px solid #eee;font-size:10px;"
           "font-weight:700;white-space:nowrap")
    tdr = "text-align:right;padding:2px 6px;border-bottom:1px solid #eee;font-size:10px;white-space:nowrap"
    tdt = ("text-align:right;padding:3px 6px;font-size:10px;font-weight:700;"
           "background:#eef3f9;border-top:2px solid #4a5a72;white-space:nowrap")
    tdtl = ("text-align:left;padding:3px 6px;font-size:10px;font-weight:700;"
            "background:#eef3f9;border-top:2px solid #4a5a72;white-space:nowrap")
    cap = ("text-align:center;padding:4px 0;background:#0b3766;color:#fff;"
           "font-size:11px;font-weight:700")

    # Anchos porcentuales fijos, IGUALES en las 8 tablas: columna instrumento +
    # una por cada columna común (T / T+1 / Acum 5d. / Mes), repartiendo el 100%.
    label_pct = 30
    col_pct = (100 - label_pct) / len(columns)
    colgroup = (
        f'<colgroup><col style="width:{label_pct}%">'
        + "".join(f'<col style="width:{col_pct:.2f}%">' for _ in columns)
        + "</colgroup>"
    )

    cards: list[str] = []
    for agent in agents:
        agent_data = data.get(agent, {})
        totals = {c: sum(agent_data.get(c, {}).values()) for c in columns}
        body = ""
        for inst in rows:
            cells = "".join(
                f'<td style="{tdr}">{_dcv_num(agent_data.get(c, {}).get(inst))}</td>'
                for c in columns
            )
            body += f'<tr><td style="{tdl}">{_esc(inst)}</td>{cells}</tr>'
        body += (
            f'<tr><td style="{tdtl}">{_esc(total_label)}</td>'
            + "".join(f'<td style="{tdt}">{_dcv_num(totals[c])}</td>' for c in columns)
            + "</tr>"
        )
        cards.append(
            '<div>'
            '<table style="width:100%;border-collapse:collapse;table-layout:fixed">'
            + f'<caption style="{cap}">{_esc(agent)}</caption>'
            + colgroup
            + f'<thead><tr><th style="{thl}"></th>'
            + "".join(f'<th style="{th}">{_esc(c)}</th>' for c in columns)
            + "</tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )

    return (
        '<div style="display:grid;grid-template-columns:repeat(4, minmax(0,1fr));'
        'gap:10px;margin:6px auto;max-width:960px;align-items:start">'
        + "".join(cards) + "</div>"
    )


# ── Tablas del Informe Cambiario AM ──────────────────────────────────────────
#
# Tres piezas del original que no son series: la tabla de percentiles bajo cada
# gráfico de soporte/resistencia, el snapshot de drivers de la portada y el mapa
# de calor de gamma por strike y vencimiento. Se dibujan como HTML (no SVG)
# porque son tablas: el correo las quiere seleccionables y Outlook las respeta.

def render_kv_table_html(
    headers: list[str],
    rows: list[list[str]],
    *,
    caption: str = "",
    row_styles: list[str] | None = None,
    max_width: int = 460,
) -> str:
    """Tabla simple encabezado + filas ya formateadas como texto.

    ``row_styles`` (opcional, uno por fila) pinta la ÚLTIMA columna: es el semáforo
    de la tabla de drivers (aprecia / deprecia) y queda vacío en las tablas de
    percentiles, que no tienen signo que destacar."""
    th = "text-align:right;padding:5px 7px;background:#4a5a72;color:#fff;font-size:11px;white-space:nowrap"
    thl = th.replace("text-align:right", "text-align:left")
    tdl = "text-align:left;padding:4px 7px;border-bottom:1px solid #eee;font-size:11px;font-weight:700"
    tdr = "text-align:right;padding:4px 7px;border-bottom:1px solid #eee;font-size:11px;white-space:nowrap"

    body = ""
    for i, row in enumerate(rows):
        cells = f'<td style="{tdl}">{_esc(str(row[0]))}</td>'
        for j, cell in enumerate(row[1:], start=1):
            extra = ""
            if row_styles and i < len(row_styles) and j == len(row) - 1:
                extra = f";{row_styles[i]}"
            cells += f'<td style="{tdr}{extra}">{_esc(str(cell))}</td>'
        body += f"<tr>{cells}</tr>"

    head = f'<th style="{thl}">{_esc(headers[0])}</th>' + "".join(
        f'<th style="{th}">{_esc(h)}</th>' for h in headers[1:]
    )
    note = f'<div style="font-size:11px;color:#777;margin:4px 0 0">{_esc(caption)}</div>' if caption else ""
    return (
        f'<div style="overflow-x:auto;max-width:{max_width}px;margin:6px auto">'
        '<table style="width:100%;border-collapse:collapse">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}</div>"
    )


# Escala del mapa de calor de gamma: rojo (poca) → amarillo → verde (mucha), la
# misma RdYlGn del original. Se interpola sobre BLANCO en vez de usar alpha
# porque Outlook no compone rgba() sobre el fondo de la celda.
_HEAT_STOPS = ((0.0, (198, 60, 45)), (0.5, (235, 190, 70)), (1.0, (35, 140, 80)))


def _heat_color(frac: float) -> tuple[int, int, int]:
    frac = min(1.0, max(0.0, frac))
    for (f0, c0), (f1, c1) in itertools.pairwise(_HEAT_STOPS):
        if frac <= f1:
            t = 0.0 if f1 == f0 else (frac - f0) / (f1 - f0)
            return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1, strict=True))  # type: ignore[return-value]
    return _HEAT_STOPS[-1][1]


# Escala DIVERGENTE centrada en 0, para el color del treemap: azul (cae) → gris
# pálido (sin cambio) → rojo (sube). Réplica en 3 paradas de
# ``color_continuous_scale='RdBu_r'`` del treemap Plotly original (RdBu_r: rojo
# = valor alto = positivo, azul = valor bajo = negativo).
_DIVERGING_STOPS = ((-1.0, (35, 90, 168)), (0.0, (240, 240, 238)), (1.0, (190, 40, 35)))


def _diverging_color(frac: float) -> tuple[int, int, int]:
    """``frac`` en [-1, 1] (ya normalizado contra el cap del caller) → RGB."""
    frac = min(1.0, max(-1.0, frac))
    for (f0, c0), (f1, c1) in itertools.pairwise(_DIVERGING_STOPS):
        if frac <= f1:
            t = 0.0 if f1 == f0 else (frac - f0) / (f1 - f0)
            return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1, strict=True))  # type: ignore[return-value]
    return _DIVERGING_STOPS[-1][1]


# ── Treemap: layout squarified (Bruls, Huizing, van Wijk, 1999) ──────────────
#
# Reparte un rectángulo (x, y, w, h) entre ``sizes`` (pesos positivos, no hace
# falta que sumen w*h) tratando de mantener cada pieza lo más cuadrada posible
# —a diferencia de un "slice-and-dice" ingenuo (una fila o columna única), que
# con pesos muy dispares deja piezas larguísimas e ilegibles—. Es el mismo
# algoritmo que usa ``squarify`` (PyPI) y el treemap de Plotly/D3.

def _worst_aspect(row: list[float], side: float) -> float:
    """Peor relación de aspecto (>= 1, mejor cuanto más cerca de 1) si ``row``
    se dispone como una franja de largo ``side``."""
    if not row or side <= 0:
        return float("inf")
    row_area = sum(row)
    if row_area <= 0:
        return float("inf")
    thickness = row_area / side
    if thickness <= 0:
        return float("inf")
    t2 = thickness * thickness
    return max(max(t2 / v if v > 0 else float("inf") for v in row),
              max(v / t2 for v in row))


def _squarify(sizes: list[float], x: float, y: float, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """``(x, y, w, h)`` de cada rectángulo, EN EL MISMO ORDEN que ``sizes``.

    El área total del rectángulo (``w*h``) se reparte a prorrata de ``sizes``
    (no hace falta normalizar antes). Internamente ordena de mayor a menor —así
    da mejores proporciones— y reordena el resultado al final.
    """
    n = len(sizes)
    if n == 0:
        return []
    if n == 1:
        return [(x, y, w, h)]
    total = sum(sizes)
    if total <= 0:
        return [(x, y, 0.0, 0.0)] * n

    area_total = w * h
    scaled = [max(s, 0.0) / total * area_total for s in sizes]
    order = sorted(range(n), key=lambda i: -scaled[i])
    remaining = [scaled[i] for i in order]
    remaining_idx = list(order)
    out: dict[int, tuple[float, float, float, float]] = {}
    cx, cy, cw, ch = float(x), float(y), float(w), float(h)

    while remaining:
        side = min(cw, ch)
        row, row_idx = [remaining[0]], [remaining_idx[0]]
        i = 1
        while i < len(remaining):
            trial = row + [remaining[i]]
            if _worst_aspect(trial, side) <= _worst_aspect(row, side):
                row.append(remaining[i])
                row_idx.append(remaining_idx[i])
                i += 1
            else:
                break
        row_area = sum(row)
        thickness = row_area / side if side > 0 else 0.0
        if cw >= ch:
            # lado corto = altura: la fila es una franja VERTICAL de ancho
            # `thickness`, sus piezas se apilan de arriba hacia abajo.
            ry = cy
            for v, idx in zip(row, row_idx):
                piece_h = (v / row_area * ch) if row_area > 0 else 0.0
                out[idx] = (cx, ry, thickness, piece_h)
                ry += piece_h
            cx += thickness
            cw -= thickness
        else:
            # lado corto = ancho: la fila es una franja HORIZONTAL de alto
            # `thickness`, sus piezas van de izquierda a derecha.
            rx = cx
            for v, idx in zip(row, row_idx):
                piece_w = (v / row_area * cw) if row_area > 0 else 0.0
                out[idx] = (rx, cy, piece_w, thickness)
                rx += piece_w
            cy += thickness
            ch -= thickness
        remaining = remaining[len(row):]
        remaining_idx = remaining_idx[len(row):]

    return [out[i] for i in range(n)]


# Meses abreviados en español (mismo orden/formato que
# ``series_transforms._fmt_date``: "DD-mmm-YYYY", p.ej. "10-ago-2026") — para
# poder ORDENAR columnas de fecha cuyo label ya llegó como texto legible en vez
# de ISO. Duplicado deliberado de la tabla de ``series_transforms.py``: importar
# desde ahí crearía un ciclo (``series_transforms`` ya importa de este módulo).
_MONTHS_ES_IDX = {
    m: i for i, m in enumerate(
        ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"), start=1,
    )
}
_DATE_LABEL_RE = re.compile(r"^(\d{1,2})-([a-záéíóúñ]{3})-(\d{4})$", re.IGNORECASE)


def _heatmap_col_sort_key(label: str) -> tuple[int, int, int] | None:
    """``(año, mes, día)`` si ``label`` sigue el formato "DD-mmm-YYYY"; ``None``
    si no matchea (columnas no-fecha, p.ej. instrumentos) — el caller entonces
    deja el orden de aparición tal cual, sin reventar."""
    m = _DATE_LABEL_RE.match(label)
    if not m:
        return None
    day, mon, year = m.groups()
    month_num = _MONTHS_ES_IDX.get(mon.lower())
    return None if month_num is None else (int(year), month_num, int(day))


def _render_heatmap(plot: PlotData, width: int, height: int) -> str:
    """Matriz fila x columna pintada con la escala RdYlGn: réplica de
    ``go.Heatmap(colorscale='RdYlGn')`` del dashboard original, en SVG puro (sin
    JS ni Plotly) — mismo mecanismo ``viewBox`` + ``width="100%"`` que TODO el
    resto de los gráficos del informe, así se auto-ajusta al ancho de su
    tarjeta (comparte fila con otro gráfico) exactamente como Plotly, en vez de
    quedar fijo como una tabla HTML de celdas de ancho constante.

    Convención de ``plot.series``: UNA serie por FILA (``label`` = etiqueta de
    fila), cuyos ``points`` son ``(columna, valor)`` — mismo ``PlotSeries`` que
    ``grouped``/``snapshot``, leído distinto. Barra de escala (colorbar) a la
    derecha con marcas en 100/75/50/25/0% del rango."""
    rows = [s for s in plot.series if s.points]
    cols: list[str] = []
    seen_cols: set[str] = set()
    cell: dict[tuple[str, str], float] = {}
    values: list[float] = []
    for s in rows:
        for c, v in s.points:
            if c not in seen_cols:
                seen_cols.add(c)
                cols.append(c)
            cell[(s.label, c)] = v
            values.append(v)
    if not rows or not cols:
        return _no_axis_message(plot, width, height)

    # Orden de columnas: cronológico si TODAS parsean como fecha (el caso real,
    # venc x strike). Sin esto quedan en orden de aparición del parquet —
    # arbitrario, no la fecha— porque la transform no ordena por fila global,
    # solo por fila individual (cada strike trae sus propios vencimientos).
    sort_keys = [_heatmap_col_sort_key(c) for c in cols]
    if all(k is not None for k in sort_keys):
        cols = [c for _k, c in sorted(zip(sort_keys, cols, strict=True))]

    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0

    left, right = 62, 92
    top, bottom = 86, 16
    plot_w = width - left - right
    plot_h = height - top - bottom
    cell_w = plot_w / len(cols)
    cell_h = plot_h / len(rows)
    show_values = cell_w > 26 and cell_h > 14

    out = _svg_open(width, height, f"{plot.dataset_id} — mapa de calor")
    # Etiquetas de columna, rotadas sobre la grilla (fechas largas: "10-ago-26").
    for j, c in enumerate(cols):
        cx = left + (j + 0.5) * cell_w
        out.append(
            f'<text transform="translate({cx:.1f},{top - 8}) rotate(-45)" '
            f'font-size="10" fill="#555" text-anchor="start">{_esc(c)}</text>'
        )
    # Celdas + etiquetas de fila. Sin dato en (fila, columna) → sin <rect>: se ve
    # el fondo blanco, igual que un ``go.Heatmap`` sobre datos dispersos.
    for i, s in enumerate(rows):
        cy = top + i * cell_h
        out.append(
            f'<text x="{left - 8}" y="{cy + cell_h / 2 + 3.5:.1f}" font-size="11" '
            f'fill="#333" text-anchor="end">{_esc(s.label)}</text>'
        )
        for j, c in enumerate(cols):
            v = cell.get((s.label, c))
            if v is None:
                continue
            frac = (v - vmin) / span
            red, green, blue = _heat_color(frac)
            ink = "#fff" if frac < 0.18 or frac > 0.82 else "#1c1c1c"
            x = left + j * cell_w
            color = f"rgb({red},{green},{blue})"
            attrs = _tip_attrs(color, s=c, k=s.label, v=_val_unit(v, plot.unit))
            out.append(
                f'<rect x="{x:.1f}" y="{cy:.1f}" width="{max(0.0, cell_w - 1.5):.1f}" '
                f'height="{max(0.0, cell_h - 1.5):.1f}" fill="{color}"{attrs}/>'
            )
            if show_values:
                out.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{cy + cell_h / 2 + 3.5:.1f}" '
                    f'font-size="9.5" fill="{ink}" text-anchor="middle" pointer-events="none">'
                    f"{_esc(_fmt_num(v))}</text>"
                )

    # Barra de escala (colorbar) a la derecha: franjas finas del MISMO
    # _heat_color, no un <linearGradient> — coherente con el resto del render,
    # que evita degradados y hace todo con formas explícitas.
    lg_x, lg_w = left + plot_w + 16, 14
    steps = max(2, round(plot_h / 6))
    for i in range(steps):
        frac = 1.0 - i / (steps - 1)
        red, green, blue = _heat_color(frac)
        y = top + i * (plot_h / steps)
        out.append(
            f'<rect x="{lg_x}" y="{y:.1f}" width="{lg_w}" height="{plot_h / steps + 0.5:.1f}" '
            f'fill="rgb({red},{green},{blue})"/>'
        )
    for frac in (1.0, 0.75, 0.5, 0.25, 0.0):
        y = top + (1.0 - frac) * plot_h
        val = vmin + frac * (vmax - vmin)
        out.append(f'<text x="{lg_x + lg_w + 5}" y="{y + 3.5:.1f}" font-size="10" fill="#666">{_esc(_fmt_num(val))}</text>')

    if plot.unit:
        out.append(f'<text x="{left}" y="16" font-size="11" fill="#777">{_esc(plot.unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)
