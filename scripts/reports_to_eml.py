#!/usr/bin/env python3
"""Convierte un informe HTML a un correo ``.eml`` autocontenido.
<br><br>
Dos modos:
<br><br>
**Pass-through (por defecto)** — el correo lleva TU informe editado tal cual. Se toma
el HTML (el que editaste a mano / con el chartbuilder), se lo deja "plano" y se arma el
``.eml`` con ESE cuerpo. NO reconstruye nada: los gráficos que insertaste (PNG del
chartbuilder) y tu texto pasan al correo. Pasos:
<br><br>
  - **Quita el chrome de edición** (panel 💾/📄 + ``contenteditable``) y el **Plotly
    interactivo** (deja solo los PNG).
  - **Inline CSS**: el CSS del ``<head>`` (clases de ``html_render.py``) se vuelca a
    estilos inline por elemento, porque el motor de Word de Outlook ignora el ``<style>``.
  - Los **gráficos SVG** se rasterizan a PNG y, junto con las imágenes ``data:``
    (fotos pegadas), pasan a ``cid:`` (multipart/related) — lo único que Outlook
    incrusta en el cuerpo.
  - **Adjunta TU HTML final** tal cual (SVG vectorial + tooltips), con su nombre.
  - Guarda una **copia PLANA** en ``--plain-dir``: el mismo cuerpo del correo pero
    autocontenido (``cid:`` → ``data:``), sin interacción, para revisar en el
    navegador exactamente lo que llega al destinatario.
  - Las salidas se agrupan **por familia** (``eml/fx/``, ``plano/fx/``…), igual que
    ``build_family_report.py``. Un HTML que no sea de una familia (el informe
    descriptivo ``reporte_…``) queda en la raíz.
<br><br>
**Reconstruir desde spec (``--from-spec``, legado)** — para los informes CURADOS
(``build_family_report.py``): re-renderiza cada gráfico a PNG con matplotlib desde el
parquet (la familia se infiere del nombre ``<familia>_<fecha>.html``) y arma un cuerpo
email-safe con tablas. Cambia el formato de los gráficos respecto del HTML original.
<br><br>
Uso:
    python scripts/reports_to_eml.py --src data/parquet_reports/html          # pass-through (default)
    python scripts/reports_to_eml.py --src reporte_final.html                 # un solo archivo
    python scripts/reports_to_eml.py --src data/parquet_reports/curated/fx            # una familia
    python scripts/reports_to_eml.py --from-spec --src data/parquet_reports/curated   # legado curado (recursivo)
    python scripts/reports_to_eml.py --to analista@bcch.cl --from informes@bcch.cl
<br><br>
``.msg`` (formato propietario de Outlook) requiere Outlook COM; este script emite el
estándar ``.eml``, que Outlook abre sin problemas.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime as dt
import html as _html
import io
import pathlib
import re
import sys
import uuid
from email.message import EmailMessage
from email.policy import SMTP as _SMTP_POLICY  # serializa con CRLF (RFC 5322) — Outlook lo exige
from email.utils import formatdate, make_msgid

import matplotlib

matplotlib.use("Agg")  # backend sin display (offline / servidor)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.reporting import (  # noqa: E402
    build_curated_report,
    rasterize_inline_svgs,
    section_slot_ids,
    strip_editable_chrome,
)
from banks_rag.application.reporting.parquet_facts import HtmlTable  # noqa: E402
from banks_rag.application.reporting.specs import available_families, get_spec  # noqa: E402

# ── Paleta (misma que el SVG del informe, para consistencia visual) ──────────
_PALETTE = ["#0b3766", "#c8102e", "#0a8a5f", "#e08a00", "#6a3d9a", "#1f9bcf",
            "#8a6d3b", "#e5559e", "#5c5c5c", "#17807e"]
_STACK_PALETTE = ["#1f6fb2", "#b08d57", "#5a6b7b", "#0a8a5f", "#e08a00", "#6a3d9a"]
_OVERLAY = "#c8102e"
_BANNER = "#4a5a72"
_BLUE = "#0b3766"

# ── Re-render de gráficos a PNG (matplotlib) ─────────────────────────────────

def _as_date(x: str):
    try:
        return dt.date.fromisoformat(str(x)[:10])
    except (TypeError, ValueError):
        return None

def _fmt_axis(v: float, _pos=None) -> str:
    a = abs(v)
    if a >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if a >= 1_000:
        return f"{v/1_000:.0f}k"
    if a >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}"

def _cats_in_order(series) -> list[str]:
    seen: list[str] = []
    for s in series:
        for c, _v in s.points:
            if c not in seen:
                seen.append(c)
    return seen

def _split_overlay(plot):
    ov = set(plot.overlay or ())
    base = [s for s in plot.series if s.label not in ov]
    over = [s for s in plot.series if s.label in ov]
    return base, over

def _style(ax, unit: str) -> None:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_axis))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors="#555")
    if unit:
        ax.set_ylabel(unit, fontsize=8, color="#777")

def _draw_lines(ax, series, unit: str) -> None:
    for i, s in enumerate(series):
        pts = [(_as_date(x), v) for x, v in s.points if _as_date(x)]
        if not pts:
            continue
        xs, ys = zip(*pts, strict=False)
        ax.plot(xs, ys, color=_PALETTE[i % len(_PALETTE)], linewidth=1.7, label=s.label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
    _style(ax, unit)

def _draw_dual(ax, plot, right_axis, right_unit: str) -> None:
    right = set(right_axis or ())
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    for i, s in enumerate(plot.series):
        pts = [(_as_date(x), v) for x, v in s.points if _as_date(x)]
        if not pts:
            continue
        xs, ys = zip(*pts, strict=False)
        tgt = ax2 if s.label in right else ax
        col = "#b9c0cc" if s.label in right else _PALETTE[i % len(_PALETTE)]
        tgt.plot(xs, ys, color=col, linewidth=1.7, label=s.label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
    _style(ax, plot.unit)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_axis))
    ax2.tick_params(labelsize=8, colors="#999")
    if right_unit:
        ax2.set_ylabel(right_unit, fontsize=8, color="#999")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", frameon=False, ncol=2)

def _draw_stacked_area(ax, series, overlays, unit: str) -> None:
    dates = sorted({_as_date(x) for s in series for x, _ in s.points if _as_date(x)})
    if not dates:
        return _draw_lines(ax, series, unit)
    lut = [{_as_date(x): v for x, v in s.points if _as_date(x)} for s in series]
    pos = [0.0] * len(dates)
    neg = [0.0] * len(dates)
    for i, s in enumerate(series):
        col = _STACK_PALETTE[i % len(_STACK_PALETTE)] if overlays else _PALETTE[i % len(_PALETTE)]
        vals = [lut[i].get(d, 0.0) for d in dates]
        up = [v if v > 0 else 0.0 for v in vals]
        dn = [v if v < 0 else 0.0 for v in vals]
        if any(up):
            ax.fill_between(dates, pos, [p + u for p, u in zip(pos, up, strict=False)], color=col, label=s.label, linewidth=0)
            pos = [p + u for p, u in zip(pos, up, strict=False)]
        if any(dn):
            lbl = None if any(up) else s.label
            ax.fill_between(dates, neg, [n + d for n, d in zip(neg, dn, strict=False)], color=col, label=lbl, linewidth=0)
            neg = [n + d for n, d in zip(neg, dn, strict=False)]
    for s in overlays:
        pts = [(_as_date(x), v) for x, v in s.points if _as_date(x)]
        if pts:
            xs, ys = zip(*pts, strict=False)
            ax.plot(xs, ys, color=_OVERLAY, linewidth=1.8, label=s.label)
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
    _style(ax, unit)

def _draw_grouped(ax, series, overlays, *, stacked: bool, unit: str) -> None:
    cats = _cats_in_order(series + overlays)
    if not cats:
        return
    x = list(range(len(cats)))
    lut = [{c: v for c, v in s.points} for s in series]
    pal = _STACK_PALETTE if (overlays or stacked) else _PALETTE
    if stacked:
        pos = [0.0] * len(cats)
        neg = [0.0] * len(cats)
        for i, s in enumerate(series):
            vals = [lut[i].get(c, 0.0) for c in cats]
            up = [v if v > 0 else 0.0 for v in vals]
            dn = [v if v < 0 else 0.0 for v in vals]
            col = pal[i % len(pal)]
            if any(up):
                ax.bar(x, up, 0.6, bottom=pos, color=col, label=s.label)
                pos = [p + u for p, u in zip(pos, up, strict=False)]
            if any(dn):
                ax.bar(x, dn, 0.6, bottom=neg, color=col, label=(None if any(up) else s.label))
                neg = [n + d for n, d in zip(neg, dn, strict=False)]
    else:
        n = max(1, len(series))
        w = 0.8 / n
        for i, s in enumerate(series):
            vals = [lut[i].get(c, 0.0) for c in cats]
            off = (i - (n - 1) / 2) * w
            ax.bar([xi + off for xi in x], vals, w * 0.92, color=pal[i % len(pal)], label=s.label)
    for s in overlays:
        ov = {c: v for c, v in s.points}
        ax.scatter(x, [ov.get(c, 0.0) for c in cats], color=_OVERLAY, s=22, zorder=5, label=s.label)
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_xticks(x)
    long = any(len(str(c)) > 8 for c in cats)
    ax.set_xticklabels(cats, fontsize=8, rotation=35 if long else 0, ha="right" if long else "center")
    _style(ax, unit)

def _draw_snapshot(ax, series, *, pie: bool, unit: str) -> None:
    s = series[0] if series else None
    if not s or not s.points:
        return
    cats = [c for c, _ in s.points]
    vals = [v for _, v in s.points]
    if pie:
        pv = [(c, v) for c, v in zip(cats, vals, strict=False) if v > 0]
        if pv:
            ax.pie([v for _, v in pv], labels=[c for c, _ in pv],
                   colors=[_PALETTE[i % len(_PALETTE)] for i in range(len(pv))],
                   autopct="%1.0f%%", textprops={"fontsize": 7})
            ax.axis("equal")
        return
    x = list(range(len(cats)))
    ax.bar(x, vals, 0.6, color=[_PALETTE[i % len(_PALETTE)] for i in range(len(cats))])
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_xticks(x)
    long = any(len(str(c)) > 8 for c in cats)
    ax.set_xticklabels(cats, fontsize=8, rotation=35 if long else 0, ha="right" if long else "center")
    _style(ax, unit)

def plot_to_png(plot, chart: str, *, right_axis=None, right_unit: str = "") -> bytes | None:
    """``PlotData`` → PNG (bytes) replicando el tipo de gráfico del informe."""
    if plot is None or getattr(plot, "is_empty", lambda: True)():
        return None
    base, over = _split_overlay(plot)
    pie = chart == "pie"
    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=130)
    try:
        if plot.kind == "grouped":
            _draw_grouped(ax, base, over, stacked=(chart == "stacked_bar"), unit=plot.unit)
        elif plot.kind == "snapshot":
            _draw_snapshot(ax, base, pie=pie, unit=plot.unit)
        elif chart == "dual_axis":
            _draw_dual(ax, plot, right_axis, right_unit)
        elif chart in ("area", "stacked_area"):
            _draw_stacked_area(ax, base, over, plot.unit)
        else:
            _draw_lines(ax, base + over, plot.unit)
        if not pie and chart != "dual_axis":
            _handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=7, loc="best", frameon=False, ncol=min(3, len(labels)))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)

# ── Parseo del texto ya inyectado en el HTML curado ──────────────────────────

_RE_SLOT_OPEN = re.compile(r'<div class="section-text"[^>]*data-text-slot="([^"]+)"[^>]*>')
_RE_SYNTH_OPEN = re.compile(r'<div[^>]*class="synthesis-body"[^>]*data-synthesis-body[^>]*>')
_RE_DIV_TAG = re.compile(r'<div\b[^>]*>|</div>')

def _balanced_div_content(html: str, open_end: int) -> str:
    """Contenido de un ``<div>`` cuya etiqueta de apertura termina en ``open_end``,
    respetando ``<div>`` anidados (a diferencia de una regex no-greedy, que se
    detiene en el primer ``</div>`` y trunca en silencio texto/fotos pegados por
    el usuario en el editor — el navegador envuelve el contenido pegado en divs)."""
    depth = 1
    for m in _RE_DIV_TAG.finditer(html, open_end):
        depth += -1 if m.group().startswith("</") else 1
        if depth == 0:
            return html[open_end:m.start()]
    return html[open_end:]  # div sin cerrar (HTML malformado): resto del documento

def parse_text(html: str) -> tuple[str, dict[str, str]]:
    """``(síntesis_html, {slot: párrafo_html})`` del HTML curado relleno."""
    slots = {
        m.group(1): _balanced_div_content(html, m.end()).strip()
        for m in _RE_SLOT_OPEN.finditer(html)
    }
    m = _RE_SYNTH_OPEN.search(html)
    synthesis = _balanced_div_content(html, m.end()).strip() if m else ""
    return synthesis, slots

def _div_span(html: str, open_start: int) -> tuple[int, int]:
    """``(open_end, close_end)`` del ``<div>`` que EMPIEZA en ``open_start``:
    ``open_end`` es el final de su tag de apertura, ``close_end`` el final de su
    ``</div>`` de cierre — respeta anidamiento igual que ``_balanced_div_content``,
    pero devuelve también dónde termina el bloque para poder recortarlo/reemplazarlo
    dentro del HTML que lo contiene (lo usa ``_group_grid_cards``)."""
    open_end = _RE_DIV_TAG.match(html, open_start).end()
    depth = 1
    for m in _RE_DIV_TAG.finditer(html, open_end):
        depth += -1 if m.group().startswith("</") else 1
        if depth == 0:
            return open_end, m.end()
    return open_end, len(html)  # div sin cerrar (HTML malformado): resto del documento

# ── Layout "grid" (2 gráficos/fila) → tabla Outlook-safe ─────────────────────
# CSS Grid (``.cards-grid``, incluido ``grid-template-rows:subgrid`` en ``.card``,
# ver html_render.py) no lo soporta el motor Word de Outlook AUNQUE se vuelque a
# ``style=`` inline (a diferencia del resto del CSS, que sí se recupera así). La
# única forma de lograr 2 gráficos por fila en el CUERPO del correo es una tabla
# real — mismo patrón que ``_cidify_images.repl_run`` ya usa para fotos pegadas
# lado a lado.
_RE_CARDS_GRID_OPEN = re.compile(r'<div class="cards-grid">')
_RE_CARD_OPEN = re.compile(r'<div class="card(?: card-wide)?"')

def _grid_row(cells: list[str]) -> str:
    """Fila de la tabla que reemplaza ``.cards-grid``: 1 celda a ancho completo
    (``card-wide`` o el impar sobrante) o 2 celdas de 50%."""
    if len(cells) == 1:
        return f'<tr><td colspan="2" valign="top" style="padding:5px">{cells[0]}</td></tr>'
    left, right = cells
    return (
        "<tr>"
        f'<td width="50%" valign="top" style="padding:5px 10px 5px 5px">{left}</td>'
        f'<td width="50%" valign="top" style="padding:5px 5px 5px 10px">{right}</td>'
        "</tr>"
    )

def _group_grid_cards(html: str) -> str:
    """``<div class="cards-grid">…</div>`` → ``<table>`` de 2 columnas.

    Las tarjetas ``card-wide`` (el "hero" de la sección o el impar sobrante,
    decidido en ``curated_report._wide_block_ids`` — MISMA fuente que dibujó el
    HTML original) ocupan la fila completa; el resto se empareja de a 2 en el
    orden en que aparece. El contenido de cada ``<div class="card">`` (título,
    unidad, nota, el PNG ya cidificado) viaja intacto adentro de su ``<td>`` — sus
    clases (``block-title`` etc.) siguen recibiendo el inline CSS de siempre."""
    out: list[str] = []
    pos = 0
    for m in _RE_CARDS_GRID_OPEN.finditer(html):
        out.append(html[pos:m.start()])
        open_end, close_end = _div_span(html, m.start())
        inner = html[open_end:close_end - len("</div>")]

        rows: list[str] = []
        pending: str | None = None
        for cm in _RE_CARD_OPEN.finditer(inner):
            _, c_close_end = _div_span(inner, cm.start())
            card_html = inner[cm.start():c_close_end]
            if "card-wide" in cm.group():
                if pending is not None:
                    rows.append(_grid_row([pending]))
                    pending = None
                rows.append(_grid_row([card_html]))
            elif pending is None:
                pending = card_html
            else:
                rows.append(_grid_row([pending, card_html]))
                pending = None
        if pending is not None:  # impar sin marcar wide (defensivo): fila propia igual
            rows.append(_grid_row([pending]))

        out.append(
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            'style="margin:10px 0 26px">' + "".join(rows) + "</table>"
        )
        pos = close_end
    out.append(html[pos:])
    return "".join(out)

# ── Imágenes pegadas por el usuario (data: URI) → adjunto cid: ───────────────
# Outlook (motor Word) no renderiza <img src="data:..."> en el cuerpo del correo;
# solo imágenes embebidas como adjunto 'related' referenciadas por cid:, igual
# que los gráficos matplotlib. Sin esto, cualquier foto pegada en el editor
# (síntesis o un text-slot) se guarda bien en el HTML pero desaparece en el .eml.
_RE_DATA_IMG = re.compile(
    r'<img\b[^>]*\ssrc="data:(image/(?:png|jpe?g|gif|webp));base64,([^"]+)"[^>]*>', re.I
)
# 2+ fotos pegadas una junto a otra (sin nada entre medio): al pegar dos gráficos
# recortados en fila, el navegador los deja como <img><img> consecutivos, que
# fluyen inline uno al lado del otro en el editor. Se detectan como grupo para
# reproducir ese layout en 2+ columnas en vez de apilarlas a ancho completo.
_RE_IMG_RUN = re.compile(r"(?:" + _RE_DATA_IMG.pattern + r"\s*){2,}", re.I)

def _trim_pasted_whitespace(subtype: str, data: bytes, *, pad: int = 14) -> tuple[str, bytes]:
    """Recorta el margen de fondo horneado en una captura de pantalla pegada a
    mano (cada una recortada a ojo, con un margen distinto). Sin esto, dos
    gráficos puestos lado a lado se ven con "aire" y bordes desparejos aunque el
    CSS que los envuelve sea idéntico — la diferencia está en los píxeles.
<br><br>
    Detecta el fondo por la esquina superior izquierda y recorta al bounding box
    de lo que difiere de ese fondo (con tolerancia a ruido JPEG/antialiasing),
    dejando ``pad`` px de aire. Si no hay nada que recortar (o falta Pillow, o la
    imagen no se puede abrir) devuelve los bytes tal cual — nunca revienta el
    correo por esto."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return subtype, data
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return subtype, data
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    diff = ImageChops.add(ImageChops.difference(rgb, bg), ImageChops.difference(rgb, bg), 2.0, -24)
    bbox = diff.getbbox()
    if bbox is None:
        return subtype, data  # imagen ~uniforme: nada que recortar
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(img.width, right + pad)
    bottom = min(img.height, bottom + pad)
    if (left, top, right, bottom) == (0, 0, img.width, img.height):
        return subtype, data  # ya estaba ajustada
    out = io.BytesIO()
    img.crop((left, top, right, bottom)).save(out, format="PNG")
    return "png", out.getvalue()

def _decode_cid(mime: str, b64: str, images: dict[str, tuple[str, bytes]]) -> str | None:
    """Decodifica una imagen pegada y la registra en ``images``; devuelve su cid."""
    subtype = mime.split("/", 1)[1].replace("jpg", "jpeg")
    try:
        data = base64.b64decode(b64)
    except (binascii.Error, ValueError):
        return None
    subtype, data = _trim_pasted_whitespace(subtype, data)
    cid = f"pasted{uuid.uuid4().hex[:8]}@banks"
    images[cid] = (subtype, data)
    return cid

def _cidify_images(html: str, images: dict[str, tuple[str, bytes]]) -> str:
    """Reemplaza ``<img src="data:...">`` por ``cid:`` y registra los bytes en
    ``images`` (mutado in-place) para que ``build_eml`` los incruste inline.
    Las fotos pegadas en fila (2+ seguidas) se arman en columnas lado a lado."""

    def repl_run(m: re.Match[str]) -> str:
        cids = [_decode_cid(mm.group(1), mm.group(2), images) for mm in _RE_DATA_IMG.finditer(m.group(0))]
        cids = [c for c in cids if c]
        if not cids:
            return ""
        if len(cids) == 1:
            return _img_tag(cids[0], width="100%")
        cell = f"{100 // len(cids)}%"
        cells = "".join(
            f'<td width="{cell}" valign="top" style="padding:2px">{_img_tag(c, width="100%")}</td>'
            for c in cids
        )
        return f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>{cells}</tr></table>'

    def repl_single(m: re.Match[str]) -> str:
        cid = _decode_cid(m.group(1), m.group(2), images)
        return _img_tag(cid, width="100%") if cid else m.group(0)

    html = _RE_IMG_RUN.sub(repl_run, html)
    return _RE_DATA_IMG.sub(repl_single, html)

def _uncidify_images(html: str, images: dict[str, tuple[str, bytes]]) -> str:
    """``cid:`` → ``data:`` URI: vuelve autocontenido el HTML del cuerpo del correo.
<br><br>
    Es el inverso de ``_cidify_images``/``_cidify_charts``. Se aplica al MISMO HTML
    que viaja en el correo, así el archivo que queda en disco es exactamente lo que
    ve quien lo recibe —solo que abrible directo en el navegador, sin las partes
    MIME que resuelven los ``cid:``."""
    for cid, (subtype, data) in images.items():
        b64 = base64.b64encode(data).decode("ascii")
        html = html.replace(f"cid:{cid}", f"data:image/{subtype};base64,{b64}")
    return html

# El navegador deja un "renglón fantasma" (<div>/<p> con solo <br> o espacios)
# como resto de posicionar el cursor tras pegar una imagen o un salto de línea en
# un área contenteditable. Sin colapsarlos, cada pegado suma un hueco vertical de
# más — la causa más común de espaciado "desproporcionado" alrededor de fotos
# pegadas a mano. Iterativo (no recursivo): un <div> puede quedar vacío recién
# tras limpiar SU hijo vacío (<div><div><br></div></div>).
# SOLO sin atributos (o con ``style=`` nomás): un contenedor funcional real
# (``<div id="chart-tip" role="tooltip">``, un ``data-text-slot`` vacío) SIEMPRE
# trae un id/clase/data-*, así que nunca matchea — no es basura de pegado.
_RE_EMPTY_BLOCK = re.compile(r'<(div|p)(?:\s+style="[^"]*")?>(?:\s|&nbsp;|<br\s*/?>)*</\1>', re.I)

def strip_paste_ghost_lines(html: str) -> str:
    """Colapsa ``<div>``/``<p>`` vacíos (solo espacios/``<br>``) dejados por el
    navegador al pegar contenido en el editor. No toca bloques con contenido real."""
    prev = None
    while prev != html:
        prev = html
        html = _RE_EMPTY_BLOCK.sub("", html)
    return html

def _img_tag(cid: str, *, width: str) -> str:
    return (
        f'<img src="cid:{cid}" alt="" '
        f'style="width:{width};max-width:100%;height:auto;display:block;margin:6px auto">'
    )

# ── Gráficos SVG del informe curado → PNG inline (cid:) ──────────────────────
# Outlook no renderiza SVG: sin esto los gráficos del informe curado (svg_chart.py,
# SVG inline) desaparecen del cuerpo del correo. Se rasteriza el SVG TAL CUAL está
# en el HTML (no se redibuja desde los datos), así el correo lleva exactamente el
# gráfico revisado en el navegador.

def _cidify_charts(html: str, images: dict[str, tuple[str, bytes]]) -> tuple[str, int]:
    """``<svg>`` inline → ``<img src="cid:…">`` con el PNG registrado en ``images``."""

    def emit(png: bytes, width: int, _height: int) -> str:
        cid = f"chart{uuid.uuid4().hex[:8]}@banks"
        images[cid] = ("png", png)
        return (
            f'<img src="cid:{cid}" width="{width}" alt="" '
            f'style="width:100%;max-width:{width}px;height:auto;display:block;margin:6px auto">'
        )
    return rasterize_inline_svgs(html, emit)

# ── Ensamblado del cuerpo email-safe (inline styles + tablas) ────────────────

def _esc(s: str) -> str:
    return _html.escape(s or "", quote=True)

def _row(inner: str) -> str:
    return f'<tr><td style="padding:0 18px">{inner}</td></tr>'

def build_email_html(report, synthesis: str, text_by_slot: dict[str, str],
                     cids: dict[int, str]) -> str:
    """HTML del cuerpo: solo estilos inline + tabla (Outlook-safe). ``cids`` mapea el
    índice de bloque → Content-ID de su PNG."""
    spec = report.spec
    parts: list[str] = [
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="background:#f4f4f4;font-family:Arial,Helvetica,sans-serif;color:#1f1f1f">',
        '<tr><td align="center" style="padding:18px 8px">',
        '<table role="presentation" width="900" cellpadding="0" cellspacing="0" '
        'style="width:900px;max-width:100%;background:#ffffff">',
        _row(
            f'<div style="background:{_BANNER};color:#fff;text-align:center;font-size:22px;'
            f'font-weight:bold;padding:12px 16px;margin:0 -18px 6px">{_esc(spec.title)}</div>'
        ),
    ]
    if synthesis:
        parts.append(_row(
            f'<div style="border:1px solid #d8dee8;'
            f'background:#f6f8fb;padding:10px 14px;margin:10px 0;font-size:13px">'
            f'<div style="color:{_BLUE};font-weight:bold;font-size:15px;margin-bottom:4px">'
            f'Síntesis</div>{synthesis}</div>'
        ))
    # Un texto por TÓPICO (bajo el banner de sección), igual que el HTML curado.
    slot_ids = section_slot_ids(spec)
    seen_section = None
    for i, cb in enumerate(report.blocks):
        b = cb.block
        if b.section and b.section != seen_section:
            seen_section = b.section
            parts.append(_row(
                f'<div style="background:{_BANNER};color:#fff;text-align:center;font-size:16px;'
                f'font-weight:bold;padding:7px 12px;margin:22px -18px 6px">{_esc(b.section)}</div>'
            ))
            topic_text = text_by_slot.get(slot_ids.get(b.section, ""), "")
            if topic_text:
                parts.append(_row(
                    f'<div style="font-size:14px;margin:8px 0 12px">{topic_text}</div>'
                ))
        parts.append(_row(
            f'<div style="color:{_BLUE};font-size:15px;font-weight:bold;margin:12px 0 2px">{_esc(b.title)}</div>'
        ))
        if b.unit:
            parts.append(_row(f'<div style="color:#777;font-size:12px">({_esc(b.unit)})</div>'))
        if b.note:
            parts.append(_row(f'<div style="color:#777;font-size:11px;font-style:italic">{_esc(b.note)}</div>'))
        if cb.date_note:
            parts.append(_row(
                f'<div style="color:{_BLUE};font-size:11px;font-weight:bold;background:#eef3f9;'
                f'border:1px solid #d8dee8;padding:2px 8px;margin:3px 0;display:inline-block">'
                f'📅 {_esc(cb.date_note)}</div>'
            ))
        # Gráfico: PNG inline (cid) o, si es tabla heatmap, el HTML inline tal cual.
        if cb.render_kind == "chart" and isinstance(cb.plot, HtmlTable):
            parts.append(_row(f'<div style="margin:6px 0">{cb.plot.html}</div>'))
        elif i in cids:
            parts.append(_row(
                f'<img src="cid:{cids[i]}" width="860" '
                f'style="width:860px;max-width:100%;display:block;margin:6px 0" alt="{_esc(b.title)}">'
            ))
        elif cb.render_kind != "chart":
            parts.append(_row(
                '<div style="border:1px dashed #bcbcbc;background:#fafafa;color:#777;'
                'padding:10px;font-size:12px;text-align:center;margin:6px 0">'
                '(sin gráfico reproducible)</div>'
            ))
    parts.append(_row(
        f'<div style="color:#999;font-size:11px;border-top:1px solid #eee;'
        f'margin:18px 0 0;padding:8px 0">Generado {_esc(report.generated_at)} · '
        f'el informe interactivo completo va adjunto.</div>'
    ))
    parts += ["</table>", "</td></tr>", "</table>"]
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
        '<body style="margin:0;padding:0">' + "".join(parts) + "</body></html>"
    )

# ── Ensamblado del .eml ──────────────────────────────────────────────────────

def build_eml(*, subject: str, sender: str, to: str, html_body: str,
              images: dict[str, tuple[str, bytes]], attach_name: str, attach_html: str) -> bytes:
    """Arma el ``.eml`` con la estructura MIME que Outlook incrusta bien:
<br><br>
        multipart/mixed
          multipart/related          ← los cid: son HERMANOS del cuerpo
            multipart/alternative
              text/plain
              text/html
            image/png … (inline)
          text/html (adjunto: el informe navegable)
<br><br>
    El orden importa: si el ``multipart/related`` va DENTRO del ``alternative``
    (imágenes anidadas bajo la parte HTML), Outlook no las resuelve como parte del
    cuerpo y las lista como **datos adjuntos** — que es el síntoma que se veía."""
    alternative = EmailMessage()
    alternative.set_content("Este informe se ve mejor en un cliente de correo con HTML.")
    # HTML en base64 (no quoted-printable): evita los soft-breaks '=\n' que Outlook
    # descodifica mal (se comía un carácter cada ~76 y rompía las tags y los cid).
    alternative.add_alternative(html_body, subtype="html", cte="base64")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.make_mixed()

    if images:
        related = EmailMessage()
        related.make_related()
        related.attach(alternative)
        for cid, (subtype, data) in images.items():
            part = EmailMessage()
            # Sin filename: un adjunto con nombre reaparece en la lista de adjuntos
            # aunque esté referenciado por cid.
            part.set_content(data, maintype="image", subtype=subtype,
                             cid=f"<{cid}>", disposition="inline")
            related.attach(part)
        msg.attach(related)
    else:
        msg.attach(alternative)

    # El adjunto va como STR, no como bytes: con bytes la librería no declara
    # charset y una parte text/* sin charset es us-ascii por RFC 2045 — el HTML se
    # abría con los acentos rotos ("informacián", "—" → basura) aunque los bytes
    # fueran UTF-8 válidos. Con str, EmailMessage escribe charset="utf-8".
    msg.add_attachment(attach_html, subtype="html", filename=attach_name)
    # CRLF (RFC 5322): sin esto el .eml sale con LF y Outlook descodifica mal el cuerpo.
    return msg.as_bytes(policy=_SMTP_POLICY)

# ── Orquestación ─────────────────────────────────────────────────────────────

def _family_from_name(path: pathlib.Path) -> str | None:
    """Familia del informe: prefijo del archivo (``fx_2026-07-18.html``) y, si ese
    no resuelve, la carpeta que lo contiene (``curated/fx/reporte_final.html``, el
    caso de un HTML renombrado a mano tras editarlo). ``None`` si no es de familia
    (p.ej. el informe descriptivo ``reporte_ffmm_…``)."""
    for candidate in (path.stem.split("_", 1)[0], path.parent.name):
        fam = candidate.strip().lower()
        if get_spec(fam) is not None:
            return fam
    return None

def _resolve_family_dir(template: str | pathlib.Path, fam: str | None) -> pathlib.Path:
    """Resuelve el directorio de salida para la familia ``fam`` (``None`` si el
    archivo no es de una familia reconocida, ej. el informe descriptivo).

    Si ``template`` trae ``{familia}`` se formatea directo — permite mandar el
    ``.eml`` y el plano a árboles totalmente distintos, ej.
    ``.../<familia>/Correo`` vs. ``.../<familia>/`` (mismo mecanismo que
    ``build_family_report.py``). Si no lo trae, se mantiene el comportamiento
    viejo: se asume una carpeta RAÍZ y se le agrega ``<familia>/`` como
    subcarpeta. Sin familia (informe descriptivo), el placeholder se limpia y
    el archivo queda plano en la raíz del template.
    """
    template = str(template)
    if "{familia}" in template:
        # Sin familia, "{familia}" se sustituye por "" y pathlib colapsa el
        # segmento vacío solo (".../IA//Correo" -> ".../IA/Correo").
        return pathlib.Path(template.format(familia=fam or ""))
    return pathlib.Path(template) / fam if fam else pathlib.Path(template)

def process_file_from_spec(path: pathlib.Path, out_dir: str | pathlib.Path, *, sender: str, to: str,
                           subject_prefix: str) -> str:
    fam = _family_from_name(path)
    if fam is None:
        return f"SKIP {path.name} (familia no reconocida; conocidas: {', '.join(available_families())})"
    html = path.read_text(encoding="utf-8")
    synthesis, text_by_slot = parse_text(html)
    synthesis = strip_paste_ghost_lines(synthesis)
    text_by_slot = {slot: strip_paste_ghost_lines(text) for slot, text in text_by_slot.items()}
    report = build_curated_report(get_spec(fam))

    images: dict[str, tuple[str, bytes]] = {}
    # Fotos pegadas por el usuario en el editor (síntesis o cualquier text-slot):
    # data: URI → adjunto cid: (Outlook no renderiza data: URIs en el cuerpo).
    synthesis = _cidify_images(synthesis, images)
    text_by_slot = {slot: _cidify_images(text, images) for slot, text in text_by_slot.items()}
    cids: dict[int, str] = {}
    for i, cb in enumerate(report.blocks):
        if cb.render_kind != "chart" or cb.plot is None or isinstance(cb.plot, HtmlTable):
            continue
        png = plot_to_png(
            cb.plot, cb.block.chart,
            right_axis=cb.block.params.get("right_axis"),
            right_unit=cb.block.params.get("right_unit", ""),
        )
        if png:
            cid = f"chart{i}.{uuid.uuid4().hex[:8]}@banks"
            images[cid] = ("png", png)
            cids[i] = cid
            
    body = build_email_html(report, synthesis, text_by_slot, cids)
    subject = f"{subject_prefix}{report.spec.title} — {path.stem.split('_', 1)[-1]}".strip()
    eml = build_eml(
        subject=subject, sender=sender, to=to, html_body=body, images=images,
        attach_name=path.name, attach_html=path.read_text(encoding="utf-8"),
    )
    eml_out = _resolve_family_dir(out_dir, fam)
    eml_out.mkdir(parents=True, exist_ok=True)
    out = eml_out / f"{path.stem}.eml"
    out.write_bytes(eml)
    return f"OK   {path.name} -> {out}  ({len(images)} gráficos PNG, adjunto el HTML interactivo)"

# ── Modo pass-through: TU HTML editado → correo (sin reconstruir) ─────────────
# Plotly interactivo del reporte (lo inserta el chartbuilder en modo "interactivo").
# Para el correo se quita: solo entran los gráficos PNG (data: URI → cid:).
_RE_PLOTLY_LIB = re.compile(r'<script id="cb-plotly-lib"[^>]*>.*?</script>', re.S)
_RE_CB_FIGURE = re.compile(r'<figure class="report-chart cb-inserted".*?</figure>', re.S)
_RE_NEWPLOT_SCRIPT = re.compile(r"<script\b(?:(?!</script>)[\s\S])*?Plotly\.newPlot(?:(?!</script>)[\s\S])*?</script>")

def strip_interactive_plotly(html: str) -> tuple[str, int]:
    """Deja el informe "plano": quita la librería Plotly vendorizada, las figuras
    interactivas (``Plotly.newPlot`` / ``<div id="cbfig…">``) y cualquier script de
    newPlot suelto. Conserva los gráficos PNG. Devuelve ``(html, n_interactivas)``."""
    html = _RE_PLOTLY_LIB.sub("", html)
    removed = 0

    def _drop(m: re.Match[str]) -> str:
        nonlocal removed
        block = m.group(0)
        if "Plotly.newPlot" in block or 'id="cbfig' in block or "id='cbfig" in block:
            removed += 1
            return ""
        return block  # figura con <img> (PNG): se conserva

    html = _RE_CB_FIGURE.sub(_drop, html)
    html = _RE_NEWPLOT_SCRIPT.sub("", html)
    return html, removed

# Mapa clase/etiqueta → estilo inline. Outlook ignora el <style> del <head>; esto
# vuelca ese CSS a cada elemento. Hay DOS plantillas con clases distintas (y algunas
# homónimas con estilos distintos, p.ej. .subtitle), así que las reglas van separadas
# y se elige el set según el documento. ACOPLADO a esas plantillas: si cambian sus
# clases/estilos, actualizar aquí.
# curated_report.py — informe CURADO por familia (build_family_report.py).
_CURATED_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'<div class="page">'), "max-width:1200px;margin:18px auto 40px"),
    (re.compile(r'<div class="report-title">'),
     "background:#4a5a72;color:#fff;text-align:center;font-size:24px;font-weight:800;padding:12px 16px"),
    (re.compile(r'<div class="subtitle">'), "text-align:center;color:#777;font-size:12px;margin:8px 0 4px"),
    (re.compile(r'<div class="report-synthesis">'),
     "border:1px solid #d8dee8;background:#f6f8fb;padding:12px 18px;margin:14px 0 8px"),
    (re.compile(r'<div class="synthesis-title">'), "color:#0b3766;font-size:16px;font-weight:800;margin:0 0 6px"),
    (re.compile(r'<div class="synthesis-body"[^>]*>'), "color:#1f1f1f;font-size:13px"),
    (re.compile(r'<div class="section-banner">'),
     "background:#4a5a72;color:#fff;text-align:center;font-size:18px;font-weight:800;padding:8px 14px;margin:30px 0 6px"),
    (re.compile(r'<div class="block"[^>]*>'), "margin:14px 0 8px"),
    (re.compile(r'<div class="block-title">'), "color:#0b3766;font-size:16px;font-weight:700;margin:12px 0 2px"),
    (re.compile(r'<div class="block-unit">'), "color:#777;font-size:12px;margin:0 0 6px"),
    (re.compile(r'<div class="block-note">'), "color:#777;font-size:11px;font-style:italic;margin:0 0 6px"),
    (re.compile(r'<div class="block-dates">'),
     "color:#0b3766;font-size:11px;font-weight:600;background:#eef3f9;border:1px solid #d8dee8;"
     "padding:2px 8px;margin:0 0 6px;display:inline-block"),
    (re.compile(r'<div class="prelim-note">'), "color:#9a4b00;font-size:11px;margin:0 0 2px"),
    (re.compile(r'<div class="section-text"[^>]*>'), "margin:4px 0 10px;color:#1f1f1f;font-size:14px"),
    (re.compile(r'<div class="placeholder-card[^"]*">'),
     "border:1px dashed #bcbcbc;background:#fafafa;color:#777;padding:18px;text-align:center;"
     "font-size:13px;max-width:760px;margin:6px auto"),
    # ``.card``/``.card-companion``: layout "grid" (html_render.py). El PAREO en 2
    # columnas lo arma ``_group_grid_cards`` (tabla real, Outlook no soporta CSS
    # Grid); esto solo da el look de tarjeta (borde/fondo/padding) al contenido de
    # cada ``<td>`` — mismo box que ``.card`` en el navegador, sin ``display:grid``.
    (re.compile(r'<div class="card(?: card-wide)?"[^>]*>'),
     "border:1px solid #e3e7ee;border-radius:8px;padding:12px 14px 10px;background:#fff"),
    (re.compile(r'<div class="card-companion">'), "margin:2px 0"),
    (re.compile(r"<p>"), "margin:0 0 6px;font-size:14px"),
    (re.compile(r"<li>"), "margin:2px 0;font-size:14px"),
]

# html_render.py — informe DESCRIPTIVO de datasets (parquet_report.py).
_DESCRIPTIVE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'<div class="page">'), "max-width:1400px;margin:20px auto 36px"),
    (re.compile(r'<div class="hero">'), "background:#d9d9d9;padding:18px 20px 12px;text-align:center"),
    (re.compile(r"<h1>"), "margin:0;color:#0b3766;font-size:30px;line-height:1.2;font-weight:800"),
    (re.compile(r'<div class="subtitle">'),
     "margin-top:14px;background:#cfcfcf;padding:8px 14px;font-size:14px;font-weight:700;color:#2a2a2a;text-align:center"),
    (re.compile(r'<div class="content">'), "margin-top:26px"),
    (re.compile(r'<section class="dataset-block"[^>]*>'), "margin-bottom:6px"),
    (re.compile(r'<h2 class="block-heading">'), "color:#0b3766;font-size:22px;font-weight:800;margin:26px 0 8px"),
    (re.compile(r'<p class="dataset-meta">'), "color:#0a4a86;font-size:13px;font-weight:700;margin:0 0 8px"),
    (re.compile(r'<div class="synthesis-body"[^>]*>'), "color:#1f1f1f;font-size:15px"),
    (re.compile(r'<div class="section-text"[^>]*>'), "margin:4px 0 10px;color:#1f1f1f;font-size:15px"),
    (re.compile(r'<ul class="bullet-list">'), "margin:8px 0 14px;padding-left:24px"),
    (re.compile(r'<hr class="separator">'), "border:0;border-top:1px solid #c8c8c8;margin:16px 0 18px"),
    (re.compile(r"<p>"), "margin:8px 0;font-size:16px"),
    (re.compile(r"<li>"), "margin:6px 0;font-size:16px"),
]

# Comunes a ambas plantillas (inline markup del texto).
_COMMON_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"<code>"), "font-family:Consolas,monospace;font-size:14px;background:#f2f2f2;padding:1px 4px"),
    (re.compile(r"<strong>"), "font-weight:800"),
    (re.compile(r"<em>"), "font-style:italic"),
]

def inline_report_css(html: str) -> str:
    """Vuelca el CSS de la plantilla a estilos inline (Outlook-safe). Detecta si el
    HTML es el informe curado o el descriptivo y aplica su set de reglas. No toca
    tags que ya traen ``style=`` (p.ej. las figuras/imgs del chartbuilder)."""
    def _add(style: str):
        def repl(m: re.Match[str]) -> str:
            tag = m.group(0)
            return tag if " style=" in tag else tag[:-1] + f' style="{style}">'
        return repl

    curated = 'class="report-title"' in html or 'class="section-banner"' in html
    rules = (_CURATED_RULES if curated else _DESCRIPTIVE_RULES) + _COMMON_RULES
    for rx, style in rules:
        html = rx.sub(_add(style), html)
    return html

# Restos de interactividad que no sirven en un correo (todo cliente descarta el JS):
# el <script> del tooltip del informe curado y su contenedor vacío #chart-tip. Solo
# se quitan del CUERPO del correo; el HTML plano adjunto los conserva para el navegador.
_RE_ANY_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script>", re.S | re.I)
_RE_CHART_TIP = re.compile(r'<div id="chart-tip"[^>]*>\s*</div>', re.S | re.I)

def strip_body_scripts(html: str) -> str:
    """Quita ``<script>`` y el contenedor del tooltip (peso muerto en el correo)."""
    return _RE_CHART_TIP.sub("", _RE_ANY_SCRIPT.sub("", html))

_RE_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_RE_H1 = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S | re.I)

def _extract_title(html: str) -> str:
    m = _RE_TITLE.search(html) or _RE_H1.search(html)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""

def process_file_passthrough(path: pathlib.Path, out_dir: str | pathlib.Path, *, sender: str, to: str,
                             subject_prefix: str, plain_dir: str | pathlib.Path) -> str:
    """Correo con TU HTML editado tal cual + copia PLANA del informe.
<br><br>
    Tres artefactos, cada uno con su rol:
<br><br>
    - **adjunto del correo**: tu HTML final SIN tocar (SVG vectorial + tooltips), con
      su nombre de archivo — se abre en el navegador con fidelidad total.
    - **cuerpo del correo**: el mismo informe pasado a plano (SVG → PNG ``cid:``, sin
      JS, CSS inline), que es lo único que el motor de Word de Outlook renderiza.
    - **copia en ``--plain-dir``**: ese MISMO cuerpo pero autocontenido (``cid:`` →
      ``data:``), así se abre solo y muestra exactamente lo que llega al correo.
    """
    fam = _family_from_name(path)
    raw = path.read_text(encoding="utf-8")
    clean = strip_editable_chrome(raw)          # fuera panel 💾/📄 + contenteditable
    plain, n_inter = strip_interactive_plotly(clean)  # fuera Plotly interactivo (queda PNG)

    images: dict[str, tuple[str, bytes]] = {}
    body = _cidify_images(plain, images)        # data: (PNG + fotos pegadas) → cid:
    body, n_svg = _cidify_charts(body, images)  # <svg> del informe curado → PNG cid:
    body = strip_body_scripts(body)             # JS del tooltip: inútil en un correo
    body = strip_paste_ghost_lines(body)        # renglones fantasma del pegado a mano (solo CUERPO,
                                                 # el adjunto -attach_html=plain- no se toca)
    body = _group_grid_cards(body)              # .cards-grid → tabla de 2 columnas (Outlook)
    body = inline_report_css(body)              # CSS del <head> → inline (Outlook)

    # Copia PLANA (sin interacción) agrupada por familia, igual que build_family_report.py.
    plano_out = _resolve_family_dir(plain_dir, fam)
    plano_out.mkdir(parents=True, exist_ok=True)
    plano_path = plano_out / f"{path.stem}.html"
    plano_path.write_text(_uncidify_images(body, images), encoding="utf-8")

    title = _extract_title(plain) or path.stem
    subject = f"{subject_prefix}{title}".strip()
    eml = build_eml(
        subject=subject, sender=sender, to=to, html_body=body, images=images,
        # El adjunto conserva TU nombre de archivo y TU contenido (interactivo): quien
        # recibe el correo espera abrir el informe que editaste, no la copia plana.
        attach_name=path.name, attach_html=plain,
    )
    eml_out = _resolve_family_dir(out_dir, fam)
    eml_out.mkdir(parents=True, exist_ok=True)
    out = eml_out / f"{path.stem}.eml"
    out.write_bytes(eml)
    warn = f" · OJO {n_inter} gráfico(s) interactivo(s) omitido(s): reinsértalos en modo PNG" if n_inter else ""
    return (f"OK   {path.name} -> {out}  ({len(images)} imágenes inline, "
            f"{n_svg} desde SVG · plano -> {plano_path}){warn}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Informe HTML → correo .eml (pass-through de tu HTML editado).")
    ap.add_argument("--src", default="data/parquet_reports/html",
                    help="Archivo HTML o carpeta con los HTML a convertir (default: la salida de parquet_report.py).")
    ap.add_argument(
        "--out",
        default="T:/GMN/DACE/Practicantes/Leandro/Informes Generados Con IA/{familia}/Correo",
        help="Carpeta de salida de los .eml. Admite '{familia}' como placeholder; sin él, "
             "se asume RAÍZ y se le agrega <familia>/.",
    )
    ap.add_argument(
        "--plain-dir",
        default=r"T:\GMN\DACE-DOMA\0- Traspaso servidores datascience\Outputs\Repo_GOEM\reports\historia",
        help="Carpeta de salida del HTML PLANO (sin interacción: gráficos como PNG embebido, "
             "sin JS). Admite '{familia}' como placeholder igual que --out; sin él, se agrupa "
             "por familia en subcarpetas (comportamiento actual).",
    )
    ap.add_argument("--to", default="lvenegas.ext@bcentral.cl", help="Destinatario del correo.")
    ap.add_argument("--from", dest="sender", default="lvenegas.ext@bcentral.cl", help="Remitente.")
    ap.add_argument("--subject-prefix", default="", help="Prefijo del asunto (ej. '[BCCh] ').")
    ap.add_argument("--glob", default="*.html", help="Patrón de archivos si --src es carpeta.")
    ap.add_argument("--from-spec", action="store_true",
                    help="Modo legado: RECONSTRUYE los gráficos desde el spec de la familia (informes curados), "
                         "en vez de pasar tu HTML editado tal cual.")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out_dir = args.out          # str: se resuelve por archivo (_resolve_family_dir), no acá.
    plain_dir = args.plain_dir  # ídem.
    # rglob: los informes curados viven en una subcarpeta POR FAMILIA
    # (curated/ffmm/, curated/fx/, …). En una carpeta plana se comporta igual que glob.
    files = [src] if src.is_file() else sorted(src.rglob(args.glob))
    if not files:
        print(f"No hay archivos {args.glob} en {src}")
        return
    for f in files:
        try:
            if args.from_spec:
                print(process_file_from_spec(f, out_dir, sender=args.sender, to=args.to,
                                             subject_prefix=args.subject_prefix))
            else:
                print(process_file_passthrough(f, out_dir, sender=args.sender, to=args.to,
                                               subject_prefix=args.subject_prefix, plain_dir=plain_dir))
        except Exception as exc:
            print(f"ERR  {f.name}: {exc}")

if __name__ == "__main__":
    main()
