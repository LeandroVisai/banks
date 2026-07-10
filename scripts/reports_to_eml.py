#!/usr/bin/env python3
"""Convierte los informes curados (HTML) a correos ``.eml`` autocontenidos.

Recorre una carpeta de HTML curados (los que produce ``build_family_report.py`` +
``fill_report_texts.py``) y, por cada uno, genera un ``.eml`` listo para abrir en
Outlook / Apple Mail / Thunderbird con:

  - CUERPO **email-safe**: estilos inline + tablas (sin ``<style>`` ni JS), porque el
    motor de Word de Outlook ignora el CSS del ``<head>`` y no renderiza SVG.
  - Cada GRÁFICO re-renderizado a **PNG con matplotlib** (desde los datos reales del
    parquet, no del SVG) e incrustado inline vía ``cid:`` (multipart/related).
  - El INFORME interactivo original (el HTML completo, con SVG + tooltips) como
    **adjunto** (se abre en el navegador con fidelidad total).

El texto de cada bloque (el que redacta el LLM) se toma TAL CUAL del HTML de entrada;
los gráficos se reconstruyen del modelo (la familia se infiere del nombre del archivo,
``<familia>_<fecha>.html``), así el PNG sale limpio y en la misma escala que el texto.

Uso:
    python scripts/reports_to_eml.py                          # carpeta por defecto
    python scripts/reports_to_eml.py --src data/parquet_reports/curated --out data/parquet_reports/eml
    python scripts/reports_to_eml.py --to analista@bcch.cl --from informes@bcch.cl

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

from banks_rag.application.reporting import build_curated_report  # noqa: E402
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


def _decode_cid(mime: str, b64: str, images: dict[str, tuple[str, bytes]]) -> str | None:
    """Decodifica una imagen pegada y la registra en ``images``; devuelve su cid."""
    subtype = mime.split("/", 1)[1].replace("jpg", "jpeg")
    try:
        data = base64.b64decode(b64)
    except (binascii.Error, ValueError):
        return None
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


def _img_tag(cid: str, *, width: str) -> str:
    return (
        f'<img src="cid:{cid}" alt="" '
        f'style="width:{width};max-width:100%;height:auto;display:block;margin:6px auto">'
    )


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
            f'<div style="border:1px solid #d8dee8;border-left:5px solid {_BLUE};'
            f'background:#f6f8fb;padding:10px 14px;margin:10px 0;font-size:13px">'
            f'<div style="color:{_BLUE};font-weight:bold;font-size:15px;margin-bottom:4px">'
            f'Síntesis</div>{synthesis}</div>'
        ))

    seen_section = None
    for i, cb in enumerate(report.blocks):
        b = cb.block
        if b.section and b.section != seen_section:
            seen_section = b.section
            parts.append(_row(
                f'<div style="background:{_BANNER};color:#fff;text-align:center;font-size:16px;'
                f'font-weight:bold;padding:7px 12px;margin:22px -18px 6px">{_esc(b.section)}</div>'
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
        slot = b.text_slot
        if slot and text_by_slot.get(slot):
            parts.append(_row(f'<div style="font-size:14px;margin:6px 0 8px">{text_by_slot[slot]}</div>'))
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
              images: dict[str, tuple[str, bytes]], attach_name: str, attach_bytes: bytes) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.set_content("Este informe se ve mejor en un cliente de correo con HTML.")
    # HTML en base64 (no quoted-printable): evita los soft-breaks '=\n' que Outlook
    # descodifica mal (se comía un carácter cada ~76 y rompía las tags y los cid).
    msg.add_alternative(html_body, subtype="html", cte="base64")
    html_part = msg.get_payload()[-1]  # la alternativa HTML
    for cid, (subtype, data) in images.items():
        html_part.add_related(data, maintype="image", subtype=subtype,
                              cid=f"<{cid}>", disposition="inline")
    msg.add_attachment(attach_bytes, maintype="text", subtype="html", filename=attach_name)
    # CRLF (RFC 5322): sin esto el .eml sale con LF y Outlook descodifica mal el cuerpo.
    return msg.as_bytes(policy=_SMTP_POLICY)


# ── Orquestación ─────────────────────────────────────────────────────────────

def _family_from_name(path: pathlib.Path) -> str | None:
    fam = path.stem.split("_", 1)[0].strip().lower()
    return fam if get_spec(fam) is not None else None


def process_file(path: pathlib.Path, out_dir: pathlib.Path, *, sender: str, to: str,
                 subject_prefix: str) -> str:
    fam = _family_from_name(path)
    if fam is None:
        return f"SKIP {path.name} (familia no reconocida; conocidas: {', '.join(available_families())})"
    html = path.read_text(encoding="utf-8")
    synthesis, text_by_slot = parse_text(html)
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
        attach_name=path.name, attach_bytes=path.read_bytes(),
    )
    out = out_dir / f"{path.stem}.eml"
    out.write_bytes(eml)
    return f"OK   {path.name} -> {out}  ({len(images)} gráficos PNG, adjunto el HTML interactivo)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Informes curados (HTML) → correos .eml autocontenidos.")
    ap.add_argument("--src", default="data/parquet_reports/curated", help="Carpeta con los HTML curados.")
    ap.add_argument("--out", default="data/parquet_reports/eml", help="Carpeta de salida de los .eml.")
    ap.add_argument("--to", default="destinatario@ejemplo.cl", help="Destinatario del correo.")
    ap.add_argument("--from", dest="sender", default="informes@ejemplo.cl", help="Remitente.")
    ap.add_argument("--subject-prefix", default="", help="Prefijo del asunto (ej. '[BCCh] ').")
    ap.add_argument("--glob", default="*.html", help="Patrón de archivos a procesar.")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob(args.glob))
    if not files:
        print(f"No hay archivos {args.glob} en {src}")
        return
    for f in files:
        try:
            print(process_file(f, out_dir, sender=args.sender, to=args.to, subject_prefix=args.subject_prefix))
        except Exception as exc:
            print(f"ERR  {f.name}: {exc}")


if __name__ == "__main__":
    main()
