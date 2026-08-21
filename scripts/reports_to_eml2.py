#!/usr/bin/env python3
"""Informe HTML → correo ``.eml`` que se ve **igual que en el navegador**.

Sucesor de ``reports_to_eml.py``. El problema que resuelve: Outlook de escritorio no
renderiza con un motor web sino con el de **Word**, que ignora el ``<style>`` del
``<head>`` y no entiende CSS moderno (``grid``, ``max-width``, ``overflow``, SVG…).
El script viejo traducía ese CSS con un **mapa hardcodeado clase → estilo**, que
había que mantener a mano: lo que no estaba en la lista se perdía (empezando por
``body``, de ahí que el correo cayera a Times New Roman) y las tablas —que además
dependen de ``max-width``/``overflow-x``— se desarmaban. El caso testigo es el
informe del DCV.

Dos modos, según cuánto importe la fidelidad frente a poder copiar el texto:

**``--mode pixel``** — el cuerpo del correo es el informe **rasterizado** con el
Chrome/Edge del sistema (``headless.py``), partido en tiras verticales que van como
``cid:``. Es idéntico al navegador **por construcción**: no hay CSS que traducir, así
que no hay nada que Outlook pueda malinterpretar. A cambio, el texto del cuerpo no se
puede seleccionar (sí en el HTML adjunto).

**``--mode hibrido``** — el texto viaja como HTML real (seleccionable) y solo se
rasterizan las piezas que Word arruina: los gráficos SVG y las **tablas**. El CSS se
vuelca a inline con un inliner **genérico** que parsea el ``<style>`` de verdad
(``email_css.py``), no con una lista a mano.

Los dos adjuntan TU HTML original intacto (SVG vectorial + tooltips) y dejan una copia
"plana" autocontenida para revisar en el navegador exactamente lo que llega al correo.

Uso:
    python scripts/reports_to_eml2.py --src data/parquet_reports/curated/dcv
    python scripts/reports_to_eml2.py --src data/parquet_reports/curated --mode pixel
    python scripts/reports_to_eml2.py --src informe.html --mode ambos --width 1240
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for _path in (_SRC, _HERE):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# La plomería del correo (armado MIME, cid:, resolución de familia) ya está resuelta
# y probada en el script v1: se reusa en vez de duplicarla.
import reports_to_eml as v1  # noqa: E402
from banks_rag.application.reporting import (  # noqa: E402
    headless,
    rasterize_inline_svgs,
    strip_editable_chrome,
)
from banks_rag.application.reporting.email_css import inline_css  # noqa: E402

MODES = ("pixel", "hibrido", "ambos")

# Tipografía de respaldo. Word NO hereda la familia del ``body`` hacia adentro de las
# tablas: sin forzarla por elemento, cada celda cae a Times New Roman.
_FONT = "Arial, Helvetica, sans-serif"
_FORCE_FONT = {t: f"font-family:{_FONT}" for t in
               ("body", "div", "p", "li", "ul", "ol", "td", "th", "span", "table", "h1", "h2", "h3")}


# ── Modo pixel ───────────────────────────────────────────────────────────────

def _strip_img(cid: str, max_width: int) -> str:
    """Una tira de la captura. ``line-height``/``font-size`` en 0 y ``display:block``
    matan el hueco que Word deja debajo de una imagen (espacio para el descendente
    de la línea de texto): sin eso aparece una franja blanca entre tira y tira."""
    return (
        f'<tr><td style="padding:0;margin:0;line-height:0;font-size:0" bgcolor="#ffffff">'
        f'<img src="cid:{cid}" width="100%" alt="" border="0" '
        f'style="display:block;width:100%;max-width:{max_width}px;height:auto;border:0"></td></tr>'
    )


def body_pixel(html: str, images: dict, *, width: int, scale: float, strip_height: int,
               browser: pathlib.Path | None) -> tuple[str, int]:
    """Cuerpo del correo = el informe rasterizado en tiras. Devuelve ``(html, n_tiras)``.

    Las tiras se muestran con ``width="100%"`` (atributo, que es lo que mira Word) en
    vez de un ancho fijo: así el correo se adapta al panel de lectura en vez de forzar
    scroll horizontal, y como el PNG viene a ``scale`` (2x), al achicarse sigue nítido.
    Todas las tiras escalan por el mismo factor, así que las costuras siguen calzando.
    """
    shot = headless.capture_html(html, width=width, scale=scale, browser=browser)
    strips = headless.split_strips(shot, strip_css_height=strip_height, scale=scale)

    rows = []
    for strip in strips:
        cid = headless.new_cid("page")
        images[cid] = ("png", headless.to_png(strip))
        rows.append(_strip_img(cid, width))

    return (
        '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '</head><body style="margin:0;padding:0;background:#ffffff">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;background:#ffffff"><tr>'
        '<td align="center" style="padding:0">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;width:100%;max-width:{width}px">'
        + "".join(rows) +
        "</table></td></tr></table></body></html>"
    ), len(strips)


# ── Modo híbrido ─────────────────────────────────────────────────────────────

def _email_img(cid: str, css_width: int, *, margin: str = "6px 0") -> str:
    """Imagen del correo a su tamaño NATIVO y centrada.

    El ancho va como ATRIBUTO en píxeles, no ``width="100%"``: Word prioriza el
    atributo y con "100%" estira la imagen a todo el panel de lectura, que es
    justo lo que hace que un gráfico de 760px o una tabla angosta se vean
    deformados respecto del informe. ``max-width:100%`` en el style deja que los
    clientes que sí leen CSS la achiquen en pantallas chicas.

    El centrado va en un contenedor con ``align``: Word no entiende ``margin:auto``.
    """
    return (
        f'<div align="center" style="text-align:center;margin:{margin}">'
        f'<img data-email-final src="cid:{cid}" width="{css_width}" alt="" border="0" '
        f'style="max-width:100%;height:auto;display:inline-block;border:0"></div>'
    )


def rasterize_charts(html: str, images: dict, *, scale: float) -> tuple[str, int]:
    """``<svg>`` inline → PNG ``cid:`` con PyMuPDF. Outlook no dibuja SVG: sin esto
    los gráficos desaparecen del cuerpo. Se rasteriza el SVG TAL CUAL quedó en el
    HTML (no se redibuja desde los datos), así el correo lleva el gráfico que se
    revisó en el navegador, ediciones a mano incluidas."""

    def emit(png: bytes, width: int, _height: int) -> str:
        cid = headless.new_cid("grafico")
        images[cid] = ("png", png)
        return _email_img(cid, width)

    return rasterize_inline_svgs(html, emit, scale=scale)


_RE_TABLE_TAG = re.compile(r"</?table\b[^>]*>", re.I)
_RE_STYLE_INNER = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S | re.I)
_RE_BODY = re.compile(r"(<body\b[^>]*>)(.*)(</body>)", re.S | re.I)
_RE_TABLE_OPEN = re.compile(r"<table\b(?![^>]*\bcellpadding=)", re.I)


def _table_spans(html: str) -> list[tuple[int, int]]:
    """Rangos de las ``<table>`` de primer nivel (las anidadas viajan adentro)."""
    spans, depth, start = [], 0, 0
    for m in _RE_TABLE_TAG.finditer(html):
        if m.group(0).lstrip("<").startswith("/"):
            depth -= 1
            if depth == 0:
                spans.append((start, m.end()))
        else:
            if depth == 0:
                start = m.start()
            depth += 1
    return spans


# Anchos del informe curado (``curated_report.py``): la página son 1200px con 20px de
# gutter, la grilla parte en 2 columnas con 20px de separación, y cada tarjeta lleva
# 14px de padding por lado. Las tablas se rasterizan al ancho REAL de su contenedor:
# una tabla ``width:100%`` dibujada a 760px y mostrada dentro de una tarjeta de 560
# queda con la letra un 26% más chica que en el informe.
_GUTTER, _GAP, _CARD_PAD = 40, 20, 28
_BLOCK_MAX = 760            # .report-chart / el wrapper de tabla, max-width del template

_RE_OPEN_DIV = re.compile(r'<div\b[^>]*\bclass="([^"]*)"[^>]*>|<div\b[^>]*>|</div>', re.I)


def _container_width(html: str, position: int, page_width: int) -> int:
    """Ancho en px del contenedor de la tabla que empieza en ``position``.

    Se recorre la pila de ``<div>`` abiertos hasta ese punto: dentro de una tarjeta a
    fila completa manda el ancho de la página, dentro de una tarjeta normal la mitad
    de la grilla, y suelta en la sección el ``max-width`` del bloque."""
    page = page_width - _GUTTER
    stack: list[str] = []
    for m in _RE_OPEN_DIV.finditer(html, 0, position):
        if m.group(0).startswith("</"):
            if stack:
                stack.pop()
        else:
            stack.append(m.group(1) or "")
    classes = {c for entry in stack for c in entry.split()}
    if "card-wide" in classes:
        return page - _CARD_PAD
    if "card" in classes:
        return (page - _GAP) // 2 - _CARD_PAD
    return min(_BLOCK_MAX, page)


def rasterize_tables(html: str, images: dict, *, page_width: int, scale: float,
                     browser: pathlib.Path | None) -> tuple[str, int]:
    """Cada ``<table>`` del informe → PNG ``cid:``.

    Es la pieza que arregla el informe del DCV: sus matrices se sostienen sobre
    ``max-width``, ``overflow-x`` y ``white-space:nowrap``, tres cosas que Word
    descarta, y por eso llegaban irreconocibles. Rasterizadas se ven exactamente
    como en el navegador.

    Todas se capturan en UNA corrida del navegador (arrancarlo cuesta ~2s y un
    informe trae varias).
    """
    spans = _table_spans(html)
    if not spans:
        return html, 0

    # Se le pasa el <style> del informe para que la tabla se dibuje con el mismo CSS
    # que en el navegador (hoy usan estilos inline, pero así no depende de eso).
    head_css = "\n".join(_RE_STYLE_INNER.findall(html))
    shots = headless.capture_fragments(
        [(html[a:b], _container_width(html, a, page_width)) for a, b in spans],
        head_css=f"<style>{head_css}</style>" if head_css.strip() else "",
        scale=scale, browser=browser,
    )

    out, cursor = [], 0
    for (a, b), shot in zip(spans, shots, strict=True):
        cid = headless.new_cid("tabla")
        images[cid] = ("png", headless.to_png(shot))
        out.append(html[cursor:a])
        out.append(_email_img(cid, round(shot.width / scale)))
        cursor = b
    out.append(html[cursor:])
    return "".join(out), len(spans)


_RE_CID_IMG = re.compile(r'<img\b(?=[^>]*src="cid:)', re.I)


def _mark_final_images(html: str) -> str:
    """Marca las imágenes que ya pusimos nosotros (gráfico o tabla rasterizada) para
    que el inliner no las normalice: su ``style`` se escribió pensando en el correo."""
    return _RE_CID_IMG.sub("<img data-email-final", html)


def _wrap_for_word(html: str, *, max_width: int) -> str:
    """Encierra el cuerpo en la tabla contenedora que Word necesita para centrar y
    acotar el ancho (un ``<div>`` con ``margin:auto`` y ``max-width`` no le sirve),
    y le pone a cada ``<table>`` los atributos que el motor de Word mira de verdad
    (``cellpadding``/``cellspacing``/``border``) en vez del CSS equivalente."""
    html = _RE_TABLE_OPEN.sub('<table cellpadding="0" cellspacing="0" border="0"', html)

    match = _RE_BODY.search(html)
    if not match:
        return html
    inner = (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;background:#ffffff"><tr>'
        '<td align="center" style="padding:0">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;width:100%;max-width:{max_width}px">'
        f'<tr><td style="padding:0 20px;font-family:{_FONT};font-size:14px;color:#1f1f1f">'
        f"{match.group(2)}"
        "</td></tr></table></td></tr></table>"
    )
    return html[:match.start(2)] + inner + html[match.end(2):]


def body_hybrid(html: str, images: dict, *, page_width: int, scale: float,
                browser: pathlib.Path | None) -> tuple[str, dict[str, int]]:
    """Cuerpo del correo con el texto en HTML real y lo que Word rompe, en imagen."""
    body, n_tables = rasterize_tables(html, images, page_width=page_width, scale=scale, browser=browser)
    body, n_svg = rasterize_charts(body, images, scale=scale)   # SVG → PNG (PyMuPDF)
    body = v1._cidify_images(body, images)               # data: (fotos pegadas) → cid:
    body = v1.strip_body_scripts(body)                   # el JS no corre en un correo
    body = v1.strip_paste_ghost_lines(body)              # renglones fantasma del pegado
    body = v1._group_grid_cards(body)                    # CSS Grid → tabla de 2 columnas
    body = _mark_final_images(body)                      # lo ya resuelto no se re-estiliza
    body = inline_css(body, extra=_FORCE_FONT)           # <style> → style= por elemento
    body = _wrap_for_word(body, max_width=page_width)
    return body, {"tablas": n_tables, "graficos": n_svg}


# ── Orquestación ─────────────────────────────────────────────────────────────

# Aviso de tamaño: el .eml va en base64 (+33%) y los buzones corporativos suelen
# cortar entre 10 y 25 MB. Bajar --scale es lo que más pesa (a 1.5 el PNG cae ~40%).
_SIZE_WARN_MB = 9.0


def _write(path: pathlib.Path, out_dir, plain_dir, fam, *, suffix: str, body: str,
           images: dict, subject: str, sender: str, to: str, attach: str) -> tuple[pathlib.Path, pathlib.Path]:
    eml_dir = v1._resolve_family_dir(out_dir, fam)
    eml_dir.mkdir(parents=True, exist_ok=True)
    eml_path = eml_dir / f"{path.stem}.{suffix}.eml"
    eml_path.write_bytes(v1.build_eml(subject=subject, sender=sender, to=to, html_body=body,
                                      images=images, attach_name=path.name, attach_html=attach))

    plano_dir = v1._resolve_family_dir(plain_dir, fam)
    plano_dir.mkdir(parents=True, exist_ok=True)
    plano_path = plano_dir / f"{path.stem}.{suffix}.html"
    plano_path.write_text(v1._uncidify_images(body, images), encoding="utf-8")
    return eml_path, plano_path


def _size_note(eml: pathlib.Path) -> str:
    mb = eml.stat().st_size / 1e6
    return f" · OJO pesa {mb:.1f} MB: bajá --scale si el buzón lo rebota" if mb > _SIZE_WARN_MB else ""


def process_file(path: pathlib.Path, *, mode: str, out_dir, plain_dir, sender: str, to: str,
                 subject_prefix: str, width: int, scale: float, strip_height: int,
                 browser: pathlib.Path | None) -> list[str]:
    fam = v1._family_from_name(path)
    raw = path.read_text(encoding="utf-8")
    clean = strip_editable_chrome(raw)                    # fuera el panel 💾/📄 de edición
    plain, n_inter = v1.strip_interactive_plotly(clean)   # solo para el modo híbrido

    title = v1._extract_title(clean) or path.stem
    subject = f"{subject_prefix}{title}".strip()
    lines = []

    if mode in ("pixel", "ambos"):
        # El navegador SÍ dibuja Plotly: la captura se hace sobre el informe completo,
        # sin quitar los gráficos interactivos — en el correo quedan como se ven.
        images: dict = {}
        body, n_strips = body_pixel(clean, images, width=width, scale=scale,
                                    strip_height=strip_height, browser=browser)
        eml, plano = _write(path, out_dir, plain_dir, fam, suffix="pixel", body=body,
                            images=images, subject=subject, sender=sender, to=to, attach=raw)
        lines.append(f"OK   {path.name} [pixel]   -> {eml}  ({n_strips} tiras · "
                     f"plano -> {plano}){_size_note(eml)}")

    if mode in ("hibrido", "ambos"):
        images = {}
        body, counts = body_hybrid(plain, images, page_width=width, scale=scale, browser=browser)
        eml, plano = _write(path, out_dir, plain_dir, fam, suffix="hibrido", body=body,
                            images=images, subject=subject, sender=sender, to=to, attach=raw)
        warn = (f" · OJO {n_inter} gráfico(s) interactivo(s) omitido(s): reinsertalos en modo PNG"
                if n_inter else "")
        lines.append(f"OK   {path.name} [hibrido] -> {eml}  ({counts['tablas']} tablas, "
                     f"{counts['graficos']} gráficos · plano -> {plano}){_size_note(eml)}{warn}")

    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Informe HTML → correo .eml fiel al navegador (rasterizado o híbrido).")
    ap.add_argument("--src", default="data/parquet_reports/curated",
                    help="Archivo HTML o carpeta con los informes a convertir.")
    ap.add_argument("--mode", default="ambos", choices=MODES,
                    help="pixel = cuerpo rasterizado (idéntico al navegador); hibrido = texto "
                         "HTML + tablas y gráficos en imagen; ambos (default) emite los dos.")
    ap.add_argument("--out", default="T:/GMN/DACE/Practicantes/Leandro/Informes Generados Con IA/{familia}/Correo",
                    help="Carpeta de los .eml. Admite '{familia}'; sin él se le agrega <familia>/.")
    ap.add_argument("--plain-dir", default="data/parquet_reports/plano",
                    help="Carpeta del HTML PLANO (lo que realmente llega al correo, autocontenido).")
    ap.add_argument("--to", default="lvenegas.ext@bcentral.cl", help="Destinatario.")
    ap.add_argument("--from", dest="sender", default="lvenegas.ext@bcentral.cl", help="Remitente.")
    ap.add_argument("--subject-prefix", default="", help="Prefijo del asunto (ej. '[BCCh] ').")
    ap.add_argument("--glob", default="*.html", help="Patrón de archivos si --src es carpeta.")
    ap.add_argument("--width", type=int, default=headless.DEFAULT_WIDTH,
                    help="Ancho CSS de render del informe (default 1240: deja el .page en 1200).")
    ap.add_argument("--scale", type=float, default=headless.DEFAULT_SCALE,
                    help="Factor de rasterizado (2 = HiDPI, nítido al reescalar).")
    ap.add_argument("--strip-height", type=int, default=1600,
                    help="Alto CSS objetivo de cada tira del modo pixel.")
    ap.add_argument("--browser", default="",
                    help="Ruta a Chrome/Edge. Por defecto se autodetecta (o BANKS_HEADLESS_BROWSER).")
    args = ap.parse_args()

    browser = pathlib.Path(args.browser) if args.browser else headless.find_browser()
    if browser is None:
        sys.exit("No se encontró Chrome/Edge headless. Instalá uno o pasá --browser <ruta>.")
    print(f"navegador: {browser}")

    src = pathlib.Path(args.src)
    files = [src] if src.is_file() else sorted(src.rglob(args.glob))
    if not files:
        print(f"No hay archivos {args.glob} en {src}")
        return

    for f in files:
        try:
            for line in process_file(
                f, mode=args.mode, out_dir=args.out, plain_dir=args.plain_dir,
                sender=args.sender, to=args.to, subject_prefix=args.subject_prefix,
                width=args.width, scale=args.scale, strip_height=args.strip_height,
                browser=browser,
            ):
                print(line, flush=True)
        except Exception as exc:
            print(f"ERR  {f.name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
