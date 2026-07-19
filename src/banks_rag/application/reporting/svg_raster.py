"""Rasteriza a PNG los ``<svg>`` inline de un informe (paso previo al correo).

El informe curado (``svg_chart.py``) dibuja sus gráficos como **SVG inline**, que se
ve perfecto en el navegador pero que el motor de Word de Outlook **no renderiza**:
al pasar el informe a ``.eml`` los gráficos simplemente desaparecen del cuerpo.

Acá cada ``<svg>`` se convierte a PNG con **PyMuPDF** (ya es dependencia del
proyecto: sin binarios extra, sin red, funciona offline en el H100). Se rasteriza
el SVG TAL CUAL quedó en el HTML —no se vuelve a dibujar desde los datos—, así el
correo lleva exactamente el gráfico que se revisó en el navegador, incluidas las
ediciones hechas a mano.

Tres normalizaciones antes de rasterizar (MuPDF es más estricto que el navegador):

- ``width="100%"`` no tiene sentido fuera de un documento: sin ancho explícito
  MuPDF cae a una página carta y escala mal. Se fija ``width``/``height`` desde el
  ``viewBox``.
- ``fill="transparent"`` (las áreas de click de la leyenda) MuPDF lo pinta NEGRO;
  el equivalente que sí entiende es ``fill="none"``.
- ``font-family`` en el ``<svg>`` raíz no se hereda a los ``<text>``: sin él la
  tipografía cae a una serif. Se pone la familia en cada ``<text>``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Escala de rasterizado: el PNG sale al doble del tamaño CSS para que se vea nítido
# en pantallas HiDPI (el <img> se muestra al ancho del viewBox).
DEFAULT_SCALE = 2.0

# Familia que MuPDF resuelve siempre (base-14), en vez de depender de que Arial esté
# instalada en la máquina que genera el correo.
_RASTER_FONT = "Helvetica"

_RE_SVG = re.compile(r"<svg\b[^>]*>.*?</svg>", re.S | re.I)
_RE_VIEWBOX = re.compile(r'viewBox="\s*([-\d.eE]+)[,\s]+([-\d.eE]+)[,\s]+([-\d.eE]+)[,\s]+([-\d.eE]+)\s*"')
_RE_SIZE_ATTR = re.compile(r'\s(?:width|height)="[^"]*"', re.I)
_RE_TEXT_NO_FONT = re.compile(r"<text\b(?![^>]*font-family)", re.I)


def _standalone_svg(svg: str) -> tuple[str, float, float] | None:
    """SVG del informe → SVG autocontenido rasterizable + su tamaño CSS (w, h).

    Devuelve ``None`` si no se puede determinar el tamaño (sin ``viewBox``)."""
    m = _RE_VIEWBOX.search(svg)
    if not m:
        return None
    width, height = float(m.group(3)), float(m.group(4))
    if width <= 0 or height <= 0:
        return None

    end = svg.index(">") + 1
    head = _RE_SIZE_ATTR.sub("", svg[:end])  # fuera width="100%" / height relativo
    head = head.replace("<svg", f'<svg width="{width:g}" height="{height:g}"', 1)
    body = svg[end:]

    out = (head + body).replace('="transparent"', '="none"')
    return _RE_TEXT_NO_FONT.sub(f'<text font-family="{_RASTER_FONT}"', out), width, height


def svg_to_png(svg: str, *, scale: float = DEFAULT_SCALE) -> tuple[bytes, int, int] | None:
    """``<svg>…</svg>`` → ``(png_bytes, ancho_css, alto_css)``.

    Devuelve ``None`` si el SVG no trae ``viewBox`` o si PyMuPDF no está disponible
    / no logra rasterizarlo (el llamador decide el fallback)."""
    prepared = _standalone_svg(svg)
    if prepared is None:
        return None
    normalized, width, height = prepared
    try:
        import fitz  # lazy: PyMuPDF solo se necesita al generar correos
    except ImportError:  # pragma: no cover - depende del entorno
        logger.warning("PyMuPDF no disponible: los gráficos SVG no se rasterizan")
        return None
    try:
        with fitz.open(stream=normalized.encode("utf-8"), filetype="svg") as doc:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale))
            return pix.tobytes("png"), round(width), round(height)
    except Exception as exc:  # un SVG raro no debe tumbar la generación del informe
        logger.warning("No se pudo rasterizar un SVG: %s", exc)
        return None


def rasterize_inline_svgs(
    html: str,
    emit: Callable[[bytes, int, int], str],
    *,
    scale: float = DEFAULT_SCALE,
) -> tuple[str, int]:
    """Reemplaza cada ``<svg>`` inline del HTML por lo que devuelva ``emit``.

    ``emit(png_bytes, ancho_css, alto_css)`` devuelve el HTML que sustituye al SVG
    (típicamente un ``<img src="cid:…">``); así este módulo no sabe nada de correo.
    Los SVG que no se puedan rasterizar quedan intactos. Devuelve
    ``(html, n_rasterizados)``."""
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        rendered = svg_to_png(m.group(0), scale=scale)
        if rendered is None:
            return m.group(0)
        count += 1
        return emit(*rendered)

    return _RE_SVG.sub(repl, html), count
