"""Renderiza HTML **tal cual lo ve el navegador** usando el Chrome/Edge del sistema.

Es la pieza que hace posible el correo píxel-perfecto: el motor de Word que usa
Outlook no entiende CSS moderno (grid, ``max-width``, ``overflow``, SVG…), así que
la única forma de que el correo se vea EXACTAMENTE igual al informe es rasterizar
el informe con un navegador de verdad y mandar esos píxeles.

No agrega dependencias: se invoca el navegador **por línea de comandos** (headless),
no CDP ni Playwright. En el H100 Windows siempre está ``msedge.exe``; en el Mac,
Chrome. Ruta manual con ``BANKS_HEADLESS_BROWSER``.

Dos detalles aprendidos a los golpes con ``--headless=new``:

- ``--dump-dom`` **cuelga** (era una feature de la headless vieja, ya removida), así
  que el alto de la página NO se puede consultar al DOM: se captura con una ventana
  deliberadamente alta y se recorta el fondo sobrante (``_trim_bottom``).
- ``--virtual-time-budget`` **también cuelga** el proceso. No usarlo: el navegador
  ya espera el ``load`` antes de sacar la captura.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 1240        # ancho CSS: deja el .page en sus 1200px con gutter
DEFAULT_SCALE = 2.0         # HiDPI: nítido aunque el cliente lo reescale
DEFAULT_MAX_HEIGHT = 24_000  # techo de la ventana de medición (px CSS)
DEFAULT_TIMEOUT = 240
_CAPTURE_PAD = 24     # px CSS de aire entre lo medido y la ventana de captura

# Separador entre fragmentos en una captura por lotes. Magenta puro no aparece en
# ningún informe, así que una fila entera de este color marca un corte sin ambigüedad.
_SEP_RGB = (255, 0, 255)
_SEP_PX = 6


class HeadlessError(RuntimeError):
    """No hay navegador headless disponible, o falló la captura."""


# ── Localizar el navegador ───────────────────────────────────────────────────

_WIN_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)
_MAC_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)
_PATH_CANDIDATES = ("google-chrome", "google-chrome-stable", "chromium",
                    "chromium-browser", "microsoft-edge", "msedge", "chrome")


def find_browser() -> pathlib.Path | None:
    """Ruta al navegador headless, o ``None``. ``BANKS_HEADLESS_BROWSER`` manda."""
    override = os.getenv("BANKS_HEADLESS_BROWSER", "").strip()
    if override:
        p = pathlib.Path(override)
        return p if p.exists() else None

    if sys.platform == "win32":
        candidates = _WIN_CANDIDATES
        local = os.getenv("LOCALAPPDATA")
        if local:
            candidates += (str(pathlib.Path(local) / r"Google\Chrome\Application\chrome.exe"),)
    elif sys.platform == "darwin":
        candidates = _MAC_CANDIDATES
    else:
        candidates = ()

    for c in candidates:
        if pathlib.Path(c).exists():
            return pathlib.Path(c)
    for name in _PATH_CANDIDATES:
        found = shutil.which(name)
        if found:
            return pathlib.Path(found)
    return None


def _flags(profile: pathlib.Path) -> list[str]:
    """Flags comunes. Todo apagado: sin red, sin extensiones, sin primera corrida
    (el informe es local y la máquina puede estar offline)."""
    return [
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-lcd-text",          # antialiasing en gris: no deja franjas de color al reescalar
        "--disable-dev-shm-usage",
        "--allow-file-access-from-files",
        "--default-background-color=FFFFFFFF",
        f"--user-data-dir={profile}",
    ]


def _png_complete(path: pathlib.Path) -> bool:
    """¿El PNG está entero? Se comprueba el chunk final ``IEND``.

    Mirar solo si el archivo existe (o si dejó de crecer) no alcanza: una captura de
    varios MB se puede leer a medio escribir, y PIL abre ese PNG truncado dejando la
    cola en NEGRO. Como el alto de la página se deduce de dónde termina el fondo, esa
    cola negra se lee como "hay más informe" y dispara una espiral de capturas cada
    vez más altas hasta reventar por tamaño. Con IEND el archivo está completo.
    """
    try:
        if path.stat().st_size < 64:
            return False
        with path.open("rb") as fh:
            fh.seek(-12, 2)
            return fh.read(12).endswith(b"IEND\xaeB`\x82")
    except OSError:
        return False


def _run(browser: pathlib.Path, args: list[str], out: pathlib.Path, timeout: int) -> None:
    """Corre el navegador y espera **el archivo**, no el proceso.

    Con ``--user-data-dir`` (perfil aislado, para no pisar el Chrome del usuario)
    ``--headless=new`` escribe la captura y después **se queda colgado** cerrando el
    perfil. Esperar al proceso sería esperar para siempre. Así que se espera a que el
    PNG exista y deje de crecer, y ahí se mata el navegador — sirve igual si algún
    día una versión de Edge/Chrome sale con otra maña parecida.
    """
    out.unlink(missing_ok=True)
    proc = subprocess.Popen([str(browser), *args],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _png_complete(out):
                return
            if proc.poll() is not None and not out.exists():
                raise HeadlessError(f"{browser.name} terminó sin generar la captura")
            time.sleep(0.25)
        raise HeadlessError(f"{browser.name} no respondió en {timeout}s")
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ── Recorte / medición ───────────────────────────────────────────────────────

def _trim_bottom(img, background=None):
    """Recorta las filas de fondo del final. Devuelve ``(img, tocó_el_techo)``.

    Con ``background`` el color de fondo se da por sabido; sin él se deduce de la
    ÚLTIMA fila (no del píxel de arriba a la izquierda: la ventana se pide más alta que
    la página, así que abajo siempre sobra fondo, mientras que arriba puede empezar
    cualquier cosa). Deducirlo NO sirve cuando la página termina en algo de color a
    ras del borde —la captura por lotes cierra con una banda separadora magenta—:
    ahí el separador se tomaría POR fondo y se borraría junto con el último fragmento.

    Si la última fila no es fondo, la página era más alta que la ventana:
    ``tocó_el_techo`` sale ``True`` y ``capture_html`` reintenta con el doble de alto.
    """
    import numpy as np

    arr = np.asarray(img)
    bottom = arr[-1]
    if background is None:
        if not (bottom == bottom[0]).all():
            return img, True
        bg = bottom[0]
    else:
        bg = np.array(background, dtype=arr.dtype)
        if not (bottom == bg).all():
            return img, True
    rows = np.flatnonzero((arr != bg).any(axis=(1, 2)))
    if not rows.size:
        return img.crop((0, 0, img.width, 1)), False
    return img.crop((0, 0, img.width, int(rows[-1]) + 1)), False


def trim_margins(img):
    """Recorta el fondo sobrante de los CUATRO lados.

    Una tabla angosta capturada en una página de 760px queda con aire a los costados;
    sin sacarlo, al mostrarla en el correo se ve más chica de lo que corresponde
    (la imagen entera se escala al contenedor, tabla + aire). Recortada a su caja
    real se puede mostrar a su tamaño natural, igual que en el navegador.
    """
    import numpy as np

    arr = np.asarray(img)
    bg = arr[-1, -1]
    mask = (arr != bg).any(axis=2)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size or not cols.size:
        return img
    return img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def _uniform_rows(arr) -> list[int]:
    """Índices de las filas de un único color (candidatas a corte invisible)."""
    import numpy as np

    return np.flatnonzero((arr == arr[:, :1, :]).all(axis=(1, 2))).tolist()


# ── API pública ──────────────────────────────────────────────────────────────

def _open_rgb(path: pathlib.Path):
    """Abre una captura propia sin el tope antibomba de PIL (una página larga a 2x
    supera de sobra sus 179M de píxeles y el archivo lo generamos nosotros)."""
    from PIL import Image

    previous, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            rgb.load()
            return rgb
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def capture_html(
    html: str,
    *,
    width: int = DEFAULT_WIDTH,
    scale: float = DEFAULT_SCALE,
    max_height: int = DEFAULT_MAX_HEIGHT,
    background: tuple[int, int, int] | None = None,
    browser: pathlib.Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
):
    """Renderiza ``html`` y devuelve la página COMPLETA como imagen PIL.

    Dos pasadas: la primera a escala 1 con una ventana altísima solo para medir el
    alto real (se recorta el fondo sobrante); la segunda captura ese alto exacto a
    ``scale``. Medir aparte evita cargar en memoria una imagen de 24.000px a 2x.

    La imagen sale a ``width*scale`` píxeles de ancho: al mostrarla a ``width`` CSS
    se ve nítida en pantallas HiDPI.
    """
    browser = browser or find_browser()
    if browser is None:
        raise HeadlessError(
            "no se encontró Chrome/Edge para rasterizar el informe. Instalá uno o "
            "apuntá BANKS_HEADLESS_BROWSER al ejecutable."
        )

    # ``ignore_cleanup_errors``: en Windows el navegador puede seguir soltando handles
    # del perfil unos instantes después del kill, y el borrado del temporal tiraría
    # PermissionError justo después de una captura que salió bien.
    with tempfile.TemporaryDirectory(prefix="banks-headless-", ignore_cleanup_errors=True) as tmp:
        tmpdir = pathlib.Path(tmp)
        page = tmpdir / "report.html"
        page.write_text(html, encoding="utf-8")
        profile = tmpdir / "profile"
        url = page.as_uri()

        # Pasada 1 — medir. Si el contenido llega al borde inferior, la ventana se
        # queda corta: se duplica y se reintenta (informes muy largos).
        probe_height = max_height
        for _ in range(3):
            probe = tmpdir / "probe.png"
            _run(browser, [*_flags(profile), f"--window-size={width},{probe_height}",
                           "--force-device-scale-factor=1", f"--screenshot={probe}", url],
                 probe, timeout)
            if not probe.exists():
                raise HeadlessError(f"{browser.name} no generó la captura de medición")
            trimmed, clipped = _trim_bottom(_open_rgb(probe), background)
            content_height = trimmed.height
            probe.unlink(missing_ok=True)
            if not clipped:
                break
            probe_height *= 2
        else:
            logger.warning("el informe supera %spx de alto: se captura truncado", probe_height)

        # Pasada 2 — capturar a escala. El aire NO es cosmético: la maquetación redondea
        # distinto a 1x que a 2x y la página puede salir unos píxeles más alta que lo
        # medido; sin margen, eso corta el final.
        shot = tmpdir / "shot.png"
        _run(browser, [*_flags(profile), f"--window-size={width},{content_height + _CAPTURE_PAD}",
                       f"--force-device-scale-factor={scale:g}", f"--screenshot={shot}", url],
             shot, timeout)
        if not shot.exists():
            raise HeadlessError(f"{browser.name} no generó la captura del informe")
        full, _ = _trim_bottom(_open_rgb(shot))
        return full


def split_strips(img, *, strip_css_height: int, scale: float) -> list:
    """Parte la captura en tiras horizontales para el cuerpo del correo.

    Una sola imagen de 9.000px de alto es frágil (clientes que la reescalan mal,
    adjunto pesado). Los cortes se buscan en una fila **de color uniforme** cerca
    del objetivo, así la costura entre dos tiras cae siempre sobre fondo liso y es
    invisible aunque el cliente redondee el ancho.
    """
    import numpy as np

    target = max(200, int(strip_css_height * scale))
    if img.height <= target * 1.35:
        return [img]

    arr = np.asarray(img)
    quiet = set(_uniform_rows(arr))
    slack = max(40, int(target * 0.25))

    strips, top = [], 0
    while img.height - top > target * 1.35:
        want = top + target
        cut = next((y for d in range(slack) for y in (want - d, want + d)
                    if top + 100 < y < img.height and y in quiet), want)
        strips.append(img.crop((0, top, img.width, cut)))
        top = cut
    strips.append(img.crop((0, top, img.width, img.height)))
    return strips


def capture_fragments(
    fragments: list[tuple[str, int]],
    *,
    head_css: str = "",
    scale: float = DEFAULT_SCALE,
    browser: pathlib.Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list:
    """Rasteriza varios fragmentos HTML (tablas) en **una sola** corrida del navegador.

    Cada fragmento viene como ``(html, ancho_css)``: el ancho importa porque una tabla
    con ``width:100%`` se dibuja al ancho de SU contenedor, así que rasterizarlas a
    todas con la misma medida las deja con la letra de otro tamaño que en el informe.

    Se apilan en una página con separadores magenta de ancho completo y después la
    captura se corta por esas bandas. Una corrida por informe en vez de una por tabla:
    arrancar el navegador cuesta ~2s y un informe trae varias.

    Devuelve una imagen PIL por fragmento, recortada a su caja real, en el mismo orden.
    """
    import numpy as np

    if not fragments:
        return []

    page = max(w for _f, w in fragments)
    sep = (f'<div style="height:{_SEP_PX}px;background:rgb({_SEP_RGB[0]},{_SEP_RGB[1]},'
           f'{_SEP_RGB[2]});margin:0"></div>')
    blocks = sep.join(f'<div style="width:{w}px;padding:6px 0">{f}</div>' for f, w in fragments)
    # Colchón blanco al final: sin él la banda separadora de cierre queda pegada al
    # borde inferior y ``_trim_bottom`` —que deduce el fondo de la última fila— la toma
    # POR fondo y la borra, con lo que se pierde el último fragmento. Aparece solo
    # cuando el redondeo del alto deja la banda justo en el borde, así que sin el
    # colchón el error es intermitente y depende de la escala.
    tail = f'<div style="height:{_SEP_PX * 4}px;background:#fff"></div>'
    doc = (f'<!doctype html><html><head><meta charset="utf-8">{head_css}</head>'
           f'<body style="margin:0;background:#fff;width:{page}px">'
           f'{sep}{blocks}{sep}{tail}</body></html>')

    full = capture_html(doc, width=page, scale=scale, background=(255, 255, 255),
                        browser=browser, timeout=timeout)
    arr = np.asarray(full)
    sep_rgb = np.array(_SEP_RGB, dtype=arr.dtype)
    is_sep = (arr == sep_rgb).all(axis=2).all(axis=1)

    # Filas magenta contiguas → un corte por banda.
    bands, start = [], None
    for y, flag in enumerate([*is_sep.tolist(), False]):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            bands.append((start, y))
            start = None

    if len(bands) != len(fragments) + 1:
        raise HeadlessError(
            f"la captura por lotes devolvió {max(len(bands) - 1, 0)} fragmentos y se "
            f"esperaban {len(fragments)}"
        )
    return [trim_margins(full.crop((0, bands[i][1], full.width, bands[i + 1][0])))
            for i in range(len(fragments))]


def to_png(img, *, colors: int = 256) -> bytes:
    """Imagen PIL → bytes PNG.

    La captura de un informe es texto y relleno plano: cuantizarla a paleta la deja
    idéntica a la vista y pesa ~4x menos que RGB, lo que importa cuando el cuerpo
    del correo son varias tiras de 2.400px de ancho. ``optimize=True`` NO se usa: en
    una imagen de decenas de millones de píxeles tarda minutos y gana poco.
    """
    import io

    out = io.BytesIO()
    if colors and img.mode == "RGB":
        from PIL import Image

        img = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    img.save(out, format="PNG", compress_level=6)
    return out.getvalue()


def new_cid(prefix: str = "shot") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}@banks"
