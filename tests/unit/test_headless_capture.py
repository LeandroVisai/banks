"""Unit tests del renderizador headless (``headless.py``) — sin navegador ni red.

Se prueban las piezas que deciden si la captura salió bien: detección del navegador,
que el PNG esté completo, el recorte del fondo y el corte en tiras. Todas ellas
nacieron de fallas reales al rasterizar informes largos.
"""

from __future__ import annotations

import io

import pytest

from banks_rag.application.reporting import headless

PIL = pytest.importorskip("PIL.Image")


def _img(width: int, height: int, color=(255, 255, 255)):
    return PIL.new("RGB", (width, height), color)


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Navegador ────────────────────────────────────────────────────────────────

def test_find_browser_respeta_la_variable_de_entorno(tmp_path, monkeypatch):
    fake = tmp_path / "msedge.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("BANKS_HEADLESS_BROWSER", str(fake))
    assert headless.find_browser() == fake


def test_find_browser_devuelve_none_si_la_ruta_no_existe(tmp_path, monkeypatch):
    monkeypatch.setenv("BANKS_HEADLESS_BROWSER", str(tmp_path / "no-existe.exe"))
    assert headless.find_browser() is None


# ── PNG completo ─────────────────────────────────────────────────────────────

def test_png_completo_reconoce_el_chunk_iend(tmp_path):
    path = tmp_path / "ok.png"
    path.write_bytes(_png_bytes(_img(4, 4)))
    assert headless._png_complete(path)


def test_png_a_medio_escribir_no_se_da_por_bueno(tmp_path):
    # Sin este chequeo, PIL abre el PNG truncado con la cola en negro; como el alto de
    # la página se deduce de dónde termina el fondo, esa cola se lee como "hay más
    # informe" y dispara capturas cada vez más altas hasta reventar por tamaño.
    data = _png_bytes(_img(64, 64))
    path = tmp_path / "parcial.png"
    path.write_bytes(data[: len(data) // 2])
    assert not headless._png_complete(path)


def test_png_inexistente_no_revienta(tmp_path):
    assert not headless._png_complete(tmp_path / "nada.png")


# ── Recortes ─────────────────────────────────────────────────────────────────

def test_trim_bottom_recorta_el_fondo_sobrante():
    img = _img(20, 100)
    img.paste((10, 20, 30), (0, 0, 20, 40))          # contenido en las primeras 40 filas
    out, clipped = headless._trim_bottom(img)
    assert (out.height, clipped) == (40, False)


def test_trim_bottom_avisa_cuando_la_pagina_no_entro():
    img = _img(20, 50)
    img.paste((10, 20, 30), (0, 0, 10, 50))          # contenido hasta la última fila
    out, clipped = headless._trim_bottom(img)
    assert clipped and out.height == 50


def test_trim_bottom_toma_el_fondo_de_abajo_no_del_borde_superior():
    # La captura por lotes empieza con una banda magenta: si el fondo se dedujera del
    # píxel de arriba a la izquierda, TODO el resto contaría como contenido.
    img = _img(20, 100)
    img.paste((255, 0, 255), (0, 0, 20, 6))
    out, clipped = headless._trim_bottom(img)
    assert (out.height, clipped) == (6, False)


def test_trim_bottom_con_fondo_explicito_no_se_come_el_cierre_de_color():
    # La captura por lotes termina en una banda separadora magenta: si el fondo se
    # dedujera de la última fila, esa banda pasaría por fondo y se borraría con el
    # último fragmento adentro.
    img = _img(20, 100)
    img.paste((0, 0, 0), (0, 0, 20, 40))
    img.paste((255, 0, 255), (0, 40, 20, 100))       # cierra en magenta, a ras del borde
    _out, clipped = headless._trim_bottom(img, (255, 255, 255))
    assert clipped                                   # no hay fondo abajo: falta página
    sin_fondo, _ = headless._trim_bottom(img)
    assert sin_fondo.height == 40                    # deduciéndolo, se pierde la banda


def test_trim_margins_recorta_los_cuatro_lados():
    img = _img(100, 60)
    img.paste((0, 0, 0), (30, 10, 70, 40))
    assert headless.trim_margins(img).size == (40, 30)


# ── Tiras ────────────────────────────────────────────────────────────────────

def test_split_strips_deja_pasar_una_imagen_corta():
    img = _img(100, 200)
    assert headless.split_strips(img, strip_css_height=1600, scale=2.0) == [img]


def test_split_strips_corta_en_una_fila_de_fondo():
    # Un corte a mitad de un renglón dejaría una costura visible; el corte debe caer
    # sobre una fila lisa cercana al objetivo.
    img = _img(50, 900, (0, 0, 0))       # todo "contenido": ninguna fila es lisa…
    for y in range(0, 900):
        img.paste((y % 200, 0, 0), (0, y, 25, y + 1))
    limpia = 305                         # …salvo esta
    img.paste((255, 255, 255), (0, limpia, 50, limpia + 1))

    strips = headless.split_strips(img, strip_css_height=150, scale=2.0)
    assert sum(s.height for s in strips) == 900
    assert strips[0].height == limpia    # cortó justo en la fila lisa


def test_to_png_devuelve_un_png_legible():
    data = headless.to_png(_img(8, 8, (12, 34, 56)))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    with PIL.open(io.BytesIO(data)) as im:
        assert im.size == (8, 8)


def test_capture_html_sin_navegador_falla_con_mensaje_claro(monkeypatch):
    monkeypatch.setattr(headless, "find_browser", lambda: None)
    with pytest.raises(headless.HeadlessError, match="Chrome/Edge"):
        headless.capture_html("<html><body>x</body></html>")


def test_capture_fragments_sin_fragmentos_no_abre_el_navegador(monkeypatch):
    monkeypatch.setattr(headless, "capture_html", lambda *a, **k: pytest.fail("no debía capturar"))
    assert headless.capture_fragments([]) == []


def test_capture_fragments_corta_por_las_bandas_magenta(monkeypatch):
    # Página simulada: separador, fragmento A, separador, fragmento B, separador.
    page = _img(40, 60)
    magenta, tramos = (255, 0, 255), []
    y = 0
    for alto, color in ((6, magenta), (14, (0, 0, 0)), (6, magenta), (24, (0, 0, 200)), (6, magenta)):
        page.paste(color, (0, y, 40, y + alto))
        tramos.append((y, alto, color))
        y += alto
    page.paste((255, 255, 255), (0, y, 40, 60))

    monkeypatch.setattr(headless, "capture_html", lambda *a, **k: page)
    out = headless.capture_fragments([("<table>a</table>", 40), ("<table>b</table>", 40)], scale=1.0)
    assert [im.height for im in out] == [14, 24]


def test_capture_fragments_deja_fondo_despues_de_la_ultima_banda(monkeypatch):
    # Si la banda de cierre queda pegada al borde inferior, _trim_bottom la toma por
    # fondo y se come el último fragmento (falla intermitente, según la escala).
    visto = {}

    def _fake(html, **kwargs):
        visto["html"] = html
        return _img(40, 40)

    monkeypatch.setattr(headless, "capture_html", _fake)
    with pytest.raises(headless.HeadlessError):
        headless.capture_fragments([("<table>a</table>", 40)], scale=1.0)
    cuerpo = visto["html"].split("</body>")[0]
    assert cuerpo.rstrip().endswith('background:#fff"></div>')


def test_capture_fragments_falla_si_no_calzan_las_bandas(monkeypatch):
    monkeypatch.setattr(headless, "capture_html", lambda *a, **k: _img(40, 40))
    with pytest.raises(headless.HeadlessError, match="esperaban"):
        headless.capture_fragments([("<table>a</table>", 40)], scale=1.0)
