"""Unit tests del conversor HTML → correo fiel al navegador (``scripts/reports_to_eml2.py``).

El navegador headless se mockea: acá se prueba el armado del correo, no el
rasterizado (eso vive en ``test_headless_capture.py``).
"""

from __future__ import annotations

import email
import importlib.util
import pathlib
import re

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "reports_to_eml2.py"


def _load():
    spec = importlib.util.spec_from_file_location("reports_to_eml2_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


r2 = _load()
PIL = pytest.importorskip("PIL.Image")


def _img(w: int, h: int, color=(255, 255, 255)):
    return PIL.new("RGB", (w, h), color)


_REPORT = """<!doctype html><html><head><meta charset="utf-8"><title>Informe DCV</title>
<style>
  body { font-family:Arial, sans-serif; color:#1f1f1f; }
  .report-title { background:#4a5a72; color:#fff; }
  .block-title { color:#0b3766; font-size:16px; }
</style></head>
<body><div class="page">
  <div class="report-title">Informe Stocks DCV</div>
  <div class="block-title">Portafolio por agente</div>
  <div style="overflow-x:auto;max-width:760px;margin:6px auto">
    <table style="width:100%"><tr><td>PDBC</td><td>9.773</td></tr></table>
  </div>
  <svg viewBox="0 0 760 300" width="100%"><text>g</text></svg>
</div></body></html>"""


# ── Piezas ───────────────────────────────────────────────────────────────────

def test_table_spans_ignora_las_tablas_anidadas():
    html = "<div><table>a<table>interna</table>b</table></div><table>otra</table>"
    spans = r2._table_spans(html)
    assert len(spans) == 2
    assert html[spans[0][0]:spans[0][1]].count("<table") == 2


def test_container_width_segun_la_tarjeta_que_la_contiene():
    suelta = '<div class="block"><table>x</table></div>'
    card = '<div class="cards-grid"><div class="card"><table>x</table></div></div>'
    wide = '<div class="cards-grid"><div class="card card-wide"><table>x</table></div></div>'
    ancho_suelta = r2._container_width(suelta, suelta.index("<table"), 1240)
    ancho_card = r2._container_width(card, card.index("<table"), 1240)
    ancho_wide = r2._container_width(wide, wide.index("<table"), 1240)
    # Una tabla width:100% dibujada al ancho equivocado queda con la letra de otro
    # tamaño que en el informe: cada contenedor tiene que dar una medida distinta.
    assert ancho_card < ancho_suelta < ancho_wide
    assert ancho_suelta == r2._BLOCK_MAX


def test_container_width_cierra_bien_los_div():
    # La tarjeta ya cerró antes de la tabla: no debe contarla como contenedor.
    html = '<div class="cards-grid"><div class="card">g</div><table>x</table></div>'
    assert r2._container_width(html, html.index("<table"), 1240) == r2._BLOCK_MAX


def test_email_img_usa_ancho_en_pixeles_y_va_centrada():
    # width="100%" haría que Word estire la imagen a todo el panel de lectura.
    tag = r2._email_img("x@banks", 560)
    assert 'width="560"' in tag and 'width="100%"' not in tag
    assert 'align="center"' in tag
    assert "max-width:100%" in tag
    assert "data-email-final" in tag


def test_mark_final_images_solo_toca_las_cid():
    html = '<img src="cid:a@b"><img src="https://x/y.png">'
    out = r2._mark_final_images(html)
    assert out.count("data-email-final") == 1


def test_wrap_for_word_agrega_los_atributos_de_tabla():
    out = r2._wrap_for_word("<body><table style='x'><tr><td>1</td></tr></table></body>",
                            max_width=1200)
    assert 'cellpadding="0"' in out and 'cellspacing="0"' in out and 'border="0"' in out
    assert 'align="center"' in out


# ── Cuerpos ──────────────────────────────────────────────────────────────────

def test_body_pixel_arma_una_tira_por_corte(monkeypatch):
    monkeypatch.setattr(r2.headless, "capture_html", lambda *a, **k: _img(2480, 6000))
    images: dict = {}
    # 6000px a escala 2 = 3000px CSS, en tiras de 1000px CSS.
    body, n = r2.body_pixel(_REPORT, images, width=1240, scale=2.0, strip_height=1000, browser=None)
    assert n == len(images) == 3
    assert body.count("<img") == 3
    for cid in images:
        assert f"cid:{cid}" in body
    # El texto del informe NO viaja en el cuerpo: es todo imagen.
    assert "Portafolio por agente" not in body


def test_body_pixel_mata_el_hueco_bajo_la_imagen(monkeypatch):
    # Sin line-height/font-size en 0, Word deja una franja blanca entre tira y tira.
    monkeypatch.setattr(r2.headless, "capture_html", lambda *a, **k: _img(2480, 6000))
    body, _ = r2.body_pixel(_REPORT, {}, width=1240, scale=2.0, strip_height=1000, browser=None)
    assert "line-height:0" in body and "font-size:0" in body


def test_body_hybrid_deja_el_texto_y_pasa_tablas_y_graficos_a_imagen(monkeypatch):
    monkeypatch.setattr(r2.headless, "capture_fragments",
                        lambda frags, **k: [_img(1120, 200) for _ in frags])
    images: dict = {}
    body, counts = r2.body_hybrid(_REPORT, images, page_width=1240, scale=2.0, browser=None)

    assert counts["tablas"] == 1
    assert "<table" not in body.split("</body>")[0].replace(
        '<table role="presentation"', "")            # la del informe se fue a imagen
    assert "Portafolio por agente" in body           # el texto sigue siendo texto
    assert "<svg" not in body                        # Outlook no dibuja SVG
    assert "<style>" not in body                     # el CSS quedó inline
    assert "font-family:Arial" in body


def test_body_hybrid_conserva_el_ancho_de_las_imagenes_que_inyecta(monkeypatch):
    # El normalizador borra max-width de todo… menos de lo que ya resolvimos nosotros.
    monkeypatch.setattr(r2.headless, "capture_fragments",
                        lambda frags, **k: [_img(1120, 200) for _ in frags])
    body, _ = r2.body_hybrid(_REPORT, {}, page_width=1240, scale=2.0, browser=None)
    assert 'width="560"' in body                     # 1120 px a escala 2 → 560 CSS
    assert "max-width:100%" in body


def test_el_eml_lleva_las_imagenes_inline_y_el_html_original_adjunto(monkeypatch, tmp_path):
    monkeypatch.setattr(r2.headless, "capture_html", lambda *a, **k: _img(1240, 900))
    src = tmp_path / "dcv_2026-07-18.html"
    src.write_text(_REPORT, encoding="utf-8")

    lineas = r2.process_file(
        src, mode="pixel", out_dir=str(tmp_path / "eml"), plain_dir=str(tmp_path / "plano"),
        sender="a@b.cl", to="c@d.cl", subject_prefix="[BCCh] ", width=1240, scale=2.0,
        strip_height=1600, browser=None,
    )
    assert len(lineas) == 1

    eml = next((tmp_path / "eml").rglob("*.pixel.eml"))
    msg = email.message_from_bytes(eml.read_bytes(), policy=email.policy.default)
    assert msg["Subject"] == "[BCCh] Informe DCV"

    html = msg.get_body(preferencelist=("html",)).get_content()
    referidos = set(re.findall(r'src="cid:([^"]+)"', html))
    inline = {p["Content-ID"].strip("<>") for p in msg.walk() if p.get("Content-ID")}
    assert referidos and referidos <= inline          # ningún cid: cuelga sin su imagen

    adjuntos = [p.get_filename() for p in msg.walk() if p.get_filename()]
    assert adjuntos == ["dcv_2026-07-18.html"]

    # La copia plana es el MISMO cuerpo, ya autocontenido para abrirlo en el navegador.
    plano = next((tmp_path / "plano").rglob("*.pixel.html"))
    assert "cid:" not in plano.read_text(encoding="utf-8")
    assert "data:image/png;base64," in plano.read_text(encoding="utf-8")


def test_modo_ambos_emite_los_dos_correos(monkeypatch, tmp_path):
    monkeypatch.setattr(r2.headless, "capture_html", lambda *a, **k: _img(1240, 900))
    monkeypatch.setattr(r2.headless, "capture_fragments",
                        lambda frags, **k: [_img(1120, 200) for _ in frags])
    src = tmp_path / "dcv_2026-07-18.html"
    src.write_text(_REPORT, encoding="utf-8")

    lineas = r2.process_file(
        src, mode="ambos", out_dir=str(tmp_path / "eml"), plain_dir=str(tmp_path / "plano"),
        sender="a@b.cl", to="c@d.cl", subject_prefix="", width=1240, scale=2.0,
        strip_height=1600, browser=None,
    )
    assert len(lineas) == 2
    nombres = sorted(p.name for p in (tmp_path / "eml").rglob("*.eml"))
    assert nombres == ["dcv_2026-07-18.hibrido.eml", "dcv_2026-07-18.pixel.eml"]
