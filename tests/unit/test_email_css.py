"""Unit tests del inliner genérico de CSS para correo (``email_css.py``).

Lo que se prueba es justo lo que el mapa hardcodeado del script viejo NO hacía:
resolver las custom properties, respetar la especificidad, y dejar afuera lo que el
motor de Word de Outlook no sabe interpretar.
"""

from __future__ import annotations

from banks_rag.application.reporting.email_css import inline_css, parse_stylesheet

_CSS = """
<style>
  :root { --blue:#0b3766; --muted:#777; }
  * { box-sizing: border-box; }
  body { margin:0; color:var(--blue); font-family:Arial, sans-serif; }
  .page { width:min(1200px, calc(100% - 40px)); margin:18px auto; }
  .block-title { color:var(--blue); font-size:16px; }
  .section-text { font-size:14px; }
  .card .section-text { font-size:13px; }
  .section-banner + .section-text { margin:8px 0 16px; }
  .block-dates { display:inline-block; padding:2px 8px; }
  .cards-grid { display:grid; grid-template-columns:repeat(2, 1fr); gap:10px; }
  .placeholder-card { border:1px dashed #bcbcbc; max-width:760px; overflow-x:auto; }
  .section-text:empty::before { content:"—"; }
  @media (max-width:860px) { .cards-grid { grid-template-columns:1fr; } }
</style>
"""


def _style_of(html: str, needle: str) -> str:
    """Extrae el ``style="…"`` del tag que contiene ``needle``."""
    start = html.index(needle)
    open_at = start if needle.startswith("<") else html.rindex("<", 0, start)
    tag = html[open_at:html.index(">", start) + 1]
    return tag.split('style="', 1)[1].split('"', 1)[0] if 'style="' in tag else ""


def test_resuelve_custom_properties_de_root():
    out = inline_css(f'{_CSS}<body><div class="block-title">T</div></body>')
    assert "color:#0b3766" in _style_of(out, 'class="block-title"')
    assert "var(" not in out


def test_la_regla_de_body_tambien_se_vuelca():
    # El mapa hardcodeado del script viejo no tenía 'body': sin font-family en el
    # elemento, Outlook cae a Times New Roman y el correo se ve "de otro formato".
    out = inline_css(f"{_CSS}<body><p>hola</p></body>")
    assert "font-family:Arial, sans-serif" in _style_of(out, "<body")


def test_especificidad_descendiente_gana():
    out = inline_css(f'{_CSS}<body><div class="card"><div class="section-text">x</div></div></body>')
    assert "font-size:13px" in _style_of(out, 'class="section-text"')


def test_hermano_adyacente():
    html = ('<body><div class="section-banner">B</div>'
            '<div class="section-text">tras el banner</div>'
            '<div class="section-text">suelto</div></body>')
    out = inline_css(_CSS + html)
    assert "margin:8px 0 16px" in _style_of(out, "tras el banner")
    # El segundo .section-text no sigue a un banner: no lleva ese margen.
    assert "margin:8px 0 16px" not in _style_of(out, "suelto")


def test_el_style_inline_propio_gana_sobre_la_hoja():
    out = inline_css(f'{_CSS}<body><div class="block-title" style="color:#c8102e">T</div></body>')
    style = _style_of(out, 'class="block-title"')
    assert style.endswith("color:#c8102e")
    assert style.count("color:") == 1


def test_descarta_lo_que_word_no_entiende():
    out = inline_css(f'{_CSS}<body><div class="cards-grid"><div class="placeholder-card">x</div></div></body>')
    grid = _style_of(out, 'class="cards-grid"')
    assert "display:grid" not in grid and "gap" not in grid
    card = _style_of(out, 'class="placeholder-card"')
    assert "max-width" not in card and "overflow" not in card
    assert "border:1px dashed #bcbcbc" in card       # lo que sí entiende, se conserva


def test_conserva_display_inline_block():
    out = inline_css(f'{_CSS}<body><div class="block-dates">14-jul</div></body>')
    assert "display:inline-block" in _style_of(out, 'class="block-dates"')


def test_descarta_valores_con_funciones_css():
    # Word no evalúa min()/calc(): si se dejan, descarta la declaración y el bloque
    # se va a ancho completo. El ancho lo pone la tabla contenedora del correo.
    out = inline_css(f'{_CSS}<body><div class="page">x</div></body>')
    style = _style_of(out, 'class="page"')
    assert "width:min(" not in style and "calc(" not in style
    assert "margin:18px auto" in style


def test_no_toca_elementos_marcados_como_finales():
    html = ('<body><img data-email-final src="cid:x@y" width="760" '
            'style="max-width:100%;height:auto"></body>')
    out = inline_css(_CSS + html)
    assert "max-width:100%" in out       # su style se escribió ya pensado para el correo


def test_saca_el_style_del_head():
    out = inline_css(f"{_CSS}<body><p>x</p></body>")
    assert "<style>" not in out


def test_no_altera_los_atributos_del_svg():
    # Un parser que serializara de nuevo el documento pasaría viewBox a minúsculas y
    # rompería el SVG; el recorrido por regex lo deja intacto.
    html = '<body><svg viewBox="0 0 10 10" preserveAspectRatio="none"><text>x</text></svg></body>'
    out = inline_css(_CSS + html)
    assert 'viewBox="0 0 10 10"' in out and 'preserveAspectRatio="none"' in out


def test_las_at_rules_no_entran():
    reglas = parse_stylesheet(_CSS)
    assert not any("1fr" in valor for _s, _o, _p, decls in reglas for _prop, valor in decls
                   if _prop == "grid-template-columns" and valor == "1fr")


def test_extra_es_la_prioridad_mas_baja():
    out = inline_css(f'{_CSS}<body><div class="block-title">T</div></body>',
                     extra={"div": "font-family:Tahoma;font-size:99px"})
    style = _style_of(out, 'class="block-title"')
    assert "font-family:Tahoma" in style      # nadie más define la familia
    assert "font-size:16px" in style          # la hoja gana sobre el extra
    assert "99px" not in style
