"""Unit tests del conversor HTML → .eml en modo PASS-THROUGH (scripts/reports_to_eml.py).

El script se carga por ruta (no es un paquete). Solo se prueban las piezas del modo
pass-through (tu HTML editado → correo), que no requieren parquets ni LLM.
"""

from __future__ import annotations

import base64
import email
import importlib.util
import io
import pathlib

import pytest

_R2E_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "reports_to_eml.py"


def _load_r2e():
    spec = importlib.util.spec_from_file_location("reports_to_eml_under_test", _R2E_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


r2e = _load_r2e()

# PNG 1x1 válido (base64) para ejercitar el cid: de una imagen pegada/insertada.
_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_PNG_FIG = (
    '<figure class="report-chart cb-inserted">'
    f'<img src="data:image/png;base64,{_PNG_1X1}" alt="g" style="width:100%"></figure>'
)
_INTERACTIVE_FIG = (
    '<figure class="report-chart cb-inserted"><div id="cbfig-x1"></div>'
    '<script>var f={};Plotly.newPlot("cbfig-x1",f.data,f.layout,{});</script></figure>'
)
_SAMPLE = (
    '<!DOCTYPE html><html><head><title>Informe FFMM</title><style>.x{}</style></head>'
    '<body><div class="page"><div class="hero"><h1>Informe FFMM</h1>'
    '<div class="subtitle">sub</div></div><div class="content">'
    '<section class="overview"><h2 class="block-heading">Síntesis</h2>'
    '<div class="synthesis-body" data-synthesis-body><p>Texto <strong>fuerte</strong>.</p></div></section>'
    '<hr class="separator">'
    '<section class="dataset-block" data-dataset-id="flujos_ffmm" data-chart-type="grouped_bar" data-status="ok">'
    '<h2 class="block-heading">1. Flujos</h2><p class="dataset-meta">flujos_ffmm · US$ Mill.</p>'
    '<div class="section-text" data-text-slot="flujos_ffmm"><p>Parrafo.</p></div>'
    f"{_PNG_FIG}</section></div></body></html>"
)
# Informe CURADO (curated_report.py): gráfico como <svg> inline + tooltip JS.
_CURATED_SVG = (
    '<svg class="report-chart" viewBox="0 0 760 320" width="100%" role="img" '
    'xmlns="http://www.w3.org/2000/svg" font-family="Arial, Helvetica, sans-serif">'
    '<rect x="0" y="0" width="760" height="320" fill="white"/>'
    '<rect x="60" y="100" width="40" height="140" fill="#0b3766"/>'
    '<rect x="120" y="284" width="67" height="18" fill="transparent" stroke="none"/>'
    '<text x="140" y="294" font-size="11" fill="#444">Tipo 1</text></svg>'
)
_CURATED_SAMPLE = (
    '<!DOCTYPE html><html><head><title>Informe Fondos Mutuos</title><style>.x{}</style></head>'
    '<body><div class="page"><div class="report-title">Informe Fondos Mutuos</div>'
    '<div class="subtitle">disclaimer</div>'
    '<div class="report-synthesis"><div class="synthesis-title">Síntesis</div>'
    '<div class="synthesis-body" data-synthesis-body><p>Escrito a mano.</p></div></div>'
    '<div class="section-banner">Flujos</div>'
    '<div class="block" data-chart="grouped_bar"><div class="block-title">Flujos por tipo</div>'
    '<div class="block-unit">(US$ Mill.)</div>'
    '<div class="section-text" data-text-slot="ffmm:flujos"><p>Parrafo.</p></div>'
    f"{_CURATED_SVG}</div></div>"
    '<div id="chart-tip" role="tooltip"></div><script>var tip=1;</script></body></html>'
)


def _html_body(msg: email.message.EmailMessage) -> str:
    """Cuerpo ``text/html`` del mensaje (no el adjunto)."""
    return next(
        p.get_content() for p in msg.walk()
        if p.get_content_type() == "text/html" and p.get_content_disposition() is None
    )


@pytest.mark.unit
class TestStripInteractivePlotly:
    def test_removes_interactive_keeps_png(self) -> None:
        html, n = r2e.strip_interactive_plotly(
            _SAMPLE + _INTERACTIVE_FIG + '<script id="cb-plotly-lib">x</script>'
        )
        assert n == 1
        assert "Plotly.newPlot" not in html
        assert "cb-plotly-lib" not in html
        assert "data:image/png" in html  # el gráfico PNG se conserva

    def test_no_interactive_is_noop_count(self) -> None:
        html, n = r2e.strip_interactive_plotly(_SAMPLE)
        assert n == 0 and "data:image/png" in html


@pytest.mark.unit
class TestInlineCss:
    def test_inlines_known_classes(self) -> None:
        html = r2e.inline_report_css(_SAMPLE)
        assert 'class="block-heading" style="color:#0b3766' in html
        assert 'class="dataset-meta" style=' in html
        assert "<p style=" in html

    def test_does_not_touch_tags_with_existing_style(self) -> None:
        # la img del chartbuilder ya trae style propio: no se le inyecta otro
        html = r2e.inline_report_css(_SAMPLE)
        assert html.count('style="width:100%"') == 1

    def test_extract_title(self) -> None:
        assert r2e._extract_title(_SAMPLE) == "Informe FFMM"


def _card(source_id: str, title: str, *, wide: bool = False) -> str:
    cls = "card card-wide" if wide else "card"
    return (
        f'<div class="{cls}" data-block-status="mvp" data-source="{source_id}">'
        f'<div class="card-head"><div class="block-title">{title}</div></div>'
        f'<div class="card-body">{_CURATED_SVG}</div></div>'
    )


@pytest.mark.unit
class TestGroupGridCards:
    """``.cards-grid`` (layout de dashboard) → tabla de 2 columnas Outlook-safe:
    CSS Grid no lo soporta el motor Word de Outlook aunque se inline-e."""

    def test_pairs_two_regular_cards_side_by_side(self) -> None:
        html = f'<div class="cards-grid">{_card("a", "A")}{_card("b", "B")}</div>'
        out = r2e._group_grid_cards(html)
        assert '<div class="cards-grid">' not in out
        assert out.count('<td width="50%"') == 2
        assert out.index('data-source="a"') < out.index('data-source="b"')  # orden preservado

    def test_wide_card_gets_its_own_full_row(self) -> None:
        html = f'<div class="cards-grid">{_card("hero", "Hero", wide=True)}{_card("a", "A")}{_card("b", "B")}</div>'
        out = r2e._group_grid_cards(html)
        assert '<td colspan="2"' in out  # la wide ocupa la fila completa
        assert out.count('<td width="50%"') == 2  # a+b se emparejan aparte

    def test_odd_trailing_card_without_wide_still_gets_full_row(self) -> None:
        """Defensivo: si por lo que sea la última no viene marcada ``card-wide``
        (debería, vía ``_wide_block_ids``), igual no queda huérfana a media fila."""
        html = f'<div class="cards-grid">{_card("a", "A")}{_card("b", "B")}{_card("c", "C")}</div>'
        out = r2e._group_grid_cards(html)
        assert out.count('<td width="50%"') == 2       # a+b
        assert out.count('<td colspan="2"') == 1        # c sola

    def test_multiple_grid_sections_are_all_converted(self) -> None:
        html = (
            f'<div class="cards-grid">{_card("a", "A")}{_card("b", "B")}</div>'
            "<div class=\"section-banner\">Otra sección</div>"
            f'<div class="cards-grid">{_card("c", "C")}{_card("d", "D")}</div>'
        )
        out = r2e._group_grid_cards(html)
        assert out.count('<div class="cards-grid">') == 0
        assert out.count('<table role="presentation"') == 2

    def test_content_and_balance_preserved(self) -> None:
        html = f'<div class="cards-grid">{_card("a", "Título A")}{_card("b", "Título B")}</div>'
        out = r2e._group_grid_cards(html)
        assert "Título A" in out and "Título B" in out
        assert out.count("<svg") == 2  # ningún gráfico se pierde en el reordenamiento
        assert out.count("<div") == out.count("</div>")

    def test_no_grid_sections_is_noop(self) -> None:
        assert r2e._group_grid_cards(_CURATED_SAMPLE) == _CURATED_SAMPLE

    def test_card_box_style_gets_inlined(self) -> None:
        """El pareo en tabla no alcanza solo: sin esta regla, ``.card`` queda sin
        estilo en el cuerpo (Outlook ignora el <style> del <head>)."""
        # el marcador ``section-banner`` decide el set CURADO de reglas (donde vive .card)
        html = r2e.inline_report_css(
            '<div class="section-banner">S</div><div class="card" data-source="a">x</div>'
        )
        assert 'style="border:1px solid' in html


@pytest.mark.unit
class TestProcessFilePassthrough:
    def test_writes_eml_and_plain_html(self, tmp_path: pathlib.Path) -> None:
        src = tmp_path / "reporte_ffmm.html"
        src.write_text(_SAMPLE + _INTERACTIVE_FIG, encoding="utf-8")
        out = tmp_path / "eml"
        out.mkdir()
        plain = tmp_path / "plain"

        msg = r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="[T] ", plain_dir=plain,
        )
        assert msg.startswith("OK")

        plain_file = plain / "reporte_ffmm.html"
        eml_file = out / "reporte_ffmm.eml"
        assert plain_file.exists() and eml_file.exists()

        # HTML plano: sin plotly interactivo ni JS, imágenes embebidas (data:)
        pt = plain_file.read_text(encoding="utf-8")
        assert "Plotly.newPlot" not in pt and "data:image/png" in pt
        assert "<script" not in pt and "cid:" not in pt

        # .eml parseable: cuerpo HTML + imagen inline; asunto con el título
        m = email.message_from_bytes(eml_file.read_bytes())
        types = {p.get_content_type() for p in m.walk()}
        assert "text/html" in types
        assert "image/png" in types
        assert "Informe FFMM" in m["Subject"]

    def test_curated_svg_charts_become_inline_images(self, tmp_path: pathlib.Path) -> None:
        """El informe curado trae los gráficos como <svg> inline (Outlook no los
        renderiza): deben salir del correo como PNG referenciados por cid:."""
        src = tmp_path / "ffmm_2026-01-01.html"
        src.write_text(_CURATED_SAMPLE, encoding="utf-8")
        out = tmp_path / "eml"
        out.mkdir()

        r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=tmp_path / "plain",
        )
        # salidas agrupadas por familia (ffmm/)
        m = email.message_from_bytes(
            (out / "ffmm" / "ffmm_2026-01-01.eml").read_bytes(), policy=email.policy.default,
        )
        body = _html_body(m)
        assert "<svg" not in body            # ningún SVG sobrevive en el cuerpo
        assert 'src="cid:chart' in body      # el gráfico va como PNG inline
        # el ADJUNTO sí conserva el SVG vectorial (es tu HTML final tal cual)
        adj = next(p for p in m.walk() if p.get_content_disposition() == "attachment")
        assert "<svg" in adj.get_content()
        # la copia PLANA, en cambio, va sin interacción: PNG embebido, sin SVG ni JS
        plano = (tmp_path / "plain" / "ffmm" / "ffmm_2026-01-01.html").read_text(encoding="utf-8")
        assert "<svg" not in plano and "<script" not in plano
        assert "data:image/png" in plano

    def test_attachment_declares_utf8_and_keeps_accents(self, tmp_path: pathlib.Path) -> None:
        """Una parte text/* SIN charset es us-ascii por RFC 2045: el HTML adjunto se
        abría con los acentos rotos aunque los bytes fueran UTF-8 válidos."""
        src = tmp_path / "fx_2026-01-01.html"
        src.write_text(_CURATED_SAMPLE, encoding="utf-8")
        out = tmp_path / "eml"

        r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=tmp_path / "plain",
        )
        m = email.message_from_bytes(
            (out / "fx" / "fx_2026-01-01.eml").read_bytes(), policy=email.policy.default,
        )
        adj = next(p for p in m.walk() if p.get_content_disposition() == "attachment")
        assert adj.get_content_charset() == "utf-8"
        assert "Síntesis" in adj.get_content()

    def test_attachment_is_the_final_html_the_user_fed_in(self, tmp_path: pathlib.Path) -> None:
        """Flujo real: se edita el informe en el navegador, se baja la "versión final"
        (sin panel 💾/📄) y ESE archivo se adjunta tal cual, con su nombre."""
        src = tmp_path / "fx_2026-01-01.html"
        src.write_text(r2e.make_editable_html(_CURATED_SAMPLE)
                       if hasattr(r2e, "make_editable_html") else _CURATED_SAMPLE, encoding="utf-8")
        out = tmp_path / "eml"

        r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=tmp_path / "plain",
        )
        m = email.message_from_bytes(
            (out / "fx" / "fx_2026-01-01.eml").read_bytes(), policy=email.policy.default,
        )
        adj = next(p for p in m.walk() if p.get_content_disposition() == "attachment")
        adjunto = adj.get_content()
        assert adj.get_filename() == "fx_2026-01-01.html"   # TU nombre, no el interno
        assert "cb-save-widget" not in adjunto          # sin panel 💾/📄
        assert "contenteditable" not in adjunto
        assert "<svg" in adjunto                        # conserva los gráficos vectoriales
        # es TU archivo de entrada, no la copia plana (salvo los CRLF del correo)
        assert adjunto.replace("\r\n", "\n").rstrip() == src.read_text(encoding="utf-8").rstrip()

    def test_images_are_related_siblings_of_the_body(self, tmp_path: pathlib.Path) -> None:
        """Las imágenes cid: deben colgar de un multipart/related HERMANO del cuerpo.
        Anidadas bajo la parte HTML, Outlook las lista como datos adjuntos en vez de
        incrustarlas."""
        src = tmp_path / "reporte_ffmm.html"
        src.write_text(_SAMPLE, encoding="utf-8")
        out = tmp_path / "eml"
        out.mkdir()

        r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=tmp_path / "plain",
        )
        m = email.message_from_bytes(
            (out / "reporte_ffmm.eml").read_bytes(), policy=email.policy.default,
        )
        assert m.get_content_type() == "multipart/mixed"
        related = m.get_payload(0)
        assert related.get_content_type() == "multipart/related"
        kids = [p.get_content_type() for p in related.iter_parts()]
        assert kids[0] == "multipart/alternative"
        assert "image/png" in kids
        # inline y sin filename: con nombre reaparecen en la lista de adjuntos
        img = next(p for p in related.iter_parts() if p.get_content_maintype() == "image")
        assert img.get_content_disposition() == "inline" and img.get_filename() is None
        # el único adjunto de verdad es el HTML navegable, con el nombre de entrada
        assert [p.get_filename() for p in m.iter_parts() if p.get_content_disposition() == "attachment"] == [
            "reporte_ffmm.html"
        ]


@pytest.mark.unit
class TestFamilyFolders:
    """Las salidas se agrupan por familia, igual que ``build_family_report.py``."""

    def test_family_from_filename_prefix(self) -> None:
        assert r2e._family_from_name(pathlib.Path("x/fx_2026-07-18.html")) == "fx"
        assert r2e._family_from_name(pathlib.Path("x/ffmm_2026-06-15.html")) == "ffmm"

    def test_family_falls_back_to_the_parent_folder(self) -> None:
        """HTML renombrado a mano tras editarlo: el prefijo ya no dice la familia,
        pero sigue viviendo en ``curated/<familia>/``."""
        assert r2e._family_from_name(pathlib.Path("curated/fx/reporte_final.html")) == "fx"

    def test_non_family_report_has_no_family(self) -> None:
        # el informe descriptivo (parquet_report.py) no pertenece a una familia
        assert r2e._family_from_name(pathlib.Path("html/reporte_ffmm_2026-06-21.html")) is None

    def test_family_report_lands_in_its_subfolder(self, tmp_path: pathlib.Path) -> None:
        src = tmp_path / "fx_2026-07-18.html"
        src.write_text(_SAMPLE, encoding="utf-8")
        out, plain = tmp_path / "eml", tmp_path / "plain"

        r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=plain,
        )
        assert (out / "fx" / "fx_2026-07-18.eml").exists()
        assert (plain / "fx" / "fx_2026-07-18.html").exists()
        assert [p.name for p in out.iterdir()] == ["fx"]  # nada suelto en la raíz

    def test_non_family_report_stays_flat(self, tmp_path: pathlib.Path) -> None:
        src = tmp_path / "reporte_ffmm_2026-06-21.html"
        src.write_text(_SAMPLE, encoding="utf-8")
        out, plain = tmp_path / "eml", tmp_path / "plain"

        r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=plain,
        )
        assert (out / "reporte_ffmm_2026-06-21.eml").exists()
        assert not (out / "ffmm").exists()

    def test_resolve_family_dir_formats_familia_placeholder(self, tmp_path: pathlib.Path) -> None:
        template = str(tmp_path / "{familia}" / "Correo")
        assert r2e._resolve_family_dir(template, "fx") == tmp_path / "fx" / "Correo"

    def test_resolve_family_dir_without_placeholder_appends_family(self, tmp_path: pathlib.Path) -> None:
        assert r2e._resolve_family_dir(str(tmp_path), "fx") == tmp_path / "fx"

    def test_resolve_family_dir_with_placeholder_and_no_family_collapses_it(self, tmp_path: pathlib.Path) -> None:
        # informe descriptivo (sin familia): el placeholder se limpia, no queda un
        # directorio literal "{familia}" ni un segmento vacío suelto.
        template = str(tmp_path / "{familia}" / "Correo")
        assert r2e._resolve_family_dir(template, None) == tmp_path / "Correo"

    def test_out_and_plain_dir_support_familia_placeholder_end_to_end(self, tmp_path: pathlib.Path) -> None:
        """Réplica del caso real: --out y --plain-dir apuntando a árboles
        DISTINTOS bajo la misma familia (ej. .../fx/Correo vs. .../fx/plano)."""
        src = tmp_path / "fx_2026-07-18.html"
        src.write_text(_SAMPLE, encoding="utf-8")
        out_tpl = str(tmp_path / "{familia}" / "Correo")
        plain_tpl = str(tmp_path / "{familia}" / "plano")

        r2e.process_file_passthrough(
            src, out_tpl, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=plain_tpl,
        )
        assert (tmp_path / "fx" / "Correo" / "fx_2026-07-18.eml").exists()
        assert (tmp_path / "fx" / "plano" / "fx_2026-07-18.html").exists()


@pytest.mark.unit
class TestPastedScreenshotCleanup:
    """Fotos pegadas a mano (captura recortada a ojo): recorte del margen blanco
    horneado en los píxeles + colapso de los "renglones fantasma" que el navegador
    deja al pegar contenido en un área contenteditable."""

    @staticmethod
    def _synthetic_screenshot_b64() -> str:
        """PNG blanco de 200x160 con un cuadrado rojo de 40x40 centrado: simula una
        captura recortada a mano con mucho aire de sobra alrededor del gráfico."""
        from PIL import Image
        img = Image.new("RGB", (200, 160), "white")
        for x in range(80, 120):
            for y in range(60, 100):
                img.putpixel((x, y), (200, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def test_trims_baked_in_whitespace_from_pasted_screenshot(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        b64 = self._synthetic_screenshot_b64()
        html = _SAMPLE.replace(
            '<div class="section-text" data-text-slot="flujos_ffmm"><p>Parrafo.</p></div>',
            f'<div class="section-text" data-text-slot="flujos_ffmm"><p>Parrafo.</p>'
            f'<img src="data:image/png;base64,{b64}"></div>',
        )
        src = tmp_path / "reporte_ffmm.html"
        src.write_text(html, encoding="utf-8")
        out = tmp_path / "eml"
        out.mkdir()

        r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=tmp_path / "plain",
        )
        m = email.message_from_bytes(
            (out / "reporte_ffmm.eml").read_bytes(), policy=email.policy.default,
        )
        pasted = [
            p for p in m.walk()
            if p.get_content_type() == "image/png" and (p.get("Content-ID") or "").startswith("<pasted")
        ]
        assert pasted, "no se encontró la imagen pegada como adjunto inline"
        trimmed = Image.open(io.BytesIO(pasted[0].get_content()))
        # recortada: bien más chica que el lienzo blanco original (200x160)...
        assert trimmed.width < 200 and trimmed.height < 160
        # ...pero sigue conteniendo el cuadrado de 40x40 completo.
        assert trimmed.width >= 40 and trimmed.height >= 40

    def test_collapses_paste_ghost_lines_in_body_but_not_the_attachment(self, tmp_path: pathlib.Path) -> None:
        """El "renglón fantasma" (<div><br></div>) que deja el navegador al pegar no
        debe sumar espacio en el CUERPO del correo; el adjunto (tu HTML tal cual)
        no se toca, y un contenedor funcional real (#chart-tip, con id+role) nunca
        se confunde con basura de pegado."""
        html = _CURATED_SAMPLE.replace(
            '<div class="section-text" data-text-slot="ffmm:flujos"><p>Parrafo.</p></div>',
            '<div class="section-text" data-text-slot="ffmm:flujos"><p>Parrafo.</p>'
            '<div><br></div><div><br></div></div>',
        )
        src = tmp_path / "ffmm_2026-01-01.html"
        src.write_text(html, encoding="utf-8")
        out = tmp_path / "eml"
        out.mkdir()

        r2e.process_file_passthrough(
            src, out, sender="a@x.cl", to="b@y.cl", subject_prefix="", plain_dir=tmp_path / "plain",
        )
        m = email.message_from_bytes(
            (out / "ffmm" / "ffmm_2026-01-01.eml").read_bytes(), policy=email.policy.default,
        )
        body = _html_body(m)
        assert "<div><br" not in body  # colapsado en el CUERPO

        adj = next(p for p in m.walk() if p.get_content_disposition() == "attachment")
        adjunto = adj.get_content()
        assert "<div><br></div><div><br></div>" in adjunto  # el adjunto NO se toca
        assert 'id="chart-tip"' in adjunto                   # tooltip funcional intacto
