"""Unit tests del conversor HTML → .eml en modo PASS-THROUGH (scripts/reports_to_eml.py).

El script se carga por ruta (no es un paquete). Solo se prueban las piezas del modo
pass-through (tu HTML editado → correo), que no requieren parquets ni LLM.
"""

from __future__ import annotations

import email
import importlib.util
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

        plain_file = plain / "reporte_ffmm.plain.html"
        eml_file = out / "reporte_ffmm.eml"
        assert plain_file.exists() and eml_file.exists()

        # HTML plano: sin plotly interactivo, con el PNG
        pt = plain_file.read_text(encoding="utf-8")
        assert "Plotly.newPlot" not in pt and "data:image/png" in pt

        # .eml parseable: cuerpo HTML + imagen inline; asunto con el título
        m = email.message_from_bytes(eml_file.read_bytes())
        types = {p.get_content_type() for p in m.walk()}
        assert "text/html" in types
        assert "image/png" in types
        assert "Informe FFMM" in m["Subject"]
