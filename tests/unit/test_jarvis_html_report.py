"""Tests del conversor Markdown → HTML del informe analítico (sin LLM)."""

from __future__ import annotations

import pytest

from jarvis_news.html_report import render_html_report
from jarvis_news.report import DISCLAIMER

_SAMPLE_MD = """# Informe Analítico de Coyuntura Económica y Financiera

## 1. Política Fiscal

### Síntesis técnica
El déficit estructural llegó a -3,6% del PIB en 2025.

### Implicancias económicas
- Menor capacidad contracíclica.
- Mayor vulnerabilidad ante **shocks** externos.

---

## 2. Mercados

### Síntesis técnica
El IPSA cayó 3,36% en la jornada.

## Referencias
- El Mercurio. (2026, 6 de marzo). Fondos fiscales en mínimos.
- Diario Financiero. (2026, 6 de marzo). Erosión de la regla fiscal.
"""


@pytest.mark.unit
class TestRenderHtmlReport:
    def test_is_standalone_html_document(self) -> None:
        html = render_html_report(_SAMPLE_MD)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "<html lang=\"es\">" in html
        assert html.rstrip().endswith("</html>")

    def test_hero_uses_h1_title(self) -> None:
        html = render_html_report(_SAMPLE_MD)
        assert '<div class="hero">' in html
        assert "<h1>Informe Analítico de Coyuntura Económica y Financiera</h1>" in html

    def test_disclaimer_box_present(self) -> None:
        html = render_html_report(_SAMPLE_MD)
        assert '<div class="subtitle">' in html
        assert DISCLAIMER in html

    def test_block_and_sub_headings(self) -> None:
        html = render_html_report(_SAMPLE_MD)
        assert '<h2 class="block-heading">1. Política Fiscal</h2>' in html
        assert '<h3 class="sub-heading">Síntesis técnica</h3>' in html

    def test_bullets_become_bullet_list(self) -> None:
        html = render_html_report(_SAMPLE_MD)
        assert '<ul class="bullet-list">' in html
        assert "<li>Menor capacidad contracíclica.</li>" in html

    def test_separator_rendered(self) -> None:
        assert '<hr class="separator">' in render_html_report(_SAMPLE_MD)

    def test_references_section(self) -> None:
        html = render_html_report(_SAMPLE_MD)
        assert '<h2 class="block-heading">Referencias</h2>' in html
        assert "El Mercurio. (2026, 6 de marzo). Fondos fiscales en mínimos." in html

    def test_inline_bold_converted(self) -> None:
        html = render_html_report(_SAMPLE_MD)
        assert "<strong>shocks</strong>" in html

    def test_escapes_html_in_content(self) -> None:
        html = render_html_report("## Riesgos <test> & cía\n\nTexto con <b> y & amp.")
        assert "<test>" not in html
        assert "&lt;test&gt; &amp; cía" in html

    def test_h1_optional_uses_default_title(self) -> None:
        # Sin H1 explícito, cae al título por defecto del informe.
        html = render_html_report("## 1. Tema\n\nContenido.")
        assert "<h1>Informe Analítico de Coyuntura Económica y Financiera</h1>" in html

    def test_custom_subtitle(self) -> None:
        html = render_html_report("## Tema\n\nx", subtitle="Borrador interno")
        assert "Borrador interno" in html
