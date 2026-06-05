"""Conversor determinista Markdown → HTML para el informe analítico.

El informe lo redacta el LLM en Markdown con una estructura estricta (un `# H1`,
bloques `## N. Tema`, sub-secciones `### …`, viñetas `- `, separadores `---` y un
bloque final `## Referencias`). Este módulo renderiza ese Markdown como un HTML
con el estilo de la plantilla de referencia (hero con título + recuadro-disclaimer,
``block-heading`` / ``sub-heading`` / ``bullet-list`` / ``separator``).

Es determinista y testeable sin LLM. No usa una librería Markdown externa a
propósito: necesitamos controlar las clases CSS exactas y mantener el paquete
offline. El parser es a nivel de bloque y tolerante (un bloque sin sub-heading
igual se renderiza como párrafos).
"""

from __future__ import annotations

import html
import re

from .report import DISCLAIMER, REPORT_TITLE

# ── Inline Markdown → HTML (sobre texto ya escapado) ─────────────────────────
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD = re.compile(r"(?:\*\*|__)(.+?)(?:\*\*|__)")
_ITAL = re.compile(r"(?<![\*\w])[\*_](?![\*_\s])(.+?)(?<![\*_\s])[\*_](?![\*\w])")


def _inline(text: str) -> str:
    """Escapa HTML y convierte negrita/itálica/links inline. El texto entra crudo
    y sale escapado: como se escapa primero, los marcadores ``* _ [`` sobreviven."""
    out = html.escape(text)
    out = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITAL.sub(r"<em>\1</em>", out)
    return out


_HR = re.compile(r"[-*_]{3,}")
_BULLET = re.compile(r"[-*+]\s+(.*)")
_ORDERED = re.compile(r"\d+[.)]\s+(.*)")
_HEADING = re.compile(r"(#{1,6})\s+(.*)")


def _render_body(report_md: str) -> tuple[str, str]:
    """Parsea el Markdown → (título del hero, HTML del cuerpo).

    El primer ``# H1`` se usa como título del hero; el resto del documento se
    convierte en el cuerpo (block/sub-headings, párrafos, listas, separadores)."""
    title = REPORT_TITLE
    found_title = False
    body: list[str] = []
    list_open = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            body.append("</ul>")
            list_open = False

    for raw in (report_md or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            close_list()
            continue

        if _HR.fullmatch(stripped):
            close_list()
            body.append('<hr class="separator">')
            continue

        h = _HEADING.match(stripped)
        if h:
            level, content = len(h.group(1)), h.group(2).strip()
            if level == 1 and not found_title:
                title, found_title = content, True
                continue
            close_list()
            cls = "block-heading" if level <= 2 else "sub-heading"
            tag = "h2" if level <= 2 else "h3"
            body.append(f'<{tag} class="{cls}">{_inline(content)}</{tag}>')
            continue

        m = _BULLET.match(stripped) or _ORDERED.match(stripped)
        if m:
            if not list_open:
                body.append('<ul class="bullet-list">')
                list_open = True
            body.append(f"<li>{_inline(m.group(1).strip())}</li>")
            continue

        close_list()
        body.append(f"<p>{_inline(stripped)}</p>")

    close_list()
    return title, "\n".join(body)


_HTML_SHELL = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  :root {
    --blue: #0b3766;
    --blue-2: #0a4a86;
    --gray-1: #d9d9d9;
    --gray-2: #efefef;
    --gray-3: #b9b9b9;
    --text: #1f1f1f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: white;
    color: var(--text);
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.45;
  }
  .page {
    width: min(1400px, calc(100% - 40px));
    margin: 20px auto 36px;
  }
  .hero {
    background: var(--gray-1);
    padding: 18px 20px 12px;
    text-align: center;
  }
  .hero h1 {
    margin: 0;
    color: var(--blue);
    font-size: 30px;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: 0.2px;
  }
  .hero .subtitle {
    margin-top: 14px;
    background: #cfcfcf;
    padding: 8px 14px;
    font-size: 14px;
    font-weight: 700;
    color: #2a2a2a;
    text-align: center;
  }
  .content {
    margin-top: 26px;
  }
  .main-heading {
    color: var(--blue);
    font-size: 22px;
    font-weight: 800;
    margin: 0 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--gray-3);
  }
  .section-heading {
    color: var(--blue);
    font-size: 20px;
    font-weight: 800;
    margin: 18px 0 10px;
  }
  .block-heading {
    color: var(--blue);
    font-size: 22px;
    font-weight: 800;
    margin: 26px 0 8px;
    padding-top: 2px;
  }
  .sub-heading {
    color: var(--blue-2);
    font-size: 16px;
    font-weight: 800;
    margin: 14px 0 8px;
  }
  p {
    margin: 8px 0;
    font-size: 16px;
  }
  .bullet-list {
    margin: 8px 0 14px 0;
    padding-left: 24px;
  }
  .bullet-list li {
    margin: 6px 0;
    font-size: 16px;
  }
  .separator {
    border: 0;
    border-top: 1px solid #c8c8c8;
    margin: 16px 0 18px;
  }
  .spacer { height: 10px; }
  strong { font-weight: 800; }
  em { font-style: italic; }
  @media print {
    .page { width: auto; margin: 12mm; }
    .hero { break-inside: avoid; }
    h1, h2, h3 { break-after: avoid; }
    ul, p { break-inside: avoid; }
  }
</style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>__TITLE__</h1>
      <div class="subtitle">__SUBTITLE__</div>
    </div>
    <div class="content">
__BODY__
    </div>
  </div>
</body>
</html>
"""


def render_html_report(
    report_md: str,
    *,
    subtitle: str | None = None,
    generated_date: str | None = None,  # reservado para un futuro pie de página
) -> str:
    """Markdown del informe → documento HTML completo con el estilo de referencia.

    ``subtitle`` es el texto del recuadro bajo el título (default: el disclaimer de
    IA). ``generated_date`` se acepta por compatibilidad (no se renderiza aún)."""
    title, body = _render_body(report_md)
    return (
        _HTML_SHELL.replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", html.escape(subtitle or DISCLAIMER))
        .replace("__BODY__", body)
    )
