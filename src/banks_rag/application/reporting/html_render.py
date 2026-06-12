"""Renderer HTML del informe descriptivo de parquets (determinista, sin LLM).

A diferencia del conversor MD→HTML de ``jarvis_news`` (que parsea Markdown a
ciegas), aquí se renderiza DESDE la estructura (``ParquetReport``): así cada
sección lleva exactamente su ``data-dataset-id`` / ``data-chart-type`` del
catálogo y un ``chart-placeholder`` listo para insertar el gráfico en fase 2,
sin heurísticas de re-asociación. El shell visual (hero + paleta + reglas de
impresión) sigue la plantilla de los informes del proyecto; es código propio
(``jarvis_news`` es un paquete aislado y no se importa desde ``banks_rag``).
"""

from __future__ import annotations

import html
import re

from .parquet_report import ParquetReport

DISCLAIMER = (
    "Informe generado automáticamente con IA sobre el catálogo de parquets — "
    "uso interno; validar cifras antes de citar."
)

# ── Inline Markdown → HTML (sobre texto ya escapado) ─────────────────────────
_BOLD = re.compile(r"(?:\*\*|__)(.+?)(?:\*\*|__)")
_ITAL = re.compile(r"(?<![\*\w])[\*_](?![\*_\s])(.+?)(?<![\*_\s])[\*_](?![\*\w])")
_CODE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    """Escapa HTML y convierte negrita/itálica/código inline. Se escapa primero,
    así los marcadores ``* _ ``` sobreviven al escape y se convierten después."""
    out = html.escape(text)
    out = _CODE.sub(r"<code>\1</code>", out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITAL.sub(r"<em>\1</em>", out)
    return out


_BULLET = re.compile(r"[-*+]\s+(.*)")


def _prose_to_html(text: str) -> str:
    """Convierte la prosa del LLM (párrafos + viñetas simples) a HTML. Es el
    único punto donde el HTML depende de texto libre; todo pasa por escape."""
    body: list[str] = []
    list_open = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            body.append("</ul>")
            list_open = False

    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            close_list()
            continue
        m = _BULLET.match(stripped)
        if m:
            if not list_open:
                body.append('<ul class="bullet-list">')
                list_open = True
            body.append(f"<li>{_inline(m.group(1).strip())}</li>")
            continue
        close_list()
        body.append(f"<p>{_inline(stripped)}</p>")

    close_list()
    return "\n".join(body)


def _attr(value: str) -> str:
    return html.escape(value or "", quote=True)


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
  .content { margin-top: 26px; }
  .block-heading {
    color: var(--blue);
    font-size: 22px;
    font-weight: 800;
    margin: 26px 0 8px;
    padding-top: 2px;
  }
  .dataset-meta {
    color: var(--blue-2);
    font-size: 13px;
    font-weight: 700;
    margin: 0 0 8px;
  }
  p { margin: 8px 0; font-size: 16px; }
  .bullet-list { margin: 8px 0 14px 0; padding-left: 24px; }
  .bullet-list li { margin: 6px 0; font-size: 16px; }
  .separator {
    border: 0;
    border-top: 1px solid #c8c8c8;
    margin: 16px 0 18px;
  }
  .dataset-block { margin-bottom: 6px; }
  /* Fase 2: aquí se monta el gráfico del dataset (data-chart-type del
     catálogo). En fase 1 no ocupa espacio. */
  .chart-placeholder { min-height: 0; }
  code {
    font-family: "SF Mono", Consolas, monospace;
    font-size: 14px;
    background: #f2f2f2;
    padding: 1px 4px;
  }
  strong { font-weight: 800; }
  em { font-style: italic; }
  @media print {
    .page { width: auto; margin: 12mm; }
    .hero { break-inside: avoid; }
    h1, h2, h3 { break-after: avoid; }
    section, ul, p { break-inside: avoid; }
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


def render_parquet_report_html(report: ParquetReport, *, subtitle: str | None = None) -> str:
    """``ParquetReport`` → documento HTML completo con secciones por dataset."""
    body: list[str] = []

    body.append('<section class="overview">')
    body.append('<h2 class="block-heading">Síntesis</h2>')
    body.append(_prose_to_html(report.overview_md))
    body.append("</section>")

    for i, s in enumerate(report.sections, 1):
        meta = [f"<code>{_inline(s.dataset_id)}</code>"]
        if s.unit:
            meta.append(_inline(s.unit))
        if s.segment:
            meta.append(_inline(s.segment))
        if s.last_date:
            meta.append(f"datos hasta {_inline(s.last_date)}")
        body.append('<hr class="separator">')
        body.append(
            f'<section class="dataset-block" data-dataset-id="{_attr(s.dataset_id)}" '
            f'data-chart-type="{_attr(s.chart_type)}" data-status="{_attr(s.status)}">'
        )
        body.append(f'<h2 class="block-heading">{i}. {_inline(s.name)}</h2>')
        body.append(f'<p class="dataset-meta">{" · ".join(meta)}</p>')
        body.append(f"<p>{_inline(s.paragraph)}</p>")
        body.append(
            '<div class="chart-placeholder" aria-hidden="true">'
            f"<!-- fase 2: gráfico {_attr(s.chart_type)} --></div>"
        )
        body.append("</section>")

    counts = report.status_counts()
    body.append('<hr class="separator">')
    body.append('<section class="references">')
    body.append('<h2 class="block-heading">Referencias y metodología</h2>')
    body.append('<ul class="bullet-list">')
    body.append(
        f"<li>Datasets analizados: {len(report.sections)} "
        f"(ok: {counts.get('ok', 0)}, sin datos: {counts.get('no_data', 0)}, "
        f"sin evidencia: {counts.get('ungrounded', 0)}, "
        f"con error: {counts.get('error', 0)})</li>"
    )
    if report.missing_ids:
        body.append(
            f"<li>No hallados en el catálogo: {_inline(', '.join(report.missing_ids))}</li>"
        )
    body.append(
        f"<li>Ventanas de análisis: {_inline(', '.join(report.windows))} "
        "(ancladas a la última fecha disponible de cada dataset)</li>"
    )
    body.append(f"<li>Generado: {_inline(report.generated_at)}</li>")
    body.append("<li>Fuente: sql_catalog/parquet_catalog.yaml</li>")
    body.append("</ul>")
    body.append("</section>")

    return (
        _HTML_SHELL.replace("__TITLE__", html.escape(report.title))
        .replace("__SUBTITLE__", html.escape(subtitle or DISCLAIMER))
        .replace("__BODY__", "\n".join(body))
    )
