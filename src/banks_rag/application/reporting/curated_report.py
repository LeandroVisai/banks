"""Builder + render HTML de un informe CURADO por familia (Python puro, sin LLM).

Toma un ``FamilyReportSpec`` (orden + bloques de un informe real, p.ej. ffmm),
corre la transform de cada bloque sobre el parquet REAL y arma un HTML con la
estructura del informe: banners de sección, título por bloque, un slot de texto
vacío (id estable, se llena después con el LLM en el H100) y el gráfico SVG.

MVP: dibuja los bloques cuyo gráfico el renderer ya soporta (multi-línea /
composición); los demás quedan como tarjeta-placeholder marcada con su tipo
pendiente (EXP) o "sin datos reproducibles" (SKIP). Así la estructura y el ORDEN
del informe quedan completos desde la primera iteración.
"""

from __future__ import annotations

import dataclasses
import html
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ParquetDataset,
    get_dataset,
    get_parquet_dir,
    load_parquet_catalog,
)

from .parquet_facts import HtmlTable, compute_series
from .report_spec import STATUS_SKIP, FamilyReportSpec, ReportBlock
from .series_transforms import get_transform
from .svg_chart import render_plot_svg, renders_natively

log = logging.getLogger(__name__)

# Familias objetivo que NO se pueden aproximar como serie (tabla): placeholder.
_TABLE_CHARTS = frozenset({"heatmap_table"})


@dataclass
class CuratedBlock:
    """Resultado de procesar un ``ReportBlock``: gráfico o placeholder + metadata."""

    block: ReportBlock
    render_kind: str       # "chart" | "placeholder" | "skip"
    body_html: str         # SVG inline o inner del placeholder
    reason: str = ""       # por qué placeholder (para la tarjeta)
    preliminary: bool = False  # gráfico dibujado como línea, tipo final pendiente


@dataclass
class CuratedReport:
    spec: FamilyReportSpec
    generated_at: str
    blocks: list[CuratedBlock] = field(default_factory=list)

    def render_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.blocks:
            counts[b.render_kind] = counts.get(b.render_kind, 0) + 1
        return counts

    def summary(self) -> str:
        c = self.render_counts()
        prelim = sum(1 for b in self.blocks if b.preliminary)
        return (
            f"gráficos={c.get('chart', 0)} (de ellos preliminares={prelim}) "
            f"placeholders={c.get('placeholder', 0)} skip={c.get('skip', 0)} "
            f"/ {len(self.blocks)} bloques"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────


def _process_block(
    block: ReportBlock, *, entries: list[ParquetDataset], parquet_dir: Path,
) -> CuratedBlock:
    """Dibuja el bloque siempre que el dato sea graficable como serie/composición.

    Criterio: SKIP (sin parquet) → tarjeta "sin datos". Tablas-heatmap SIN
    transform implementada → placeholder "pendiente". El resto se dibuja: si tiene
    transform propia (PlotData o HtmlTable) se usa; si no, se cae a la serie
    natural del parquet (VISTA PRELIMINAR). Best-effort: un bloque que falle no
    tumba el resto.
    """
    if block.status == STATUS_SKIP:
        return CuratedBlock(block, "skip", "", "sin parquet reproducible todavía")

    # Tabla cuya transform no está implementada aún: placeholder directo.
    transform = get_transform(block.transform)
    if block.chart in _TABLE_CHARTS and transform is None:
        return CuratedBlock(block, "placeholder", "", f"pendiente: {block.chart}")

    dataset = get_dataset(entries, block.source_id or "")
    if dataset is None:
        log.warning("[%s] source_id %r no está en el catálogo", block.title, block.source_id)
        return CuratedBlock(block, "placeholder", "", f"dataset {block.source_id} no encontrado")

    try:
        if transform is not None:
            result = transform(dataset, parquet_dir, block.params)
        else:
            # Vista preliminar: serie natural, filtrada por las categorías pedidas.
            cf = list(block.params.get("funds") or block.params.get("types") or []) or None
            result = compute_series(dataset, parquet_dir, category_filter=cf)
    except Exception:
        log.exception("[%s] el cálculo de la serie falló", block.title)
        return CuratedBlock(block, "placeholder", "", "error al calcular la serie")

    # Resultado HTML inline (tablas con color — heatmap DCV).
    if isinstance(result, HtmlTable):
        return CuratedBlock(block, "chart", result.html, preliminary=False)

    plot = result
    if plot is None or plot.is_empty():
        return CuratedBlock(block, "placeholder", "", "sin serie en el parquet")

    svg = render_plot_svg(plot, chart=block.chart)
    if svg is None:
        return CuratedBlock(block, "placeholder", "", "serie no graficable como línea")
    # Preliminar solo si la forma del dato NO permite el tipo objetivo (cae a
    # línea/composición). Con su transform propia, el bloque sale en su tipo final.
    preliminary = not renders_natively(plot.kind, block.chart)
    return CuratedBlock(block, "chart", svg, preliminary=preliminary)


def _resolve_text_slots(spec: FamilyReportSpec) -> list[ReportBlock]:
    """Asigna ``text_slot`` automáticamente al PRIMER bloque que use cada
    ``source_id``, si el bloque no tiene ya un slot explícito en el spec.

    Esto garantiza que haya exactamente UN párrafo descriptivo por parquet, justo
    encima del primer gráfico que lo usa — independientemente del orden en que el
    spec declare los bloques. Los slots explícitos del spec tienen precedencia."""
    seen: set[str] = set()
    resolved: list[ReportBlock] = []
    for b in spec.blocks:
        slot = b.text_slot
        if not slot and b.source_id and b.source_id not in seen and b.status != STATUS_SKIP:
            slot = f"{spec.family}:{b.source_id}"
        if b.source_id:
            seen.add(b.source_id)
        resolved.append(dataclasses.replace(b, text_slot=slot) if slot != b.text_slot else b)
    return resolved


def build_curated_report(
    spec: FamilyReportSpec,
    *,
    entries: list[ParquetDataset] | None = None,
    parquet_dir: Path | None = None,
    catalog_path: Path | str | None = None,
) -> CuratedReport:
    """Procesa todos los bloques del spec. ``entries``/``parquet_dir`` inyectables
    para tests; por defecto carga el catálogo real."""
    if entries is None:
        entries = load_parquet_catalog(catalog_path)
    if parquet_dir is None:
        parquet_dir = get_parquet_dir(catalog_path)

    resolved = _resolve_text_slots(spec)
    blocks = [_process_block(b, entries=entries, parquet_dir=parquet_dir) for b in resolved]
    report = CuratedReport(spec=spec, generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"), blocks=blocks)
    log.info("Informe curado %s: %s", spec.family, report.summary())
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Render HTML
# ─────────────────────────────────────────────────────────────────────────────

_SHELL = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  :root { --blue:#0b3766; --banner:#4a5a72; --text:#1f1f1f; --muted:#777; }
  * { box-sizing: border-box; }
  body { margin:0; background:#fff; color:var(--text); font-family:Arial, Helvetica, sans-serif; line-height:1.45; }
  .page { width:min(1200px, calc(100% - 40px)); margin:18px auto 40px; }
  .report-title { background:var(--banner); color:#fff; text-align:center; font-size:24px; font-weight:800; padding:12px 16px; letter-spacing:.3px; }
  .subtitle { text-align:center; color:var(--muted); font-size:12px; margin:8px 0 4px; }
  .section-banner { background:var(--banner); color:#fff; text-align:center; font-size:18px; font-weight:800; padding:8px 14px; margin:30px 0 6px; }
  .block { margin:14px 0 8px; page-break-inside:avoid; }
  .block-title { color:var(--blue); font-size:16px; font-weight:700; margin:12px 0 2px; }
  .block-unit { color:var(--muted); font-size:12px; margin:0 0 6px; }
  .block-note { color:var(--muted); font-size:11px; font-style:italic; margin:0 0 6px; }
  .prelim-note { color:#9a4b00; font-size:11px; margin:0 0 2px; }
  .section-text { min-height:18px; margin:4px 0 10px; color:var(--text); font-size:14px; }
  .section-text:empty::before { content:"—"; color:#cfcfcf; }
  .report-chart { display:block; max-width:760px; margin:6px auto; }
  .placeholder-card { border:1px dashed #bcbcbc; background:#fafafa; color:var(--muted); border-radius:6px; padding:18px; text-align:center; font-size:13px; max-width:760px; margin:6px auto; }
  .placeholder-card .kind { font-weight:700; color:#9a4b00; }
  .placeholder-card.skip .kind { color:#8a8a8a; }
  @media print { .section-banner, .report-chart, .placeholder-card, .block { break-inside:avoid; } }
</style>
</head>
<body>
  <div class="page">
    <div class="report-title">__TITLE__</div>
    <div class="subtitle">__SUBTITLE__</div>
__BODY__
  </div>
</body>
</html>
"""

_DISCLAIMER = (
    "Informe generado automáticamente desde los parquets (sin IA en el texto) — "
    "uso interno; validar cifras antes de citar."
)


def _esc(text: str) -> str:
    return html.escape(text or "")


def _attr(text: str) -> str:
    return html.escape(text or "", quote=True)


def _block_html(cb: CuratedBlock) -> str:
    b = cb.block
    parts = [f'<div class="block-title">{_esc(b.title)}</div>']
    if b.unit:
        parts.append(f'<div class="block-unit">({_esc(b.unit)})</div>')
    if b.note:
        parts.append(f'<div class="block-note">{_esc(b.note)}</div>')
    if b.text_slot:
        parts.append(f'<div class="section-text" data-text-slot="{_attr(b.text_slot)}"></div>')
    if cb.render_kind == "chart":
        if cb.preliminary:
            parts.append(
                '<div class="prelim-note">vista preliminar (línea) — el informe final '
                f"usa <code>{_esc(b.chart)}</code></div>"
            )
        parts.append(cb.body_html)
    elif cb.render_kind == "skip":
        parts.append(
            f'<div class="placeholder-card skip"><span class="kind">Sin datos</span><br>'
            f"{_esc(cb.reason)}</div>"
        )
    else:
        parts.append(
            f'<div class="placeholder-card"><span class="kind">{_esc(cb.reason)}</span><br>'
            f"gráfico de tipo <code>{_esc(b.chart)}</code> — se completa en la próxima iteración</div>"
        )
    return (
        f'<div class="block" data-block-status="{_attr(b.status)}" '
        f'data-chart="{_attr(b.chart)}" data-source="{_attr(b.source_id or "")}">'
        f"{''.join(parts)}</div>"
    )


def render_curated_html(report: CuratedReport, *, subtitle: str | None = None) -> str:
    """``CuratedReport`` → documento HTML con banners de sección + gráficos/slots."""
    body: list[str] = []
    current_section: str | None = None
    for cb in report.blocks:
        if cb.block.section != current_section:
            current_section = cb.block.section
            body.append(f'<div class="section-banner">{_esc(current_section)}</div>')
        body.append(_block_html(cb))

    return (
        _SHELL.replace("__TITLE__", _esc(report.spec.title))
        .replace("__SUBTITLE__", _esc(subtitle or _DISCLAIMER))
        .replace("__BODY__", "\n".join(body))
    )
