"""Inyección de gráficos en el HTML del informe — proceso SEPARADO y posterior.

Diseño pedido: el informe (``scripts/parquet_report.py``) genera Markdown + HTML
de SOLO TEXTO; este módulo es un segundo paso, independiente del LLM, que toma
ese HTML ya generado y le inyecta los gráficos. Por eso vive aparte y solo
necesita el sistema de archivos (catálogo + parquets), no la GPU ni el modelo:
puede correr en el H100, en el Mac o en CI.

La fuente de los datos del gráfico es el PARQUET, no los agregados que vio el
LLM (``compute_series`` los lee directo y reusa ``detect_roles`` para elegir las
mismas columnas que ``compute_facts``). Así el gráfico es una verificación
independiente de la prosa: si discrepan, hay un problema real de datos.

El HTML lo genera nuestro propio renderer (``html_render``), con estructura
determinista: cada sección es ``<section class="dataset-block" data-dataset-id=…
data-status=…>`` con un único ``<div class="chart-placeholder">``. Por eso se
parsea con regex sobre esa estructura conocida, sin meter un parser HTML nuevo.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ParquetDataset,
    get_dataset,
    get_parquet_dir,
    load_parquet_catalog,
)

from .parquet_facts import compute_series
from .svg_chart import render_mini_table_html, render_plot_svg

log = logging.getLogger(__name__)

# Captura el tag de apertura de una sección de dataset, su contenido y su
# placeholder de gráfico (re.DOTALL). El placeholder es uno por sección.
_SECTION_RE = re.compile(
    r'(?P<open><section class="dataset-block" data-dataset-id="(?P<id>[^"]*)"'
    r' data-chart-type="[^"]*" data-status="(?P<status>[^"]*)">)'
    r'(?P<mid>.*?)'
    r'<div class="chart-placeholder"[^>]*>.*?</div>',
    re.DOTALL,
)

_CHART_CSS = (
    "\n  .report-chart { display:block; max-width:760px; margin:10px 0 4px; }"
    "\n  .chart-fallback td, .chart-fallback th { border:1px solid #d8d8d8; padding:2px 8px; }"
    "\n  @media print { .report-chart, .chart-fallback { break-inside:avoid; } }\n"
)


@dataclass
class InjectStats:
    charted: int = 0       # secciones con gráfico SVG
    tabled: int = 0        # secciones con mini-tabla (familia no graficable)
    no_series: int = 0     # status ok pero sin parquet / sin serie
    skipped: int = 0       # status != ok (no_data/error): no se grafican
    unknown_id: int = 0    # dataset-id no encontrado en el catálogo

    def summary(self) -> str:
        return (
            f"gráficos={self.charted} tablas={self.tabled} "
            f"sin_serie={self.no_series} omitidos={self.skipped} "
            f"id_desconocido={self.unknown_id}"
        )


def _placeholder(inner: str) -> str:
    return f'<div class="chart-placeholder">{inner}</div>'


def _render_section_chart(
    dataset_id: str,
    status: str,
    *,
    entries: list[ParquetDataset],
    parquet_dir: Path,
    stats: InjectStats,
) -> str:
    """Contenido del placeholder para una sección. Best-effort: un dataset que
    reviente no debe tumbar la inyección del resto."""
    if status != "ok":
        stats.skipped += 1
        return _placeholder("")

    dataset = get_dataset(entries, dataset_id)
    if dataset is None:
        stats.unknown_id += 1
        log.warning("dataset-id %r del HTML no está en el catálogo", dataset_id)
        return _placeholder("")

    try:
        plot = compute_series(dataset, parquet_dir)
    except Exception:
        log.exception("[%s] compute_series falló", dataset_id)
        stats.no_series += 1
        return _placeholder("")

    if plot is None or plot.is_empty():
        stats.no_series += 1
        return _placeholder("")

    svg = render_plot_svg(plot)
    if svg is None:  # familia "table" → mini-tabla verificable
        stats.tabled += 1
        return _placeholder(render_mini_table_html(plot))
    stats.charted += 1
    return _placeholder(svg)


def inject_charts_into_html(
    html_text: str,
    *,
    entries: list[ParquetDataset] | None = None,
    parquet_dir: Path | None = None,
    catalog_path: Path | str | None = None,
) -> tuple[str, InjectStats]:
    """HTML de solo texto → HTML con gráficos SVG inyectados en los placeholders.

    ``entries``/``parquet_dir`` son inyectables para tests; por defecto se carga
    el catálogo y su ``parquet_dir`` reales.
    """
    if entries is None:
        entries = load_parquet_catalog(catalog_path)
    if parquet_dir is None:
        parquet_dir = get_parquet_dir(catalog_path)

    stats = InjectStats()

    def _sub(m: re.Match[str]) -> str:
        chart = _render_section_chart(
            m.group("id"), m.group("status"),
            entries=entries, parquet_dir=parquet_dir, stats=stats,
        )
        return f"{m.group('open')}{m.group('mid')}{chart}"

    out = _SECTION_RE.sub(_sub, html_text)
    if "</style>" in out and ".report-chart" not in html_text:
        out = out.replace("</style>", f"{_CHART_CSS}</style>", 1)
    log.info("Inyección de gráficos: %s", stats.summary())
    return out, stats
