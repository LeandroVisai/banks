"""Informes batch sobre el catálogo de parquets (caso de uso de aplicación).

Genera el informe descriptivo de datasets (estilo jarvis_news pero sobre los
datos del catálogo): selección por segmento / ids / texto libre, un párrafo
por dataset vía mini-loop de tool-calling, síntesis global y render MD/HTML.
Entrypoint CLI: ``scripts/parquet_report.py``.
"""

from .chart_inject import InjectStats, inject_charts_into_html
from .curated_report import (
    CuratedBlock,
    CuratedReport,
    build_curated_report,
    fill_synthesis_slot,
    fill_text_slots,
    render_curated_html,
)
from .html_render import render_parquet_report_html
from .parquet_facts import PlotData, PlotSeries, compute_series
from .parquet_report import (
    DatasetSection,
    DatasetSelection,
    ParquetReport,
    generate_parquet_report,
    select_datasets,
)
from .report_spec import FamilyReportSpec, ReportBlock

__all__ = [
    "CuratedBlock",
    "CuratedReport",
    "DatasetSection",
    "DatasetSelection",
    "FamilyReportSpec",
    "InjectStats",
    "ParquetReport",
    "PlotData",
    "PlotSeries",
    "ReportBlock",
    "build_curated_report",
    "compute_series",
    "fill_synthesis_slot",
    "fill_text_slots",
    "generate_parquet_report",
    "inject_charts_into_html",
    "render_curated_html",
    "render_parquet_report_html",
    "select_datasets",
]
