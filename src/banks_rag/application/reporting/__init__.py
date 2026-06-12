"""Informes batch sobre el catálogo de parquets (caso de uso de aplicación).

Genera el informe descriptivo de datasets (estilo jarvis_news pero sobre los
datos del catálogo): selección por segmento / ids / texto libre, un párrafo
por dataset vía mini-loop de tool-calling, síntesis global y render MD/HTML.
Entrypoint CLI: ``scripts/parquet_report.py``.
"""

from .html_render import render_parquet_report_html
from .parquet_report import (
    DatasetSection,
    DatasetSelection,
    ParquetReport,
    generate_parquet_report,
    select_datasets,
)

__all__ = [
    "DatasetSection",
    "DatasetSelection",
    "ParquetReport",
    "generate_parquet_report",
    "render_parquet_report_html",
    "select_datasets",
]
