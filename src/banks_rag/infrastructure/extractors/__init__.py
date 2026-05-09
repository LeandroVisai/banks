"""Adaptadores de extracción: PDF, Excel, charts, encoding fixers."""

from .chart_detector import (
    extract_visual_pages,
    is_pymupdf_available,
    page_has_visuals,
)
from .doc_metadata import (
    detect_date,
    detect_doc_type,
    detect_institution,
    slugify_document_id,
)
from .encoding_fixers import fix_all, fix_bcch_font, fix_mojibake
from .excel_extractor import cell_to_str, extract_cell_chunks, find_header_row
from .pdf_extractor import extract_pages
from .text_normalizer import normalize_page_text

__all__ = [
    # encoding
    "fix_bcch_font",
    "fix_mojibake",
    "fix_all",
    "normalize_page_text",
    # metadata
    "detect_doc_type",
    "detect_institution",
    "detect_date",
    "slugify_document_id",
    # pdf
    "extract_pages",
    # excel
    "extract_cell_chunks",
    "find_header_row",
    "cell_to_str",
    # charts
    "is_pymupdf_available",
    "page_has_visuals",
    "extract_visual_pages",
]
