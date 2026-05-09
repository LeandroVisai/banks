"""Document entity — metadata por archivo PDF/Excel."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """Metadata de un documento (1 por archivo PDF/Excel).

    Campos:
        document_id: slug único derivado de la ruta relativa.
        doc_type_category: COMUNICADO | MINUTA | FED_STATEMENT | REPORTE_RESEARCH | MONITOR_PM
        institution: banco_central_chile | federal_reserve | jpmorgan | unknown
        document_date: ISO YYYY-MM-DD o solo año si no se detecta día/mes.
    """

    document_id: str
    filename: str
    filepath: str
    doc_type_category: str
    institution: str
    document_date: str | None
    total_pages: int
    total_chunks: int
    char_count: int
    extraction_warnings: list[str] = field(default_factory=list)
