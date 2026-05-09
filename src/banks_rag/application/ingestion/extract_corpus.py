"""Orquestador del paso de extracción + chunking (legacy paso 00).

Toma un directorio raíz con PDFs y Excel, para cada archivo:

  1. Detecta tipo de doc, institución, fecha desde el path/filename.
  2. Extrae páginas (PDF) o celdas (Excel).
  3. Normaliza texto.
  4. Chunkea con el chunker jerárquico.
  5. Detecta páginas con visuales y las renderiza como PNG (PDF + PyMuPDF).

Produce ``(list[Document], list[Chunk], extraction_report)`` en memoria.
La persistencia a JSON es responsabilidad del CLI (``interface/cli/ingest.py``).

**No** importa de ``infrastructure`` directamente para los Protocols, pero sí
usa los adaptadores concretos del paquete ``banks_rag.infrastructure.extractors``
y ``banks_rag.infrastructure.chunking`` — esto es aceptable acá porque la
orquestación necesita conocer los detalles concretos del filesystem.
En Fase 5+ esto se generalizará con un Protocol ``Extractor`` si se justifica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from banks_rag.domain.documents import Chunk, ChunkKind, Document
from banks_rag.infrastructure.chunking import ChunkingConfig, chunk_pages
from banks_rag.infrastructure.extractors import (
    detect_date,
    detect_doc_type,
    detect_institution,
    extract_cell_chunks,
    extract_pages,
    extract_visual_assets,
    is_pymupdf_available,
    normalize_page_text,
    slugify_document_id,
)

EXCEL_EXTENSIONS = {".xlsx", ".xls"}


@dataclass
class ExtractionReport:
    """Estadísticas y warnings de la corrida de extracción."""

    pdf_count: int
    excel_count: int
    documents_processed: int
    documents_failed: list[str]
    documents_with_warnings: list[dict]
    chunks_total: int
    chunk_char_stats: dict
    chunking_config: dict
    pymupdf_available: bool

    def to_dict(self) -> dict:
        return {
            "pdf_count": self.pdf_count,
            "excel_count": self.excel_count,
            "documents_processed": self.documents_processed,
            "documents_failed": self.documents_failed,
            "documents_with_warnings": self.documents_with_warnings,
            "chunks_total": self.chunks_total,
            "chunk_char_stats": self.chunk_char_stats,
            "chunking_config": self.chunking_config,
            "pymupdf_available": self.pymupdf_available,
        }


@dataclass
class ExtractionResult:
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    report: ExtractionReport | None = None


def _process_pdf(
    pdf_path: Path,
    data_root: Path,
    images_dir: Path,
    chunking_config: ChunkingConfig,
) -> tuple[Document | None, list[Chunk]]:
    rel_path = str(pdf_path.relative_to(data_root))
    filename = pdf_path.name
    doc_id = slugify_document_id(rel_path)

    pages, warnings = extract_pages(pdf_path)

    if not pages or "pdf_rights_protected" in warnings or "pdf_encrypted" in warnings:
        doc = Document(
            document_id=doc_id,
            filename=filename,
            filepath=rel_path,
            doc_type_category=detect_doc_type(rel_path),
            institution=detect_institution(rel_path),
            document_date=detect_date(filename),
            total_pages=len(pages),
            total_chunks=0,
            char_count=0,
            extraction_warnings=warnings,
        )
        return doc, []

    first_page = pages[0]
    doc_type = detect_doc_type(rel_path)
    institution = detect_institution(rel_path)
    doc_date = detect_date(filename, first_page)

    normalized_pages = [normalize_page_text(p) for p in pages]
    raw_chunks = chunk_pages(normalized_pages, chunking_config)

    chunks: list[Chunk] = []
    for i, rc in enumerate(raw_chunks):
        text = rc["text"]
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_{i:04d}",
            document_id=doc_id,
            text=text,
            char_count=len(text),
            token_estimate=max(1, len(text) // 4),
            page_start=rc["page_start"],
            page_end=rc["page_end"],
            position_in_doc=i,
            section_title_raw=rc["section_title_raw"],
            kind=ChunkKind.TEXT,
        ))

    visual_assets = extract_visual_assets(pdf_path, doc_id, images_dir)
    for i, asset in enumerate(visual_assets):
        # El texto del chunk visual incluye el tipo, la caption (si existe)
        # y un snippet del texto vecino — para que BM25/full-text encuentre
        # el chunk por descripción, aunque el embedding venga de la imagen.
        text_parts: list[str] = [f"[{asset.kind} p.{asset.page}]"]
        if asset.caption:
            text_parts.append(asset.caption)
        if asset.surrounding_text:
            text_parts.append(asset.surrounding_text[:200])
        text = "\n".join(text_parts)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_img_{i:04d}",
            document_id=doc_id,
            text=text,
            char_count=len(text),
            token_estimate=max(1, len(text) // 4),
            page_start=asset.page,
            page_end=asset.page,
            position_in_doc=len(chunks) + i,
            section_title_raw=asset.caption,
            image_path=asset.image_path,
            visual_caption=asset.caption,
            kind=ChunkKind.VISUAL,
        ))

    total_chars = sum(c.char_count for c in chunks)
    doc = Document(
        document_id=doc_id,
        filename=filename,
        filepath=rel_path,
        doc_type_category=doc_type,
        institution=institution,
        document_date=doc_date,
        total_pages=len(pages),
        total_chunks=len(chunks),
        char_count=total_chars,
        extraction_warnings=warnings,
    )
    return doc, chunks


def _process_excel(
    excel_path: Path,
    data_root: Path,
) -> tuple[Document | None, list[Chunk]]:
    rel_path = str(excel_path.relative_to(data_root))
    filename = excel_path.name
    doc_id = slugify_document_id(rel_path)

    raw_chunks, warnings = extract_cell_chunks(excel_path)

    if not raw_chunks:
        doc = Document(
            document_id=doc_id,
            filename=filename,
            filepath=rel_path,
            doc_type_category=detect_doc_type(rel_path),
            institution=detect_institution(rel_path),
            document_date=detect_date(filename),
            total_pages=0,
            total_chunks=0,
            char_count=0,
            extraction_warnings=warnings or ["excel_no_content"],
        )
        return doc, []

    chunks: list[Chunk] = []
    for i, rc in enumerate(raw_chunks):
        text = rc["text"]
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_{i:04d}",
            document_id=doc_id,
            text=text,
            char_count=len(text),
            token_estimate=max(1, len(text) // 4),
            page_start=rc["page_start"],
            page_end=rc["page_end"],
            position_in_doc=i,
            section_title_raw=rc["section_title_raw"],
            chunk_date=rc["fecha_iso"],
            kind=ChunkKind.TEXT,
        ))

    first_fecha = raw_chunks[0]["fecha_iso"]
    total_chars = sum(c.char_count for c in chunks)
    total_sheets = max(rc["page_end"] for rc in raw_chunks)

    doc = Document(
        document_id=doc_id,
        filename=filename,
        filepath=rel_path,
        doc_type_category=detect_doc_type(rel_path),
        institution=detect_institution(rel_path),
        document_date=first_fecha or detect_date(filename),
        total_pages=total_sheets,
        total_chunks=len(chunks),
        char_count=total_chars,
        extraction_warnings=warnings,
    )
    return doc, chunks


def _build_chunk_char_stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"min": 0, "p25": 0, "p50_median": 0, "p75": 0, "p90": 0, "max": 0, "mean": 0}
    counts = sorted(c.char_count for c in chunks)
    n = len(counts)
    return {
        "min": counts[0],
        "p25": counts[n // 4],
        "p50_median": counts[n // 2],
        "p75": counts[(3 * n) // 4],
        "p90": counts[(9 * n) // 10],
        "max": counts[-1],
        "mean": sum(counts) // n,
    }


def extract_corpus(
    data_root: Path,
    images_dir: Path,
    chunking_config: ChunkingConfig | None = None,
    on_file_complete: callable | None = None,
) -> ExtractionResult:
    """Itera ``data_root`` y construye documentos + chunks.

    Args:
        data_root: directorio con los inputs (PDFs + Excel) — escaneo recursivo.
        images_dir: dónde guardar los PNGs renderizados de páginas visuales.
        chunking_config: configuración del chunker; default = ChunkingConfig().
        on_file_complete: callback opcional ``(rel_path, doc, chunks) -> None``
            invocado al terminar cada archivo (útil para barras de progreso).

    Retorna ``ExtractionResult`` con documentos, chunks y reporte agregado.
    """
    cfg = chunking_config or ChunkingConfig()

    pdf_files = sorted(data_root.rglob("*.pdf"))
    excel_files = sorted(p for ext in EXCEL_EXTENSIONS for p in data_root.rglob(f"*{ext}"))

    all_files: list[Path] = list(pdf_files) + list(excel_files)
    if not all_files:
        return ExtractionResult(
            report=ExtractionReport(
                pdf_count=0, excel_count=0, documents_processed=0,
                documents_failed=[], documents_with_warnings=[],
                chunks_total=0,
                chunk_char_stats=_build_chunk_char_stats([]),
                chunking_config=cfg.__dict__,
                pymupdf_available=is_pymupdf_available(),
            ),
        )

    documents: list[Document] = []
    chunks: list[Chunk] = []
    failed: list[str] = []

    for file_path in all_files:
        rel = str(file_path.relative_to(data_root))
        try:
            if file_path.suffix.lower() in EXCEL_EXTENSIONS:
                doc, doc_chunks = _process_excel(file_path, data_root)
            else:
                doc, doc_chunks = _process_pdf(file_path, data_root, images_dir, cfg)
        except Exception:  # noqa: BLE001
            failed.append(rel)
            continue

        if doc is None:
            failed.append(rel)
            continue

        documents.append(doc)
        chunks.extend(doc_chunks)

        if on_file_complete:
            on_file_complete(rel, doc, doc_chunks)

    report = ExtractionReport(
        pdf_count=len(pdf_files),
        excel_count=len(excel_files),
        documents_processed=len(documents),
        documents_failed=failed,
        documents_with_warnings=[
            {"document_id": d.document_id, "warnings": d.extraction_warnings}
            for d in documents
            if d.extraction_warnings
        ],
        chunks_total=len(chunks),
        chunk_char_stats=_build_chunk_char_stats(chunks),
        chunking_config=cfg.__dict__,
        pymupdf_available=is_pymupdf_available(),
    )

    return ExtractionResult(documents=documents, chunks=chunks, report=report)
