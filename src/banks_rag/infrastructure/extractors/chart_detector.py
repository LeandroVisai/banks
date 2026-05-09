"""Detección y renderizado de páginas con contenido visual (charts/figuras/tablas).

Usa PyMuPDF (``fitz``) — opcional. Si no está instalado, retorna lista vacía
sin abortar el pipeline.

En Fase 1 reproduce el comportamiento legacy: una página se considera "visual"
si tiene imágenes embebidas o ≥8 formas rellenas (chart). Renderiza la página
completa a PNG.

En Fase 2 esto se refina: bbox de chart en lugar de página completa, captioning
con regex sobre texto vecino, y embedding multimodal con Qwen3-VL-Embedding-8B.
"""

from __future__ import annotations

from pathlib import Path

try:
    import fitz  # noqa: F401
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

DEFAULT_IMAGE_DPI = 150
VISUAL_MIN_FILLED_SHAPES = 8  # mínimo de formas rellenas para considerar página visual


def is_pymupdf_available() -> bool:
    return _PYMUPDF_AVAILABLE


def page_has_visuals(page) -> bool:
    """True si la página tiene imágenes embebidas o ≥8 formas rellenas (charts)."""
    if page.get_images(full=False):
        return True
    filled = [d for d in page.get_drawings() if d.get("fill") is not None]
    return len(filled) >= VISUAL_MIN_FILLED_SHAPES


def extract_visual_pages(
    pdf_path: Path,
    doc_id: str,
    output_dir: Path,
    image_dpi: int = DEFAULT_IMAGE_DPI,
) -> list[dict]:
    """Renderiza como PNG las páginas con contenido visual relevante.

    Retorna lista de raw_chunks con keys ``text``, ``page_start``,
    ``page_end``, ``section_title_raw``, ``image_path``. Si PyMuPDF no
    está disponible o el PDF no se puede abrir, retorna ``[]``.
    """
    if not _PYMUPDF_AVAILABLE:
        return []

    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    image_chunks: list[dict] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:  # noqa: BLE001
        return []

    for page_num, page in enumerate(doc, start=1):
        if not page_has_visuals(page):
            continue

        img_name = f"{doc_id}_p{page_num:03d}.png"
        img_path = output_dir / img_name
        try:
            mat = fitz.Matrix(image_dpi / 72, image_dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            pix.save(str(img_path))
        except Exception:  # noqa: BLE001
            continue

        image_chunks.append({
            "text": f"[Imagen p.{page_num}]",
            "page_start": page_num,
            "page_end": page_num,
            "section_title_raw": None,
            "image_path": str(img_path),
        })

    doc.close()
    return image_chunks
