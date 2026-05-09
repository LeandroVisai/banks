"""Extracción de páginas de texto desde archivos PDF.

Usa ``pypdf`` (con fallback a ``PyPDF2``). Detecta PDFs encriptados,
PDFs protegidos por rights management de Microsoft Azure, y PDFs sin
texto extraíble (probablemente escaneados).

Función pura: recibe ``Path``, retorna ``(pages, warnings)``. No imprime,
no escribe a disco.
"""

from __future__ import annotations

from pathlib import Path

# Patrones que indican PDF protegido por DRM o sin texto extraíble.
_PROTECTED_MARKERS: tuple[str, ...] = (
    "Microsoft Azure Information Protection",
    "This is a protected document",
    "You can view it using a supported PDF reader",
)

_LOW_TEXT_THRESHOLD = 500


def extract_pages(pdf_path: Path) -> tuple[list[str], list[str]]:
    """Extrae texto página por página desde un PDF.

    Retorna:
        ``(pages, warnings)`` donde ``pages`` es una lista de strings
        (uno por página, vacía si la extracción falla) y ``warnings``
        contiene tags como ``pdf_encrypted``, ``pdf_rights_protected``,
        ``low_text_content_maybe_scanned`` o errores de lectura.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]
        except ImportError as e:
            raise ImportError(
                "Falta pypdf. Instala con: pip install pypdf"
            ) from e

    warnings: list[str] = []
    pages: list[str] = []

    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:  # noqa: BLE001
                    warnings.append("pdf_encrypted")
                    return pages, warnings

            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"page_extract_error: {e}")
                    pages.append("")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"pdf_open_error: {e}")
        return pages, warnings

    full_text = "\n".join(pages)
    if any(marker in full_text for marker in _PROTECTED_MARKERS):
        warnings.append("pdf_rights_protected")

    if len(full_text.strip()) < _LOW_TEXT_THRESHOLD and not warnings:
        warnings.append("low_text_content_maybe_scanned")

    return pages, warnings
