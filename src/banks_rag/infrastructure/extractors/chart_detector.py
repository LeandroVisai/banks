"""Detección y renderizado de páginas con contenido visual (charts/figuras/tablas).

Versión Fase 2:

  - Detecta páginas con visuales mediante imágenes embebidas + densidad de
    formas rellenas (≥ 8 = chart probable).
  - **NEW**: extrae caption del texto vecino mediante regex
    (``"Gráfico N"``, ``"Figura N"``, ``"Cuadro N"``, ``"Table N"``).
  - **NEW**: captura ``surrounding_text`` (~ 400 chars) de la página
    para usarlo como fallback cuando el embedding visual falla, y
    enriquecer la búsqueda BM25 (que de otra forma solo vería ``"[CHART p.N]"``).
  - **NEW**: produce ``VisualAsset`` con ``BoundingBox`` cuando posible
    (Fase 2.b: cropping a bbox; por ahora la bbox cubre la página completa).
  - Mantiene compat con ``extract_visual_pages(...)`` legacy que devolvía
    list[dict].

Usa PyMuPDF (``fitz``) — opcional. Si no está instalado, retorna lista vacía
sin abortar el pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

from banks_rag.domain.documents import (
    BoundingBox,
    VisualAsset,
    VisualKind,
)

try:
    import fitz  # noqa: F401
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

DEFAULT_IMAGE_DPI = 150
VISUAL_MIN_FILLED_SHAPES = 8

# Caption patterns: "Gráfico 3", "Figura 12", "Cuadro 5", "Tabla 7", "Figure 4".
# Soporta dos puntos opcional + título corto (max 120 chars) hasta fin de línea.
_CAPTION_RE = re.compile(
    r"\b(?:gr[áa]fico|figura|cuadro|tabla|table|figure|chart)\s*(?:n[°º]?\s*)?(\d{1,3})"
    r"(?:\s*[:.\-—]\s*([^\n]{0,120}))?",
    re.IGNORECASE,
)

_SURROUNDING_TEXT_CHARS = 400


def is_pymupdf_available() -> bool:
    return _PYMUPDF_AVAILABLE


def page_has_visuals(page) -> bool:
    """``True`` si la página tiene imágenes embebidas o ≥8 formas rellenas."""
    if page.get_images(full=False):
        return True
    filled = [d for d in page.get_drawings() if d.get("fill") is not None]
    return len(filled) >= VISUAL_MIN_FILLED_SHAPES


def detect_caption(page_text: str) -> str | None:
    """Busca la primera caption tipo "Gráfico N: ..." en el texto de la página.

    Devuelve la línea completa de la caption (e.g. ``"Gráfico 3: Evolución de la TPM"``).
    """
    if not page_text:
        return None
    m = _CAPTION_RE.search(page_text)
    if not m:
        return None
    # Reconstruir la caption con el match completo + título si existe.
    title = (m.group(2) or "").strip().rstrip(".:;")
    label = m.group(0).split(":")[0].strip()
    return f"{label}: {title}" if title else label


def extract_surrounding_text(page_text: str, *, max_chars: int = _SURROUNDING_TEXT_CHARS) -> str:
    """Extrae los primeros ``max_chars`` chars del texto de la página.

    En Fase 2.b refinaremos para que sea texto **cerca de la bbox**
    (requiere PyMuPDF spans con coordenadas). Por ahora usar el inicio
    es razonable: las captions y leyendas suelen estar al principio.
    """
    if not page_text:
        return ""
    cleaned = " ".join(page_text.split())  # colapsar whitespace
    return cleaned[:max_chars]


def _classify_kind(page) -> VisualKind:
    """Heurística para clasificar el tipo de visual.

    - Si la página tiene 1+ imágenes embebidas → IMAGE.
    - Si tiene muchas formas rellenas en grid → TABLE.
    - Default → CHART.
    """
    images = page.get_images(full=False)
    if images and len(images) >= 1:
        return "IMAGE"
    drawings = page.get_drawings()
    # Heurística simple: muchos rectángulos → tabla.
    rects = [d for d in drawings if d.get("type") in (None, "fill") and d.get("rect")]
    if len(rects) >= 20:
        return "TABLE"
    return "CHART"


def _page_bbox(page) -> BoundingBox:
    rect = page.rect
    return BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)


def extract_visual_assets(
    pdf_path: Path,
    doc_id: str,
    output_dir: Path,
    *,
    image_dpi: int = DEFAULT_IMAGE_DPI,
) -> list[VisualAsset]:
    """Extrae visuales de un PDF como ``VisualAsset`` con caption + surrounding text.

    Por ahora la bbox cubre la página completa (cropping a bbox real es un
    refinamiento de Fase 2.b). El PNG renderizado se guarda en ``output_dir``.
    """
    if not _PYMUPDF_AVAILABLE:
        return []

    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[VisualAsset] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:  # noqa: BLE001
        return []

    asset_idx = 0
    for page_num, page in enumerate(doc, start=1):
        if not page_has_visuals(page):
            continue

        page_text = page.get_text() or ""
        caption = detect_caption(page_text)
        surrounding = extract_surrounding_text(page_text)
        kind = _classify_kind(page)
        bbox = _page_bbox(page)

        img_name = f"{doc_id}_p{page_num:03d}.png"
        img_path = output_dir / img_name
        try:
            mat = fitz.Matrix(image_dpi / 72, image_dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            pix.save(str(img_path))
        except Exception:  # noqa: BLE001
            continue

        asset = VisualAsset(
            asset_id=f"{doc_id}_v{asset_idx:04d}",
            document_id=doc_id,
            page=page_num,
            kind=kind,
            bbox=bbox,
            image_path=str(img_path),
            caption=caption,
            surrounding_text=surrounding,
        )
        assets.append(asset)
        asset_idx += 1

    doc.close()
    return assets


# ── Backwards compat — Fase 1A produce list[dict] ───────────────────────────


def extract_visual_pages(
    pdf_path: Path,
    doc_id: str,
    output_dir: Path,
    image_dpi: int = DEFAULT_IMAGE_DPI,
) -> list[dict]:
    """Compat shim: retorna list[dict] como en Fase 1A.

    Internamente usa ``extract_visual_assets`` para aprovechar caption +
    surrounding text. Los nuevos campos viajan en el dict, ignorados por
    consumers viejos.
    """
    assets = extract_visual_assets(
        pdf_path, doc_id, output_dir, image_dpi=image_dpi,
    )
    return [
        {
            "text": _legacy_text_for_asset(a),
            "page_start": a.page,
            "page_end": a.page,
            "section_title_raw": a.caption,
            "image_path": a.image_path,
            # Nuevos campos (no rompen el legacy):
            "visual_caption": a.caption,
            "visual_surrounding_text": a.surrounding_text,
            "visual_kind": a.kind,
            "asset_id": a.asset_id,
        }
        for a in assets
    ]


def _legacy_text_for_asset(asset: VisualAsset) -> str:
    """Texto que va al chunk para que BM25 pueda encontrarlo.

    Si hay caption, la incluye. Si hay surrounding text, agrega los primeros
    200 chars (no el total para no inflar el char_count del chunk visual,
    que se penalizaba en importance).
    """
    parts: list[str] = [f"[{asset.kind} p.{asset.page}]"]
    if asset.caption:
        parts.append(asset.caption)
    if asset.surrounding_text:
        parts.append(asset.surrounding_text[:200])
    return "\n".join(parts)
