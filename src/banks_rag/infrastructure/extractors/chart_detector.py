"""Detección y renderizado de páginas con contenido visual (charts/figuras/tablas).

Versión Fase 2.b:

  - Detección **inicial** de páginas con visuales (heurística rápida:
    imágenes embebidas o ≥ 8 formas rellenas).
  - **Bbox real del chart** (no la página completa): clustering por
    proximidad de drawings vectoriales + bboxes de imágenes embebidas.
  - **Cropping del PNG** a esos bboxes — una página puede generar varios
    visuales o ninguno (skip si la heurística inicial fue falso positivo).
  - Caption regex (``Gráfico N``, ``Figura N``, ``Cuadro N``, ``Table N``)
    + surrounding text (~ 400 chars) para el chunk visual.
  - ``VisualAsset`` con ``BoundingBox`` real, ``image_path`` apunta al PNG
    cropeado.
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

# Fase 2.b — bbox detection + cropping
MIN_IMAGE_AREA_RATIO = 0.12       # imágenes embebidas < 12% página = logo (incluye logos institucionales BCCh/Fed)
MAX_REGION_AREA_RATIO = 0.95      # bbox > 95% página = no aporta vs full-page
MIN_CLUSTER_AREA_RATIO = 0.05     # cluster vectorial < 5% página = ruido
MIN_SHAPES_PER_CLUSTER = 5        # cluster con < 5 formas = no es chart
CLUSTER_DISTANCE_PT = 25          # distancia (pt) entre formas para unirlas en un cluster
BACKGROUND_MIN_RATIO = 0.05       # un rect ≥ 5% página y ancho/alto suficiente = fondo de chart
BACKGROUND_MIN_WIDTH_PT = 100     # ancho mínimo (pt) para considerar fondo de chart
BACKGROUND_MIN_HEIGHT_PT = 50     # alto mínimo (pt) para considerar fondo de chart
BBOX_PADDING_PT = 6               # margen al recortar para no cortar ejes
TITLE_SEARCH_DISTANCE_PT = 70     # cuánto arriba/abajo del chart buscar título o leyenda
TITLE_MAX_CHARS = 400             # bloques más largos = párrafo, no título

# Caption patterns: "Gráfico 3", "Figura 12", "Cuadro 5", "Tabla 7", "Figure 4".
# Soporta dos puntos opcional + título corto (max 120 chars) hasta fin de línea.
_CAPTION_RE = re.compile(
    r"\b(?:gr[áa]fico|figura|cuadro|tabla|table|figure|chart|exhibit|panel)"
    r"\s*(?:n[°º]?\s*)?(\d{1,3})"
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


RectTuple = tuple[float, float, float, float]


def _rects_close(r1: RectTuple, r2: RectTuple, distance: float) -> bool:
    """``True`` si dos rectángulos se solapan o están a < ``distance`` pt."""
    return not (
        r1[2] + distance < r2[0]
        or r2[2] + distance < r1[0]
        or r1[3] + distance < r2[1]
        or r2[3] + distance < r1[1]
    )


def _union_of_rects(rects: list[RectTuple]) -> RectTuple:
    """Bbox que envuelve todos los rectángulos."""
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _cluster_rects(rects: list[RectTuple], distance: float) -> list[list[RectTuple]]:
    """Agrupa rectángulos por proximidad usando union-find iterativo.

    Dos rectángulos quedan en el mismo cluster si están a < ``distance`` pt
    (en cualquier dirección) o se solapan.
    """
    n = len(rects)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _rects_close(rects[i], rects[j], distance):
                union(i, j)

    clusters: dict[int, list[RectTuple]] = {}
    for i, r in enumerate(rects):
        clusters.setdefault(find(i), []).append(r)
    return list(clusters.values())


def _merge_overlapping_regions(
    regions: list[RectTuple],
    distance: float = 0.0,
) -> list[RectTuple]:
    """Fusiona bboxes que se solapan o están muy cerca (transitivamente)."""
    if not regions:
        return []
    clusters = _cluster_rects(regions, distance)
    return [_union_of_rects(c) for c in clusters]


def _pad_bbox(r: RectTuple, padding: float, page_w: float, page_h: float) -> RectTuple:
    """Expande el bbox por ``padding`` pt en cada lado, clamped a página."""
    return (
        max(0.0, r[0] - padding),
        max(0.0, r[1] - padding),
        min(page_w, r[2] + padding),
        min(page_h, r[3] + padding),
    )


def detect_chart_regions(
    *,
    embedded_image_rects: list[RectTuple],
    filled_drawing_rects: list[RectTuple],
    page_width: float,
    page_height: float,
) -> list[BoundingBox]:
    """Detecta los bboxes de charts/figuras en una página.

    Función pura: no depende de PyMuPDF, recibe rectángulos como tuplas.
    Permite testear la lógica con datos sintéticos.

    Estrategia:
      1. **Imágenes embebidas (raster)**: cada instancia es un bbox candidato.
         Se filtran las menores a ``MIN_IMAGE_AREA_RATIO`` (logos).
      2. **Drawings vectoriales rellenos**: si hay ≥ ``VISUAL_MIN_FILLED_SHAPES``,
         se clusterizan por proximidad (``CLUSTER_DISTANCE_PT``). Cada cluster
         con ≥ ``MIN_SHAPES_PER_CLUSTER`` formas y área dentro del rango
         válido es un candidato.
      3. **Merge** final de regiones solapadas o muy cercanas.
      4. **Padding** suave para no recortar ejes.

    Devuelve lista de :class:`BoundingBox`. Vacía si no se detecta nada
    (skip de la página, sin fallback a full-page).
    """
    page_area = page_width * page_height
    if page_area <= 0:
        return []

    candidates: list[RectTuple] = []

    # 1) Imágenes embebidas
    for r in embedded_image_rects:
        area = max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
        ratio = area / page_area
        if MIN_IMAGE_AREA_RATIO <= ratio <= MAX_REGION_AREA_RATIO:
            candidates.append(r)

    # 2) Drawings vectoriales:
    #    a) Si hay "fondos de chart" (rects grandes y bien dimensionados),
    #       usarlos directamente como regiones — esto separa charts adyacentes
    #       que el clustering bridgeaba via el rect de fondo.
    #    b) Si no, fallback a clustering por proximidad de formas chicas.
    backgrounds: list[RectTuple] = []
    small_drawings: list[RectTuple] = []
    for r in filled_drawing_rects:
        w = r[2] - r[0]
        h = r[3] - r[1]
        area = w * h
        if (
            area / page_area >= BACKGROUND_MIN_RATIO
            and w >= BACKGROUND_MIN_WIDTH_PT
            and h >= BACKGROUND_MIN_HEIGHT_PT
        ):
            backgrounds.append(r)
        else:
            small_drawings.append(r)

    if backgrounds:
        for bg in backgrounds:
            ratio = ((bg[2] - bg[0]) * (bg[3] - bg[1])) / page_area
            if ratio <= MAX_REGION_AREA_RATIO:
                candidates.append(bg)
    elif len(small_drawings) >= VISUAL_MIN_FILLED_SHAPES:
        for cluster in _cluster_rects(small_drawings, CLUSTER_DISTANCE_PT):
            if len(cluster) < MIN_SHAPES_PER_CLUSTER:
                continue
            bbox = _union_of_rects(cluster)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            ratio = area / page_area
            if MIN_CLUSTER_AREA_RATIO <= ratio <= MAX_REGION_AREA_RATIO:
                candidates.append(bbox)

    # 3) Merge solapados + padding
    merged = _merge_overlapping_regions(candidates, distance=BBOX_PADDING_PT)

    return [
        BoundingBox(*_pad_bbox(r, BBOX_PADDING_PT, page_width, page_height))
        for r in merged
    ]


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


def find_caption_block_for_region(
    text_blocks: list[tuple[float, float, float, float, str]],
    region: RectTuple,
) -> tuple[str | None, RectTuple | None]:
    """Busca el bloque de texto con caption más cercano a ``region`` para mergear.

    Función pura: recibe ``text_blocks`` como lista de tuplas
    ``(x0, y0, x1, y1, text)``. Esto la hace testeable sin PyMuPDF.

    Criterios:
      - El bloque debe matchear ``_CAPTION_RE`` (Gráfico/Figure/Cuadro/...).
      - Debe estar **encima o debajo** de la región (no solapado).
      - Debe tener **overlap horizontal** con la región (caption del mismo chart).
      - Se elige el **más cercano verticalmente**.

    Devuelve ``(caption_text, bbox)`` o ``(None, None)`` si no hay match.
    """
    best: tuple[float, str, RectTuple] | None = None
    region_x0, region_y0, region_x1, region_y1 = region

    for block in text_blocks:
        if len(block) < 5:
            continue
        bx0, by0, bx1, by1, text = block[0], block[1], block[2], block[3], block[4]
        if not text or not _CAPTION_RE.search(text):
            continue

        # Vertical: caption debe estar arriba o abajo, no solapando
        if by1 <= region_y0:           # bloque arriba del chart
            v_dist = region_y0 - by1
        elif by0 >= region_y1:         # bloque abajo del chart
            v_dist = by0 - region_y1
        else:
            continue                    # solapa verticalmente, no es caption del chart

        # Horizontal: overlap requerido con la región
        h_overlap = min(bx1, region_x1) - max(bx0, region_x0)
        if h_overlap <= 0:
            continue

        clean = detect_caption(text) or _CAPTION_RE.search(text).group(0)
        block_bbox = (bx0, by0, bx1, by1)
        if best is None or v_dist < best[0]:
            best = (v_dist, clean, block_bbox)

    if best is None:
        return None, None
    return best[1], best[2]


def find_text_blocks_around_region(
    text_blocks: list[tuple[float, float, float, float, str]],
    region: RectTuple,
    *,
    max_distance: float = TITLE_SEARCH_DISTANCE_PT,
    max_chars: int = TITLE_MAX_CHARS,
    position: str = "above",  # "above" | "below" | "both"
) -> list[tuple[RectTuple, str]]:
    """Devuelve bloques de texto cerca del chart (título/leyenda/subtítulo).

    Función pura: no depende de PyMuPDF. Útil para incluir títulos en el
    bbox del chart aunque no matcheen el regex de caption (Figure/Gráfico).

    Criterios:
      - Bloque está arriba/debajo de la región (no solapado verticalmente).
      - Distancia vertical ≤ ``max_distance``.
      - Overlap horizontal con la región.
      - Texto no vacío y ≤ ``max_chars`` (descarta párrafos largos).

    Devuelve lista ordenada por proximidad ``[(bbox, text), ...]``.
    """
    region_x0, region_y0, region_x1, region_y1 = region
    candidates: list[tuple[float, RectTuple, str]] = []

    for block in text_blocks:
        if len(block) < 5:
            continue
        bx0, by0, bx1, by1, text = block[0], block[1], block[2], block[3], block[4]
        if not text or not text.strip():
            continue
        if len(text) > max_chars:
            continue

        # Posición vertical
        if by1 <= region_y0:           # arriba
            if position not in ("above", "both"):
                continue
            v_dist = region_y0 - by1
        elif by0 >= region_y1:         # abajo
            if position not in ("below", "both"):
                continue
            v_dist = by0 - region_y1
        else:
            continue                    # solapa, no es título/leyenda

        if v_dist > max_distance:
            continue

        # Overlap horizontal requerido
        h_overlap = min(bx1, region_x1) - max(bx0, region_x0)
        if h_overlap <= 0:
            continue

        candidates.append((v_dist, (bx0, by0, bx1, by1), text.strip()))

    candidates.sort(key=lambda c: c[0])
    return [(bbox, text) for _, bbox, text in candidates]


def _collect_page_rects(page) -> tuple[list[RectTuple], list[RectTuple]]:
    """Devuelve (image_rects, filled_drawing_rects) para una página fitz."""
    image_rects: list[RectTuple] = []
    for img_info in page.get_images(full=False):
        xref = img_info[0]
        for r in page.get_image_rects(xref) or []:
            image_rects.append((r.x0, r.y0, r.x1, r.y1))

    filled_rects: list[RectTuple] = []
    for d in page.get_drawings():
        if d.get("fill") is None:
            continue
        rect = d.get("rect")
        if rect is None:
            continue
        filled_rects.append((rect.x0, rect.y0, rect.x1, rect.y1))

    return image_rects, filled_rects


def extract_visual_assets(
    pdf_path: Path,
    doc_id: str,
    output_dir: Path,
    *,
    image_dpi: int = DEFAULT_IMAGE_DPI,
) -> list[VisualAsset]:
    """Extrae visuales reales de un PDF (Fase 2.b — bbox cropping).

    Para cada página:
      1. Detecta bboxes de charts/figuras vía :func:`detect_chart_regions`.
      2. Si no hay bboxes válidos → skip (no full-page fallback).
      3. Por cada bbox: renderiza solo esa región como PNG.
      4. Crea un :class:`VisualAsset` por bbox con caption + surrounding text.

    Una página puede generar **múltiples** assets (varios charts) o **cero**
    (la heurística inicial era falso positivo: solo un logo o cosmética).
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

        image_rects, filled_rects = _collect_page_rects(page)
        regions = detect_chart_regions(
            embedded_image_rects=image_rects,
            filled_drawing_rects=filled_rects,
            page_width=page.rect.width,
            page_height=page.rect.height,
        )
        if not regions:
            continue  # skip — heurística inicial fue falso positivo

        page_text = page.get_text() or ""
        page_caption = detect_caption(page_text)  # fallback global
        surrounding = extract_surrounding_text(page_text)
        kind = _classify_kind(page)
        page_w, page_h = page.rect.width, page.rect.height

        # Bloques de texto con bbox para detectar captions por región
        raw_blocks = page.get_text("blocks") or []
        text_blocks = [
            (b[0], b[1], b[2], b[3], b[4])
            for b in raw_blocks
            if len(b) >= 7 and b[6] == 0  # type 0 = text block
        ]

        for region_idx, bbox in enumerate(regions):
            # detect_chart_regions devuelve bboxes ya paddeadas. Para buscar
            # captions/títulos correctamente debemos usar la región INTERIOR
            # (sin padding) — sino el padding "se come" la separación con la
            # caption inmediatamente arriba/abajo del chart.
            padded_tuple = bbox.to_tuple()
            inner_tuple = (
                padded_tuple[0] + BBOX_PADDING_PT,
                padded_tuple[1] + BBOX_PADDING_PT,
                padded_tuple[2] - BBOX_PADDING_PT,
                padded_tuple[3] - BBOX_PADDING_PT,
            )

            region_caption, caption_bbox = find_caption_block_for_region(
                text_blocks, inner_tuple,
            )

            # 1) Caption regex (Figure N, Gráfico N, etc.) — provee texto preciso
            to_merge: list[RectTuple] = [padded_tuple]
            if caption_bbox is not None:
                to_merge.append(caption_bbox)

            # 2) Bloques de título/subtítulo arriba (no requieren regex match)
            #    Captura títulos de chart que no usan "Figure N" pattern.
            for tbox, _ in find_text_blocks_around_region(
                text_blocks, inner_tuple, position="above",
            ):
                to_merge.append(tbox)

            if len(to_merge) > 1:
                merged = _union_of_rects(to_merge)
                merged = _pad_bbox(merged, BBOX_PADDING_PT, page_w, page_h)
                bbox = BoundingBox(*merged)

            asset_caption = region_caption or page_caption

            suffix = f"_r{region_idx + 1}" if len(regions) > 1 else ""
            img_name = f"{doc_id}_p{page_num:03d}{suffix}.png"
            img_path = output_dir / img_name
            try:
                mat = fitz.Matrix(image_dpi / 72, image_dpi / 72)
                clip = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                pix = page.get_pixmap(matrix=mat, clip=clip)
                pix.save(str(img_path))
            except Exception:  # noqa: BLE001
                continue

            assets.append(VisualAsset(
                asset_id=f"{doc_id}_v{asset_idx:04d}",
                document_id=doc_id,
                page=page_num,
                kind=kind,
                bbox=bbox,
                image_path=str(img_path),
                caption=asset_caption,
                surrounding_text=surrounding,
            ))
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
