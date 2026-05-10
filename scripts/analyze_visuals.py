"""Analiza las imágenes extraídas y las clasifica por contenido.

Para cada chunk con image_path, abre la página correspondiente del PDF y mide:
  - text_chars   : caracteres de texto extraíble en la página
  - num_images   : imágenes embebidas (raster)
  - filled_shapes: formas con relleno (drawings → indicador de chart vectorial)
  - text_ratio   : área de bloques de texto / área de página
  - image_ratio  : área de imágenes / área de página

Clasificación:
  CHART_ONLY  : text_chars < 200  (página casi sin texto, dominada por gráfico)
  CHART_HEAVY : text_chars 200-800 y (image_ratio > 0.3 o filled_shapes > 30)
  MIXED       : todo lo demás (texto + chart)
  TEXT_HEAVY  : text_chars > 1500 y image_ratio < 0.2  (probablemente mal etiquetado)
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import fitz

ENRICHED = Path("logs/chunks_enriched_full.json")
DOCS = Path("logs/documents.json")
OUT = Path("logs/visual_analysis.json")


DATOS_ROOT = Path("Datos_prueba")


def build_doc_path_index() -> dict[str, Path]:
    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    out = {}
    for d in docs:
        fp = d.get("filepath")
        if not fp:
            continue
        p = Path(fp)
        if not p.is_absolute() and not p.exists():
            p = DATOS_ROOT / fp
        out[d["document_id"]] = p
    return out


_DOC_PATHS = build_doc_path_index()


def find_pdf(doc_id: str) -> Path | None:
    p = _DOC_PATHS.get(doc_id)
    return p if p and p.exists() else None


def analyze_page(page) -> dict:
    rect = page.rect
    page_area = rect.width * rect.height
    text = page.get_text("text") or ""
    text_chars = len(text.strip())

    text_blocks = page.get_text("blocks") or []
    text_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in text_blocks if b[6] == 0)

    images = page.get_images(full=True) or []
    num_images = len(images)

    image_area = 0.0
    for img in images:
        for inst in page.get_image_rects(img[0]) or []:
            image_area += inst.width * inst.height

    drawings = page.get_drawings() or []
    filled_shapes = sum(1 for d in drawings if d.get("fill"))

    return {
        "page_area": page_area,
        "text_chars": text_chars,
        "text_ratio": round(text_area / page_area, 3) if page_area else 0,
        "num_images": num_images,
        "image_ratio": round(image_area / page_area, 3) if page_area else 0,
        "filled_shapes": filled_shapes,
    }


def classify(stats: dict) -> str:
    tc = stats["text_chars"]
    ir = stats["image_ratio"]
    fs = stats["filled_shapes"]
    if tc < 200:
        return "CHART_ONLY"
    if tc < 800 and (ir > 0.3 or fs > 30):
        return "CHART_HEAVY"
    if tc > 1500 and ir < 0.2 and fs < 15:
        return "TEXT_HEAVY"
    return "MIXED"


def main() -> int:
    chunks = json.loads(ENRICHED.read_text(encoding="utf-8"))
    visual_chunks = [c for c in chunks if c.get("image_path")]

    print(f"Encontrados {len(visual_chunks)} chunks con image_path")

    results = []
    by_doc: dict[str, fitz.Document] = {}

    for c in visual_chunks:
        doc_id = c.get("document_id", "")
        page_num = c.get("page_start") or c.get("page", 1)
        img_path = c.get("image_path", "")

        pdf_path = find_pdf(doc_id)
        if not pdf_path:
            print(f"⚠ No encontré PDF para {doc_id}")
            continue

        if str(pdf_path) not in by_doc:
            by_doc[str(pdf_path)] = fitz.open(str(pdf_path))
        doc = by_doc[str(pdf_path)]

        try:
            page = doc[page_num - 1]
        except IndexError:
            print(f"⚠ Página {page_num} fuera de rango en {pdf_path.name}")
            continue

        stats = analyze_page(page)
        cls = classify(stats)
        results.append({
            "image_path": img_path,
            "document_id": doc_id,
            "page": page_num,
            "doc_type": c.get("doc_type_category", ""),
            "classification": cls,
            **stats,
        })

    for d in by_doc.values():
        d.close()

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Distribución ({len(results)} imágenes) ===")
    counts = Counter(r["classification"] for r in results)
    for cls in ("CHART_ONLY", "CHART_HEAVY", "MIXED", "TEXT_HEAVY"):
        print(f"  {cls:12s} {counts.get(cls, 0):>3d}")

    print("\n=== Detalle por imagen (ordenado por text_chars asc) ===")
    print(f"{'classification':14s} {'text_chars':>10s} {'img_ratio':>9s} {'shapes':>6s}  image_path")
    for r in sorted(results, key=lambda x: x["text_chars"]):
        print(f"  {r['classification']:12s} {r['text_chars']:>10d} {r['image_ratio']:>9.3f} {r['filled_shapes']:>6d}  {r['image_path']}")

    print(f"\nDetalle completo: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
