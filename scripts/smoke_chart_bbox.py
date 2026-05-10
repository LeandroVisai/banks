"""Smoke test Fase 2.b — corre el nuevo chart_detector contra todos los PDFs.

Compara: cuántos visuales se extraen ahora (con bbox + cropping) vs los 32
de la versión página-completa anterior. Renderiza los PNGs en
``images_bbox/`` para inspección visual.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from banks_rag.infrastructure.extractors.chart_detector import (
    extract_visual_assets,
    is_pymupdf_available,
)

DOCS = Path("logs/documents.json")
DATOS_ROOT = Path("Datos_prueba")
OUTPUT_DIR = Path("images_bbox")


def main() -> int:
    if not is_pymupdf_available():
        print("❌ PyMuPDF no instalado")
        return 1

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    pdf_docs = [d for d in docs if d.get("filepath", "").lower().endswith(".pdf")]

    by_doc: dict[str, list] = defaultdict(list)
    total_assets = 0
    total_pages_with_visuals = 0
    captions_detected = 0

    for d in pdf_docs:
        doc_id = d["document_id"]
        fp = Path(d["filepath"])
        if not fp.is_absolute() and not fp.exists():
            fp = DATOS_ROOT / fp
        if not fp.exists():
            print(f"⚠ No existe: {fp}")
            continue

        assets = extract_visual_assets(fp, doc_id, OUTPUT_DIR)
        by_doc[doc_id] = assets
        total_assets += len(assets)
        total_pages_with_visuals += len({a.page for a in assets})
        captions_detected += sum(1 for a in assets if a.caption)

        if assets:
            print(f"\n📄 {doc_id} ({fp.name})")
            for a in assets:
                area_pt = a.bbox.area if a.bbox else 0
                cap = (a.caption or "—")[:50]
                print(f"   p{a.page:>3d} {a.kind:<6s} {area_pt:>8.0f}pt²  cap='{cap}'  → {Path(a.image_path).name}")

    print(f"\n{'='*60}")
    print("RESUMEN Fase 2.b vs legacy")
    print(f"{'='*60}")
    print(f"PDFs procesados:           {len(pdf_docs)}")
    print(f"Páginas con visuales:      {total_pages_with_visuals}  (legacy: 6)")
    print(f"Assets visuales extraídos: {total_assets}              (legacy: 32 [con falsos positivos])")
    print(f"Captions detectadas:       {captions_detected}")
    print(f"\nPNGs en: {OUTPUT_DIR}/")

    print("\nDistribución por kind:")
    from collections import Counter
    kinds = Counter(a.kind for assets in by_doc.values() for a in assets)
    for k, c in kinds.most_common():
        print(f"  {k:<8s} {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
