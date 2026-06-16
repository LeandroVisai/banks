#!/usr/bin/env python3
"""Inyecta los párrafos de parquet_report.py en el HTML curado.

Lee un JSON {dataset_id: párrafo} producido por ``parquet_report.py
--paragraphs-out`` y rellena los ``<div data-text-slot="…">`` vacíos del HTML
curado generado por ``build_family_report.py``.

Uso típico (H100, después de correr parquet_report.py):

    python scripts/parquet_report.py --segment ffmm \\
        --paragraphs-out data/parquet_reports/paragraphs_ffmm.json

    python scripts/fill_report_texts.py \\
        --curated data/parquet_reports/curated/ffmm_2026-06-15.html \\
        --paragraphs data/parquet_reports/paragraphs_ffmm.json

El HTML se sobreescribe en-place (usa --out para otra ruta).
La familia se infiere del primer data-text-slot del HTML; pasa --family para
forzarla (útil si el HTML tiene slots de varias familias).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.reporting.curated_report import fill_text_slots  # noqa: E402


def _infer_family(html_content: str) -> str:
    m = re.search(r'data-text-slot="([^:]+):', html_content)
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rellena los slots de texto de un informe curado con párrafos del LLM."
    )
    ap.add_argument("--curated", required=True,
                    help="HTML curado generado por build_family_report.py.")
    ap.add_argument("--paragraphs", required=True,
                    help="JSON {dataset_id: párrafo} de parquet_report.py --paragraphs-out.")
    ap.add_argument("--family", default="",
                    help="Prefijo de familia (ej. ffmm). Default: inferido del HTML.")
    ap.add_argument("--out", default=None,
                    help="Ruta de salida. Default: sobreescribe --curated.")
    args = ap.parse_args()

    curated_path = pathlib.Path(args.curated)
    paragraphs_path = pathlib.Path(args.paragraphs)

    if not curated_path.exists():
        sys.exit(f"ERROR: no existe {curated_path}")
    if not paragraphs_path.exists():
        sys.exit(f"ERROR: no existe {paragraphs_path}")

    html_content = curated_path.read_text(encoding="utf-8")
    paragraphs: dict[str, str] = json.loads(paragraphs_path.read_text(encoding="utf-8"))

    family = args.family or _infer_family(html_content)
    slots = {f"{family}:{did}": p for did, p in paragraphs.items()} if family else paragraphs

    patched = fill_text_slots(html_content, slots)

    filled = sum(
        1 for did in paragraphs
        if f'data-text-slot="{family}:{did}"' in html_content
    ) if family else len(paragraphs)

    out_path = pathlib.Path(args.out) if args.out else curated_path
    out_path.write_text(patched, encoding="utf-8")
    print(f"OK {out_path}  ({filled} slots rellenos de {len(paragraphs)} párrafos)")


if __name__ == "__main__":
    main()
