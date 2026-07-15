#!/usr/bin/env python3
"""Construye el informe CURADO de una familia (réplica de un informe real del BCCh).

Python puro, SIN modelo de lenguaje: los gráficos se dibujan leyendo los parquets
directo; el texto queda en slots vacíos para llenarse después (con el LLM, en el
H100). MVP: dibuja los gráficos que el renderer ya soporta (multi-línea /
composición); el resto queda como placeholder marcado con su tipo pendiente.

Uso:
    python scripts/build_family_report.py --family ffmm
    python scripts/build_family_report.py            # lista familias disponibles

Salida: data/parquet_reports/curated/<familia>_<fecha>.html
Además, una versión EDITABLE junto a esa (``<familia>_<fecha>_editable.html``):
panel flotante 💾 Guardar / 📄 Versión final para escribir el texto A MANO en el
navegador (sin pasar por parquet_report.py/fill_report_texts.py) y guardar
(--no-editable para omitirla).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.reporting import (  # noqa: E402
    build_curated_report,
    make_editable_html,
    render_curated_html,
)
from banks_rag.application.reporting.specs import available_families, get_spec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Informe curado por familia (gráficos desde parquets, sin LLM).")
    ap.add_argument("--family", default=None, help="Familia a construir (ej. ffmm). Sin valor: lista las disponibles.")
    ap.add_argument("--out", default="data/parquet_reports/curated", help="Carpeta de salida.")
    ap.add_argument("--verbose", action="store_true", help="Log a nivel INFO.")
    ap.add_argument(
        "--no-editable", action="store_true",
        help="No generar la versión editable (panel 💾/📄 para escribir el texto a mano).",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    families = available_families()
    if not args.family:
        print("Familias disponibles:", ", ".join(families) or "(ninguna)")
        return

    spec = get_spec(args.family)
    if spec is None:
        print(f"No hay spec curado para {args.family!r}. Disponibles: {', '.join(families)}")
        raise SystemExit(1)

    report = build_curated_report(spec)
    html = render_curated_html(report)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.family}_{datetime.date.today().isoformat()}"
    dst = out / f"{stem}.html"
    dst.write_text(html, encoding="utf-8")
    print(f"OK {spec.title} -> {dst}  ({report.summary()})")

    # Versión editable (misma que "Guardar editable" del chartbuilder): útil ACÁ
    # sobre todo si se va a escribir el texto a mano, sin pasar por el LLM.
    if not args.no_editable:
        ed_dst = out / f"{stem}_editable.html"
        ed_dst.write_text(make_editable_html(html), encoding="utf-8")
        print(f"OK editable -> {ed_dst}  (contenteditable + panel 💾 Guardar / 📄 Versión final)")


if __name__ == "__main__":
    main()
