#!/usr/bin/env python3
"""Construye el informe CURADO de una familia (réplica de un informe real del BCCh).
<br><br>
Python puro, SIN modelo de lenguaje: los gráficos se dibujan leyendo los parquets
directo; el texto queda en slots vacíos para llenarse después (con el LLM, en el
H100). MVP: dibuja los gráficos que el renderer ya soporta (multi-línea /
composición); el resto queda como placeholder marcado con su tipo pendiente.
<br><br>
Uso:
    python scripts/build_family_report.py --family ffmm
    python scripts/build_family_report.py            # lista familias disponibles
    python scripts/build_family_report.py --all      # construye todas
<br><br>
Salida: **un directorio por familia y variante**, resuelto vía ``--out``/``--out-editable``
(ambos admiten el placeholder ``{familia}``). Default:
    <raíz>/{familia}/no_editable/<familia>_<fecha>.html
    <raíz>/{familia}/Editable/<familia>_<fecha>_editable.html
Cada informe se acumula por fecha. La versión EDITABLE trae panel flotante
💾 Guardar / 📄 Versión final para escribir el texto A MANO en el navegador
(sin pasar por parquet_report.py/fill_report_texts.py); ``--no-editable`` la omite.
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


def _resolve_out_dir(template: str, family: str) -> pathlib.Path:
    """Resuelve el directorio final para ``family``.

    Si el template trae ``{familia}`` se formatea directo (permite separar
    editable/no-editable en árboles distintos, ej. ``.../<familia>/Editable``).
    Si no lo trae, se mantiene el comportamiento viejo: se asume una carpeta
    RAÍZ y se le agrega ``<familia>/`` como subcarpeta.
    """
    if "{familia}" in template:
        return pathlib.Path(template.format(familia=family))
    return pathlib.Path(template) / family


def build_one(
    family: str,
    out_dir: pathlib.Path,
    editable_dir: pathlib.Path | None,
    *,
    editable: bool,
) -> bool:
    """Construye una familia: HTML final en ``out_dir``, editable en ``editable_dir``.

    ``False`` si no hay spec para la familia.
    """
    spec = get_spec(family)
    if spec is None:
        return False

    report = build_curated_report(spec)
    html = render_curated_html(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.family}_{datetime.date.today().isoformat()}"
    dst = out_dir / f"{stem}.html"
    dst.write_text(html, encoding="utf-8")
    print(f"OK {spec.title} -> {dst}  ({report.summary()})")

    # Versión editable (misma que "Guardar editable" del chartbuilder): útil ACÁ
    # sobre todo si se va a escribir el texto a mano, sin pasar por el LLM.
    if editable:
        assert editable_dir is not None
        editable_dir.mkdir(parents=True, exist_ok=True)
        ed_dst = editable_dir / f"{stem}_editable.html"
        ed_dst.write_text(make_editable_html(html), encoding="utf-8")
        print(f"OK editable -> {ed_dst}  (contenteditable + panel 💾 Guardar / 📄 Versión final)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Informe curado por familia (gráficos desde parquets, sin LLM).")
    ap.add_argument("--family", default=None, help="Familia a construir (ej. ffmm). Sin valor: lista las disponibles.")
    ap.add_argument("--all", action="store_true", help="Construye TODAS las familias disponibles.")
    ap.add_argument(
        "--out",
        default="T:/GMN/DACE/Practicantes/Leandro/Informes Generados Con IA/{familia}/no_editable",
        help="Carpeta de salida del HTML final (no editable). Admite '{familia}' como placeholder.",
    )
    ap.add_argument(
        "--out-editable",
        default="T:/GMN/DACE/Practicantes/Leandro/Informes Generados Con IA/{familia}/Editable",
        help="Carpeta de salida del HTML editable. Admite '{familia}' como placeholder (--no-editable la omite).",
    )
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
    if not args.family and not args.all:
        print("Familias disponibles:", ", ".join(families) or "(ninguna)")
        return

    targets = families if args.all else [args.family]
    for fam in targets:
        out_dir = _resolve_out_dir(args.out, fam)
        editable_dir = None if args.no_editable else _resolve_out_dir(args.out_editable, fam)
        if not build_one(fam, out_dir, editable_dir, editable=not args.no_editable):
            print(f"No hay spec curado para {fam!r}. Disponibles: {', '.join(families)}")
            raise SystemExit(1)

if __name__ == "__main__":
    main()
