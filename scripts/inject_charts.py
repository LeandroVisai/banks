#!/usr/bin/env python3
"""Inyecta gráficos SVG en un informe HTML ya generado (proceso SEPARADO).

Segundo paso, independiente del LLM: ``scripts/parquet_report.py`` produce el
HTML de solo texto; este script lo toma y le incrusta los gráficos leyendo las
series DIRECTO de los parquets (no de los agregados del informe). No usa GPU ni
modelo — solo el catálogo y los parquets en el sistema de archivos.

Uso:
    python scripts/inject_charts.py data/parquet_reports/html/reporte_ffmm_2026-06-13.html
    python scripts/inject_charts.py reporte.html --out reporte_charts.html
    python scripts/inject_charts.py data/parquet_reports/html/   # todos los .html del dir

Por defecto escribe junto al original con sufijo ``_charts.html``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.reporting import inject_charts_into_html  # noqa: E402


def _out_path(src: pathlib.Path, out_arg: str | None) -> pathlib.Path:
    if out_arg:
        return pathlib.Path(out_arg)
    return src.with_name(f"{src.stem}_charts{src.suffix}")


def _process_one(src: pathlib.Path, out_arg: str | None) -> None:
    if src.stem.endswith("_charts"):
        return  # no re-inyectar una salida previa
    html_text = src.read_text(encoding="utf-8")
    out_html, stats = inject_charts_into_html(html_text)
    dst = _out_path(src, out_arg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(out_html, encoding="utf-8")
    print(f"OK {src.name} -> {dst}  ({stats.summary()})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inyecta gráficos SVG en un informe HTML ya generado.")
    ap.add_argument("input", help="Archivo .html del informe, o un directorio con .html.")
    ap.add_argument("--out", default=None, help="Ruta de salida (solo si input es un archivo).")
    ap.add_argument("--verbose", action="store_true", help="Log a nivel INFO.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    path = pathlib.Path(args.input)
    if path.is_dir():
        htmls = sorted(p for p in path.glob("*.html") if not p.stem.endswith("_charts"))
        if not htmls:
            print(f"No hay .html (sin sufijo _charts) en {path}")
            return
        for p in htmls:
            _process_one(p, None)
    elif path.is_file():
        _process_one(path, args.out)
    else:
        print(f"No existe: {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
