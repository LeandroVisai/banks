#!/usr/bin/env python3
"""Empaqueta el Chart Builder en un HTML autónomo (offline, sin CDN ni servidor).

Toma la plantilla legible `chart_builder.src.html` (que contiene los marcadores
`/*__PLOTLY__*/` y `/*__HYPARQUET__*/`) y empotra las librerías vendorizadas de
`vendor/`, produciendo `chart_builder.html`: un único archivo que se copia a
cualquier computador y corre con doble-click, sin red.

Uso:
    python tools/build_chart_builder.py

Para actualizar Plotly/hyparquet: reemplaza el archivo en `vendor/`, ajusta las
rutas abajo y vuelve a correr. Editar la LÓGICA se hace en `chart_builder.src.html`,
nunca en el `chart_builder.html` generado (es un blob de ~4.6 MB).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "chart_builder.src.html"
OUT = HERE / "chart_builder.html"
PLOTLY = HERE / "vendor" / "plotly-2.35.2.min.js"
HYPARQUET = HERE / "vendor" / "hyparquet-1.26.2.esm.js"
FZSTD = HERE / "vendor" / "fzstd-0.1.1.esm.js"
CATALOG_YAML = HERE.parent / "sql_catalog" / "parquet_catalog.yaml"
CATALOG_JSON = HERE / "vendor" / "catalog_segments.json"

MARKERS = {
    "/*__PLOTLY__*/": PLOTLY,
    "/*__HYPARQUET__*/": HYPARQUET,
    "/*__FZSTD__*/": FZSTD,
    "/*__CATALOG__*/": CATALOG_JSON,
}


def build_catalog() -> None:
    """Genera vendor/catalog_segments.json: { basename_parquet: {seg:[...], name} }.

    Sale del sql_catalog/parquet_catalog.yaml para poder filtrar datasets por
    segmento dentro del HTML offline. Si falta el YAML o pyyaml, escribe {}.
    """
    mapping: dict[str, dict] = {}
    try:
        import yaml  # type: ignore
        cat = yaml.safe_load(CATALOG_YAML.read_text(encoding="utf-8")) or {}
        for d in cat.get("datasets", []) or []:
            f = d.get("file") or ""
            base = Path(f).stem
            if not base:
                continue
            entry = mapping.setdefault(base, {"seg": [], "name": d.get("name", "")})
            seg = d.get("segment")
            if seg and seg not in entry["seg"]:
                entry["seg"].append(seg)
            if not entry["name"] and d.get("name"):
                entry["name"] = d["name"]
    except Exception as e:  # YAML ausente o pyyaml no instalado → filtro vacío
        print(f"AVISO: catálogo no embebido ({e}); el filtro por segmento saldrá vacío", file=sys.stderr)
    CATALOG_JSON.write_text(json.dumps(mapping, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"catálogo: {len(mapping)} parquets mapeados a segmento")


def main() -> int:
    build_catalog()
    html = SRC.read_text(encoding="utf-8")
    for marker, lib in MARKERS.items():
        if marker not in html:
            print(f"ERROR: falta el marcador {marker} en {SRC.name}", file=sys.stderr)
            return 1
        code = lib.read_text(encoding="utf-8")
        # Guardas: nada que rompa el <script> contenedor o el propio empaquetado.
        if "</script>" in code:
            print(f"ERROR: {lib.name} contiene '</script>' — no es seguro inlinear", file=sys.stderr)
            return 1
        if marker in code:
            print(f"ERROR: {lib.name} contiene el marcador {marker}", file=sys.stderr)
            return 1
        html = html.replace(marker, code)

    OUT.write_text(html, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"OK · {OUT.relative_to(HERE.parent)} generado ({kb:,.0f} KB, standalone offline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
