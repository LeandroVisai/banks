#!/usr/bin/env python
"""Fusiona el diccionario AUTO-GENERADO de parquets con nuestro catálogo curado.

Problema que resuelve
---------------------
``sql_catalog/diccionario_parquets_IA.yaml`` lo regenera el pipeline cada vez
que cambia un parquet en las bases: es la VERDAD sobre el esquema real
(columnas, tipos, enums, ``date_range``, renombres de dataset). Pero lo escribe
una máquina y por lo tanto NO sabe nada de lo que agregamos a mano en
``sql_catalog/parquet_catalog.yaml`` a lo largo de los informes:

  * los 41 datasets ``cam_*`` del Informe Cambiario AM (no existen en las bases
    del tablero: los produce ``scripts/build_cambiario_parquets.py``),
  * ``value_scale`` / ``value_kind`` / ``facts`` — pistas nuestras para que el
    TEXTO del informe describa el mismo corte y la misma unidad que el gráfico,
  * la de-duplicación de ids: el diccionario emite el MISMO ``id`` una vez por
    segmento (``posicion_spot_derivados`` aparece 5 veces) y el loader indexa
    por id, así que sin sufijo se pisan entre ellos,
  * ``segment`` canónico en snake_case: el diccionario a veces trae la etiqueta
    de la sección del tablero ("Flujos por fondo", "Stocks") donde nosotros
    necesitamos el segmento (``ffmm``) que filtra ``parquet_report.py``.

Copiar el diccionario encima del catálogo borra todo eso. Este script hace lo
contrario: toma el diccionario como BASE (esquema) y le vuelve a aplicar
nuestra capa curada (overlay), que se deriva sola del catálogo actual — no hay
que mantener una lista a mano.

Reglas de fusión
----------------
1. **Identidad** la fija nuestro catálogo: cada entrada del diccionario se
   casa con la nuestra por ``(file, segment)``; así conserva el id con sufijo
   que ya le habíamos puesto (``posicion_spot_derivados_ffmm``). Si el id
   quedara duplicado igual, se sufija con el segmento.
2. **Esquema** (``columns``) sale del PARQUET REAL cuando el archivo está en
   disco — ni del diccionario ni de nuestro catálogo, que pueden ir atrasados
   respecto del dato. Los ``values`` (enums) se heredan por nombre de columna.
   Si el parquet todavía no se copió, se usan las columnas del diccionario.
3. **``name`` / ``description`` / ``unit`` / ``date_range`` / ``chart_type``**:
   del diccionario (es el que se regenera con los datos).
4. **``value_scale`` / ``value_kind`` / ``facts``**: nuestros, siempre.
5. **``segment``**: del diccionario solo si es un segmento canónico (los que ya
   usa nuestro catálogo); si no, se conserva el nuestro.
6. **Datasets que solo existen en nuestro catálogo** (``cam_*``, y los que
   apuntan a un parquet que sí está en disco) se anexan intactos.
7. Nunca se emiten ``values`` en columnas payload (``_sparkline_json``,
   ``_range_json``, ``_title_override``): son blobs por fila, no enums, y
   además rompen el YAML (el generador los dumpea con comillas sin escapar).

Uso
---
    python scripts/merge_parquet_catalog.py                  # dry-run + reporte
    python scripts/merge_parquet_catalog.py --write          # escribe el catálogo
    python scripts/merge_parquet_catalog.py --write --out /tmp/x.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_DICC = _ROOT / "sql_catalog" / "diccionario_parquets_IA.yaml"
_CATALOG = _ROOT / "sql_catalog" / "parquet_catalog.yaml"

log = logging.getLogger("merge_catalog")

# Columnas de payload: llevan un blob por fila (JSON de sparkline, título ya
# formateado), no una categoría. Declararles `values` infla el catálogo con
# cientos de KB de basura y, tal como los emite el generador, rompe el parser.
_PAYLOAD_COLS = ("_sparkline_json", "_range_json", "_title_override")

# Campos que son NUESTROS: el diccionario no los conoce y hay que reinyectarlos.
_OVERLAY_FIELDS = ("value_scale", "value_kind", "facts")

# `values: ["[{"d": ...}]"]` — enum de payload con comillas sin escapar.
_BAD_VALUES = re.compile(r',\s*values:\s*\[".*\]\s*\}\s*$')


# ─────────────────────────── carga ────────────────────────────


def load_dicc(path: Path) -> dict[str, Any]:
    """Carga el diccionario auto-generado saneando las ``values`` con JSON crudo.

    El generador dumpea el enum de las columnas payload como una lista de
    strings JSON SIN escapar las comillas, lo que deja el YAML inválido (y
    ~300KB de ruido). Se descarta ese enum antes de parsear.
    """
    lines, fixed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if 'values: ["[{' in line or 'values: ["{' in line:
            line, fixed = _BAD_VALUES.sub("}", line), fixed + 1
        lines.append(line)
    if fixed:
        log.warning("diccionario: %d columna(s) payload con enum JSON inválido descartado", fixed)
    return yaml.safe_load("\n".join(lines))


def real_columns(parquet: Path) -> list[tuple[str, str]] | None:
    """``[(nombre, tipo)]`` leídos del parquet real, o ``None`` si no está."""
    if not parquet.exists():
        return None
    import duckdb

    con = duckdb.connect()
    try:
        rows = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet)]
        ).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception as exc:  # parquet corrupto / ilegible: no es fatal
        log.warning("no se pudo leer %s: %s", parquet.name, exc)
        return None
    finally:
        con.close()


# ─────────────────────────── fusión ────────────────────────────


def _enum_index(*entries: dict[str, Any] | None) -> dict[str, list[str]]:
    """``{nombre_columna: values}`` juntando varias entradas (la 1a gana)."""
    out: dict[str, list[str]] = {}
    for entry in entries:
        for col in (entry or {}).get("columns", []) or []:
            name, vals = col.get("name"), col.get("values")
            if name and vals and name not in out and name not in _PAYLOAD_COLS:
                out[name] = vals
    return out


def _build_columns(
    dicc: dict[str, Any], ours: dict[str, Any] | None, parquet: Path
) -> list[dict[str, Any]]:
    """Columnas finales: las del PARQUET REAL si está, si no las del diccionario.

    Los enums se heredan por nombre (diccionario primero, luego los nuestros),
    de modo que renombrar una columna en las bases no arrastre un enum viejo
    pegado a un nombre que ya no existe.
    """
    enums = _enum_index(dicc, ours)
    real = real_columns(parquet)
    if real is None:
        cols = [dict(c) for c in dicc.get("columns", [])]
        for c in cols:
            if c.get("name") in _PAYLOAD_COLS:
                c.pop("values", None)
        return cols
    out: list[dict[str, Any]] = []
    for name, typ in real:
        col: dict[str, Any] = {"name": name, "type": typ}
        if name not in _PAYLOAD_COLS and name in enums:
            col["values"] = enums[name]
        out.append(col)
    return out


def merge(dicc: dict[str, Any], catalog: dict[str, Any], parquet_dir: Path) -> tuple[list[dict], dict]:
    """Devuelve ``(datasets_fusionados, reporte)``."""
    ours = catalog.get("datasets", [])
    by_id = {d["id"]: d for d in ours}
    # Identidad por (file, segment): así una entrada del diccionario recupera el
    # id con sufijo que le habíamos puesto para de-duplicar. La clave puede ser
    # AMBIGUA (tenemos `posicion_spot_derivados` y `..._nr` con el mismo file y
    # segmento), así que gana el primero y el match por id exacto tiene
    # prioridad — si no, el id resultante dependería del orden del dict y
    # fusionar dos veces daría catálogos distintos.
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for d in ours:
        by_key.setdefault((d["file"], d["segment"]), d)
    canon_segments = {d["segment"] for d in ours}

    rep: dict[str, Any] = {
        "actualizados": [], "nuevos": [], "conservados": [],
        "overlay": [], "segment_conservado": [], "colisiones": [],
        "sin_parquet": [], "chart_type_cambiado": [], "columnas_cambiadas": [],
    }

    merged: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    consumed: set[str] = set()  # ids nuestros ya fusionados

    for entry in dicc.get("datasets", []):
        file_, seg = entry["file"], entry["segment"]
        same_id = by_id.get(entry["id"])
        # 1) mismo id Y mismo segmento → es literalmente el mismo dataset;
        # 2) si no, (file, segment) → recupera nuestro id sufijado;
        # 3) si no, mismo id (le cambió el segmento en las bases).
        if same_id is not None and same_id["segment"] == seg:
            mine = same_id
        else:
            mine = by_key.get((file_, seg)) or same_id
        ident = (mine or entry)["id"]

        if ident in used_ids:  # el diccionario repite id+file+segment
            rep["colisiones"].append(f"{ident} (chart_type descartado: {entry['chart_type']})")
            continue

        parquet = parquet_dir / file_
        out: dict[str, Any] = {
            "id": ident,
            "file": file_,
            "chart_type": entry["chart_type"],
            "name": entry["name"],
            "description": entry["description"],
            # El diccionario a veces trae la etiqueta de la sección del tablero
            # ("Stocks") en vez del segmento; nos quedamos con el nuestro.
            "segment": seg if seg in canon_segments else (mine or entry)["segment"],
            "unit": entry["unit"],
            "date_range": entry.get("date_range", entry.get("date", [])),
        }
        if mine and out["segment"] != seg:
            rep["segment_conservado"].append(f"{ident}: {seg!r} → {out['segment']!r}")

        for f in _OVERLAY_FIELDS:  # capa curada nuestra
            if mine and f in mine:
                out[f] = mine[f]
        if mine and any(f in mine for f in _OVERLAY_FIELDS):
            rep["overlay"].append(ident)

        out["columns"] = _build_columns(entry, mine, parquet)
        if not parquet.exists():
            rep["sin_parquet"].append(ident)

        if mine:
            consumed.add(mine["id"])
            rep["actualizados"].append(ident)
            if mine["chart_type"] != out["chart_type"]:
                rep["chart_type_cambiado"].append(
                    f"{ident}: {mine['chart_type']} → {out['chart_type']}")
            a = [c["name"] for c in mine.get("columns", [])]
            b = [c["name"] for c in out["columns"]]
            if a != b:
                rep["columnas_cambiadas"].append(f"{ident}: {a} → {b}")
        else:
            rep["nuevos"].append(ident)

        merged.append(out)
        used_ids.add(ident)

    # Lo que solo existe en nuestro catálogo: cam_* + los custom cuyo parquet
    # sigue en disco. Se anexan intactos (el diccionario no los conoce).
    for d in ours:
        if d["id"] in consumed or d["id"] in used_ids:
            continue
        merged.append(d)
        used_ids.add(d["id"])
        rep["conservados"].append(d["id"])
        if not (parquet_dir / d["file"]).exists():
            rep["sin_parquet"].append(d["id"])

    return merged, rep


# ─────────────────────────── salida ────────────────────────────


def _scalar(value: Any) -> str:
    """Escalar/lista como JSON — que es YAML válido y siempre queda bien citado.

    (``yaml.safe_dump`` de un escalar suelto agrega el marcador de fin de
    documento ``...``, que no se puede incrustar en una línea ``clave: valor``.)
    """
    return json.dumps(value, ensure_ascii=False)


def dump_yaml(parquet_dir: str, datasets: list[dict[str, Any]]) -> str:
    """Serializa respetando el estilo del catálogo (columnas en flow-style)."""
    out = [f"parquet_dir: {parquet_dir}", "", "datasets:"]
    for d in datasets:
        out.append(f"  - id: {d['id']}")
        out.append(f"    file: {d['file']}")
        out.append(f"    chart_type: {_scalar(d['chart_type'])}")
        out.append(f"    name: {_scalar(d['name'])}")
        desc = " ".join(str(d.get("description", "")).split())
        out.append("    description: >")
        out.append(f"      {desc}")
        out.append(f"    segment: {_scalar(d['segment'])}")
        out.append(f"    unit: {_scalar(d['unit'])}")
        out.append(f"    date_range: {_scalar(d.get('date_range') or [])}")
        for f in _OVERLAY_FIELDS:
            if f in d:
                out.append(f"    {f}: {_scalar(d[f])}")
        out.append("    columns:")
        for c in d["columns"]:
            item = {"name": c["name"], "type": c["type"]}
            if c.get("values"):
                item["values"] = c["values"]
            out.append(f"      - {_scalar(item)}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicc", type=Path, default=_DICC, help="diccionario auto-generado")
    ap.add_argument("--catalog", type=Path, default=_CATALOG, help="nuestro catálogo curado")
    ap.add_argument("--out", type=Path, default=None, help="destino (default: --catalog)")
    ap.add_argument("--write", action="store_true", help="escribir (si no, dry-run)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dicc = load_dicc(args.dicc)
    catalog = yaml.safe_load(args.catalog.read_text(encoding="utf-8"))
    parquet_dir = _ROOT / (catalog.get("parquet_dir") or "data_pipeline/parquet")

    merged, rep = merge(dicc, catalog, parquet_dir)

    print(f"\ndiccionario : {len(dicc['datasets'])} entradas  ({args.dicc.name})")
    print(f"catálogo    : {len(catalog['datasets'])} datasets  ({args.catalog.name})")
    print(f"fusionado   : {len(merged)} datasets\n")
    for key, label in (
        ("actualizados", "actualizados desde el diccionario"),
        ("nuevos", "NUEVOS del diccionario"),
        ("conservados", "solo nuestros (conservados intactos)"),
        ("overlay", "con overlay curado reinyectado (value_*/facts)"),
        ("colisiones", "ids duplicados en el diccionario (descartados)"),
        ("segment_conservado", "segment no canónico → conservado el nuestro"),
        ("chart_type_cambiado", "chart_type actualizado"),
        ("sin_parquet", "sin parquet local todavía"),
    ):
        items = rep[key]
        print(f"  {label}: {len(items)}")
        if args.verbose or key in ("nuevos", "colisiones", "segment_conservado",
                                   "chart_type_cambiado"):
            for i in items:
                print(f"      · {i}")
    if args.verbose:
        print(f"\n  columnas re-sincronizadas con el parquet real: {len(rep['columnas_cambiadas'])}")
        for i in rep["columnas_cambiadas"]:
            print(f"      · {i}")

    text = dump_yaml(catalog.get("parquet_dir", "data_pipeline/parquet"), merged)
    # Falla temprano si lo generado no vuelve a parsear.
    reparsed = yaml.safe_load(text)
    assert len(reparsed["datasets"]) == len(merged), "el YAML generado no round-trippea"

    dest = args.out or args.catalog
    if args.write:
        dest.write_text(text, encoding="utf-8")
        print(f"\nOK escrito -> {dest}")
    else:
        print(f"\n(dry-run: nada escrito; usar --write para actualizar {dest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
