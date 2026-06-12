#!/usr/bin/env python3
"""Audita los parquets reales vs el catálogo (sql_catalog/parquet_catalog.yaml).

Para cada parquet en data_pipeline/parquet/: detecta roles de columnas (fecha /
categóricas / numéricas) desde el esquema REAL y reporta los desajustes contra
el catálogo:

  - columnas declaradas en el catálogo que NO existen en el parquet,
  - ids del catálogo sin parquet,
  - parquets sin entrada en el catálogo,
  - snapshots (sin columna de fecha).

Es el "identificar cómo vienen los datos" tras convertir el Excel. No modifica
nada. Para actualizar los date_range usa scripts/refresh_catalog_dates.py.

Uso:
    python3 scripts/audit_parquets.py            # tabla + resumen de mismatches
    python3 scripts/audit_parquets.py --segment ffmm
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import duckdb

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.reporting.parquet_facts import detect_roles  # noqa: E402
from banks_rag.infrastructure.sql.parquet_catalog_loader import (  # noqa: E402
    get_parquet_dir,
    load_parquet_catalog,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audita parquets reales vs catálogo.")
    ap.add_argument("--segment", default=None, help="Filtrar por segmento del catálogo.")
    args = ap.parse_args()

    entries = load_parquet_catalog()
    if args.segment:
        entries = [e for e in entries if e.segment == args.segment]
    pdir = get_parquet_dir()
    con = duckdb.connect()

    missing_parquet: list[str] = []
    col_mismatches: list[tuple[str, list[str]]] = []
    snapshots: list[str] = []

    print(f"{'dataset':32} {'shape':12} {'date_col':14} {'#cat':>4} {'#val':>4}  cols_faltantes")
    print("-" * 100)
    for e in sorted(entries, key=lambda x: x.id):
        p = e.parquet_path(pdir)
        if not p.exists():
            missing_parquet.append(e.id)
            print(f"{e.id:32} {'NO PARQUET':12}")
            continue
        roles = detect_roles(p, con)
        real_cols = {c[0] for c in con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{p.as_posix()}')"
        ).fetchall()}
        declared = [c.name for c in e.columns]
        faltan = [c for c in declared if c not in real_cols]
        if faltan:
            col_mismatches.append((e.id, faltan))
        shape = "snapshot" if roles.date_col is None else "timeseries"
        if roles.date_col is None:
            snapshots.append(e.id)
        print(
            f"{e.id:32} {shape:12} {roles.date_col!s:14} "
            f"{len(roles.category_cols):>4} {len(roles.value_cols):>4}  "
            f"{', '.join(faltan) if faltan else ''}"
        )

    # Parquets sin entrada de catálogo (datasets nuevos del Excel).
    cat_files = {e.file for e in load_parquet_catalog()}
    on_disk = {p.name for p in pdir.glob("*.parquet")}
    huerfanos = sorted(on_disk - cat_files)

    print("\n=== RESUMEN ===")
    print(f"ids sin parquet: {missing_parquet or 'ninguno'}")
    print(f"snapshots (sin fecha): {snapshots or 'ninguno'}")
    print(f"mismatches de columnas: {col_mismatches or 'ninguno'}")
    print(f"parquets sin entrada en catálogo ({len(huerfanos)}): {huerfanos}")


if __name__ == "__main__":
    main()
