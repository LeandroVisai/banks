"""Refresca los ``date_range`` de sql_catalog/parquet_catalog.yaml desde los parquets reales.

Problema que resuelve: los parquets de data_pipeline/parquet/ se regeneran fuera
del repo y se copian con datos nuevos, pero el ``date_range`` del catálogo queda
congelado. El LLM ve ese rango vía ``discover_query`` y, en modo thinking, se lo
toma literal: acota sus queries a la fecha vieja y responde con datos
desactualizados aunque el parquet ya tenga filas más recientes.

Uso (correr DESPUÉS de cada copia de parquets nuevos):

    python scripts/refresh_catalog_dates.py            # actualiza el YAML in-place
    python scripts/refresh_catalog_dates.py --check    # solo reporta drift (exit 1 si hay)

El YAML se edita a nivel de TEXTO (solo las líneas ``date_range:``), preservando
comentarios, orden y formato del catálogo curado a mano. La columna de fecha se
detecta desde el propio catálogo (primera columna DATE/TIMESTAMP; fallback por
nombre). Datasets sin columna de fecha o sin ``date_range`` se omiten.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "sql_catalog" / "parquet_catalog.yaml"

# Nombres típicos de columna temporal cuando el tipo no basta para decidir.
_DATE_NAME_HINTS = ("fecha", "date", "periodo", "mes", "dia")

_ID_RE = re.compile(r"^\s*-\s+id:\s*(\S+)\s*$")
_DATE_RANGE_RE = re.compile(r"^(\s*)date_range:\s*\[.*\]\s*$")


def _find_date_column(columns: list[dict]) -> str | None:
    """Primera columna DATE/TIMESTAMP del catálogo; fallback por nombre."""
    for col in columns:
        ctype = str(col.get("type", "")).upper()
        if "DATE" in ctype or "TIMESTAMP" in ctype:
            return col["name"]
    for col in columns:
        if str(col.get("name", "")).lower() in _DATE_NAME_HINTS:
            return col["name"]
    return None


def _real_date_range(parquet_path: Path, date_col: str) -> tuple[str, str] | None:
    """min/max reales de la columna de fecha, como 'YYYY-MM-DD'."""
    ident = '"' + date_col.replace('"', '""') + '"'
    fname = parquet_path.as_posix().replace("'", "''")
    # TRY_CAST: hay parquets con la fecha como VARCHAR y valores malformados
    # (p. ej. "2024-01-" truncado); se ignoran en vez de abortar el refresh.
    row = duckdb.sql(
        f"SELECT min(d), max(d) FROM ("
        f"  SELECT TRY_CAST({ident} AS DATE) AS d FROM read_parquet('{fname}')"
        f") WHERE d IS NOT NULL"
    ).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    return str(row[0]), str(row[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="no escribe: reporta drift y sale con código 1 si lo hay",
    )
    args = parser.parse_args()

    text = CATALOG_PATH.read_text(encoding="utf-8")
    catalog = yaml.safe_load(text)
    parquet_dir = REPO_ROOT / catalog.get("parquet_dir", "data_pipeline/parquet")

    # dataset_id → date_range real calculado desde el parquet.
    real_ranges: dict[str, tuple[str, str]] = {}
    skipped: list[str] = []
    missing: list[str] = []
    for entry in catalog.get("datasets", []):
        ds_id = entry.get("id", "?")
        if not entry.get("date_range"):
            skipped.append(ds_id)
            continue
        date_col = _find_date_column(entry.get("columns", []))
        if date_col is None:
            skipped.append(ds_id)
            continue
        parquet_path = parquet_dir / entry["file"]
        if not parquet_path.exists():
            missing.append(ds_id)
            continue
        rng = _real_date_range(parquet_path, date_col)
        if rng is not None:
            real_ranges[ds_id] = rng

    # Reescritura a nivel de línea: solo se tocan las líneas date_range.
    out_lines: list[str] = []
    current_id: str | None = None
    changes: list[tuple[str, str, str]] = []  # (id, viejo, nuevo)
    for line in text.splitlines(keepends=True):
        m_id = _ID_RE.match(line)
        if m_id:
            current_id = m_id.group(1)
        m_dr = _DATE_RANGE_RE.match(line)
        if m_dr and current_id in real_ranges:
            lo, hi = real_ranges[current_id]
            new_line = f'{m_dr.group(1)}date_range: ["{lo}", "{hi}"]'
            old = line.rstrip("\n")
            if old.strip() != new_line.strip():
                changes.append((current_id, old.strip(), new_line.strip()))
            line = new_line + ("\n" if line.endswith("\n") else "")
        out_lines.append(line)

    for ds_id, old, new in changes:
        print(f"  {ds_id}:\n    - {old}\n    + {new}")
    print(
        f"\n{len(changes)} dataset(s) con drift, "
        f"{len(real_ranges) - len(changes)} al día, "
        f"{len(skipped)} sin fecha (omitidos), {len(missing)} parquet(s) faltante(s)."
    )
    if missing:
        print(f"  parquets faltantes: {', '.join(missing)}", file=sys.stderr)

    if args.check:
        return 1 if changes else 0
    if changes:
        CATALOG_PATH.write_text("".join(out_lines), encoding="utf-8")
        print(f"Catálogo actualizado: {CATALOG_PATH}")
    else:
        print("Sin cambios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
