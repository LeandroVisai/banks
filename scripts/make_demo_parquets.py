#!/usr/bin/env python3
"""Genera parquets con datos FICTICIOS para probar el render de un informe.

⚠️  Las cifras que produce son INVENTADAS. Sirven para ver cómo queda el informe
(layout, gráficos, tablas, el correo) cuando todavía no llegaron los parquets reales
del servidor — NO para analizar nada ni para mandar a nadie como si fueran del BCCh.

El esquema NO se inventa: sale de ``sql_catalog/parquet_catalog.yaml`` (columnas,
tipos, valores de enum y ``date_range`` de cada dataset), así que los parquets tienen
exactamente la forma que van a tener los de verdad y las transforms/gráficos se
ejercitan igual que en producción.

Uso:
    python scripts/make_demo_parquets.py --family dcv --out data_pipeline/demo/parquet
    python scripts/make_demo_parquets.py --datasets flujos_ffmm,duracion_ffmm --out /tmp/p
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import math
import pathlib
import random
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.infrastructure.sql.parquet_catalog_loader import load_parquet_catalog  # noqa: E402

_DATE_TYPES = ("TIMESTAMP", "DATE", "DATETIME")
_INT_TYPES = ("BIGINT", "INTEGER", "INT", "SMALLINT")
_MAX_COMBOS = 600          # techo de combinaciones categóricas por dataset
_DEFAULT_DAYS = 130        # historia máxima (días hábiles); alcanza para T-7 y T-30


def _is_date(col) -> bool:
    return str(col.type).upper().split("(")[0] in _DATE_TYPES


def _is_numeric(col) -> bool:
    t = str(col.type).upper().split("(")[0]
    return t in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", *_INT_TYPES)


def _business_days(start: dt.date, end: dt.date, limit: int) -> list[dt.date]:
    """Días hábiles del rango, quedándose con los ``limit`` ÚLTIMOS (el informe mira
    el corte más reciente; arrastrar 9 años de historia solo infla el parquet)."""
    days, cursor = [], start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += dt.timedelta(days=1)
    return days[-limit:] if len(days) > limit else days or [end]


def _range_of(dataset, days: int) -> list[dt.date]:
    lo, hi = (dataset.date_range or [None, None])[:2] or (None, None)
    try:
        start = dt.date.fromisoformat(str(lo)[:10])
        end = dt.date.fromisoformat(str(hi)[:10])
    except (TypeError, ValueError):
        end = dt.date.today()
        start = end - dt.timedelta(days=days * 2)
    return _business_days(start, end, days)


def _scale_for(unit: str) -> tuple[float, float, float, float]:
    """``(base, volatilidad, mínimo, máximo)`` plausibles según la unidad del dataset.

    Una duración en años y un stock en millones de dólares no pueden salir del mismo
    rango, y sin techo el sorteo lognormal escupe duraciones de 30 años que estiran el
    eje del gráfico y delatan que el dato es de mentira."""
    u = (unit or "").lower()
    if "año" in u or "anio" in u:
        return 3.5, 0.02, 0.2, 12.0
    if "%" in u or "porcentaje" in u or "tasa" in u:
        return 6.0, 0.03, 0.1, 25.0
    if "mill" in u or "mm" in u or "usd" in u:
        return 4_000.0, 0.015, 20.0, 120_000.0
    return 1_000.0, 0.02, 1.0, 50_000.0


def _combos(dataset, rng: random.Random) -> tuple[list, list[tuple[str, ...]]]:
    """Columnas categóricas del dataset y sus combinaciones (recortadas al techo)."""
    cats = [c for c in dataset.columns
            if not _is_date(c) and not _is_numeric(c) and getattr(c, "values", None)]
    if not cats:
        return [], [()]
    combos = list(itertools.product(*[list(c.values) for c in cats]))
    if len(combos) > _MAX_COMBOS:
        combos = rng.sample(combos, _MAX_COMBOS)
    return cats, combos


def build_dataset(dataset, out_dir: pathlib.Path, *, days: int, seed: int) -> tuple[pathlib.Path, int]:
    """Escribe un parquet ficticio con el esquema real de ``dataset``."""
    import duckdb

    rng = random.Random(f"{seed}:{dataset.id}")
    dates = _range_of(dataset, days)
    cats, combos = _combos(dataset, rng)

    date_cols = [c for c in dataset.columns if _is_date(c)]
    num_cols = [c for c in dataset.columns if _is_numeric(c)]
    free_text = [c for c in dataset.columns
                 if not _is_date(c) and not _is_numeric(c) and not getattr(c, "values", None)]
    base, vol, lo, hi = _scale_for(dataset.unit)

    rows = []
    for combo in combos:
        # Un nivel por combinación (lognormal: convive un PDBC gigante con un BCCh
        # marginal) y un paseo aleatorio suave arriba, para que las variaciones
        # T-7 / T-30 que calcula el informe den números coherentes.
        levels = [min(hi, max(lo, base * math.exp(rng.gauss(0, 0.9)))) for _ in num_cols]
        for date in dates:
            values = []
            for i, _c in enumerate(num_cols):
                levels[i] = min(hi, max(lo, levels[i] * (1 + rng.gauss(0, vol))))
                values.append(round(levels[i], 4))
            row = []
            for col in dataset.columns:
                if col in date_cols:
                    row.append(dt.datetime.combine(date, dt.time()))
                elif col in num_cols:
                    row.append(values[num_cols.index(col)])
                elif col in free_text:
                    row.append(f"demo-{rng.randint(1, 4)}")
                else:
                    row.append(combo[cats.index(col)])
            rows.append(tuple(row))

    def _sql_type(col) -> str:
        if _is_date(col):
            return "TIMESTAMP"
        if _is_numeric(col):
            return "BIGINT" if str(col.type).upper() in _INT_TYPES else "DOUBLE"
        return "VARCHAR"

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / dataset.file
    schema = ", ".join(f'"{c.name}" {_sql_type(c)}' for c in dataset.columns)
    placeholders = ", ".join("?" for _ in dataset.columns)

    con = duckdb.connect()
    try:
        con.execute(f"CREATE TABLE demo ({schema})")
        con.executemany(f"INSERT INTO demo VALUES ({placeholders})", rows)
        con.execute(f"COPY demo TO '{target.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
    return target, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parquets con datos FICTICIOS (esquema real del catálogo) para probar el render.")
    ap.add_argument("--family", default=None,
                    help="Genera los datasets que consume el spec de esa familia (ej. dcv).")
    ap.add_argument("--datasets", default="", help="Lista de dataset_id separada por comas.")
    ap.add_argument("--out", default="data_pipeline/demo/parquet", help="Carpeta destino.")
    ap.add_argument("--days", type=int, default=_DEFAULT_DAYS,
                    help=f"Días hábiles de historia como máximo (default {_DEFAULT_DAYS}).")
    ap.add_argument("--seed", type=int, default=7, help="Semilla (mismos datos entre corridas).")
    args = ap.parse_args()

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.family:
        from banks_rag.application.reporting.specs import get_spec

        spec = get_spec(args.family)
        if spec is None:
            sys.exit(f"No hay spec para la familia '{args.family}'.")
        for block in spec.blocks:
            if block.source_id and block.source_id not in wanted:
                wanted.append(block.source_id)
    if not wanted:
        sys.exit("Indicá --family o --datasets.")

    catalog = {d.id: d for d in load_parquet_catalog()}
    out_dir = pathlib.Path(args.out)
    print("⚠️  DATOS FICTICIOS — solo para probar el render, no son cifras del BCCh")
    for dataset_id in wanted:
        dataset = catalog.get(dataset_id)
        if dataset is None:
            print(f"SKIP {dataset_id}: no está en el catálogo")
            continue
        target, n = build_dataset(dataset, out_dir, days=args.days, seed=args.seed)
        print(f"OK   {dataset_id:36s} -> {target}  ({n:,} filas)")


if __name__ == "__main__":
    main()
