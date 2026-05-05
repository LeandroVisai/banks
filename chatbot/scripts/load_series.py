#!/usr/bin/env python3
"""
load_series.py — Carga datos de series históricas desde CSV a `historical_data`.

Uso:
    python -m scripts.load_series tpm datos_tpm.csv
    python -m scripts.load_series ipc_anual datos_ipc.csv --replace

CSV esperado (con header):
    date,value[,notes]
    2024-01-01,11.25,
    2024-02-01,11.25,
    2024-04-01,7.25,recorte 100pb

`--replace` borra los datos existentes de la serie antes de insertar.
Sin `--replace` hace UPSERT (los duplicados de fecha actualizan el valor).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

# Añade chatbot/ al sys.path para importar `app.*` cuando se corre desde scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import close_pool, get_conn, init_pool  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402

log = logging.getLogger(__name__)


def _read_csv(path: Path) -> list[tuple[str, str | None, str | None]]:
    rows: list[tuple[str, str | None, str | None]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "date" not in reader.fieldnames or "value" not in reader.fieldnames:
            raise ValueError(f"CSV inválido: faltan columnas 'date' y/o 'value' en {path}")
        for row in reader:
            date_s = (row["date"] or "").strip()
            val_s = (row["value"] or "").strip()
            notes = (row.get("notes") or "").strip() or None
            if not date_s:
                continue
            rows.append((date_s, val_s if val_s else None, notes))
    return rows


async def _series_exists(series_id: str) -> bool:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM historical_series WHERE series_id = %s", [series_id])
            return (await cur.fetchone()) is not None


async def _delete_series_data(series_id: str) -> int:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM historical_data WHERE series_id = %s", [series_id])
            n = cur.rowcount
        await conn.commit()
    return n


async def _insert_rows(series_id: str, rows: Iterable[tuple]) -> int:
    n = 0
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            for date_s, val_s, notes in rows:
                value = float(val_s) if val_s is not None else None
                await cur.execute(
                    """
                    INSERT INTO historical_data (series_id, date, value, notes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (series_id, date)
                    DO UPDATE SET value = EXCLUDED.value, notes = EXCLUDED.notes
                    """,
                    [series_id, date_s, value, notes],
                )
                n += 1
        await conn.commit()
    return n


async def run(series_id: str, csv_path: Path, replace: bool) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        log.warning("CSV vacío: %s", csv_path)
        return

    await init_pool()
    try:
        if not await _series_exists(series_id):
            raise SystemExit(
                f"La serie '{series_id}' no está registrada en `historical_series`. "
                f"Agrégala primero a schema/002_historical_series.sql o vía INSERT."
            )

        if replace:
            deleted = await _delete_series_data(series_id)
            log.info("--replace: eliminadas %d filas previas de %s", deleted, series_id)

        n = await _insert_rows(series_id, rows)
        log.info("Cargadas %d observaciones a %s desde %s", n, series_id, csv_path)
    finally:
        await close_pool()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("series_id", help="ID de la serie (ej. 'tpm', 'ipc_anual')")
    ap.add_argument("csv_path", type=Path, help="Ruta al CSV con columnas date,value[,notes]")
    ap.add_argument("--replace", action="store_true", help="Borra los datos existentes antes de cargar")
    args = ap.parse_args()

    if not args.csv_path.exists():
        ap.error(f"No existe: {args.csv_path}")

    configure_logging()
    asyncio.run(run(args.series_id, args.csv_path, args.replace))


if __name__ == "__main__":
    main()
