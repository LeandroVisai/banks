"""
extract.py — Ejecuta las queries del catálogo y materializa parquets (snapshots offline).

Modos:
  --mode dw     : consulta el DW vía gd.get_data() (requiere GET_DATA_PATH)
  --mode mock   : genera datos sintéticos (para desarrollo/testing offline)

Ejemplo:
  python -m data_pipeline.extract --mode mock
  python -m data_pipeline.extract --mode dw --get-data-path "D:\\GOM\\DACE\\Nacho\\Modulos"
  python -m data_pipeline.extract --mode dw --get-data-path /ruta/Modulos --since 2018-01-01

El output es un parquet por categoría en snapshots/, en formato long:
    columnas: date, series_id, value, [tenor]

Nota: en producción los chatbots consultan el DW directamente vía dw_store.py.
Los parquets son solo backups opcionales para desarrollo offline.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from data_pipeline.parquet_store import DEFAULT_SNAPSHOT_DIR, load_catalog  # noqa: E402

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conector al DW vía gd.get_data() — mismo módulo que Monitor.py
# ─────────────────────────────────────────────────────────────────────────────

def _load_gd(get_data_path: str):
    """Importa Get_Data desde la ruta configurada (igual que dw_store._load_gd)."""
    import importlib.util
    p = Path(get_data_path)
    if not p.is_dir():
        raise RuntimeError(f"GET_DATA_PATH no existe: {get_data_path!r}")
    module_file = p / "Get_Data.py"
    if not module_file.exists():
        raise RuntimeError(f"Get_Data.py no encontrado en {get_data_path!r}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("Get_Data", module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_sql(gd, sql: str) -> pd.DataFrame:
    log.debug("SQL: %s", sql[:200] + ("..." if len(sql) > 200 else ""))
    return gd.get_data(sql)


# ─────────────────────────────────────────────────────────────────────────────
# Modo DW: traduce el catálogo a SQL y normaliza
# ─────────────────────────────────────────────────────────────────────────────

def _build_query(s: dict, since: Optional[date]) -> Optional[str]:
    """
    Construye una query SELECT para una serie del catálogo.
    Las series con `derivation` (agregaciones complejas) se manejan caso por caso
    en _run_derived_series; aquí retornamos None.
    """
    if "derivation" in s:
        return None
    table = s.get("sql_table")
    col = s.get("sql_column")
    if not table or not col:
        return None

    where = "WHERE 1=1"
    if "sql_filter" in s:
        where += f" AND {s['sql_filter']}"
    if since:
        where += f" AND Fecha >= '{since.isoformat()}'"
    # Algunas tablas usan otra columna de fecha; default a Fecha
    return (
        f"SELECT Fecha AS date, {col} AS value FROM {table} {where} "
        f"ORDER BY Fecha ASC"
    )


def _normalize_value(df: pd.DataFrame, multiplier: Optional[float]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if multiplier and multiplier != 1:
        df["value"] = df["value"] * multiplier
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def extract_from_dw(
    catalog_path: Path,
    output_dir: Path,
    *,
    get_data_path: str,
    since: Optional[date] = None,
    only_categories: Optional[list[str]] = None,
) -> dict:
    """Itera el catálogo, ejecuta las queries vía gd.get_data() y escribe los parquets."""
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    gd = _load_gd(get_data_path)
    summary = {"categories": {}, "errors": []}

    for cat_name, cat_def in (raw.get("categories") or {}).items():
        if only_categories and cat_name not in only_categories:
            continue

        parquet_file = cat_def.get("parquet_file", f"{cat_name}.parquet")
        log.info("→ Categoría: %s (parquet: %s)", cat_name, parquet_file)
        frames: list[pd.DataFrame] = []

        for s in (cat_def.get("series") or []):
            sid = s["id"]
            sql = _build_query(s, since)
            if sql is None:
                log.warning("   skip %s (derivation no implementada)", sid)
                continue
            try:
                raw_df = _run_sql(gd, sql)
            except Exception as e:
                msg = f"{sid}: {e}"
                log.error("   ERROR %s", msg)
                summary["errors"].append(msg)
                continue

            if raw_df.empty:
                log.warning("   %s: vacío", sid)
                continue

            df = _normalize_value(raw_df, s.get("multiply_by"))
            df["series_id"] = sid
            if s.get("tenor"):
                df["tenor"] = s["tenor"]
            frames.append(df[["date", "series_id", "value"] + (["tenor"] if "tenor" in df.columns else [])])
            log.info("   %s: %d filas", sid, len(df))

        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)
        out_path = output_dir / parquet_file
        merged.to_parquet(out_path, compression="snappy", index=False)
        summary["categories"][cat_name] = {
            "n_series": merged["series_id"].nunique(),
            "n_rows": len(merged),
            "file": str(out_path),
        }
        log.info("   ✓ %s escrito (%d filas)", out_path.name, len(merged))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Modo mock: datos sintéticos para desarrollo offline
# ─────────────────────────────────────────────────────────────────────────────

def _mock_series(series_id: str, unit: str, frequency: str,
                 days: int = 1500) -> pd.DataFrame:
    """Genera una serie sintética plausible según el tipo/unidad."""
    rng = np.random.default_rng(seed=hash(series_id) & 0xFFFFFFFF)

    end = date.today() - timedelta(days=1)
    if frequency == "diario":
        idx = pd.bdate_range(end - timedelta(days=days), end, freq="B")
    elif frequency == "mensual":
        idx = pd.date_range(end - timedelta(days=days), end, freq="ME")
    elif frequency == "trimestral":
        idx = pd.date_range(end - timedelta(days=days * 4), end, freq="QE")
    else:
        idx = pd.bdate_range(end - timedelta(days=days), end, freq="B")

    n = len(idx)
    if n == 0:
        return pd.DataFrame(columns=["date", "series_id", "value"])

    # Centro y volatilidad razonables según unidad
    if "CLP por USD" in unit:
        center, vol = 920.0, 8.0
    elif "USD/lb" in unit:
        center, vol = 4.2, 0.05
    elif unit in ("bp", "puntos"):
        center, vol = 50.0, 4.0
    elif unit == "índice":
        center, vol = 100.0, 1.5
    elif "ratio" in unit:
        center, vol = 1.3, 0.05
    elif "% anual" in unit or unit == "%":
        # Yields: usar rango plausible 0–12%
        center, vol = 6.0, 0.20
    elif "USD millones" in unit:
        center, vol = 800.0, 50.0
    elif "USD" in unit:
        center, vol = 80.0, 1.5
    else:
        center, vol = 1.0, 0.02

    # Random walk con reversion a la media — más realista que ruido puro
    walk = rng.normal(0, vol, size=n).cumsum()
    walk = walk - 0.05 * walk.cumsum() / (np.arange(n) + 1)
    values = center + walk

    df = pd.DataFrame({
        "date": idx.tz_localize(None),
        "series_id": series_id,
        "value": values,
    })
    return df


def extract_mock(
    catalog_path: Path,
    output_dir: Path,
    *,
    only_categories: Optional[list[str]] = None,
) -> dict:
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"categories": {}}

    for cat_name, cat_def in (raw.get("categories") or {}).items():
        if only_categories and cat_name not in only_categories:
            continue
        parquet_file = cat_def.get("parquet_file", f"{cat_name}.parquet")
        frames: list[pd.DataFrame] = []
        for s in (cat_def.get("series") or []):
            df = _mock_series(s["id"], s.get("unit", ""), s.get("frequency", "diario"))
            if s.get("tenor"):
                df["tenor"] = s["tenor"]
            frames.append(df)

        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        out_path = output_dir / parquet_file
        merged.to_parquet(out_path, compression="snappy", index=False)
        summary["categories"][cat_name] = {
            "n_series": merged["series_id"].nunique(),
            "n_rows": len(merged),
            "file": str(out_path),
        }
        log.info("✓ MOCK %s (%d series, %d filas)",
                 out_path.name, merged["series_id"].nunique(), len(merged))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["dw", "mock"], required=True)
    ap.add_argument("--get-data-path", default=os.environ.get("GET_DATA_PATH", ""),
                    help="Directorio con Get_Data.py (modo dw). "
                         "Equivale a sys.path.append en Monitor.py. "
                         "También se lee de la variable de entorno GET_DATA_PATH.")
    ap.add_argument("--catalog", type=Path,
                    default=ROOT / "series_catalog.yaml",
                    help="Ruta al YAML del catálogo")
    ap.add_argument("--output", type=Path, default=DEFAULT_SNAPSHOT_DIR,
                    help="Directorio donde escribir los parquets")
    ap.add_argument("--since", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                    default=None, help="Fecha desde (YYYY-MM-DD), solo modo dw")
    ap.add_argument("--category", action="append", default=None,
                    help="Filtra a una o más categorías (repetible)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "mock":
        summary = extract_mock(args.catalog, args.output, only_categories=args.category)
    else:
        if not args.get_data_path:
            ap.error("--get-data-path (o GET_DATA_PATH env) es requerido en modo dw")
        summary = extract_from_dw(
            args.catalog, args.output,
            get_data_path=args.get_data_path,
            since=args.since,
            only_categories=args.category,
        )

    log.info("─" * 60)
    log.info("Resumen:")
    for name, info in summary.get("categories", {}).items():
        log.info("  %s: %d series, %d filas → %s",
                 name, info["n_series"], info["n_rows"], info["file"])
    if summary.get("errors"):
        log.warning("  Errores: %d (%s)", len(summary["errors"]), summary["errors"][:3])


if __name__ == "__main__":
    main()
