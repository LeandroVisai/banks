"""
dw_store.py — Series históricas vía gd.get_data() del módulo Get_Data.

Usa la misma función que querys/Monitor.py para consultar el Data Warehouse:
    import Get_Data as gd
    df = gd.get_data("SELECT Fecha, [Col] FROM Tabla WHERE ...")

La ruta al módulo Get_Data.py se configura con GET_DATA_PATH en el .env.
En producción (Windows): D:\\GOM\\DACE\\Nacho\\Modulos

API pública compatible con parquet_store:
  - get_series_meta, list_categories, list_series  (solo catálogo YAML, sin DW)
  - fetch_series, fetch_multiple                   (async, usa gd.get_data())
  - format_table, format_value                     (formateo puro)
"""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from data_pipeline.parquet_store import (
    DEFAULT_CATALOG_PATH,
    SeriesMeta,
    format_table,
    format_value,
    list_categories as _list_categories_raw,
    list_series as _list_series_raw,
    load_catalog,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Carga dinámica de Get_Data
# ─────────────────────────────────────────────────────────────────────────────

_gd_module = None  # cacheado tras primera carga


def _load_gd(get_data_path: str) -> object:
    """
    Importa Get_Data desde la ruta configurada.
    Equivalente a: sys.path.append(get_data_path); import Get_Data as gd
    """
    global _gd_module
    if _gd_module is not None:
        return _gd_module

    p = Path(get_data_path)
    if not p.is_dir():
        raise RuntimeError(
            f"GET_DATA_PATH no existe o no es un directorio: {get_data_path!r}"
        )

    module_file = p / "Get_Data.py"
    if not module_file.exists():
        raise RuntimeError(f"Get_Data.py no encontrado en {get_data_path!r}")

    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

    spec = importlib.util.spec_from_file_location("Get_Data", module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _gd_module = mod
    log.info("Get_Data cargado desde %s", module_file)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de catálogo
# ─────────────────────────────────────────────────────────────────────────────

def _catalog_str(catalog_path: Optional[Path]) -> Optional[str]:
    return str(catalog_path) if catalog_path else None


# ─────────────────────────────────────────────────────────────────────────────
# API de catálogo (solo YAML, no toca el DW)
# ─────────────────────────────────────────────────────────────────────────────

def get_series_meta(series_id: str, catalog_path: Optional[Path] = None) -> Optional[dict]:
    cat = load_catalog(_catalog_str(catalog_path))
    s = cat.get(series_id)
    return s.to_dict() if s else None


def list_categories(catalog_path: Optional[Path] = None) -> list[dict]:
    return _list_categories_raw(catalog_path=_catalog_str(catalog_path))


def list_series(
    category: Optional[str] = None,
    variable: Optional[str] = None,
    tenor: Optional[str] = None,
    *,
    catalog_path: Optional[Path] = None,
    include_coverage: bool = False,
    **_ignored,
) -> list[dict]:
    """
    Lista series del catálogo (solo metadata YAML).
    La cobertura temporal no se incluye para no lanzar queries al DW en el listado.
    """
    return _list_series_raw(
        category=category,
        variable=variable,
        tenor=tenor,
        catalog_path=_catalog_str(catalog_path),
        include_coverage=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de queries SQL (mismo patrón que Monitor.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_series_sql(
    meta: SeriesMeta,
    date_from: Optional[date | datetime],
    date_to: Optional[date | datetime],
    limit: Optional[int],
) -> str:
    """SELECT simple: sql_table + sql_column, con filtros de fecha."""
    table = meta.sql_table
    col = meta.sql_column

    where_parts = ["1=1"]
    if meta.sql_filter:
        where_parts.append(meta.sql_filter)
    if date_from is not None:
        d = date_from if isinstance(date_from, date) else date_from.date()
        where_parts.append(f"Fecha >= '{d.isoformat()}'")
    if date_to is not None:
        d = date_to if isinstance(date_to, date) else date_to.date()
        where_parts.append(f"Fecha <= '{d.isoformat()}'")

    where = " AND ".join(where_parts)

    if limit is not None and date_from is None and date_to is None:
        # TOP N más recientes, reordenados asc
        return (
            f"SELECT TOP {limit} date, value FROM ("
            f"  SELECT Fecha AS date, {col} AS value"
            f"  FROM {table} WHERE {where} ORDER BY Fecha DESC"
            f") AS sub ORDER BY date ASC"
        )

    return (
        f"SELECT Fecha AS date, {col} AS value"
        f" FROM {table} WHERE {where} ORDER BY Fecha ASC"
    )


def _build_tib_sql(
    meta: SeriesMeta,
    date_from: Optional[date | datetime],
    date_to: Optional[date | datetime],
    limit: Optional[int],
) -> str:
    """Query especial para series con derivation (TIB ponderado, sumas, etc.)."""
    table = meta.sql_table
    derivation = meta.derivation or ""

    where_parts = ["1=1"]
    if date_from is not None:
        d = date_from if isinstance(date_from, date) else date_from.date()
        where_parts.append(f"Fecha >= '{d.isoformat()}'")
    if date_to is not None:
        d = date_to if isinstance(date_to, date) else date_to.date()
        where_parts.append(f"Fecha <= '{d.isoformat()}'")
    where = " AND ".join(where_parts)

    if "weighted_avg" in derivation:
        inner = (
            f"SELECT Fecha AS date,"
            f" SUM(Tasa * Monto) / NULLIF(SUM(Monto), 0) AS value"
            f" FROM {table} WHERE {where} GROUP BY Fecha"
        )
    elif "sum" in derivation:
        inner = (
            f"SELECT Fecha AS date, SUM(Monto) AS value"
            f" FROM {table} WHERE {where} GROUP BY Fecha"
        )
    else:
        raise ValueError(f"derivation no soportada: {derivation!r}")

    if limit is not None and date_from is None and date_to is None:
        return (
            f"SELECT TOP {limit} date, value FROM ({inner} ORDER BY date DESC) AS sub"
            f" ORDER BY date ASC"
        )
    return f"SELECT date, value FROM ({inner}) AS sub ORDER BY date ASC"


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución síncrona vía gd.get_data()
# ─────────────────────────────────────────────────────────────────────────────

def _run_query_sync(get_data_path: str, sql: str) -> pd.DataFrame:
    """
    Llama a gd.get_data(sql) en el hilo del executor.
    get_data() retorna un DataFrame igual que pd.read_sql().
    """
    gd = _load_gd(get_data_path)
    return gd.get_data(sql)


def _normalize_df(df: pd.DataFrame, multiply_by: Optional[float]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if multiply_by and multiply_by != 1.0:
        df["value"] = df["value"] * multiply_by
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# API de series (async — corre gd.get_data en executor)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_series(
    series_id: str,
    *,
    get_data_path: str,
    catalog_path: Optional[Path] = None,
    date_from: Optional[date | datetime] = None,
    date_to: Optional[date | datetime] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retorna DataFrame (date, value) para series_id usando gd.get_data().
    La call síncrona se ejecuta en ThreadPoolExecutor para no bloquear asyncio.
    """
    cat = load_catalog(_catalog_str(catalog_path))
    meta = cat.get(series_id)
    if meta is None:
        raise KeyError(f"Series desconocida: {series_id!r}")

    if meta.derivation:
        if not meta.sql_table:
            log.warning("Serie %s con derivation pero sin sql_table — omitida", series_id)
            return pd.DataFrame(columns=["date", "value"])
        sql = _build_tib_sql(meta, date_from=date_from, date_to=date_to, limit=limit)
    elif meta.sql_table and meta.sql_column:
        sql = _build_series_sql(meta, date_from=date_from, date_to=date_to, limit=limit)
    else:
        log.warning("Serie %s sin sql_table/sql_column en catálogo — omitida", series_id)
        return pd.DataFrame(columns=["date", "value"])

    log.debug("DW [%s]: %s", series_id, sql[:160] + ("..." if len(sql) > 160 else ""))

    loop = asyncio.get_event_loop()
    try:
        raw_df = await loop.run_in_executor(
            None, functools.partial(_run_query_sync, get_data_path, sql)
        )
    except Exception as e:
        log.error("gd.get_data falló para %s: %s", series_id, e)
        raise

    if raw_df.empty:
        return pd.DataFrame(columns=["date", "value"])

    return _normalize_df(raw_df, meta.multiply_by)


async def fetch_multiple(
    series_ids: list[str],
    *,
    get_data_path: str,
    catalog_path: Optional[Path] = None,
    date_from: Optional[date | datetime] = None,
    date_to: Optional[date | datetime] = None,
) -> dict[str, pd.DataFrame]:
    """Trae varias series en paralelo (asyncio.gather)."""
    tasks = [
        fetch_series(
            sid,
            get_data_path=get_data_path,
            catalog_path=catalog_path,
            date_from=date_from,
            date_to=date_to,
        )
        for sid in series_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[str, pd.DataFrame] = {}
    for sid, res in zip(series_ids, results):
        if isinstance(res, Exception):
            log.warning("fetch_multiple: %s falló: %s", sid, res)
            out[sid] = pd.DataFrame(columns=["date", "value"])
        else:
            out[sid] = res
    return out
