"""
parquet_store.py — API de lectura de series históricas para los chatbots.

Carga el catálogo (series_catalog.yaml) en memoria y resuelve consultas
contra los archivos parquet en snapshots/. Es el único módulo que ambos
chatbots importan; el resto del pipeline (extract.py) solo se usa offline.

Formato canónico de cada parquet (long format):
    columnas: date (datetime64), series_id (str), value (float64), [tenor]

Esquema lógico:
    catálogo  → metadata (id, name, unit, source, frequency, category, tenor)
    parquet   → time series (una fila por (series_id, date))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeriesMeta:
    id: str
    name: str
    category: str
    unit: str
    frequency: str
    source: str
    parquet_file: str
    variable: Optional[str] = None
    tenor: Optional[str] = None
    description: Optional[str] = None
    sql_table: Optional[str] = None
    sql_column: Optional[str] = None
    sql_filter: Optional[str] = None
    multiply_by: Optional[float] = None
    derivation: Optional[str] = None

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "id", "name", "category", "unit", "frequency", "source",
            "variable", "tenor", "description",
        )}
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Catalog:
    series_by_id: dict[str, SeriesMeta] = field(default_factory=dict)
    categories: dict[str, dict] = field(default_factory=dict)  # name → {description, parquet_file, series_ids}

    def list_series(
        self,
        category: Optional[str] = None,
        variable: Optional[str] = None,
        tenor: Optional[str] = None,
    ) -> list[SeriesMeta]:
        out: list[SeriesMeta] = []
        for s in self.series_by_id.values():
            if category and s.category != category:
                continue
            if variable and s.variable != variable:
                continue
            if tenor and s.tenor != tenor:
                continue
            out.append(s)
        return out

    def get(self, series_id: str) -> Optional[SeriesMeta]:
        return self.series_by_id.get(series_id)


# ─────────────────────────────────────────────────────────────────────────────
# Carga del catálogo
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CATALOG_PATH = Path(__file__).parent / "series_catalog.yaml"
DEFAULT_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


@lru_cache(maxsize=1)
def load_catalog(path: Optional[str] = None) -> Catalog:
    """Lee y normaliza el YAML. Cached para evitar reparsear en cada query."""
    p = Path(path) if path else DEFAULT_CATALOG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    cat = Catalog()
    for cat_name, cat_def in (raw.get("categories") or {}).items():
        parquet_file = cat_def.get("parquet_file", f"{cat_name}.parquet")
        cat.categories[cat_name] = {
            "description": cat_def.get("description", ""),
            "parquet_file": parquet_file,
            "series_ids": [],
        }
        for s in (cat_def.get("series") or []):
            sid = s["id"]
            meta = SeriesMeta(
                id=sid,
                name=s.get("name", sid),
                category=cat_name,
                unit=s.get("unit", ""),
                frequency=s.get("frequency", "diario"),
                source=s.get("source", ""),
                parquet_file=parquet_file,
                variable=s.get("variable"),
                tenor=s.get("tenor"),
                description=s.get("description"),
                sql_table=s.get("sql_table"),
                sql_column=s.get("sql_column"),
                sql_filter=s.get("sql_filter"),
                multiply_by=s.get("multiply_by"),
                derivation=s.get("derivation"),
            )
            cat.series_by_id[sid] = meta
            cat.categories[cat_name]["series_ids"].append(sid)

    log.info("Catálogo cargado: %d series en %d categorías",
             len(cat.series_by_id), len(cat.categories))
    return cat


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de parquet
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def _load_parquet(path_str: str) -> pd.DataFrame:
    """Carga un parquet completo. LRU cache evita re-leer el mismo archivo."""
    path = Path(path_str)
    if not path.exists():
        log.warning("Parquet no encontrado: %s", path)
        return pd.DataFrame(columns=["date", "series_id", "value"])
    df = pd.read_parquet(path)
    # Normalizar
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_series(
    series_id: str,
    *,
    snapshot_dir: Optional[Path] = None,
    catalog_path: Optional[str] = None,
    date_from: Optional[date | datetime] = None,
    date_to: Optional[date | datetime] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retorna un DataFrame con (date, value) para `series_id`, ordenado ascendente.

    `limit` aplica AL FINAL después del filtro temporal y trae los más recientes.
    """
    cat = load_catalog(catalog_path)
    meta = cat.get(series_id)
    if meta is None:
        raise KeyError(f"Series desconocida: {series_id!r}")

    snap = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR
    df = _load_parquet(str(snap / meta.parquet_file))

    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    sub = df.loc[df["series_id"] == series_id, ["date", "value"]].copy()
    if sub.empty:
        return sub

    if date_from is not None:
        sub = sub.loc[sub["date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        sub = sub.loc[sub["date"] <= pd.Timestamp(date_to)]

    sub = sub.sort_values("date").reset_index(drop=True)

    if limit is not None and len(sub) > limit:
        sub = sub.tail(limit).reset_index(drop=True)

    return sub


def fetch_multiple(
    series_ids: list[str],
    *,
    snapshot_dir: Optional[Path] = None,
    catalog_path: Optional[str] = None,
    date_from: Optional[date | datetime] = None,
    date_to: Optional[date | datetime] = None,
) -> dict[str, pd.DataFrame]:
    """Atajo para traer varias series — agrupa lecturas por archivo."""
    return {
        sid: fetch_series(sid, snapshot_dir=snapshot_dir, catalog_path=catalog_path,
                          date_from=date_from, date_to=date_to)
        for sid in series_ids
    }


# ─────────────────────────────────────────────────────────────────────────────
# API de inspección (para tools del agente y endpoints)
# ─────────────────────────────────────────────────────────────────────────────

def list_categories(catalog_path: Optional[str] = None) -> list[dict]:
    cat = load_catalog(catalog_path)
    return [
        {"category": name, **info, "n_series": len(info["series_ids"])}
        for name, info in cat.categories.items()
    ]


def list_series(
    category: Optional[str] = None,
    variable: Optional[str] = None,
    tenor: Optional[str] = None,
    *,
    snapshot_dir: Optional[Path] = None,
    catalog_path: Optional[str] = None,
    include_coverage: bool = True,
) -> list[dict]:
    """
    Lista series con su cobertura (primer y último dato disponible).
    `include_coverage=False` omite el lookup en parquet — más rápido.
    """
    cat = load_catalog(catalog_path)
    out: list[dict] = []
    for s in cat.list_series(category=category, variable=variable, tenor=tenor):
        item = s.to_dict()
        if include_coverage:
            cov = _coverage(s, snapshot_dir)
            item.update(cov)
        out.append(item)
    return out


def _coverage(meta: SeriesMeta, snapshot_dir: Optional[Path]) -> dict:
    snap = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR
    df = _load_parquet(str(snap / meta.parquet_file))
    sub = df.loc[df.get("series_id", pd.Series([], dtype=object)) == meta.id, "date"]
    if len(sub) == 0:
        return {"n_observations": 0, "first_date": None, "last_date": None}
    return {
        "n_observations": int(len(sub)),
        "first_date": sub.min().date().isoformat() if pd.notna(sub.min()) else None,
        "last_date": sub.max().date().isoformat() if pd.notna(sub.max()) else None,
    }


def get_series_meta(series_id: str, catalog_path: Optional[str] = None) -> Optional[dict]:
    cat = load_catalog(catalog_path)
    s = cat.get(series_id)
    return s.to_dict() if s else None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de formateo (usados por sql_context y los tools del agente)
# ─────────────────────────────────────────────────────────────────────────────

def format_value(value, unit: str) -> str:
    if pd.isna(value):
        return "N/D"
    val = float(value)
    if any(u in unit for u in ("CLP por", "USD/lb", "USD/barril", "índice")):
        return f"{val:,.2f}"
    if "bp" in unit or unit == "puntos":
        return f"{val:,.0f}"
    if "%" in unit or "ratio" in unit:
        return f"{val:.2f}"
    return f"{val:.4f}"


def format_table(meta: dict, df: pd.DataFrame) -> str:
    """Convierte un DF (date, value) en tabla ASCII para el prompt."""
    if df.empty:
        return ""
    name = meta["name"]
    unit = meta.get("unit", "")
    src = meta.get("source", "")
    freq = meta.get("frequency", "")

    header = f"{name}  [fuente: {src} | {freq} | unidad: {unit}]"
    lines = [header, "-" * len(header)]
    for _, row in df.iterrows():
        d = row["date"]
        date_str = d.strftime("%Y-%m") if freq in ("mensual", "trimestral") else d.strftime("%Y-%m-%d")
        val_str = format_value(row["value"], unit)
        lines.append(f"  {date_str}  {val_str}")
    return "\n".join(lines)


def clear_cache() -> None:
    """Útil tras refrescar parquets sin reiniciar el servicio."""
    _load_parquet.cache_clear()
    load_catalog.cache_clear()
