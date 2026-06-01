"""Helper compartido: arma SQL safe sobre parquet_catalog y la ejecuta.

``execute_query`` y las 5 analytics tools comparten este módulo. La SQL la
arma SIEMPRE la tool — con identificadores y valores validados contra el
esquema del dataset — y NUNCA el LLM. Esto elimina la superficie de SQL
injection y desliga a cada tool de la lógica de construcción/validación.

Filosofía:
- Solo lectura: SELECT, FROM read_parquet, WHERE (range de fecha + igualdad
  contra enums), ORDER BY fecha DESC, LIMIT acotado.
- Identificadores comillados (case-sensitive — las columnas del parquet pueden
  ser "Fecha", "CLP", "Banco"…).
- Strings con escape de apóstrofe.
- Dates validadas como ISO YYYY-MM-DD (no se aceptan relativas como ``-90d`` a
  este nivel; el caller resuelve si las necesita).
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
from pathlib import Path
from typing import Any

from banks_rag.infrastructure.sql.duckdb_runner import MAX_ROWS, run_duckdb
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ParquetDataset,
    get_dataset,
    get_parquet_dir,
    load_parquet_catalog,
)

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 200

# Umbrales de similitud para resolver identificadores que el LLM alucina
# (p.ej. 'spread_btp_spc_plazo' → 'spread_btp_spc', 'usdclp' → 'CLP'). El de
# dataset es más estricto (nombres largos, menos falsos positivos); el de
# columna algo más laxo (nombres cortos).
_DATASET_FUZZY_CUTOFF = 0.85  # estricto: evita resolver a un dataset equivocado
_COLUMN_FUZZY_CUTOFF = 0.6


def resolve_dataset_id(
    entries: list[ParquetDataset], dataset_id: str,
) -> tuple[str | None, str | None]:
    """Resuelve un ``dataset_id`` al id real más cercano.

    Returns ``(resolved_id, note)``: exacto → ``(id, None)``; case-insensitive o
    fuzzy → ``(id_real, nota)``; sin match razonable → ``(None, None)``. Hace que
    un id alucinado por el LLM (sufijo inventado, mayúsculas) no rompa el fetch.
    """
    ids = [e.id for e in entries]
    if dataset_id in ids:
        return dataset_id, None
    lower = {i.lower(): i for i in ids}
    key = dataset_id.lower().strip()
    if key in lower:
        return lower[key], f"dataset_id {dataset_id!r} interpretado como {lower[key]!r}."
    # Prefijo: el LLM suele AÑADIR un sufijo inventado ('_plazo', '_10y'). El id
    # real más largo que sea prefijo del pedido es el match correcto y seguro
    # (mucho más fiable que difflib, que no es semántico).
    prefix_hits = [real for low, real in lower.items() if key.startswith(low + "_")]
    if prefix_hits:
        real = max(prefix_hits, key=len)
        return real, f"dataset_id {dataset_id!r} no existe; se usó {real!r} (prefijo)."
    # difflib ESTRICTO como último recurso (typos), no para sufijos espurios.
    match = difflib.get_close_matches(key, list(lower), n=1, cutoff=_DATASET_FUZZY_CUTOFF)
    if match:
        real = lower[match[0]]
        return real, f"dataset_id {dataset_id!r} no existe; se usó el más cercano {real!r}."
    return None, None


def resolve_columns(
    dataset: ParquetDataset, columns: list[str],
) -> tuple[list[str], list[str]]:
    """Resuelve columnas pedidas a las reales del esquema (case-insensitive +
    fuzzy). Returns ``(columnas_resueltas, notas)``. Las que no tengan match
    razonable se devuelven tal cual (``_validate_columns`` luego levanta con la
    lista válida)."""
    valid = [c.name for c in dataset.columns]
    lower = {c.lower(): c for c in valid}
    resolved: list[str] = []
    notes: list[str] = []
    for col in columns:
        if col in valid:
            resolved.append(col)
            continue
        key = col.lower().strip()
        if key in lower:
            resolved.append(lower[key])
            notes.append(f"columna {col!r} → {lower[key]!r}")
            continue
        # Substring: 'usdclp' ⊃ 'clp'; 'monto' ⊂ 'monto transado'. Más fiable
        # que difflib para nombres compuestos. Se exige len ≥ 3 para evitar
        # matches triviales.
        subs = [
            real for low, real in lower.items()
            if len(low) >= 3 and len(key) >= 3 and (low in key or key in low)
        ]
        if subs:
            real = max(subs, key=len)
            resolved.append(real)
            notes.append(f"columna {col!r} no existe; se usó {real!r}")
            continue
        match = difflib.get_close_matches(key, list(lower), n=1, cutoff=_COLUMN_FUZZY_CUTOFF)
        if match:
            resolved.append(lower[match[0]])
            notes.append(f"columna {col!r} no existe; se usó {lower[match[0]]!r}")
        else:
            resolved.append(col)  # sin match → _validate_columns dará el error útil
    return resolved, notes

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_FALLBACK_NAMES = ("Fecha", "fecha", "date", "Date")
_DATE_TYPES = frozenset({"TIMESTAMP", "DATE", "TIMESTAMP_NS", "TIMESTAMP_S", "TIMESTAMP_MS"})


# ─────────────────────────────────────────────────────────────────────────────
# Resolución / validación
# ─────────────────────────────────────────────────────────────────────────────

def find_date_column(dataset: ParquetDataset) -> str | None:
    """Nombre de la columna fecha del dataset, o ``None`` si no tiene.

    Prefiere la primera columna de tipo TIMESTAMP/DATE; fallback a nombres
    comunes (``Fecha``, ``fecha``, ``date``, ``Date``). Muchos datasets son
    *snapshots* transversales (posición/composición por categoría, p. ej.
    ``posicion_rfl_afp``, ``attribution``, ``cambiario_afp``) y NO tienen
    columna temporal: para esos retorna ``None`` (se leen igual con
    ``execute_query``, sin orden ni filtro por fecha)."""
    for col in dataset.columns:
        if col.type.upper() in _DATE_TYPES:
            return col.name
    for name in _DATE_FALLBACK_NAMES:
        for col in dataset.columns:
            if col.name == name:
                return col.name
    return None


def date_column(dataset: ParquetDataset) -> str:
    """Como :func:`find_date_column` pero LANZA si el dataset no tiene fecha.

    Para código que exige una serie temporal (analytics de variación/spread/
    estadística/anomalía); ``execute_query`` usa ``find_date_column``."""
    col = find_date_column(dataset)
    if col is None:
        raise ValueError(
            f"Dataset {dataset.id!r}: no se identificó columna de fecha "
            f"(tipos {sorted(_DATE_TYPES)} o nombres {_DATE_FALLBACK_NAMES})."
        )
    return col


def _validate_columns(dataset: ParquetDataset, columns: list[str]) -> None:
    valid = {c.name for c in dataset.columns}
    bad = [c for c in columns if c not in valid]
    if bad:
        raise ValueError(
            f"Dataset {dataset.id!r}: columnas inválidas {bad}. "
            f"Disponibles: {sorted(valid)}"
        )


def _validate_date(value: str, field: str) -> None:
    if not _DATE_RE.match(value):
        raise ValueError(f"{field}={value!r} no es una fecha ISO YYYY-MM-DD válida.")


def _normalize_filters(
    dataset: ParquetDataset,
    filters: dict[str, Any],
) -> dict[str, list[str]]:
    """Valida y normaliza filtros a ``{col: [valores]}``. Solo igualdad/IN.

    Si la columna declara ``values`` (enum) en el catálogo, valida que cada
    valor pasado esté en esa lista (case-sensitive)."""
    cols_by_name = {c.name: c for c in dataset.columns}
    normalized: dict[str, list[str]] = {}
    for col_name, value in filters.items():
        if col_name not in cols_by_name:
            raise ValueError(
                f"Filter columna {col_name!r} no existe en {dataset.id!r}. "
                f"Columnas: {sorted(cols_by_name)}"
            )
        col = cols_by_name[col_name]
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ValueError(f"Filter para {col_name!r} con lista vacía.")
        values_str = [str(v) for v in values]
        if col.values:
            bad = [v for v in values_str if v not in col.values]
            if bad:
                raise ValueError(
                    f"Filter {col_name}={bad} fuera del enum del catálogo. "
                    f"Permitidos: {col.values}"
                )
        normalized[col_name] = values_str
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# SQL building (todo quoted/validado — el LLM nunca aporta SQL)
# ─────────────────────────────────────────────────────────────────────────────

def _quote_ident(name: str) -> str:
    """Comilla doble para identificadores (preserva case y permite espacios)."""
    return '"' + name.replace('"', '""') + '"'


def _quote_string(value: str) -> str:
    """Comilla simple para literales; escape de apóstrofe."""
    return "'" + value.replace("'", "''") + "'"


def build_fetch_sql(
    dataset: ParquetDataset,
    *,
    parquet_dir: Path,
    columns: list[str] | None = None,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[str, str | None, list[str], list[str]]:
    """Construye SQL safe para leer del parquet del dataset.

    Returns:
        ``(sql, date_col, select_cols, notes)``. ``date_col`` es ``None`` para
        datasets *snapshot* sin columna temporal (sin ``ORDER BY`` ni filtro de
        fecha). ``notes`` lista correcciones de columnas resueltas por similitud.
    """
    date_col = find_date_column(dataset)
    notes: list[str] = []

    # Columnas a seleccionar: si hay fecha, va primero; si no se pidieron
    # columnas, vuelca todas (útil para descubrimiento). Los snapshots sin
    # fecha se leen igual con las columnas pedidas. Las columnas se resuelven
    # por similitud antes de validar (tolera nombres alucinados por el LLM).
    if columns:
        columns, notes = resolve_columns(dataset, columns)
        _validate_columns(dataset, columns)
        if date_col:
            select_cols = [date_col] + [c for c in columns if c != date_col]
        else:
            select_cols = list(columns)
    else:
        select_cols = [c.name for c in dataset.columns]

    parquet_path = dataset.parquet_path(parquet_dir)
    select_clause = ", ".join(_quote_ident(c) for c in select_cols)
    sql_parts: list[str] = [
        f"SELECT {select_clause}",
        f"FROM read_parquet({_quote_string(str(parquet_path))})",
    ]

    if (fecha_inicio or fecha_fin) and date_col is None:
        raise ValueError(
            f"Dataset {dataset.id!r} no tiene columna de fecha (es un snapshot "
            "transversal); no admite filtro temporal fecha_inicio/fecha_fin."
        )

    where: list[str] = []
    if fecha_inicio:
        _validate_date(fecha_inicio, "fecha_inicio")
        where.append(f"{_quote_ident(date_col)} >= {_quote_string(fecha_inicio)}")
    if fecha_fin:
        _validate_date(fecha_fin, "fecha_fin")
        where.append(f"{_quote_ident(date_col)} <= {_quote_string(fecha_fin)}")
    if filters:
        norm = _normalize_filters(dataset, filters)
        for col_name, values in norm.items():
            if len(values) == 1:
                where.append(
                    f"{_quote_ident(col_name)} = {_quote_string(values[0])}"
                )
            else:
                in_list = ", ".join(_quote_string(v) for v in values)
                where.append(f"{_quote_ident(col_name)} IN ({in_list})")
    if where:
        sql_parts.append("WHERE " + " AND ".join(where))

    # ORDER BY solo si hay columna de fecha; los snapshots no tienen orden temporal.
    if date_col:
        sql_parts.append(f"ORDER BY {_quote_ident(date_col)} DESC")

    lim = min(int(limit), MAX_ROWS) if limit is not None else DEFAULT_LIMIT
    sql_parts.append(f"LIMIT {lim}")
    return " ".join(sql_parts), date_col, select_cols, notes


# ─────────────────────────────────────────────────────────────────────────────
# Carga + ejecución (async, para tools)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_rows_from_dataset(
    dataset_id: str,
    *,
    columns: list[str] | None = None,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[ParquetDataset, list[dict], str | None, list[str], list[str]]:
    """Resuelve el dataset, arma SQL safe y la ejecuta.

    Resuelve por similitud el ``dataset_id`` (y las columnas, dentro de
    ``build_fetch_sql``) cuando el LLM alucina un nombre que no existe.

    Returns:
        ``(dataset, rows, date_col, select_cols, notes)`` — ``notes`` lista las
        correcciones de identificadores aplicadas (para reportarlas al modelo).

    Raises:
        ValueError: dataset irresoluble o validación de columnas/filters/fechas.
        FileNotFoundError: el archivo parquet declarado no existe en disco.
    """
    entries = await asyncio.to_thread(load_parquet_catalog)
    notes: list[str] = []
    dataset = get_dataset(entries, dataset_id)
    if dataset is None:
        resolved, note = resolve_dataset_id(entries, dataset_id)
        if resolved is not None:
            dataset = get_dataset(entries, resolved)
            if note:
                log.info("plot/exec: %s", note)
                notes.append(note)
    if dataset is None:
        raise ValueError(
            f"dataset_id desconocido: {dataset_id!r} "
            f"({len(entries)} datasets en el catálogo — usa discover_query)."
        )

    parquet_dir = await asyncio.to_thread(get_parquet_dir)
    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_id!r}: archivo parquet no existe en {parquet_path}"
        )

    sql, date_col, select_cols, col_notes = build_fetch_sql(
        dataset,
        parquet_dir=parquet_dir,
        columns=columns,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        filters=filters,
        limit=limit,
    )
    notes.extend(col_notes)
    rows = await asyncio.to_thread(run_duckdb, sql)
    return dataset, rows, date_col, select_cols, notes
