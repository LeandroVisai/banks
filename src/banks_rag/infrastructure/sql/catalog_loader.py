"""Loader del catálogo SQL analítico (sql_catalog/catalog.yaml).

El catálogo contiene queries DuckDB sobre parquets del DW. Cada entrada
(``CatalogEntry``) describe una pregunta analítica que el agente puede
responder sin inventar SQL libre: el LLM descubre la query y pasa parámetros.

Funciones públicas:
    load_catalog(path)   → list[CatalogEntry]
    get_entry(entries, query_id) → CatalogEntry | None
    render_sql(entry, snapshots_dir, params) → str
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from banks_rag.config.paths import ROOT

# Solo estos tipos de parámetro pasan por una coacción segura en
# ``_resolve_param``. Cualquier otro tipo permitiría inyectar SQL arbitrario
# vía la interpolación de ``render_sql`` — se rechaza al cargar el catálogo.
_ALLOWED_PARAM_TYPES = frozenset({"date", "int"})

# Placeholder con forma de identificador: ``{nombre_param}``.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class ParamSpec:
    name: str
    type: str
    default: str
    description: str = ""


@dataclass
class CatalogEntry:
    query_id: str
    name: str
    description: str
    segment: str
    tags: list[str]
    parquet: str
    sql: str
    params: list[ParamSpec]
    unit: str
    frequency: str
    columns: list[str]

    def to_discovery_dict(self) -> dict:
        """Versión compacta para que el LLM decida si usar esta query."""
        return {
            "query_id": self.query_id,
            "name": self.name,
            "description": self.description.strip(),
            "segment": self.segment,
            "tags": self.tags,
            "unit": self.unit,
            "frequency": self.frequency,
            "params": [
                {"name": p.name, "type": p.type, "default": p.default, "description": p.description}
                for p in self.params
            ],
        }


def load_catalog(path: Path | str | None = None) -> list[CatalogEntry]:
    """Carga el catálogo desde YAML. Path por defecto: sql_catalog/catalog.yaml."""
    if path is None:
        path = ROOT / "sql_catalog" / "catalog.yaml"
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return [_parse_entry(q) for q in data.get("queries", [])]


def get_entry(entries: list[CatalogEntry], query_id: str) -> CatalogEntry | None:
    """Busca una entrada por query_id exacto."""
    for e in entries:
        if e.query_id == query_id:
            return e
    return None


def render_sql(
    entry: CatalogEntry,
    snapshots_dir: Path | str,
    params: dict[str, Any] | None = None,
    *,
    parquet_dir: Path | str | None = None,
) -> str:
    """Rellena los placeholders {param} del SQL template.

    Maneja:
      - ``{snapshots_dir}`` → ruta absoluta al directorio snapshots (legacy).
      - ``{parquet_dir}``   → ruta absoluta al directorio parquet/ (preferido).
      - ``{fecha_inicio}`` / ``{fecha_fin}`` → fechas ISO.
        Defaults relativos como ``"-365d"`` o ``"-90d"`` se resuelven vs hoy.
        El string ``"hoy"`` se resuelve como ``date.today()``.
      - ``{limit}`` → int, default del ParamSpec o 200.
    """
    params = params or {}
    resolved: dict[str, str] = {"snapshots_dir": str(snapshots_dir)}
    if parquet_dir is not None:
        resolved["parquet_dir"] = str(parquet_dir)

    for spec in entry.params:
        raw = params.get(spec.name, spec.default)
        resolved[spec.name] = _resolve_param(spec, raw)

    sql = entry.sql
    for key, val in resolved.items():
        sql = sql.replace("{" + key + "}", val)

    # Defensa en profundidad: ningún placeholder debe quedar sin resolver.
    # Uno sin ``ParamSpec`` indicaría un catálogo mal formado y podría
    # filtrar texto no validado al SQL.
    leftover = sorted(set(_PLACEHOLDER_RE.findall(sql)))
    if leftover:
        raise ValueError(
            f"Query '{entry.query_id}': placeholders sin resolver {leftover}. "
            "Cada {placeholder} del SQL debe tener un ParamSpec correspondiente."
        )
    return sql


# ── Internos ──────────────────────────────────────────────────────────────────

def _parse_entry(raw: dict) -> CatalogEntry:
    params = []
    for p in raw.get("params", []):
        ptype = p.get("type", "")
        if ptype not in _ALLOWED_PARAM_TYPES:
            raise ValueError(
                f"Query '{raw.get('query_id')}': el parámetro '{p.get('name')}' "
                f"declara type={ptype!r}. Tipos permitidos: {sorted(_ALLOWED_PARAM_TYPES)}. "
                "Un tipo no whitelisted permitiría SQL injection vía render_sql."
            )
        params.append(
            ParamSpec(
                name=p["name"],
                type=ptype,
                default=str(p.get("default", "")),
                description=p.get("description", ""),
            )
        )
    return CatalogEntry(
        query_id=raw["query_id"],
        name=raw["name"],
        description=raw.get("description", ""),
        segment=raw.get("segment", ""),
        tags=raw.get("tags", []),
        parquet=raw.get("parquet", ""),
        sql=raw["sql"],
        params=params,
        unit=raw.get("unit", ""),
        frequency=raw.get("frequency", ""),
        columns=raw.get("columns", []),
    )


def _resolve_param(spec: ParamSpec, raw: str) -> str:
    """Convierte un valor de parámetro a string listo para SQL."""
    raw = str(raw).strip()

    if spec.type == "date":
        return _resolve_date(raw)

    if spec.type == "int":
        try:
            return str(int(raw))
        except ValueError:
            raise ValueError(
                f"Parámetro '{spec.name}': se esperaba un entero, se recibió {raw!r}."
            ) from None

    # _parse_entry garantiza type ∈ _ALLOWED_PARAM_TYPES; este punto es
    # inalcanzable salvo regresión — fallar es más seguro que interpolar texto crudo.
    raise ValueError(f"Tipo de parámetro no soportado: {spec.type!r}")


def _resolve_date(raw: str) -> str:
    """Convierte expresiones de fecha a ISO YYYY-MM-DD.

    Soporta:
      - ``"hoy"``        → date.today()
      - ``"-365d"``      → today - 365 days
      - ``"-12m"``       → today - 12 months (aprox 30d/month)
      - ``"YYYY-MM-DD"`` → pasado tal cual
    """
    today = date.today()
    lower = raw.lower()

    if lower == "hoy" or lower == "today":
        return today.isoformat()

    if lower.startswith("-") and lower.endswith("d"):
        try:
            days = int(lower[1:-1])
            return (today - timedelta(days=days)).isoformat()
        except ValueError:
            pass

    if lower.startswith("-") and lower.endswith("m"):
        try:
            months = int(lower[1:-1])
            return (today - timedelta(days=months * 30)).isoformat()
        except ValueError:
            pass

    # Validar que sea una fecha ISO
    try:
        date.fromisoformat(raw)
        return raw
    except ValueError:
        raise ValueError(
            f"Fecha inválida: {raw!r}. Use formato ISO YYYY-MM-DD, "
            "'hoy', o un offset relativo como '-365d' / '-12m'."
        ) from None
