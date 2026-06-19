"""Cómputo de hechos descriptivos de un parquet — en Python, SIN tools ni LLM.

Reemplaza el mini-loop de tool-calling del informe (que se caía en el
llama-server). Aquí Python lee el parquet REAL con DuckDB, detecta los roles de
las columnas desde el esquema real (no del catálogo, así es robusto al drift) y
calcula los números relevantes reutilizando las funciones puras de
``series_analytics``. El resultado es un dict estructurado que el informe pasa
al LLM solo para redactar prosa — el LLM no hace aritmética ni elige columnas.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

from banks_rag.infrastructure.sql import series_analytics as sa
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset

log = logging.getLogger(__name__)

# Tipos DuckDB que consideramos fecha y numéricos.
_DATE_TYPES = ("TIMESTAMP", "DATE", "TIMESTAMP_NS", "TIMESTAMP_S", "TIMESTAMP_MS")
_NUMERIC_PREFIXES = ("DOUBLE", "FLOAT", "DECIMAL", "BIGINT", "INTEGER", "HUGEINT", "SMALLINT", "TINYINT", "REAL")
# Una columna VARCHAR es "categórica" si su cardinalidad no supera esto.
_MAX_CATEGORY_CARDINALITY = 60
# Tope de categorías/columnas que se detallan en los facts (las de mayor nivel).
_TOP_CATEGORIES = 8
# Tope de series por gráfico (legibilidad de un multi-línea) y de puntos por
# serie (downsample): el gráfico es para verificar la prosa de un vistazo, no
# un tablero interactivo.
_MAX_PLOT_SERIES = 6
_MAX_PLOT_POINTS = 200
# Tope de categorías en un gráfico de composición (snapshot / pie).
_MAX_PLOT_CATEGORIES = 12


@dataclass
class ParquetRoles:
    date_col: str | None
    category_cols: list[str]
    value_cols: list[str]


def _quote_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_columns(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> list[tuple[str, str]]:
    src = f"read_parquet({_quote_str(parquet_path.as_posix())})"
    return [(r[0], r[1]) for r in con.sql(f"DESCRIBE SELECT * FROM {src}").fetchall()]


def detect_roles(parquet_path: Path, con: duckdb.DuckDBPyConnection | None = None) -> ParquetRoles:
    """Detecta (date_col, category_cols, value_cols) desde el esquema REAL.

    - date_col: primera columna de tipo fecha; si no hay, una llamada ``fecha``/
      ``date`` de cualquier tipo. ``None`` → snapshot transversal.
    - value_cols: columnas numéricas (excluida la fecha).
    - category_cols: columnas VARCHAR de baja cardinalidad (excluida la fecha).
    """
    own_con = con is None
    con = con or duckdb.connect()
    try:
        cols = _read_columns(con, parquet_path)
        src = f"read_parquet({_quote_str(parquet_path.as_posix())})"

        date_col: str | None = None
        for name, typ in cols:
            if any(typ.upper().startswith(t) for t in _DATE_TYPES):
                date_col = name
                break
        if date_col is None:
            for name, _typ in cols:
                if name.lower() in ("fecha", "date"):
                    date_col = name
                    break

        value_cols: list[str] = []
        varchar_cols: list[str] = []
        for name, typ in cols:
            if name == date_col:
                continue
            up = typ.upper()
            if any(up.startswith(p) for p in _NUMERIC_PREFIXES):
                value_cols.append(name)
            elif up.startswith("VARCHAR"):
                varchar_cols.append(name)

        category_cols: list[str] = []
        for name in varchar_cols:
            n = con.sql(
                f"SELECT count(DISTINCT {_quote_ident(name)}) FROM {src}"
            ).fetchone()[0]
            if n and n <= _MAX_CATEGORY_CARDINALITY:
                category_cols.append(name)

        return ParquetRoles(date_col=date_col, category_cols=category_cols, value_cols=value_cols)
    finally:
        if own_con:
            con.close()


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _read_rows(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    *,
    date_col: str | None,
    columns: list[str],
) -> list[dict]:
    """Lee filas como list[dict]; castea la fecha a DATE → ISO 'YYYY-MM-DD'."""
    src = f"read_parquet({_quote_str(parquet_path.as_posix())})"
    select_parts = []
    for c in columns:
        if c == date_col:
            select_parts.append(f"CAST({_quote_ident(c)} AS DATE) AS {_quote_ident(c)}")
        else:
            select_parts.append(_quote_ident(c))
    rel = con.sql(f"SELECT {', '.join(select_parts)} FROM {src}")
    names = rel.columns
    out: list[dict] = []
    for row in rel.fetchall():
        d = dict(zip(names, row, strict=False))
        if date_col is not None and d.get(date_col) is not None:
            d[date_col] = str(d[date_col])  # date → 'YYYY-MM-DD'
        out.append(d)
    return out


def _scale_rows(rows: list[dict], value_cols: list[str], scale: float) -> None:
    """Multiplica IN-PLACE las columnas de valor por ``scale`` (corrección de
    unidad del dataset). No-op si ``scale == 1.0``. Así los hechos del texto
    quedan en la misma escala que el gráfico del informe."""
    if scale == 1.0:
        return
    for row in rows:
        for col in value_cols:
            v = row.get(col)
            if v is None:
                continue
            # float() cubre int/float/Decimal (DuckDB devuelve Decimal para literales).
            with contextlib.suppress(TypeError, ValueError):
                row[col] = float(v) * scale


def _windows(last_date: str, windows: list[tuple[str, int]]) -> list[dict]:
    """``[(label, days)]`` → ``[{label, start, end}]`` anclados a ``last_date``."""
    anchor = date.fromisoformat(last_date)
    out = []
    for label, days in windows:
        out.append({
            "label": label,
            "start": (anchor - timedelta(days=days)).isoformat(),
            "end": last_date,
        })
    return out


def _slice(series: list[sa.Point], start: str, end: str) -> list[sa.Point]:
    return [p for p in series if start <= p[0] <= end]


def _window_variations(series: list[sa.Point], windows: list[dict]) -> list[dict]:
    out = []
    for w in windows:
        v = sa.variation(_slice(series, w["start"], w["end"]))
        out.append({"label": w["label"], "desde": w["start"], "hasta": w["end"], "variacion": v})
    return out


def _days_between(d0: str, d1: str) -> int:
    try:
        return max(1, (date.fromisoformat(d1[:10]) - date.fromisoformat(d0[:10])).days)
    except ValueError:
        return 1


def _trend_signal(window_variations: list[dict]) -> dict | None:
    """Aceleración / desaceleración / reversión comparando ventana corta vs larga.

    Usa la tasa de cambio por día (normalizada por el span real de observaciones)
    de la ventana más corta frente a la más larga. ``None`` si no hay al menos
    dos ventanas con variación medible.
    """
    usable = [w for w in window_variations if w.get("variacion")]
    if len(usable) < 2:
        return None
    short, long = usable[0], usable[-1]
    if short["label"] == long["label"]:
        return None
    vs, vl = short["variacion"], long["variacion"]
    rate_s = vs["cambio_absoluto"] / _days_between(vs["fecha_inicio_obs"], vs["fecha_fin_obs"])
    rate_l = vl["cambio_absoluto"] / _days_between(vl["fecha_inicio_obs"], vl["fecha_fin_obs"])
    sign_s = (rate_s > 0) - (rate_s < 0)
    sign_l = (rate_l > 0) - (rate_l < 0)

    if sign_s == 0:
        clasif = "estable"
    elif sign_l != 0 and sign_s != sign_l:
        clasif = "reversión"
    elif abs(rate_s) > abs(rate_l) * 1.2:
        clasif = "aceleración"
    elif abs(rate_s) < abs(rate_l) * 0.8:
        clasif = "desaceleración"
    else:
        clasif = "en línea con la tendencia"
    return {
        "clasificacion": clasif,
        "ventana_corta": short["label"],
        "ventana_larga": long["label"],
    }


# Palabras del id/nombre que marcan una serie de FLUJOS (entradas/salidas), donde
# el SIGNO del flujo (no su variación) define la dirección. Para estas series una
# variación negativa con flujo aún positivo es MENOR ENTRADA, no una salida.
_FLOW_KEYWORDS = (
    "flujo", "flow", "movimiento", "traspaso", "aporte", "rescate",
    "suscrip", "vencim", "captacion", "captación",
)


def _is_flow(dataset: ParquetDataset) -> bool:
    """``True`` si el dataset es de flujos (su valor es una entrada/salida, no un
    stock): el signo del valor indica dirección. Heurística por id + nombre."""
    blob = f"{dataset.id} {dataset.name}".lower()
    return any(k in blob for k in _FLOW_KEYWORDS)


def compute_facts(
    dataset: ParquetDataset,
    parquet_dir: Path,
    windows: list[tuple[str, int]],
) -> dict[str, Any] | None:
    """Calcula los hechos descriptivos relevantes de un dataset.

    ``windows`` es ``[(etiqueta, días)]`` (p.ej. ``[("última semana", 7), …]``).
    Devuelve ``None`` si el parquet no existe (→ el caller marca ``no_data``).
    El dict resultante incluye ``shape`` y un bloque de hechos por forma de dato.
    """
    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None

    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)
        n_rows = con.sql(
            f"SELECT count(*) FROM read_parquet({_quote_str(parquet_path.as_posix())})"
        ).fetchone()[0]

        base: dict[str, Any] = {
            "dataset_id": dataset.id,
            "unit": dataset.unit,
            "n_rows": int(n_rows),
            "date_col": roles.date_col,
            "value_cols": roles.value_cols,
            "category_cols": roles.category_cols,
            "is_flow": _is_flow(dataset),
            "last_date": None,
        }

        scale = dataset.value_scale

        # ── Snapshot transversal (sin columna de fecha) ──────────────────────
        if roles.date_col is None:
            if roles.category_cols and roles.value_cols:
                rows = _read_rows(
                    con, parquet_path, date_col=None,
                    columns=[roles.category_cols[0], roles.value_cols[0]],
                )
                _scale_rows(rows, roles.value_cols, scale)
                comp = _snapshot_composition(rows, roles.category_cols[0], roles.value_cols[0])
                base.update(shape="snapshot", composition=comp)
            else:
                base.update(shape="snapshot", composition=None)
            return base

        last_date = con.sql(
            f"SELECT CAST(max({_quote_ident(roles.date_col)}) AS DATE) "
            f"FROM read_parquet({_quote_str(parquet_path.as_posix())})"
        ).fetchone()[0]
        if last_date is None:
            base.update(shape="empty")
            return base
        last_date = str(last_date)
        wins = _windows(last_date, windows)
        base.update(last_date=last_date, windows=[{"label": w["label"], "desde": w["start"], "hasta": w["end"]} for w in wins])

        # ── Serie temporal con categoría ─────────────────────────────────────
        if roles.category_cols and roles.value_cols:
            cat_col, val_col = roles.category_cols[0], roles.value_cols[0]
            rows = _read_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, cat_col, val_col])
            _scale_rows(rows, roles.value_cols, scale)
            base.update(
                shape="timeseries_categorical",
                category_col=cat_col, value_col=val_col,
                **_categorical_facts(rows, roles.date_col, cat_col, val_col, wins, last_date),
            )
            return base

        # ── Serie temporal "ancha" (varias columnas de valor, sin categoría) ──
        if len(roles.value_cols) > 1:
            rows = _read_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, *roles.value_cols])
            _scale_rows(rows, roles.value_cols, scale)
            base.update(
                shape="timeseries_wide",
                **_wide_facts(rows, roles.date_col, roles.value_cols, wins, last_date),
            )
            return base

        # ── Serie temporal simple (una columna de valor) ─────────────────────
        if roles.value_cols:
            val_col = roles.value_cols[0]
            rows = _read_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, val_col])
            _scale_rows(rows, roles.value_cols, scale)
            series = sa.clean_series(rows, roles.date_col, val_col)
            wv = _window_variations(series, wins)
            base.update(
                shape="timeseries_single", value_col=val_col,
                windows_variation=wv,
                tendencia=_trend_signal(wv),
                estadisticas=sa.descriptive_stats(series),
                anomalia=sa.anomaly_check(series),
            )
            return base

        base.update(shape="unknown")
        return base
    finally:
        con.close()


def _snapshot_composition(rows: list[dict], cat_col: str, val_col: str) -> dict | None:
    sums: dict[str, float] = {}
    for row in rows:
        cat, raw = row.get(cat_col), row.get(val_col)
        if cat is None or raw is None:
            continue
        try:
            sums[str(cat)] = sums.get(str(cat), 0.0) + float(raw)
        except (TypeError, ValueError):
            continue
    if not sums:
        return None
    total = sum(sums.values())
    breakdown = [
        {"categoria": c, "valor": round(v, 6), "share_pct": round(100.0 * v / total, 2) if total else None}
        for c, v in sorted(sums.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {"total": round(total, 6), "n_categorias": len(breakdown), "breakdown": breakdown}


def _categorical_facts(
    rows: list[dict], date_col: str, cat_col: str, val_col: str,
    wins: list[dict], last_date: str,
) -> dict:
    """Total agregado por ventana + composición a la fecha de corte + top categorías."""
    # Total por fecha (suma de todas las categorías) → variación por ventana.
    by_date: dict[str, float] = {}
    for row in rows:
        f, raw = row.get(date_col), row.get(val_col)
        if f is None or raw is None:
            continue
        try:
            by_date[str(f)] = by_date.get(str(f), 0.0) + float(raw)
        except (TypeError, ValueError):
            continue
    total_series = sorted(by_date.items(), key=lambda kv: kv[0])
    composition = sa.composition(rows, date_col, cat_col, val_col, fecha=last_date)

    # Serie por categoría AGREGANDO por fecha: si el dataset tiene más de una
    # columna categórica (p.ej. Tipo_instrumento x Tipo_fondo en allocation),
    # filtrar por una sola deja varias filas por fecha; hay que sumarlas, si no
    # la variación compararía dos valores del mismo día (basura).
    agg: dict[str, dict[str, float]] = {}
    for row in rows:
        cat, f, raw = row.get(cat_col), row.get(date_col), row.get(val_col)
        if cat is None or f is None or raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        per_date = agg.setdefault(str(cat), {})
        per_date[str(f)] = per_date.get(str(f), 0.0) + val
    series_by_cat: dict[str, list[sa.Point]] = {
        cat: sorted(per_date.items(), key=lambda kv: kv[0]) for cat, per_date in agg.items()
    }
    cut_vals = {cat: s[-1][1] for cat, s in series_by_cat.items() if s}
    top = sorted(cut_vals, key=lambda c: abs(cut_vals[c]), reverse=True)[:_TOP_CATEGORIES]

    by_category = []
    for cat in top:
        series = series_by_cat[cat]
        by_category.append({
            "categoria": cat,
            "ultimo_valor": round(series[-1][1], 6),
            "ultima_fecha": series[-1][0],
            "ventanas": _window_variations(series, wins),
            "anomalia": sa.anomaly_check(series),
        })

    total_ventanas = _window_variations(total_series, wins)
    return {
        "total_ventanas": total_ventanas,
        "tendencia": _trend_signal(total_ventanas),
        "contribuciones": _contributions(series_by_cat, wins),
        "composicion_corte": composition,
        "por_categoria": by_category,
    }


def _contributions(series_by_cat: dict[str, list[sa.Point]], wins: list[dict]) -> list[dict]:
    """Qué categorías explican el movimiento del total, por ventana.

    Para cada ventana, ordena las categorías por |cambio absoluto| y devuelve las
    de mayor aporte (los "drivers" del movimiento agregado)."""
    out = []
    for w in wins:
        cambios = []
        for cat, series in series_by_cat.items():
            v = sa.variation(_slice(series, w["start"], w["end"]))
            if v and v["cambio_absoluto"]:
                cambios.append({
                    "categoria": cat,
                    "cambio_absoluto": v["cambio_absoluto"],
                    "variacion_pct": v["variacion_pct"],
                    "nivel_fin": v.get("valor_fin"),  # nivel al cierre (signo = dirección si es flujo)
                })
        cambios.sort(key=lambda c: abs(c["cambio_absoluto"]), reverse=True)
        if cambios:
            out.append({"label": w["label"], "drivers": cambios[:3]})
    return out


def _wide_facts(
    rows: list[dict], date_col: str, value_cols: list[str],
    wins: list[dict], last_date: str,
) -> dict:
    """Variación por columna + composición wide a la fecha de corte (top columnas)."""
    series_by_col = {c: sa.clean_series(rows, date_col, c) for c in value_cols}
    last_vals = {c: (s[-1][1] if s else 0.0) for c, s in series_by_col.items()}
    top = sorted(value_cols, key=lambda c: abs(last_vals[c]), reverse=True)[:_TOP_CATEGORIES]
    por_columna = [
        {
            "columna": c,
            "ultimo_valor": round(series_by_col[c][-1][1], 6) if series_by_col[c] else None,
            "ultima_fecha": series_by_col[c][-1][0] if series_by_col[c] else None,
            "ventanas": _window_variations(series_by_col[c], wins),
        }
        for c in top
    ]
    return {
        "por_columna": por_columna,
        "composicion_corte": sa.composition_wide(rows, date_col, value_cols, fecha=last_date),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Series para graficar — leídas del parquet REAL, no de los agregados del LLM
# ─────────────────────────────────────────────────────────────────────────────
#
# El gráfico del informe es una verificación INDEPENDIENTE de la prosa del LLM:
# lee la serie directo del parquet (la fuente de verdad), no re-dibuja los
# agregados que vio el modelo. La consistencia con los números del párrafo se
# garantiza por CÓDIGO COMPARTIDO — ``compute_series`` reusa ``detect_roles``,
# así elige la misma columna de fecha/valor/categoría que ``compute_facts`` —
# no por pasar datos de un proceso a otro. Si la curva discrepa del párrafo es
# porque hay un problema real de datos (p.ej. fecha que no castea a DATE), y eso
# se quiere ver, no esconder.


@dataclass
class PlotSeries:
    """Una serie a dibujar: una línea (por categoría/columna) o un grupo de
    barras (composición). ``points`` es ``[(x, y)]`` ya ordenado: ``x`` es fecha
    ISO en series temporales y nombre de categoría en snapshots."""

    label: str
    points: list[tuple[str, float]]


@dataclass
class PlotData:
    """Datos listos para el renderer SVG. ``kind='timeseries'`` → multi-línea
    (eje X temporal); ``kind='snapshot'`` → barras de composición (eje X
    categórico). ``family`` es la familia renderizable del catálogo
    (``chart_family``); ``table`` => el caller cae a una mini-tabla HTML."""

    dataset_id: str
    family: str
    kind: str  # "timeseries" | "snapshot" | "grouped"
    unit: str
    series: list[PlotSeries]
    # Etiquetas de series que se dibujan SUPERPUESTAS (no apiladas): una línea
    # sobre el área/barras en series temporales, un punto por categoría en barras.
    # Réplica del "Neto"/"Total" de los informes BCCh sobre apilados divergentes.
    overlay: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not self.series or all(len(s.points) < 1 for s in self.series)


@dataclass
class HtmlTable:
    """Resultado alternativo a ``PlotData`` para transforms que producen tablas
    HTML con color condicional (heatmaps DCV). El HTML se incrusta directamente
    en el bloque del informe sin pasar por ``render_plot_svg``."""

    html: str
    dataset_id: str = ""


def _downsample(points: list[tuple[str, float]], max_points: int = _MAX_PLOT_POINTS) -> list[tuple[str, float]]:
    """Reduce a lo más ``max_points`` por muestreo uniforme, conservando SIEMPRE
    el primer y el último punto (el dato más reciente importa)."""
    n = len(points)
    if n <= max_points:
        return points
    step = (n - 1) / (max_points - 1)
    idx = sorted({round(i * step) for i in range(max_points)} | {0, n - 1})
    return [points[i] for i in idx if 0 <= i < n]


def _read_series_rows(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    *,
    date_col: str,
    columns: list[str],
) -> list[dict]:
    """Como ``_read_rows`` pero con ``TRY_CAST`` de la fecha a DATE: una fecha
    que no castea (VARCHAR en formato no-ISO) queda en NULL y la fila se descarta
    en ``clean_series`` — la serie se ve recortada, señal visible del problema."""
    src = f"read_parquet({_quote_str(parquet_path.as_posix())})"
    select_parts = []
    for c in columns:
        if c == date_col:
            select_parts.append(f"TRY_CAST({_quote_ident(c)} AS DATE) AS {_quote_ident(c)}")
        else:
            select_parts.append(_quote_ident(c))
    rel = con.sql(f"SELECT {', '.join(select_parts)} FROM {src}")
    names = rel.columns
    out: list[dict] = []
    for row in rel.fetchall():
        d = dict(zip(names, row, strict=False))
        if d.get(date_col) is not None:
            d[date_col] = str(d[date_col])
        out.append(d)
    return out


def _aggregate_by_category(
    rows: list[dict], date_col: str, cat_col: str, val_col: str,
) -> dict[str, list[sa.Point]]:
    """Serie por categoría sumando por fecha (mismo criterio que _categorical_facts:
    si hay más de una categórica, varias filas por fecha se agregan)."""
    agg: dict[str, dict[str, float]] = {}
    for row in rows:
        cat, f, raw = row.get(cat_col), row.get(date_col), row.get(val_col)
        if cat is None or f is None or raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        per_date = agg.setdefault(str(cat), {})
        per_date[str(f)] = per_date.get(str(f), 0.0) + val
    return {
        cat: sorted(per_date.items(), key=lambda kv: kv[0])
        for cat, per_date in agg.items()
    }


def compute_series(
    dataset: ParquetDataset,
    parquet_dir: Path,
    *,
    category_filter: list[str] | None = None,
) -> PlotData | None:
    """Lee del parquet REAL la(s) serie(s) a graficar, reusando ``detect_roles``.

    Devuelve ``None`` si el parquet no existe (→ sin gráfico). La familia
    renderizable sale del ``chart_type`` del catálogo vía ``chart_family``; el
    renderer decide la marca concreta y degrada barras-con-muchos-puntos a línea.

    ``category_filter`` (opcional): en datos categóricos o snapshot, restringe a
    esas categorías y RESPETA su orden (sin el tope de top-N). Lo usa el transform
    ``filter_fund`` del informe curado (p.ej. Duración solo T1/T2) sin duplicar la
    detección de roles ni la agregación.
    """
    from banks_rag.domain.agent.chart_types import chart_family

    parquet_path = dataset.parquet_path(parquet_dir)
    if not parquet_path.exists():
        return None

    family = chart_family(dataset.chart_type)
    con = duckdb.connect()
    try:
        roles = detect_roles(parquet_path, con)

        # ── Snapshot transversal (sin fecha) → composición categórica ────────
        if roles.date_col is None:
            if not (roles.category_cols and roles.value_cols):
                return None
            cat_col, val_col = roles.category_cols[0], roles.value_cols[0]
            rows = _read_rows(con, parquet_path, date_col=None, columns=[cat_col, val_col])
            sums: dict[str, float] = {}
            for row in rows:
                c, raw = row.get(cat_col), row.get(val_col)
                if c is None or raw is None:
                    continue
                try:
                    sums[str(c)] = sums.get(str(c), 0.0) + float(raw)
                except (TypeError, ValueError):
                    continue
            if not sums:
                return None
            if category_filter:
                keep = set(category_filter)
                ordered = [(c, v) for c, v in sums.items() if c in keep]
                ordered.sort(key=lambda kv: abs(kv[1]), reverse=True)
            else:
                ordered = sorted(sums.items(), key=lambda kv: abs(kv[1]), reverse=True)[:_MAX_PLOT_CATEGORIES]
            return PlotData(
                dataset_id=dataset.id, family=family, kind="snapshot", unit=dataset.unit,
                series=[PlotSeries(label=cat_col, points=[(c, round(v, 6)) for c, v in ordered])],
            )

        # ── Serie temporal con categoría → multi-línea (top categorías) ──────
        if roles.category_cols and roles.value_cols:
            cat_col, val_col = roles.category_cols[0], roles.value_cols[0]
            rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, cat_col, val_col])
            by_cat = _aggregate_by_category(rows, roles.date_col, cat_col, val_col)
            if category_filter:
                # Respeta el orden pedido; ignora categorías ausentes en el parquet.
                cats = [c for c in category_filter if c in by_cat]
            else:
                cats = sorted(by_cat, key=lambda c: abs(by_cat[c][-1][1]) if by_cat[c] else 0.0, reverse=True)[:_MAX_PLOT_SERIES]
            series = [PlotSeries(label=c, points=_downsample(by_cat[c])) for c in cats if by_cat[c]]
            return _nonempty(PlotData(dataset.id, family, "timeseries", dataset.unit, series))

        # ── Serie temporal "ancha" (varias columnas de valor) → multi-línea ──
        if len(roles.value_cols) > 1:
            rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, *roles.value_cols])
            by_col = {c: sa.clean_series(rows, roles.date_col, c) for c in roles.value_cols}
            top = sorted(roles.value_cols, key=lambda c: abs(by_col[c][-1][1]) if by_col[c] else 0.0, reverse=True)[:_MAX_PLOT_SERIES]
            series = [PlotSeries(label=c, points=_downsample(by_col[c])) for c in top if by_col[c]]
            return _nonempty(PlotData(dataset.id, family, "timeseries", dataset.unit, series))

        # ── Serie temporal simple ────────────────────────────────────────────
        if roles.value_cols:
            val_col = roles.value_cols[0]
            rows = _read_series_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, val_col])
            pts = _downsample(sa.clean_series(rows, roles.date_col, val_col))
            return _nonempty(PlotData(dataset.id, family, "timeseries", dataset.unit, [PlotSeries(label=val_col, points=pts)]))

        return None
    finally:
        con.close()


def _nonempty(plot: PlotData) -> PlotData | None:
    return None if plot.is_empty() else plot


# ─────────────────────────────────────────────────────────────────────────────
# Render de los hechos a texto compacto para el LLM
# ─────────────────────────────────────────────────────────────────────────────


def _num(x: float | None) -> str:
    if x is None:
        return "s/d"
    return f"{x:,.2f}"


def _pct(x: float | None) -> str:
    return "s/d" if x is None else f"{x:+.2f}%"


def _flow_tag(nivel: float | None, cambio: float | None = None) -> str:
    """Etiqueta determinista de DIRECCIÓN para series de flujos, según el SIGNO del
    nivel (no de la variación). Evita que una variación negativa de un flujo aún
    positivo se lea como 'salida': eso es MENOR ENTRADA."""
    if nivel is None:
        return ""
    if nivel < 0:
        return " [SALIDA: flujo negativo]"
    if cambio is not None and cambio < 0:
        return " [MENOR ENTRADA: flujo sigue positivo, solo desacelera — NO es salida]"
    return " [ENTRADA: flujo positivo]"


def _variation_line(v: dict) -> str:
    """Una ventana → texto. ``v`` es ``{label, desde, hasta, variacion}``."""
    var = v.get("variacion")
    if not var:
        return f"{v['label']} ({v['desde']}→{v['hasta']}): sin observaciones suficientes en la ventana"
    return (
        f"{v['label']} ({var['fecha_inicio_obs']}→{var['fecha_fin_obs']}): "
        f"de {_num(var['valor_inicio'])} a {_num(var['valor_fin'])}, "
        f"cambio {_num(var['cambio_absoluto'])} ({_pct(var['variacion_pct'])}); "
        f"rango [{_num(var['minimo'])}, {_num(var['maximo'])}], n={var['n_observaciones']}"
    )


def _composition_lines(comp: dict, unit: str) -> list[str]:
    out = [f"Composición a {comp.get('fecha', 'corte')}: total {_num(comp['total'])} {unit}".rstrip()]
    for item in comp["breakdown"]:
        out.append(f"  - {item['categoria']}: {_num(item['valor'])} ({_pct_share(item['share_pct'])})")
    return out


def _pct_share(x: float | None) -> str:
    return "s/d" if x is None else f"{x:.2f}%"


def _trend_lines(trend: dict | None) -> list[str]:
    if not trend:
        return []
    return [
        f"Tendencia: {trend['clasificacion']} "
        f"({trend['ventana_corta']} vs {trend['ventana_larga']})."
    ]


def _contribution_lines(contribuciones: list[dict] | None, *, is_flow: bool = False) -> list[str]:
    if not contribuciones:
        return []
    out = ["Drivers del movimiento del total:"]
    for c in contribuciones:
        partes = ", ".join(
            f"{d['categoria']} (cambio {_num(d['cambio_absoluto'])}, {_pct(d['variacion_pct'])}"
            + (f", flujo final {_num(d.get('nivel_fin'))}{_flow_tag(d.get('nivel_fin'), d.get('cambio_absoluto'))}" if is_flow else "")
            + ")"
            for d in c["drivers"]
        )
        out.append(f"  - {c['label']}: {partes}")
    return out


def facts_to_text(facts: dict) -> str:
    """Renderiza el dict de ``compute_facts`` a un bloque de texto para el LLM."""
    unit = facts.get("unit") or ""
    shape = facts.get("shape")
    is_flow = bool(facts.get("is_flow"))
    lines: list[str] = [f"Filas: {facts.get('n_rows')}; unidad: {unit or 's/d'}."]

    if shape == "snapshot":
        comp = facts.get("composition")
        if comp:
            lines.append("Corte transversal (sin serie temporal).")
            lines += _composition_lines(comp, unit)
        return "\n".join(lines)

    lines.append(f"Última fecha con datos: {facts.get('last_date')}.")
    if is_flow:
        lines.append(
            "NOTA — serie de FLUJOS: el SIGNO del flujo indica dirección (positivo = "
            "ENTRADA/aportes, negativo = SALIDA/rescates). Una variación negativa con "
            "el flujo aún positivo es MENOR ENTRADA (desaceleración), NO una salida; "
            "solo hay salida cuando el flujo en sí es negativo."
        )

    if shape == "timeseries_categorical":
        lines.append(f"Categoría: {facts.get('category_col')}; métrica: {facts.get('value_col')}.")
        lines.append("Total agregado (suma de categorías):")
        for v in facts.get("total_ventanas", []):
            lines.append(f"  - {_variation_line(v)}")
        lines += _trend_lines(facts.get("tendencia"))
        lines += _contribution_lines(facts.get("contribuciones"), is_flow=is_flow)
        comp = facts.get("composicion_corte")
        if comp:
            lines += _composition_lines(comp, unit)
        lines.append("Por categoría (top por nivel):")
        for cat in facts.get("por_categoria", []):
            tag = _flow_tag(cat["ultimo_valor"]) if is_flow else ""
            lines.append(f"  · {cat['categoria']}: último {_num(cat['ultimo_valor'])} ({cat['ultima_fecha']}){tag}")
            for v in cat["ventanas"]:
                lines.append(f"      {_variation_line(v)}")
        return "\n".join(lines)

    if shape == "timeseries_wide":
        lines.append("Por columna (top por nivel):")
        for col in facts.get("por_columna", []):
            tag = _flow_tag(col["ultimo_valor"]) if is_flow else ""
            lines.append(f"  · {col['columna']}: último {_num(col['ultimo_valor'])} ({col['ultima_fecha']}){tag}")
            for v in col["ventanas"]:
                lines.append(f"      {_variation_line(v)}")
        comp = facts.get("composicion_corte")
        if comp:
            lines += _composition_lines(comp, unit)
        return "\n".join(lines)

    if shape == "timeseries_single":
        lines.append(f"Métrica: {facts.get('value_col')}.")
        for v in facts.get("windows_variation", []):
            lines.append(f"  - {_variation_line(v)}")
        lines += _trend_lines(facts.get("tendencia"))
        st = facts.get("estadisticas")
        if st:
            lines.append(
                f"Histórico: último {_num(st['ultimo_valor'])} ({st['ultima_fecha']}), "
                f"rango [{_num(st['minimo'])}, {_num(st['maximo'])}], "
                f"percentil del último {_pct_share(st['percentil_ultimo_valor'])}."
            )
        return "\n".join(lines)

    return "\n".join(lines)
