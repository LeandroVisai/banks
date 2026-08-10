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
from dataclasses import dataclass, field
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


# Gap mediano (días) máximo para considerar un dataset de ALTA FRECUENCIA
# (diario/semanal). Por encima (mensual, p.ej. carteras a fin de mes) NO entra al
# cálculo del corte común.
_HIGH_FREQ_MAX_GAP_DAYS = 10
# Una ventana es "semanal" si cubre <= estos días.
_WEEKLY_MAX_DAYS = 7


def _windows(
    last_date: str, windows: list[tuple[str, int]], *, weekly_asof: str | None = None,
) -> list[dict]:
    """``[(label, days)]`` → ``[{label, start, end, days, is_weekly, anchored}]``.

    ``weekly_asof`` es la fecha de corte / ``T`` COMÚN del informe (solo lo reciben los
    datasets de las secciones de anclaje: Flujos + Portafolio DCV). Cuando se pasa,
    ``anchored=True`` y TODAS las ventanas (semanal Y mensual) se anclan al mismo
    ``end`` → esos datasets comparten ``T``, ``T-7`` y ``T-30`` (los mismos días). Sin
    corte (``weekly_asof=None``), cada ventana se ancla al ``last_date`` del propio
    parquet. ``is_weekly`` / ``anchored`` le indican a ``_window_value`` qué resolución
    usar (ver ahí)."""
    anchored = weekly_asof is not None
    anchor = weekly_asof or last_date
    out = []
    for label, days in windows:
        start = (date.fromisoformat(anchor[:10]) - timedelta(days=days)).isoformat()
        out.append({"label": label, "start": start, "end": anchor, "days": days,
                    "is_weekly": days <= _WEEKLY_MAX_DAYS, "anchored": anchored})
    return out


def _slice(series: list[sa.Point], start: str, end: str) -> list[sa.Point]:
    return [p for p in series if start <= p[0] <= end]


def _flow_window(series_slice: list[sa.Point]) -> dict | None:
    """Hecho de FLUJOS para una ventana: el flujo NETO del período = SUMA de los
    flujos observados en ``[start, end]`` (NO último−primero).

    El signo de la suma es la dirección (positivo = entrada/aportes, negativo =
    salida/rescates), consistente con el gráfico de suma por período
    (``monthly_sum_by_fund`` / ``monthly_diff`` / ``window_grouped``). ``None`` si
    la ventana no tiene observaciones. Devuelve un dict shape-compatible con el de
    ``sa.variation`` (mismas claves usadas por los renderers) más ``is_flow_sum`` y
    ``flujo_periodo``."""
    if not series_slice:
        return None
    valores = [v for _, v in series_slice]
    total = sum(valores)
    return {
        "is_flow_sum": True,
        "fecha_inicio_obs": series_slice[0][0],
        "fecha_fin_obs": series_slice[-1][0],
        "flujo_periodo": round(total, 6),
        # Alias para que _contributions (ordena/imprime por cambio_absoluto) y
        # _flow_tag (lee nivel_fin) operen sobre el flujo del período sin ramas.
        "cambio_absoluto": round(total, 6),
        "valor_fin": round(total, 6),
        "variacion_pct": None,
        "minimo": round(min(valores), 6),
        "maximo": round(max(valores), 6),
        "n_observaciones": len(series_slice),
    }


def _value_asof(series: list[sa.Point], d: str) -> sa.Point | None:
    """Último ``(fecha, valor)`` con ``fecha <= d`` (serie ordenada ASC); ``None``
    si no hay ningún punto en o antes de ``d``."""
    out: sa.Point | None = None
    for p in series:
        if p[0] <= d:
            out = p
        else:
            break
    return out


def weekly_delta(
    series: list[sa.Point], asof: str, *, days: int = 7, is_flow: bool = False,
) -> dict | None:
    """Variación semanal CANÓNICA anclada a ``asof`` — la MISMA que dibuja el
    gráfico, para que el texto y la tabla/barra coincidan siempre.

    - flujos: suma de las observaciones en ``[asof - days, asof]`` (idéntico a
      ``_flow_window`` sobre ese slice).
    - niveles: ``valor_asof(asof) - valor_asof(asof - days)`` con resolución
      *at-or-before* en ambos extremos (igual que ``series_transforms.stacked_by_bucket``);
      si no hay punto en o antes del inicio, cae al primer punto de la serie.

    El dict resultante es shape-compatible con ``sa.variation`` / ``_flow_window``
    (lo consumen ``_variation_line`` y los renderers). ``None`` si la serie está
    vacía o no hay punto en o antes de ``asof``."""
    if not series:
        return None
    start = (date.fromisoformat(asof[:10]) - timedelta(days=days)).isoformat()
    window = _slice(series, start, asof)
    if is_flow:
        return _flow_window(window)
    fin = _value_asof(series, asof)
    if fin is None:
        return None
    ini = _value_asof(series, start) or series[0]
    out = sa.variation([(ini[0], ini[1]), (fin[0], fin[1])])
    # min/max/n informativos sobre la ventana observada (no solo los 2 extremos).
    if out and window:
        vals = [v for _, v in window]
        out["minimo"], out["maximo"] = round(min(vals), 6), round(max(vals), 6)
        out["n_observaciones"] = len(window)
    return out


def geom_return(points: list[sa.Point], start: str, end: str) -> float | None:
    """Retorno COMPUESTO entre ``start`` y ``end`` de un ÍNDICE de retorno acumulado
    (en fracción): ``(1+i_end)/(1+i_start)-1`` con resolución at-or-before en ambos
    extremos. Para índices de retorno la "variación" de una ventana es el retorno
    compuesto, NO la resta del índice (que sobreestima al crecer el índice).
    ``None`` si no hay datos."""
    if not points:
        return None
    i_end = next((v for iso, v in reversed(points) if iso <= end), None)
    if i_end is None:
        return None
    i_start = next((v for iso, v in reversed(points) if iso <= start), points[0][1])
    denom = 1.0 + i_start
    if denom == 0:
        return None
    return (1.0 + i_end) / denom - 1.0


def geom_ytd(points: list[sa.Point], ytd_start: str) -> list[sa.Point]:
    """Serie de rentabilidad acumulada YTD geométrica: ``(1+i_t)/(1+i_base)-1`` para
    ``t >= ytd_start``, con ``i_base`` = último índice ANTES de ``ytd_start`` (cierre
    del año previo; o el primer punto si la serie arranca dentro del año)."""
    if not points:
        return []
    base = next((v for iso, v in reversed(points) if iso < ytd_start), points[0][1])
    denom = 1.0 + base
    if denom == 0:
        return []
    return [(iso, (1.0 + v) / denom - 1.0) for iso, v in points if iso >= ytd_start]


def _window_change(series_slice: list[sa.Point], *, is_flow: bool) -> dict | None:
    """Hecho de una ventana clásica: suma de flujos (``is_flow``) o variación de nivel
    (último menos primer punto DENTRO del slice)."""
    return _flow_window(series_slice) if is_flow else sa.variation(series_slice)


def _window_value(series: list[sa.Point], w: dict, *, is_flow: bool) -> dict | None:
    """Variación de UNA ventana.

    - Ventana SEMANAL (``is_weekly``) → ``weekly_delta`` (resolución canónica
      at-or-before / suma) SIEMPRE: así la barra/tabla semanal y el texto coinciden.
    - Ventana MENSUAL → ``weekly_delta`` SOLO si el dataset está ANCLADO (Flujos +
      Portafolio DCV, que comparten T, T-7 y T-30); el RESTO de parquets conserva la
      variación de ventana clásica (``_window_change``), sin cambios."""
    if w.get("is_weekly") or w.get("anchored"):
        return weekly_delta(series, w["end"], days=w.get("days", 7), is_flow=is_flow)
    return _window_change(_slice(series, w["start"], w["end"]), is_flow=is_flow)


def _window_variations(
    series: list[sa.Point], windows: list[dict], *, is_flow: bool = False,
) -> list[dict]:
    out = []
    for w in windows:
        out.append({
            "label": w["label"], "desde": w["start"], "hasta": w["end"],
            "variacion": _window_value(series, w, is_flow=is_flow),
        })
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


# Valores de ``value_kind`` (catálogo) que NO son flujos: la dirección es el signo
# de la VARIACIÓN del nivel, no del valor del período.
_NON_FLOW_KINDS = frozenset({"stock", "level", "nivel", "return", "retorno", "rate", "tasa"})
# ``value_kind`` donde SUMAR las categorías no es una métrica real (retornos/tasas):
# no se calcula total agregado ni composición, solo el detalle por categoría.
_NON_AGGREGATABLE_KINDS = frozenset({"return", "retorno", "rate", "tasa", "return_index"})

# Etiquetas (categoría o columna) que suelen ser un AGREGADO precomputado en el
# parquet (= suma de los componentes), no un componente más. Si se cuentan junto a
# los componentes, el total/composición se DUPLICA. Se excluyen del total agregado
# y de la composición SOLO si además validan numéricamente como suma del resto
# (``_validates_as_sum``), así un dataset sin agregado no se ve afectado.
_AGGREGATE_LABELS = frozenset({"total", "totales", "total general", "neto", "neto general"})


def _is_aggregate_label(name: object) -> bool:
    return str(name).strip().lower() in _AGGREGATE_LABELS


def _validates_as_sum(
    agg: dict[str, float], others: list[dict[str, float]], *, tol: float = 0.05,
) -> bool:
    """``True`` si ``agg`` ≈ suma de ``others`` en la mayoría (≥80%) de las fechas
    comunes (tolerancia relativa ``tol``). Confirma que una serie rotulada "Total"/
    "Neto" es de verdad el agregado de las demás antes de excluirla."""
    ok = n = 0
    for d, av in agg.items():
        comp = sum(o.get(d, 0.0) for o in others)
        scale = max(abs(av), abs(comp), 1.0)
        n += 1
        if abs(av - comp) <= tol * scale:
            ok += 1
    return n > 0 and ok / n >= 0.8


def _apply_facts_hints(
    rows: list[dict], hints: dict, value_cols: list[str],
) -> list[dict]:
    """Acota las filas para que el TEXTO describa el mismo corte que el gráfico.

    - ``filter`` ``{col: valor}``: deja solo esas filas (p.ej. ``Institucion=Total``).
    - ``exclude`` ``{col: [valores]}``: descarta esas filas (p.ej. ``Plazos_D`` ≠ tramo corto).
    - ``sign`` ``{col, pos, neg}``: deja solo filas ``pos``/``neg`` y NIEGA el valor de
      las ``neg`` -> el agregado pasa a ser el NETO (= Σpos - Σneg), igual que el chart.
    """
    flt = hints.get("filter") or {}
    exc = hints.get("exclude") or {}
    sign = hints.get("sign") or {}
    out = rows
    if flt:
        out = [r for r in out if all(str(r.get(c)) == str(v) for c, v in flt.items())]
    if exc:
        out = [r for r in out
               if all(str(r.get(c)) not in {str(x) for x in vals} for c, vals in exc.items())]
    if sign:
        col, pos, neg = sign.get("col"), str(sign.get("pos")), str(sign.get("neg"))
        vcol = value_cols[0] if value_cols else None
        kept: list[dict] = []
        for r in out:
            t = str(r.get(col))
            if t == pos:
                kept.append(r)
            elif t == neg and vcol is not None:
                r2 = dict(r)
                with contextlib.suppress(TypeError, ValueError):
                    r2[vcol] = -float(r2.get(vcol))
                kept.append(r2)
            # otros tipos (p.ej. Spot) se descartan: no entran al neto
        out = kept
    return out


def _aggregate_keys(series_by_key: dict[str, list[sa.Point]]) -> set[str]:
    """Claves (categorías o columnas) que son un AGREGADO precomputado: su nombre es
    de total/neto Y su serie valida como suma de las demás. Vacío si no hay ninguna
    (caso común → sin cambio de comportamiento)."""
    candidates = [k for k in series_by_key if _is_aggregate_label(k)]
    if not candidates or len(series_by_key) <= len(candidates):
        return set()
    out: set[str] = set()
    for k in candidates:
        others = [dict(s) for kk, s in series_by_key.items() if kk not in set(candidates)]
        if others and _validates_as_sum(dict(series_by_key[k]), others):
            out.add(k)
    return out


def _is_flow(dataset: ParquetDataset) -> bool:
    """``True`` si el dataset es de flujos (su valor es una entrada/salida, no un
    stock): el signo del valor indica dirección.

    Prioriza el ``value_kind`` declarado en el catálogo (robusto entre familias);
    solo cae a la heurística de palabras clave (id + nombre) si ``value_kind`` está
    vacío — así datasets de afp/nr cuyo nombre no trae una keyword de flujo
    (``var_pos_derivados``, ``spot_derivados_afp``, ``nr_var_posicion_*``) quedan
    bien marcados desde el YAML."""
    kind = (getattr(dataset, "value_kind", "") or "").strip().lower()
    if kind == "flow":
        return True
    if kind in _NON_FLOW_KINDS:
        return False
    blob = f"{dataset.id} {dataset.name}".lower()
    return any(k in blob for k in _FLOW_KEYWORDS)


def _distinct_dates(con: duckdb.DuckDBPyConnection, parquet_path: Path, date_col: str) -> list[str]:
    """Fechas distintas (ISO, ASC) del parquet — para clasificar la cadencia."""
    src = f"read_parquet({_quote_str(parquet_path.as_posix())})"
    rows = con.sql(
        f"SELECT DISTINCT CAST({_quote_ident(date_col)} AS DATE) AS d "
        f"FROM {src} WHERE {_quote_ident(date_col)} IS NOT NULL ORDER BY d"
    ).fetchall()
    return [str(r[0]) for r in rows if r[0] is not None]


def _median_gap_days(dates: list[str], *, tail: int = 12) -> float | None:
    """Gap mediano (días) entre las últimas ``tail`` fechas distintas; ``None`` si
    hay menos de 2."""
    recent = dates[-tail:]
    if len(recent) < 2:
        return None
    gaps = sorted(
        (date.fromisoformat(recent[i + 1][:10]) - date.fromisoformat(recent[i][:10])).days
        for i in range(len(recent) - 1)
    )
    mid = len(gaps) // 2
    return float(gaps[mid]) if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2.0


def is_high_frequency(dates: list[str]) -> bool:
    """True si la cadencia reciente de la serie es sub-mensual (gap mediano
    ≤ ``_HIGH_FREQ_MAX_GAP_DAYS``) → entra al corte semanal común. Series mensuales
    o con una sola fecha quedan fuera."""
    gap = _median_gap_days(dates)
    return gap is not None and gap <= _HIGH_FREQ_MAX_GAP_DAYS


def weekly_cutoff(datasets: list[ParquetDataset], parquet_dir: Path) -> str | None:
    """Fecha de corte semanal COMÚN del informe = ``min`` de las fechas máximas de
    los datasets de ALTA FRECUENCIA (diarios/semanales).

    Los mensuales NO entran (su máximo a fin de mes arrastraría el corte semanal de
    todos hacia atrás). Determinista: el mismo conjunto de datasets da el mismo corte
    en el proceso de gráficos y en el de texto, garantizando cifras semanales
    referidas a una sola fecha. ``None`` si ningún dataset califica (→ cada parquet
    se ancla a su propio máximo, comportamiento previo)."""
    maxes: list[str] = []
    con = duckdb.connect()
    try:
        for ds in datasets:
            path = ds.parquet_path(parquet_dir)
            if not path.exists():
                continue
            try:
                roles = detect_roles(path, con)
                if roles.date_col is None:
                    continue
                dates = _distinct_dates(con, path, roles.date_col)
            except Exception:
                log.exception("[weekly_cutoff] no se pudieron leer fechas de %s", ds.id)
                continue
            if dates and is_high_frequency(dates):
                maxes.append(dates[-1])
    finally:
        con.close()
    return min(maxes) if maxes else None


def compute_facts(
    dataset: ParquetDataset,
    parquet_dir: Path,
    windows: list[tuple[str, int]],
    *,
    weekly_asof: str | None = None,
) -> dict[str, Any] | None:
    """Calcula los hechos descriptivos relevantes de un dataset.

    ``windows`` es ``[(etiqueta, días)]`` (p.ej. ``[("última semana", 7), …]``).
    ``weekly_asof`` (opcional) es la fecha de corte / ``T`` COMÚN del informe (ver
    ``weekly_cutoff``): ancla TODAS las ventanas (semanal y mensual) ahí en vez del
    máximo del propio parquet, para que Flujos y DCV compartan T, T-7 y T-30 (mismos
    días) y la variación del texto coincida con la del gráfico/tabla.
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

        kind = (getattr(dataset, "value_kind", "") or "").strip().lower()
        base: dict[str, Any] = {
            "dataset_id": dataset.id,
            "unit": dataset.unit,
            "n_rows": int(n_rows),
            "date_col": roles.date_col,
            "value_cols": roles.value_cols,
            "category_cols": roles.category_cols,
            "is_flow": _is_flow(dataset),
            "value_kind": kind,
            # ¿Tiene sentido SUMAR las categorías? Para flujos/stocks sí (flujo o
            # stock total). Para RETORNOS/TASAS no: sumar la rentabilidad de fondos
            # distintos (o promediar tasas) no es una métrica real → solo se describe
            # por categoría, sin total agregado ni composición.
            "aggregatable": kind not in _NON_AGGREGATABLE_KINDS,
            "last_date": None,
        }

        scale = dataset.value_scale

        # ── Snapshot transversal (sin columna de fecha) ──────────────────────
        if roles.date_col is None:
            hints = getattr(dataset, "facts_hints", None) or {}
            if roles.category_cols and roles.value_cols:
                # ``category``/``value`` eligen el eje y la métrica que describe el chart
                # (p.ej. cambiario: el dato relevante es ``Neto``, no la 1ª col ``Spot``).
                cat_col = str(hints.get("category") or roles.category_cols[0])
                val_col = str(hints.get("value") or roles.value_cols[0])
                read_cols = (
                    list(dict.fromkeys([*roles.category_cols, *roles.value_cols])) if hints
                    else [cat_col, val_col]
                )
                rows = _read_rows(con, parquet_path, date_col=None, columns=read_cols)
                _scale_rows(rows, roles.value_cols, scale)
                if hints:
                    rows = _apply_facts_hints(rows, hints, [val_col])
                comp = _snapshot_composition(rows, cat_col, val_col)
                base.update(
                    shape="snapshot", composition=comp,
                    category_col=cat_col, value_col=val_col,
                    facts_cut=({k: hints[k] for k in ("filter", "exclude", "category", "value") if k in hints} or None),
                )
            else:
                base.update(shape="snapshot", composition=None)
            return base

        # max(TRY_CAST(... AS DATE)): castea POR FILA antes del max. Necesario porque
        # algunos parquets traen la fecha como VARCHAR y su estadística de columna en
        # la metadata viene TRUNCADA (p.ej. "2026-06-"); ``CAST(max(varchar) AS DATE)``
        # leería esa stat corrupta y lanzaría. TRY_CAST ignora valores no-fecha (NULL).
        last_date = con.sql(
            f"SELECT max(TRY_CAST({_quote_ident(roles.date_col)} AS DATE)) "
            f"FROM read_parquet({_quote_str(parquet_path.as_posix())})"
        ).fetchone()[0]
        if last_date is None:
            base.update(shape="empty")
            return base
        last_date = str(last_date)
        wins = _windows(last_date, windows, weekly_asof=weekly_asof)
        base.update(last_date=last_date, windows=[{"label": w["label"], "desde": w["start"], "hasta": w["end"]} for w in wins])

        # ── Índice de retorno acumulado (return_index) ───────────────────────
        # La "variación" de una ventana es el retorno COMPUESTO (geom_return), no la
        # resta del índice. NO se escala en filas: se computa en fracción y se lleva a
        # % con value_scale dentro de _return_index_facts.
        if kind == "return_index" and roles.value_cols:
            rows = _read_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, *roles.value_cols])
            base.update(
                shape="return_index",
                **_return_index_facts(rows, roles.date_col, roles.value_cols, wins, value_scale=scale),
            )
            return base

        # ── Serie temporal con categoría ─────────────────────────────────────
        if roles.category_cols and roles.value_cols:
            hints = getattr(dataset, "facts_hints", None) or {}
            # Con hints, la categoría primaria puede no ser la natural y filter/
            # exclude/sign necesitan otras categóricas → se leen TODAS.
            cat_col = str(hints.get("category") or roles.category_cols[0])
            net_cols = hints.get("net_cols") or {}
            if net_cols:
                # Dataset con DOS columnas de valor (p.ej. Suscripcion / Vencimiento):
                # el valor descrito es el NETO = pos - neg por fila (el rol natural solo
                # leería la primera columna y perdería la segunda y el neto).
                pos, neg = net_cols.get("pos"), net_cols.get("neg")
                val_col = "Neto"
                read_cols = [roles.date_col, *roles.category_cols, pos, neg]
                rows = _read_rows(con, parquet_path, date_col=roles.date_col, columns=read_cols)
                _scale_rows(rows, [pos, neg], scale)
                for r in rows:
                    try:
                        r[val_col] = float(r.get(pos) or 0.0) - float(r.get(neg) or 0.0)
                    except (TypeError, ValueError):
                        r[val_col] = None
            else:
                val_col = roles.value_cols[0]
                read_cols = (
                    [roles.date_col, *roles.category_cols, val_col] if hints
                    else [roles.date_col, cat_col, val_col]
                )
                rows = _read_rows(con, parquet_path, date_col=roles.date_col, columns=read_cols)
                _scale_rows(rows, roles.value_cols, scale)
            if hints:
                rows = _apply_facts_hints(rows, hints, [val_col])
            base.update(
                shape="timeseries_categorical",
                category_col=cat_col, value_col=val_col,
                facts_cut=({k: hints[k] for k in ("filter", "exclude", "sign", "net_cols", "category") if k in hints} or None),
                **_categorical_facts(
                    rows, roles.date_col, cat_col, val_col, wins, last_date,
                    is_flow=base["is_flow"], aggregatable=base["aggregatable"],
                ),
            )
            return base

        # ── Serie temporal "ancha" (varias columnas de valor, sin categoría) ──
        if len(roles.value_cols) > 1:
            hints = getattr(dataset, "facts_hints", None) or {}
            rows = _read_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, *roles.value_cols])
            _scale_rows(rows, roles.value_cols, scale)
            if hints:
                rows = _apply_facts_hints(rows, hints, roles.value_cols)
            base.update(
                shape="timeseries_wide",
                facts_cut=({k: hints[k] for k in ("filter", "exclude", "composition_exclude") if k in hints} or None),
                **_wide_facts(
                    rows, roles.date_col, roles.value_cols, wins, last_date,
                    is_flow=base["is_flow"], aggregatable=base["aggregatable"],
                    composition_exclude=hints.get("composition_exclude"),
                ),
            )
            return base

        # ── Serie temporal simple (una columna de valor) ─────────────────────
        if roles.value_cols:
            val_col = roles.value_cols[0]
            rows = _read_rows(con, parquet_path, date_col=roles.date_col, columns=[roles.date_col, val_col])
            _scale_rows(rows, roles.value_cols, scale)
            series = sa.clean_series(rows, roles.date_col, val_col)
            wv = _window_variations(series, wins, is_flow=base["is_flow"])
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


def _asof_data_date(rows: list[dict], date_col: str, ref: str) -> str:
    """Última fecha CON dato ≤ ``ref`` en ``rows`` (at-or-before); ``ref`` si no hay
    ninguna. Sirve para tomar la composición 'a corte' del ancla aunque el parquet no
    tenga dato justo ese día."""
    avail = {str(r[date_col]) for r in rows if r.get(date_col) is not None}
    return max((d for d in avail if d <= ref), default=ref)


def _categorical_facts(
    rows: list[dict], date_col: str, cat_col: str, val_col: str,
    wins: list[dict], last_date: str, *, is_flow: bool = False, aggregatable: bool = True,
) -> dict:
    """Total agregado por ventana + composición a la fecha de corte + top categorías.

    Si ``aggregatable`` es False (retornos/tasas), NO se calcula el total agregado
    ni la composición (sumar retornos de fondos distintos no es una métrica real):
    solo el detalle por categoría."""
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
    # Si el parquet trae una categoría AGREGADA ("Total"/"Neto" = suma del resto), se
    # excluye de los COMPONENTES para no duplicar el total ni la composición; su
    # variación por ventana se reporta aparte (``neto_agregado``).
    agg_cats = _aggregate_keys(series_by_cat)
    comp_cats = {c: s for c, s in series_by_cat.items() if c not in agg_cats}
    series_by_cat = comp_cats or series_by_cat  # nunca dejar el detalle vacío

    # Total por fecha = suma de los COMPONENTES (no del agregado) → variación por ventana.
    by_date: dict[str, float] = {}
    for s in series_by_cat.values():
        for d, v in s:
            by_date[d] = by_date.get(d, 0.0) + v
    total_series = sorted(by_date.items(), key=lambda kv: kv[0])
    # La composición a la fecha de corte solo tiene sentido para STOCKS (cartera a
    # un día). Para FLUJOS sería el desglose de los flujos de UN día suelto: shares
    # >100% / negativos (el total del día es chico y una categoría puede excederlo)
    # que confunden la lectura. La dirección/magnitud del flujo va en las ventanas.
    # Se cita "a corte" del ANCLA (``wins[0]['end']`` = corte común si lo hay, o el
    # máximo propio): el valor se toma at-or-before, pero la fecha mostrada es el corte
    # → consistente con las ventanas, aunque el parquet no tenga dato justo ese día.
    comp_ref = wins[0]["end"] if wins else last_date
    composition = None
    if not is_flow:
        rows_comp = [r for r in rows if str(r.get(cat_col)) not in agg_cats] if agg_cats else rows
        composition = sa.composition(
            rows_comp, date_col, cat_col, val_col, fecha=_asof_data_date(rows, date_col, comp_ref))
        if composition:
            composition["fecha"] = comp_ref
    # Variación por ventana del agregado precomputado (si lo hay), para que el texto
    # pueda citar el Neto sin recomputarlo.
    neto_agregado = None
    if agg_cats:
        cat_neto = next(iter(agg_cats))
        neto_series = sorted(agg[cat_neto].items(), key=lambda kv: kv[0])
        neto_agregado = {
            "categoria": cat_neto,
            "ventanas": _window_variations(neto_series, wins, is_flow=is_flow),
        }
    cut_vals = {cat: s[-1][1] for cat, s in series_by_cat.items() if s}
    # Para flujos, el "top" de categorías se rankea por el flujo NETO acumulado de
    # toda la serie (|suma|), no por el último valor de un día suelto (ruidoso).
    if is_flow:
        rank = {cat: abs(sum(v for _, v in s)) for cat, s in series_by_cat.items() if s}
    else:
        rank = {cat: abs(cut_vals[cat]) for cat in cut_vals}
    top = sorted(rank, key=lambda c: rank[c], reverse=True)[:_TOP_CATEGORIES]

    by_category = []
    for cat in top:
        series = series_by_cat[cat]
        by_category.append({
            "categoria": cat,
            "ultimo_valor": round(series[-1][1], 6),
            "ultima_fecha": series[-1][0],
            "ventanas": _window_variations(series, wins, is_flow=is_flow),
            "anomalia": sa.anomaly_check(series),
        })

    if not aggregatable:
        # Retornos/tasas: el total por suma no es real → solo detalle por categoría.
        return {
            "total_ventanas": [],
            "tendencia": None,
            "contribuciones": [],
            "composicion_corte": None,
            "por_categoria": by_category,
            "neto_agregado": neto_agregado,
        }
    total_ventanas = _window_variations(total_series, wins, is_flow=is_flow)
    return {
        "total_ventanas": total_ventanas,
        "tendencia": _trend_signal(total_ventanas),
        "contribuciones": _contributions(series_by_cat, wins, is_flow=is_flow),
        "composicion_corte": composition,
        "por_categoria": by_category,
        "neto_agregado": neto_agregado,
    }


def _contributions(
    series_by_cat: dict[str, list[sa.Point]], wins: list[dict], *, is_flow: bool = False,
) -> list[dict]:
    """Qué categorías explican el movimiento del total, por ventana.

    Para cada ventana, ordena las categorías por |cambio absoluto| y devuelve las
    de mayor aporte (los "drivers" del movimiento agregado). Para flujos el
    "cambio" es el flujo NETO del período (suma), no último−primero."""
    out = []
    for w in wins:
        cambios = []
        for cat, series in series_by_cat.items():
            v = _window_value(series, w, is_flow=is_flow)
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
    wins: list[dict], last_date: str, *, is_flow: bool = False, aggregatable: bool = True,
    composition_exclude: list[str] | None = None,
) -> dict:
    """Variación por columna + composición wide a la fecha de corte (top columnas).

    Si ``aggregatable`` es False (retornos/tasas), se omite la composición (sumar
    columnas de retorno no es una métrica real); solo el detalle por columna.
    ``composition_exclude`` saca columnas de la composición/ranking aunque NO sean
    agregados (p.ej. ``AUM`` en otra unidad que la allocation %): se siguen
    reportando por columna pero no entran a las shares."""
    series_by_col = {c: sa.clean_series(rows, date_col, c) for c in value_cols}
    # Columnas que son un AGREGADO precomputado ("Neto"/"Total" = suma de los tramos,
    # confirmado numéricamente): se excluyen de la composición y del ranking de
    # componentes (si no, el total se DUPLICA y el Neto sale como ~50% del corte),
    # pero se siguen reportando como columna propia (su variación es la del neto).
    agg_cols = _aggregate_keys(series_by_col)
    excl = set(agg_cols) | {c for c in (composition_exclude or []) if c in series_by_col}
    comp_cols = [c for c in value_cols if c not in excl] or list(value_cols)
    last_vals = {c: (s[-1][1] if s else 0.0) for c, s in series_by_col.items()}
    if is_flow:
        rank = {c: abs(sum(v for _, v in s)) for c, s in series_by_col.items()}
    else:
        rank = {c: abs(last_vals[c]) for c in value_cols}
    top = sorted(comp_cols, key=lambda c: rank.get(c, 0.0), reverse=True)[:_TOP_CATEGORIES]
    # Las columnas excluidas de la composición (agregado o distinta unidad) igual se
    # reportan por columna (su variación importa); solo no entran a las shares.
    report_cols = top + [c for c in value_cols if c in excl and c not in top]
    por_columna = [
        {
            "columna": c,
            "ultimo_valor": round(series_by_col[c][-1][1], 6) if series_by_col[c] else None,
            "ultima_fecha": series_by_col[c][-1][0] if series_by_col[c] else None,
            "ventanas": _window_variations(series_by_col[c], wins, is_flow=is_flow),
            "es_agregado": c in agg_cols,
        }
        for c in report_cols
    ]
    # Composición "a corte" del ancla (corte común si lo hay, o máx propio): valor
    # at-or-before, fecha mostrada = el corte (ver _categorical_facts). Solo sobre
    # los COMPONENTES (sin el agregado, que duplicaría el total).
    comp_ref = wins[0]["end"] if wins else last_date
    composicion = None
    if aggregatable:
        composicion = sa.composition_wide(
            rows, date_col, comp_cols, fecha=_asof_data_date(rows, date_col, comp_ref))
        if composicion:
            composicion["fecha"] = comp_ref
    return {
        "por_columna": por_columna,
        # Retornos/tasas: no se compone (sumar columnas de retorno no es real).
        "composicion_corte": composicion,
    }


# Fondos que el informe ffmm reporta SIEMPRE (1/2/3/6).
_RETURN_INDEX_FUNDS = ("Tipo 1", "Tipo 2", "Tipo 3", "Tipo 6")


def _return_index_facts(
    rows: list[dict], date_col: str, value_cols: list[str], wins: list[dict],
    *, value_scale: float, funds: tuple[str, ...] = _RETURN_INDEX_FUNDS,
) -> dict:
    """Hechos de un ÍNDICE de retorno acumulado (fracción): por fondo, el RETORNO
    COMPUESTO de cada ventana (``geom_return``, en %), más el índice acumulado al
    cierre. El retorno se computa en fracción y se lleva a % con ``value_scale``.
    Foco en los fondos pedidos (default Tipo 1/2/3/6)."""
    cols = [c for c in funds if c in value_cols] or list(value_cols)
    por_fondo: list[dict] = []
    for c in cols:
        series = sa.clean_series(rows, date_col, c)
        if not series:
            continue
        ventanas = []
        for w in wins:
            r = geom_return(series, w["start"], w["end"])
            ventanas.append({
                "label": w["label"], "desde": w["start"], "hasta": w["end"],
                "retorno_pct": None if r is None else round(r * value_scale, 4),
            })
        por_fondo.append({
            "fondo": c,
            "indice_acum_pct": round(series[-1][1] * value_scale, 4),
            "ultima_fecha": series[-1][0],
            "ventanas": ventanas,
        })
    return {"por_fondo": por_fondo}


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
    ISO en series temporales y nombre de categoría en snapshots/rangos.

    ``kind='scatter'`` reusa esta MISMA forma para un caso distinto (cada serie es
    UN punto, no una curva): ``label`` = etiqueta de texto del punto (p.ej. país) y
    ``points`` trae un único par ``(str(x), y)`` — el valor X va serializado como
    string en la posición que normalmente lleva la fecha/categoría, así el scatter
    no necesita un campo nuevo en el dataclass."""

    label: str
    points: list[tuple[str, float]]


@dataclass
class PlotData:
    """Datos listos para el renderer SVG. ``kind='timeseries'`` → multi-línea
    (eje X temporal); ``kind='snapshot'`` → barras de composición (eje X
    categórico); ``kind='grouped'`` → barras por categoría; ``kind='range'`` →
    caja [mín,máx] + promedio + "hoy" por categoría (series con label EXACTO
    "Mínimo"/"Máximo"/"Promedio"/"Hoy"); ``kind='scatter'`` → dispersión x/y
    etiquetada (ver convención en ``PlotSeries``; ``overlay`` marca las
    etiquetas del punto destacado). ``family`` es la familia renderizable del
    catálogo (``chart_family``); ``table`` => el caller cae a una mini-tabla HTML."""

    dataset_id: str
    family: str
    kind: str  # "timeseries" | "snapshot" | "grouped" | "range" | "scatter"
    unit: str
    series: list[PlotSeries]
    # Etiquetas de series que se dibujan SUPERPUESTAS (no apiladas): una línea
    # sobre el área/barras en series temporales, un punto por categoría en barras.
    # Réplica del "Neto"/"Total" de los informes BCCh sobre apilados divergentes.
    overlay: tuple[str, ...] = ()
    # Nota de FECHAS que cubre el gráfico (qué corte/ventana usa cada serie). Para
    # gráficos de ventana (Mes/YtD/Δ7d/…) donde el eje X NO es la fecha, hace
    # explícitas las fechas consideradas. El bloque del informe la muestra debajo
    # del título. Vacío = el eje X ya es temporal (la fecha se ve en el gráfico).
    date_note: str = ""
    # ``True`` (default) → el eje Y de las series temporales incluye el 0, que es
    # lo correcto para FLUJOS y STOCKS: la magnitud se lee contra la nada.
    #
    # ``False`` → el eje se ajusta al rango del dato. Es lo que necesitan los
    # NIVELES que nunca se acercan a cero (un tipo de cambio en 930, un índice
    # base 100, un RSI entre 30 y 70): forzarles la base 0 aplasta toda la serie
    # contra el borde superior y el movimiento —que es justamente lo que hay que
    # leer— deja de verse. Solo lo miran los renderers de línea y doble eje; las
    # barras y el área apilada siempre parten de 0, porque ahí la base ES el dato.
    zero_base: bool = True
    # Estilo por serie: ``{label: color_hex}`` fuerza el color de esa serie por
    # encima de la paleta cíclica por índice; ``muted`` marca labels que se dibujan
    # finas y en gris de solo contexto. Réplica del resaltado MA50/MA200 del
    # informe cambiario original — sin esto, un gráfico con 5-6 medias móviles del
    # mismo grosor y color aleatorio no deja ver cuál es la que importa (el cruce
    # dorado/de la muerte). Vacío (default) = comportamiento histórico: paleta
    # cíclica por orden de ``series``. Solo lo consultan los renderers de línea y
    # doble eje del EJE IZQUIERDO; el eje derecho ya tiene su propio estilo fijo
    # (área gris tenue).
    emphasis: dict[str, str] = field(default_factory=dict)
    muted: tuple[str, ...] = ()

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
    """Una ventana → texto. ``v`` es ``{label, desde, hasta, variacion}``.

    Muestra la VENTANA pedida (``desde``→``hasta`` = el corte común si lo hay), no las
    fechas observadas: así Flujos y DCV citan EXACTAMENTE la misma ventana aunque DCV
    no tenga dato justo en el corte (el valor se resuelve at-or-before, pero la fecha
    citada es la del corte)."""
    var = v.get("variacion")
    if not var:
        return f"{v['label']} ({v['desde']}→{v['hasta']}): sin observaciones suficientes en la ventana"
    if var.get("is_flow_sum"):
        return (
            f"{v['label']} ({v['desde']}→{v['hasta']}): "
            f"flujo neto del período {_num(var['flujo_periodo'])}"
            f"{_flow_tag(var['flujo_periodo'])}; "
            f"rango por obs [{_num(var['minimo'])}, {_num(var['maximo'])}], n={var['n_observaciones']}"
        )
    return (
        f"{v['label']} ({v['desde']}→{v['hasta']}): "
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
        if is_flow:
            # En flujos el "cambio" del driver ES su flujo neto del período (suma);
            # el signo (ENTRADA/SALIDA) lo da _flow_tag sobre esa suma.
            partes = ", ".join(
                f"{d['categoria']} (flujo del período {_num(d['cambio_absoluto'])}"
                f"{_flow_tag(d['cambio_absoluto'])})"
                for d in c["drivers"]
            )
        else:
            partes = ", ".join(
                f"{d['categoria']} (cambio {_num(d['cambio_absoluto'])}, {_pct(d['variacion_pct'])})"
                for d in c["drivers"]
            )
        out.append(f"  - {c['label']}: {partes}")
    return out


def _cut_note(cut: dict | None) -> list[str]:
    """Describe el CORTE aplicado a los facts (filter/exclude/sign), para que el LLM
    redacte sobre el mismo recorte que dibuja el gráfico y lo nombre bien."""
    if not cut:
        return []
    parts: list[str] = []
    for c, v in (cut.get("filter") or {}).items():
        parts.append(f"filtrado a {c}={v}")
    for c, vals in (cut.get("exclude") or {}).items():
        parts.append(f"excluye {c}: {', '.join(map(str, vals))}")
    sign = cut.get("sign") or {}
    if sign:
        parts.append(f"el valor es el NETO = {sign.get('pos')} menos {sign.get('neg')} (col {sign.get('col')})")
    nets = cut.get("net_cols") or {}
    if nets:
        parts.append(f"el valor es el NETO = {nets.get('pos')} menos {nets.get('neg')}")
    if cut.get("value"):
        parts.append(f"la métrica descrita es {cut['value']}")
    ce = cut.get("composition_exclude") or []
    if ce:
        parts.append(f"{', '.join(map(str, ce))} va aparte (no entra a la composición)")
    if not parts:
        return []
    return ["CORTE aplicado (describe ESTE recorte, igual que el gráfico): " + "; ".join(parts) + "."]


def facts_to_text(facts: dict) -> str:
    """Renderiza el dict de ``compute_facts`` a un bloque de texto para el LLM."""
    unit = facts.get("unit") or ""
    shape = facts.get("shape")
    is_flow = bool(facts.get("is_flow"))
    lines: list[str] = [f"Filas: {facts.get('n_rows')}; unidad: {unit or 's/d'}."]

    if shape == "snapshot":
        lines += _cut_note(facts.get("facts_cut"))
        comp = facts.get("composition")
        if comp:
            lines.append("Corte transversal (sin serie temporal).")
            lines += _composition_lines(comp, unit)
        return "\n".join(lines)

    lines.append(f"Última fecha con datos: {facts.get('last_date')}.")
    lines += _cut_note(facts.get("facts_cut"))
    if is_flow:
        lines.append(
            "NOTA — serie de FLUJOS: el SIGNO del flujo indica dirección (positivo = "
            "ENTRADA/aportes, negativo = SALIDA/rescates). Una variación negativa con "
            "el flujo aún positivo es MENOR ENTRADA (desaceleración), NO una salida; "
            "solo hay salida cuando el flujo en sí es negativo."
        )

    if shape == "return_index":
        lines.append(
            "RENTABILIDAD por tipo de fondo — RETORNO COMPUESTO de cada ventana (NO es "
            "la resta del índice). El signo indica si el fondo rentó positivo o negativo."
        )
        for f in facts.get("por_fondo", []):
            lines.append(
                f"  · {f['fondo']} (índice acumulado {_pct(f['indice_acum_pct'])} al "
                f"{f['ultima_fecha']}):"
            )
            for v in f["ventanas"]:
                rp = v.get("retorno_pct")
                if rp is None:
                    lines.append(f"      {v['label']} ({v['desde']}→{v['hasta']}): sin dato suficiente")
                else:
                    direccion = "positivo" if rp >= 0 else "negativo"
                    lines.append(
                        f"      {v['label']} ({v['desde']}→{v['hasta']}): rentó {direccion} {_pct(rp)}"
                    )
        return "\n".join(lines)

    if shape == "timeseries_categorical":
        lines.append(f"Categoría: {facts.get('category_col')}; métrica: {facts.get('value_col')}.")
        total_ventanas = facts.get("total_ventanas") or []
        if not facts.get("aggregatable", True):
            lines.append(
                "NOTA — serie de RETORNOS/TASAS: no se reporta total agregado (sumar "
                "retornos de categorías distintas no es una métrica real); se describe "
                "cada categoría por separado."
            )
        elif total_ventanas:
            lines.append("Total agregado (suma de categorías):")
            for v in total_ventanas:
                lines.append(f"  - {_variation_line(v)}")
        lines += _trend_lines(facts.get("tendencia"))
        lines += _contribution_lines(facts.get("contribuciones"), is_flow=is_flow)
        comp = facts.get("composicion_corte")
        if comp:
            lines += _composition_lines(comp, unit)
        # En flujos el encabezado NO lleva tag de dirección (sería el de un día
        # suelto, que puede contradecir el flujo NETO del período): la dirección
        # autoritativa va en cada línea de ventana (flujo neto).
        lines.append("Por categoría (top por flujo neto):" if is_flow else "Por categoría (top por nivel):")
        for cat in facts.get("por_categoria", []):
            head = "último flujo obs" if is_flow else "último"
            lines.append(f"  · {cat['categoria']}: {head} {_num(cat['ultimo_valor'])} ({cat['ultima_fecha']})")
            for v in cat["ventanas"]:
                lines.append(f"      {_variation_line(v)}")
        return "\n".join(lines)

    if shape == "timeseries_wide":
        lines.append("Por columna (top por flujo neto):" if is_flow else "Por columna (top por nivel):")
        for col in facts.get("por_columna", []):
            head = "último flujo obs" if is_flow else "último"
            lines.append(f"  · {col['columna']}: {head} {_num(col['ultimo_valor'])} ({col['ultima_fecha']})")
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
