#!/usr/bin/env python3
"""Piezas COMPARTIDAS por los dos extractores de parquets (``from_excel`` y
``from_sql``).

No define datasets ni lee archivos: solo el andamiaje que los dos scripts usan
igual — declarar un dataset, resolver una columna de un Excel con encabezados
sucios, escribir el parquet y emitir la entrada de catálogo.

Por qué existe
-------------
``scripts/build_cambiario_parquets.py`` ya traía todo esto, pero atado a UN
informe y a UN origen (el ``.xlsm`` del DOMA + ``Get_Data``). Al sumar el IPC
—que lee cuatro planillas del INE y dos consultas distintas— hacía falta el
mismo andamiaje con dos informes y tres orígenes. En vez de copiarlo, este
módulo lo **importa** de aquel script: ``Dataset``, ``col``/``pick``/``like`` y
la coerción de tipos siguen teniendo una sola implementación, la que ya se probó
contra el Excel real. Si mañana el ``.xlsm`` cambia un encabezado, se arregla en
un solo lugar y los dos extractores lo heredan.

    from_excel.py ─┐
                   ├─► sources.py ─► build_cambiario_parquets.py  (Dataset, col, pick…)
    from_sql.py   ─┘

Uso desde un extractor::

    import sources
    REG = sources.Registry()

    @REG.dataset(ds_id="ipc_difusion", chart_type="line", name="Difusión IPC",
                 description="…", unit="%", needs=("canasta",),
                 columns=[("Fecha", sources.TS), ("Difusión (%)", sources.D)])
    def _difusion(src):
        ...
        return frame

    sources.write_all(REG.datasets, {"canasta": df}, Path("data_pipeline/parquet"))
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("ingest")

ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = ROOT / "data_pipeline" / "parquet"
_BCP_PATH = ROOT / "scripts" / "build_cambiario_parquets.py"


def _load_bcp():
    """Carga ``build_cambiario_parquets.py`` como módulo.

    Es un script suelto (``scripts/`` no es paquete), así que se carga por ruta —
    el mismo patrón que usan los tests del repo para los demás scripts. Importarlo
    tiene el efecto lateral de registrar sus 41 datasets ``cam_*`` en
    ``bcp.DATASETS``, que es justo lo que ``from_excel.py`` necesita para
    reconstruir el informe cambiario sin volver a escribir un solo builder.
    """
    spec = importlib.util.spec_from_file_location("build_cambiario_parquets", _BCP_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - ruta rota
        raise ImportError(f"no pude cargar {_BCP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_cambiario_parquets", module)
    spec.loader.exec_module(module)
    return module


bcp = _load_bcp()

# ── Re-exports: una sola implementación, la de build_cambiario_parquets ──────
Dataset = bcp.Dataset
MissingColumnError = bcp.MissingColumnError
col = bcp.col
cols = bcp.cols
like = bcp.like
pick = bcp.pick
norm = bcp._norm

# Tipos de columna del catálogo (los mismos literales que emite el YAML).
TS = bcp._TS   # TIMESTAMP
D = bcp._D     # DOUBLE
V = bcp._V     # VARCHAR


class Registry:
    """Lista de ``Dataset`` de UN informe, poblada por el decorador ``dataset``.

    Cada extractor tiene la suya, así ``from_excel`` puede registrar los datasets
    del IPC sin mezclarlos con los ``cam_*`` que ya trae ``bcp.DATASETS``.
    """

    def __init__(self) -> None:
        self.datasets: list[Any] = []

    def dataset(self, **kwargs) -> Callable:
        def wrap(fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
            self.datasets.append(Dataset(build=fn, **kwargs))
            return fn
        return wrap

    def by_id(self, ds_id: str):
        return next((d for d in self.datasets if d.id == ds_id), None)


def to_parquet(frame, dst: Path) -> None:
    """Escribe el DataFrame como parquet, con o sin ``pyarrow`` instalado.

    ``DataFrame.to_parquet`` exige ``pyarrow`` o ``fastparquet``, que no están en
    todas las máquinas donde se corre esto (el H100 va offline y con el entorno
    mínimo). DuckDB sí está —es lo que leen las transforms del informe— y escribe
    parquet nativamente desde un DataFrame registrado, así que sirve de respaldo.
    Es además el mismo camino que usa ``scripts/make_demo_parquets.py``.
    """
    try:
        frame.to_parquet(dst, index=False)
        return
    except ImportError:
        pass
    import duckdb

    con = duckdb.connect()
    try:
        con.register("_frame", frame)
        con.execute(f"COPY _frame TO '{dst.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def read_parquet(path: Path):
    """Lee un parquet a DataFrame, con o sin ``pyarrow`` (ver ``to_parquet``)."""
    import pandas as pd

    try:
        return pd.read_parquet(path)
    except ImportError:
        pass
    import duckdb

    con = duckdb.connect()
    try:
        return con.execute(f"SELECT * FROM '{Path(path).as_posix()}'").fetchdf()
    finally:
        con.close()


def write_all(datasets: list, src: dict, out_dir: Path) -> tuple[int, int]:
    """Construye y escribe los datasets cuyos orígenes estén en ``src``.

    Devuelve ``(escritos, saltados)``. Ningún dataset puede tumbar la corrida: si
    le falta un origen, le falta una columna o falla al escribirse, se salta con
    log y los demás se generan igual. Es la misma política que
    ``bcp.build_all`` —un Excel con una celda sucia no puede dejar sin generar
    los otros cuarenta parquets— pero recibiendo la lista por parámetro en vez de
    leer una global, para poder correr un informe a la vez.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for ds in datasets:
        missing = [n for n in ds.needs if n not in src]
        if missing:
            log.warning("%-32s saltado - falta el origen %s", ds.id, ", ".join(missing))
            skipped += 1
            continue
        try:
            frame = ds.build(src)
        except MissingColumnError as exc:
            log.warning("%-32s saltado - %s", ds.id, exc)
            skipped += 1
            continue
        except Exception:
            log.exception("%-32s fallo al construirse", ds.id)
            skipped += 1
            continue
        if frame is None or len(frame) == 0:
            log.warning("%-32s saltado - sin filas", ds.id)
            skipped += 1
            continue
        frame = bcp._coerce_declared_numeric(frame, ds)
        dst = out_dir / ds.file
        try:
            to_parquet(frame, dst)
        except Exception:
            log.exception("%-32s fallo al escribir el parquet (revisar tipos)", ds.id)
            skipped += 1
            continue
        log.info("%-32s -> %s (%d filas x %d cols)", ds.id, dst.name, len(frame), len(frame.columns))
        written += 1
    return written, skipped


# ── Tabla ancha del cambiario: quién manda cuando dos orígenes traen lo mismo ──
#
# El informe cambiario arma 32 de sus 41 datasets desde UNA tabla ancha (la ex
# hoja "Datos" del .xlsm), que hoy se reconstruye entre dos scripts. Cada uno
# deja su aporte en la carpeta de staging con uno de estos nombres, y el orden de
# esta tupla decide quién gana si los dos traen la misma columna:
#
#   1. cam_datos_excel      datos_lea.xlsx — el recorte de Bloomberg. MANDA.
#   2. cam_datos_sql        lo que reconstruye from_sql.py desde el DW.
#   3. cam_datos_derivado   lo que se calcula por falta de origen (la vela OHLC
#                           sacada del intradía).
#
# datos_lea.xlsx va primero a propósito: es la planilla que el DOMA mantiene y a
# la que se le van a ir agregando las columnas que faltan. El día que sume la
# vela real de Bloomberg, esa pisa a la derivada sin tocar una línea de código;
# hasta entonces, la derivada rellena el hueco.
PRIORIDAD_CAMBIARIO = ("cam_datos_excel", "cam_datos_sql", "cam_datos_derivado")


def merge_wide(frames: list, key: str = "Fecha"):
    """Une tablas anchas por ``key``, con el PRIMER frame mandando.

    Dos reglas, en este orden:

    - una columna que ya aportó un frame anterior no se reemplaza; y
    - donde ese frame la dejó vacía, el siguiente la rellena (``combine_first``).

    La segunda regla importa: la vela derivada del intradía solo existe desde
    2023, así que si mañana Bloomberg aporta la real desde 2019, la de Bloomberg
    manda en todo su tramo y la derivada no aporta nada — pero si Bloomberg
    llegara recortada, el hueco se tapa en vez de quedar en blanco.
    """
    import pandas as pd

    frames = [f for f in frames if f is not None and key in f.columns]
    if not frames:
        return None
    out = frames[0].copy()
    out[key] = pd.to_datetime(out[key], errors="coerce")
    out = out.dropna(subset=[key])
    for extra in frames[1:]:
        extra = extra.copy()
        extra[key] = pd.to_datetime(extra[key], errors="coerce")
        extra = extra.dropna(subset=[key])
        compartidas = [c for c in extra.columns if c != key and c in out.columns]
        nuevas = [c for c in extra.columns if c != key and c not in out.columns]
        if nuevas:
            out = out.merge(extra[[key] + nuevas], on=key, how="outer")
        if compartidas:
            relleno = extra[[key] + compartidas].set_index(key)
            base = out.set_index(key)
            for c in compartidas:
                base[c] = base[c].combine_first(relleno[c])
            out = base.reset_index()
    return out.sort_values(key).reset_index(drop=True)


def stage_write(frame, stage_dir: Path, nombre: str) -> None:
    """Deja el aporte de un script a la tabla ancha, para que el otro lo levante."""
    if nombre not in PRIORIDAD_CAMBIARIO:
        raise ValueError(f"{nombre!r} no está en PRIORIDAD_CAMBIARIO: {PRIORIDAD_CAMBIARIO}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    to_parquet(frame, stage_dir / f"{nombre}.parquet")
    log.info("stage - %s.parquet (%d filas x %d cols)", nombre, len(frame), len(frame.columns))


def stage_merge(stage_dir: Path, propios: dict | None = None):
    """La tabla ancha completa: lo staged + lo que este script acaba de calcular.

    ``propios`` son los frames de esta corrida, por nombre de ``PRIORIDAD_CAMBIARIO``;
    pisan al archivo staged del mismo nombre (que es su versión anterior). El
    resultado respeta ``PRIORIDAD_CAMBIARIO`` sin importar quién corrió primero.
    """
    propios = propios or {}
    frames = []
    for nombre in PRIORIDAD_CAMBIARIO:
        if nombre in propios:
            frames.append(propios[nombre])
            continue
        path = stage_dir / f"{nombre}.parquet"
        if path.exists():
            df = read_parquet(path)
            log.info("stage - levantado %s.parquet (%d cols)", nombre, len(df.columns))
            frames.append(df)
    return merge_wide(frames)


# ── Gráficos de RANGO: un esquema común para los cuatro ──────────────────────
#
# Cuatro piezas del informe tienen la misma forma —una banda por categoría más el
# valor de hoy— y en el original eran cuatro gráficos distintos: la difusión del
# IPC general, las de bienes y servicios sin volátiles, y las incidencias
# esperadas por división. Escritos con el MISMO esquema de columnas, los cuatro
# los dibuja una sola transform (``ipc_rango_categoria``) y no hace falta un
# renderer por gráfico.
COLS_RANGO = [("Categoria", V), ("Orden", D), ("Minimo", D), ("Maximo", D),
              ("Promedio", D), ("Actual", D)]

MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
             "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def banda_mensual(df, columna: str, *, desde: int, hasta: int, cuantiles=(0.10, 0.90)):
    """Banda por mes calendario + el valor del año en curso.

    ``cuantiles`` define los bordes: el informe usa p10-p90 para las bandas de
    bienes y servicios sin volátiles. Con ``(0, 1)`` la banda es el mínimo-máximo
    absoluto, que es como se dibuja la de difusión.
    """
    ventana = df[df["Año"].between(max(int(df["Año"].min()), desde), hasta)]
    if ventana.empty:
        return None
    lo, hi = cuantiles
    banda = (ventana.groupby("Mes", observed=True)[columna]
             .agg(Minimo=lambda s: s.quantile(lo), Maximo=lambda s: s.quantile(hi),
                  Promedio="mean")
             .reset_index())
    anio_actual = int(df["Año"].max())
    actual = (df[df["Año"] == anio_actual][["Mes", columna]]
              .rename(columns={columna: "Actual"}))
    out = banda.merge(actual, on="Mes", how="left")
    out["Orden"] = out["Mes"].astype(int)
    out["Categoria"] = out["Orden"].map(lambda m: MESES_ES[m - 1])
    return out.sort_values("Orden")[[c for c, _ in COLS_RANGO]].round(3).reset_index(drop=True)



def emit_catalog(datasets: list) -> str:
    """Entradas YAML para ``sql_catalog/parquet_catalog.yaml``.

    Sin ``date_range``: lo fija ``scripts/refresh_catalog_dates.py`` leyendo el
    parquet real. Pegar la salida bajo ``datasets:`` y correr después el merge del
    diccionario (``merge_parquet_catalog.py``), que reinyecta la capa curada.
    """
    return "\n".join(ds.catalog_entry() for ds in datasets)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )


def month_start(frame, year_col: str = "Año", month_col: str = "Mes"):
    """Serie ``Fecha`` = primer día del mes a partir de las columnas Año/Mes.

    Las planillas del INE no traen fecha: traen ``Año`` y ``Mes`` como enteros.
    Todo el informe cruza por mes (analíticos del BCCh, expectativas del
    operador), así que la fecha canónica es el día 1 — la misma convención que
    usa ``data.cargar_expectativas`` del publicador original.
    """
    import pandas as pd

    return pd.to_datetime(
        dict(year=frame[year_col].astype(int), month=frame[month_col].astype(int), day=1)
    )
