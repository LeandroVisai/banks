#!/usr/bin/env python3
"""Excel -> parquets, para los DOS informes (IPC y Cambiario AM).

Un solo script porque el trabajo es el mismo: abrir planillas, resolver
encabezados sucios y dejar un parquet tidy por dataset en
``data_pipeline/parquet/``. Lo que cambia por informe es qué planillas se abren,
y eso vive en ``load_ipc()`` / ``load_cambiario()``.

    python scripts/ingest/from_excel.py --informe ipc -v
    python scripts/ingest/from_excel.py --informe cambiario -v
    python scripts/ingest/from_excel.py --informe ambos --out data_pipeline/parquet
    python scripts/ingest/from_excel.py --informe ipc --emit-catalog   # YAML, sin datos

IPC - qué es FIJO y qué es VARIABLE
-----------------------------------
Las cuatro bases del INE tienen el MISMO esquema (16 columnas) y períodos
disjuntos, así que se concatenan en una sola canasta 2009->hoy:

    ipc2008.xlsx  hoja "Historia"   2009-01 .. 2013-12   FIJO     (base cerrada)
    ipc2013.xlsx                    2014-01 .. 2018-12   FIJO     (base cerrada)
    ipc2018.xlsx                    2019-01 .. 2023-12   FIJO     (base cerrada)
    ipc2023.xlsx                    2024-01 .. hoy       VARIABLE (crece cada mes)
    diccionario ipc 2023.xlsx       286 productos        FIJO     (vive con la canasta 2023)
    entrada_operador/Expectativas   últimos meses        VARIABLE (lo llena el operador)
    outputs/Output IPC BI.xlsx      analíticos del BCCh  VARIABLE (lo deja el notebook)

Las tres bases cerradas y el diccionario se copian UNA vez y no se vuelven a
tocar: son historia publicada, el INE no las reedita. Lo único que hay que
reemplazar cada mes es ``ipc2023.xlsx`` (la descarga nueva del INE) y el Excel
del operador. El ``skiprows`` difiere entre bases (2 en 2008/2013, 3 en
2018/2023) y los encabezados viejos traen ``" ( % )"`` con espacios de más; las
dos cosas se normalizan acá, no en el consumidor.

Cambiario - el Excel ya NO es la planilla entera
------------------------------------------------
El informe original leía la hoja ``Datos`` del ``.xlsm`` (104 columnas), pero esa
hoja es en su mayoría un volcado del DW hecho con Power Query. ``datos_lea.xlsx``
es el recorte de lo que SOLO existe en Bloomberg y el DW no tiene: 23 columnas
(inventarios y posiciones de cobre, los cuatro RSI, términos de intercambio,
volatilidad implícita 1W, expectativas de TPM, tasas implícitas y el TCR). El
resto lo trae ``from_sql.py``.

Los encabezados de ese recorte perdieron el prefijo de sección del ``.xlsm``
(``"RSI RSI DXY"`` quedó ``"RSI DXY"``), así que ``_ALIAS_CAMBIARIO`` los devuelve
al nombre que esperan los builders. Los builders NO se reescriben: se importan de
``build_cambiario_parquets.py`` tal cual, que es donde ya están probados contra
el Excel real. El mapa se verificó POR VALOR contra un volcado de la hoja ``Datos``
(``paquete_cambiario/xlsx de parquets/df1.xlsx``): las 23 columnas coinciden.

De esa misma verificación salió un hallazgo que ``_DUPLICAR_CAMBIARIO`` aprovecha:
"Volúmen Futuros Londres" no es un volumen sino el precio del cobre LME, idéntico
byte a byte a "Drivers Precio Londres".

Datasets MIXTOS (Excel + DW en el mismo parquet)
------------------------------------------------
Varios datasets del cambiario cruzan una columna de Bloomberg con una del DW
(``cam_tot_gs`` = términos de intercambio + CLP; ``cam_inventarios_comex`` =
inventarios + precio HGA). Ninguno de los dos scripts los puede armar solo, así
que cada uno deja su aporte en ``--stage-dir`` (por defecto ``_stage/`` dentro de
la carpeta de salida) y ambos, antes de construir, juntan por ``Fecha`` todo lo
que haya staged.

**Quién manda**: no el que corrió primero, sino ``sources.PRIORIDAD_CAMBIARIO``,
donde ``datos_lea.xlsx`` va antes que el DW y que cualquier serie derivada. Es la
planilla que mantiene el DOMA y a la que se le van a agregar las columnas que
faltan; cuando una llegue, pisa sola a la alternativa sin tocar código.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources  # noqa: E402

log = logging.getLogger("ingest.excel")

_ROOT = sources.ROOT
_IPC_DIR = _ROOT / "Nuevo_Repo" / "IPC" / "Post IPC"
_CAM_XLSX = _ROOT / "Nuevo_Repo" / "cambiario am" / "data" / "datos_lea.xlsx"

TS, D, V = sources.TS, sources.D, sources.V


# =============================================================================
# CAMBIARIO - recorte de Bloomberg (datos_lea.xlsx)
# =============================================================================

# encabezado en datos_lea.xlsx  ->  encabezado del .xlsm que esperan los builders.
# El recorte le sacó el prefijo de sección a cada columna; sin este mapa los diez
# datasets que dependen de ellas fallan con MissingColumnError aunque el dato esté.
_ALIAS_CAMBIARIO = {
    "Inventarios Comex":            "Drivers Inventarios Comex",
    "Inv. Cobre":                   "Drivers Inv. Cobre",
    "Volúmen Futuros Londres":      "Drivers Volúmen Futuros Londres",
    "Volatilidad Futuros Londres":  "Drivers Volatilidad Futuros Londres",
    "Posiciones Cobre":             "Drivers Posiciones Cobre",
    "RSI No residentes":            "RSI RSI Pos. OffShore",
    "RSI DXY":                      "RSI RSI DXY",
    "RSI CLP":                      "RSI RSI CLP",
    "RSI Cobre":                    "RSI RSI Cobre",
    "GS Terms of Trade":            "Términos de intercambio GS Terms of Trade",
    "CT Terms of Trade.":           "Términos de intercambio CT Terms of Trade.",
    "Volatilidad implicita Opciones 1W CLP": "Misceláneos Volatilidad implicita Opciones 1W CLP",
    "Volatilidad implicita Opciones 1W MXN": "Misceláneos Volatilidad implicita Opciones 1W MXN",
    "Volatilidad implicita Opciones 1W BRL": "Misceláneos Volatilidad implicita Opciones 1W BRL",
    "Volatilidad implicita Opciones 1W COP": "Misceláneos Volatilidad implicita Opciones 1W COP",
    "Expectativas de TPM US":       "Expectativas de TPM Expectativas de TPM US",
    "Expectativas de TPM CL":       "Expectativas de TPM Expectativas de TPM CL",
    # "Implicita <país>" y "Tipo de cambio real CLP" ya vienen con el nombre final.
}

# Columnas del recorte que el ``.xlsm`` guardaba DOS veces con nombres distintos.
# Se copian, no se renombran: los builders piden los dos nombres.
#
# "Volúmen Futuros Londres" NO es un volumen: su ticker es ``LMCADY LME Comdty``
# —el cobre LME al contado— y en el .xlsm es byte por byte la misma serie que
# "Drivers Precio Londres" (1.811 filas, 100% idénticas, max|dif| = 0). El
# rótulo quedó mal en la planilla de origen y se arrastró al recorte. Sin esta
# copia, cam_inventarios_lme y cam_cobre_comex_lme se saltaban por "falta
# Drivers Precio Londres" teniendo el dato adelante.
_DUPLICAR_CAMBIARIO = {
    "Drivers Volúmen Futuros Londres": "Drivers Precio Londres",
}


def load_cambiario(path: Path) -> dict:
    """La hoja del recorte, con los encabezados devueltos al nombre del ``.xlsm``.

    ``skiprows=3``: la planilla trae dos filas de rango de fechas arriba, el
    encabezado en la 4 y los tickers de Bloomberg en la 5. Esa fila de tickers
    queda como primera fila de datos y se descarta sola en el ``dropna`` de la
    fecha, que no parsea.
    """
    import pandas as pd

    df = pd.read_excel(path, sheet_name=0, skiprows=3)
    df = df.rename(columns={df.columns[0]: "Fecha"})
    df = df.rename(columns=_ALIAS_CAMBIARIO)
    for origen, copia in _DUPLICAR_CAMBIARIO.items():
        if origen in df.columns and copia not in df.columns:
            df[copia] = df[origen]
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    log.info("Excel cambiario - %s: %d filas x %d columnas", path.name, len(df), len(df.columns))

    # Las hojas opcionales del recorte. Hoy datos_lea.xlsx trae una sola, pero el
    # plan es que sume "Intradía" y "CarryTrade" (los dos únicos orígenes que le
    # faltan al informe además de la vela y las curvas de contratos). Cuando
    # aparezcan, se leen solas: cam_clp_intradia y cam_carry_trade ya existen.
    out = {"datos": df}
    for clave, hoja in (("intradia", "Intradía"), ("carry", "CarryTrade")):
        try:
            out[clave] = pd.read_excel(path, sheet_name=hoja)
            log.info("Excel cambiario - hoja %s: %d filas", hoja, len(out[clave]))
        except Exception:
            log.info("Excel cambiario - sin hoja %s (opcional)", hoja)
    return out


# =============================================================================
# IPC - las planillas del INE, el diccionario y lo que llena el operador
# =============================================================================

# archivo -> (hoja, skiprows, etiqueta de base). Las tres primeras son historia
# cerrada; solo la última cambia de un mes a otro.
_BASES_INE = [
    ("data/ipc2008.xlsx", "Historia", 2, "2008"),
    ("data/ipc2013.xlsx", 0,          2, "2013"),
    ("data/ipc2018.xlsx", 0,          3, "2018"),
    ("data/ipc2023.xlsx", 0,          3, "2023"),
]

# La base VIGENTE: la única de las cuatro que cambia de un mes a otro. Las otras
# tres son historia cerrada y se cachean en un parquet (load_canasta_fija).
_BASE_VIGENTE = "2023"

# Nombre del parquet que congela las tres bases cerradas. Vive en la carpeta de
# salida junto al resto, así el catálogo lo ve como un dataset más.
_CACHE_FIJO = "ipc_canasta_base_cerrada.parquet"

# Las bases 2008 y 2013 escriben "Variación Mensual ( % )"; las nuevas, sin
# espacios de más. Se unifica al nombre nuevo para poder concatenarlas.
_RENAME_INE = {
    "Variación Mensual ( % )":    "Variación Mensual (%)",
    "Variación Acumulada ( % )":  "Variación Acumulada (%)",
    "Variación 12 Meses ( % )":   "Variación 12 Meses (%)",
    "Incidencia Mensual ( % )":   "Incidencia Mensual (%)",
    "Incidencia Acumulada ( % )": "Incidencia Acumulada (%)",
    "Incidencia 12 Meses ( % )":  "Incidencia 12 Meses (%)",
}

_COLS_CANASTA = [
    "Fecha", "Año", "Mes", "Base", "División", "Grupo", "Clase", "Subclase",
    "Producto", "Glosa", "Ponderación", "Índice",
    "Variación Mensual (%)", "Variación Acumulada (%)", "Variación 12 Meses (%)",
    "Incidencia Mensual (%)", "Incidencia Acumulada (%)", "Incidencia 12 Meses (%)",
]

# Ventana de la banda histórica de difusión. Excluye 2022+ a propósito: el shock
# post-pandemia deformaría la banda de comportamiento "normal". El informe Excel
# del banco usa 2010-2021 y acá se respeta, acotado a lo que la canasta alcanza.
_BANDA_DESDE, _BANDA_HASTA = 2010, 2021

# Hojas del Output IPC BI.xlsx que el informe consume. Son series del BCCh que el
# notebook baja por API (bcchapi) y deja acá: no salen de ninguna planilla del INE
# ni del DW, así que este Excel es su único origen offline.
_HOJAS_BI = {
    "analiticos": "IPC Analiticos",
    "volatiles":  "IPC Volatiles SV BCCh",
    "grupos":     "Variacion IPC por Grupo",
}


def _leer_base(root: Path, rel: str, sheet, skiprows: int, base: str):
    import pandas as pd

    path = root / rel
    if not path.exists():
        log.warning("IPC - falta %s (la base %s queda fuera de la canasta)", rel, base)
        return None
    df = pd.read_excel(path, sheet_name=sheet, skiprows=skiprows).rename(columns=_RENAME_INE)
    faltan = [c for c in ("Año", "Mes", "Glosa") if c not in df.columns]
    if faltan:
        log.warning("IPC - %s no trae %s; se descarta", rel, faltan)
        return None
    df = df[df["Año"].notna() & df["Mes"].notna()].copy()
    df["Base"] = base
    df["Fecha"] = sources.month_start(df)
    log.info("IPC - %s: %d filas (%s .. %s)", rel, len(df),
             df["Fecha"].min().date(), df["Fecha"].max().date())
    return df


def _marcar(canasta):
    """Las dos columnas derivadas que usa todo lo que consume la canasta."""
    # "Producto" es el CÓDIGO de producto: las filas sin él son agregados
    # (división, grupo, clase), no productos de la canasta. La difusión se mide
    # sobre productos, así que la marca se calcula una sola vez, acá.
    canasta["_es_producto"] = (
        canasta["Producto"].notna() & (canasta["Producto"] != "") & (canasta["Producto"] != 0)
    )
    canasta["Glosa_norm"] = canasta["Glosa"].astype(str).str.strip().str.upper()
    return canasta.sort_values(["Fecha", "División", "Grupo"]).reset_index(drop=True)


def load_canasta_fija(root: Path, cache: Path, *, rebuild: bool = False):
    """Las TRES bases cerradas del INE (2009-01 .. 2023-12), cacheadas en parquet.

    2008, 2013 y 2018 son historia publicada: el INE no las reedita, así que
    parsear 16 MB de Excel cada mes para obtener siempre las mismas 110.280 filas
    es trabajo tirado. La primera corrida las lee y deja
    ``ipc_canasta_base_cerrada.parquet``; las siguientes levantan ese parquet en
    vez de los Excel. ``--rebuild-fijo`` fuerza la relectura — que es lo que hay
    que correr el día que el INE empalme una base nueva (2028) o corrija historia.
    """
    if cache.exists() and not rebuild:
        fija = sources.read_parquet(cache)
        log.info("IPC - base cerrada desde cache: %s (%d filas)", cache.name, len(fija))
        return fija

    import pandas as pd

    partes = [_leer_base(root, rel, sh, sk, base)
              for rel, sh, sk, base in _BASES_INE if base != _BASE_VIGENTE]
    partes = [p for p in partes if p is not None]
    if not partes:
        return None
    fija = pd.concat(partes, ignore_index=True)[_COLS_CANASTA]
    cache.parent.mkdir(parents=True, exist_ok=True)
    sources.to_parquet(fija, cache)
    log.info("IPC - base cerrada RECONSTRUIDA -> %s (%d filas, %s .. %s)", cache.name, len(fija),
             fija["Fecha"].min().date(), fija["Fecha"].max().date())
    return fija


def load_ipc(root: Path, *, cache_fijo: Path, rebuild_fijo: bool = False) -> dict:
    """Las fuentes del IPC: canasta del INE, diccionario, operador y Output BI.

    La canasta entra partida en dos, que es como se comporta el dato:

        canasta_fija      2009-01 .. 2023-12   bases cerradas, del parquet cacheado
        canasta_vigente   2024-01 .. hoy       ipc2023.xlsx, se reemplaza cada mes
        canasta           las dos juntas       lo que consumen los derivados

    Un archivo que falte no aborta: los datasets que dependan de él se saltan con
    warning y el resto se genera igual (la misma política que el resto del repo).
    """
    import pandas as pd

    out: dict = {}

    fija = load_canasta_fija(root, cache_fijo, rebuild=rebuild_fijo)
    vigente = next((_leer_base(root, rel, sh, sk, base)
                    for rel, sh, sk, base in _BASES_INE if base == _BASE_VIGENTE), None)

    if fija is not None:
        out["canasta_fija"] = _marcar(fija.copy())
    if vigente is not None:
        out["canasta_vigente"] = _marcar(vigente[_COLS_CANASTA].copy())

    partes = [p for p in (fija, vigente) if p is not None]
    if partes:
        canasta = pd.concat([p[_COLS_CANASTA] for p in partes], ignore_index=True)
        out["canasta"] = _marcar(canasta)
        log.info("IPC - canasta consolidada: %d filas (%s .. %s)", len(canasta),
                 out["canasta"]["Fecha"].min().date(), out["canasta"]["Fecha"].max().date())

    dicc_path = root / "data" / "diccionario ipc 2023.xlsx"
    if dicc_path.exists():
        out["dicc"] = pd.read_excel(dicc_path)
        log.info("IPC - diccionario: %d filas", len(out["dicc"]))
    else:
        log.warning("IPC - falta %s", dicc_path.name)

    exp_path = root / "data" / "entrada_operador" / "Expectativas IPC.xlsx"
    if exp_path.exists():
        try:
            x = pd.ExcelFile(exp_path)
            if "Esperado vs Efectivo" in x.sheet_names:
                out["expectativas"] = pd.read_excel(x, sheet_name="Esperado vs Efectivo")
            if "Divisiones" in x.sheet_names:
                out["exp_divisiones"] = pd.read_excel(x, sheet_name="Divisiones")
            log.info("IPC - operador: %s",
                     ", ".join(k for k in ("expectativas", "exp_divisiones") if k in out))
        except OSError as exc:
            # Pasa si el operador lo dejó abierto en Excel: se avisa y se sigue.
            log.warning("IPC - no pude leer %s (%s). ¿Está abierto en Excel?",
                        exp_path.name, type(exc).__name__)
    else:
        log.warning("IPC - falta %s (las figuras de expectativas quedan sin datos)", exp_path.name)

    bi_path = root / "outputs" / "Output IPC BI.xlsx"
    if bi_path.exists():
        x = pd.ExcelFile(bi_path)
        for key, hoja in _HOJAS_BI.items():
            if hoja in x.sheet_names:
                out[key] = pd.read_excel(x, sheet_name=hoja)
        log.info("IPC - Output BI: %s", ", ".join(k for k in _HOJAS_BI if k in out))
    else:
        log.warning("IPC - falta outputs/Output IPC BI.xlsx (analíticos del BCCh sin origen)")

    return out


# -- Registro de datasets del IPC ---------------------------------------------

IPC = sources.Registry()

_COLS_CANASTA_YAML = [
    ("Fecha", TS), ("Año", D), ("Mes", D), ("Base", V), ("División", D), ("Grupo", D),
    ("Clase", D), ("Subclase", D), ("Producto", D), ("Glosa", V), ("Ponderación", D),
    ("Índice", D), ("Variación Mensual (%)", D), ("Variación Acumulada (%)", D),
    ("Variación 12 Meses (%)", D), ("Incidencia Mensual (%)", D),
    ("Incidencia Acumulada (%)", D), ("Incidencia 12 Meses (%)", D),
]


@IPC.dataset(
    ds_id="ipc_canasta_base_cerrada", chart_type="market_monitor_table", segment="ipc",
    name="Canasta IPC de las bases cerradas (2009-2023)",
    description="Canasta del INE a nivel de producto de las tres bases ya cerradas "
                "(2008, 2013 y 2018), empalmadas. Historia publicada: NO cambia mes a mes. "
                "Se regenera solo con --rebuild-fijo, el día que el INE empalme una base nueva.",
    unit="%", columns=_COLS_CANASTA_YAML, needs=("canasta_fija",),
)
def _canasta_base_cerrada(src):
    return src["canasta_fija"][_COLS_CANASTA].copy()


@IPC.dataset(
    ds_id="ipc_canasta_vigente", chart_type="market_monitor_table", segment="ipc",
    name="Canasta IPC de la base vigente (2023=100)",
    description="Canasta del INE a nivel de producto de la base 2023, desde 2024-01 hasta el "
                "último mes publicado. Es el único archivo del INE que hay que reemplazar cada "
                "mes.",
    unit="%", columns=_COLS_CANASTA_YAML, needs=("canasta_vigente",),
)
def _canasta_vigente(src):
    return src["canasta_vigente"][_COLS_CANASTA].copy()


@IPC.dataset(
    ds_id="ipc_canasta_historica", chart_type="market_monitor_table", segment="ipc",
    name="Canasta IPC consolidada 2009-hoy",
    description="Canasta del INE a nivel de producto, con las cuatro bases (2008/2013/2018/2023) "
                "empalmadas por Año-Mes. Es la fuente de la difusión y de la variación histórica "
                "del mes. Equivale a ipc_canasta_base_cerrada + ipc_canasta_vigente, y existe "
                "aparte porque cada transform lee UN solo parquet.",
    unit="%", columns=_COLS_CANASTA_YAML, needs=("canasta",),
)
def _canasta_historica(src):
    return src["canasta"][_COLS_CANASTA].copy()


def _difusion_de(validos):
    """Difusión mensual sobre un subconjunto de productos ya filtrado."""
    d = validos.copy()
    d["Variación Mensual Positiva"] = d["Variación Mensual (%)"] > 0
    pos = d.groupby(["Año", "Mes"], observed=True)["Variación Mensual Positiva"].sum().reset_index()
    tot = d.groupby(["Año", "Mes"], observed=True).size().reset_index(name="Total Productos")
    out = pos.merge(tot, on=["Año", "Mes"])
    out["Difusión (%)"] = (out["Variación Mensual Positiva"] / out["Total Productos"] * 100).round(2)
    out["Fecha"] = sources.month_start(out)
    cols = ["Fecha", "Año", "Mes", "Variación Mensual Positiva", "Total Productos", "Difusión (%)"]
    return out[cols].sort_values("Fecha").reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_difusion", chart_type="line", segment="ipc",
    name="Difusión inflacionaria del IPC general",
    description="Porcentaje de productos de la canasta con variación mensual positiva, por mes. "
                "Se mide sobre productos (filas con código de producto), no sobre agregados.",
    unit="%", value_kind="ratio",
    columns=[("Fecha", TS), ("Año", D), ("Mes", D), ("Variación Mensual Positiva", D),
             ("Total Productos", D), ("Difusión (%)", D)],
    needs=("canasta",),
)
def _difusion(src):
    return _difusion_de(src["canasta"][src["canasta"]["_es_producto"]])


@IPC.dataset(
    ds_id="ipc_difusion_banda", chart_type="hist_range", segment="ipc",
    name="Difusión del mes contra su banda histórica",
    description=f"Para cada mes calendario: la banda de difusión histórica "
                f"({_BANDA_DESDE}-{_BANDA_HASTA}) y el valor del año en curso. La ventana "
                "excluye 2022 en adelante a propósito: el shock post-pandemia deformaría la "
                "banda de comportamiento normal.",
    unit="%", value_kind="ratio", columns=sources.COLS_RANGO, needs=("canasta",),
)
def _difusion_banda(src):
    dif = _difusion_de(src["canasta"][src["canasta"]["_es_producto"]])
    # min-max, no p10-p90: el notebook dibuja la banda de difusión con los
    # extremos absolutos de la ventana (agg min/max), a diferencia de las bandas
    # de bienes y servicios sin volátiles, que sí usan percentiles.
    return sources.banda_mensual(dif, "Difusión (%)", desde=_BANDA_DESDE, hasta=_BANDA_HASTA,
                          cuantiles=(0.0, 1.0))


def _clasificacion(dicc):
    """``Glosa_norm -> (Clasificación, Ponderación 2023)`` del diccionario oficial.

    Las columnas se toman POR POSICIÓN, no por nombre: los encabezados del Excel
    del INE traen tildes y saltos de línea que cambian entre descargas. Las
    últimas filas son totales (conteo y ponderación) y se filtran por código.
    """
    import numpy as np

    c_nombre = dicc.columns[2]   # PRODUCTO IPC 2023=100
    c_codigo = dicc.columns[1]   # CÓDIGO IPC 2023=100
    c_pond = dicc.columns[4]     # POND 2023=100
    c_vol = dicc.columns[16]     # IPC VOLÁTILES
    return (dicc[dicc[c_codigo].notna()]
            .assign(Glosa_norm=lambda d: d[c_nombre].astype(str).str.strip().str.upper(),
                    Clasificación=lambda d: np.where(d[c_vol] == 1, "Volátil", "Sin volátiles"))
            .rename(columns={c_pond: "Ponderación 2023"})
            [["Glosa_norm", "Clasificación", "Ponderación 2023"]])


@IPC.dataset(
    ds_id="ipc_canasta_mes", chart_type="market_monitor_table", segment="ipc",
    name="Canasta del último mes por producto",
    description="Todos los productos del último mes publicado con su variación e incidencia "
                "(mensual y 12 meses), clasificados en Volátil / Sin volátiles según el "
                "diccionario IPC 2023. Es la tabla de detalle que cierra el informe.",
    unit="%",
    columns=[("Fecha", TS), ("Glosa", V), ("Variación Mensual (%)", D),
             ("Incidencia Mensual (%)", D), ("Variación 12 Meses (%)", D),
             ("Incidencia 12 Meses (%)", D), ("Clasificación", V), ("Ponderación 2023", D)],
    needs=("canasta", "dicc"),
)
def _canasta_mes(src):
    validos = src["canasta"][src["canasta"]["_es_producto"]]
    ultimo = validos["Fecha"].max()
    cols = ["Glosa", "Glosa_norm", "Variación Mensual (%)", "Incidencia Mensual (%)",
            "Variación 12 Meses (%)", "Incidencia 12 Meses (%)"]
    out = (validos.loc[validos["Fecha"] == ultimo, cols]
           .merge(_clasificacion(src["dicc"]), on="Glosa_norm", how="left")
           .drop(columns="Glosa_norm")
           .sort_values("Incidencia Mensual (%)", ascending=False)
           .reset_index(drop=True))
    sin_match = int(out["Clasificación"].isna().sum())
    if sin_match:
        log.warning("ipc_canasta_mes - %d producto(s) sin match en el diccionario", sin_match)
    out.insert(0, "Fecha", ultimo)
    return out


@IPC.dataset(
    ds_id="ipc_difusion_sv", chart_type="line", segment="ipc",
    name="Difusión inflacionaria sin volátiles",
    description="La misma difusión que ipc_difusion pero restringida a los productos que el "
                "diccionario IPC 2023 marca como no volátiles.",
    unit="%", value_kind="ratio",
    columns=[("Fecha", TS), ("Año", D), ("Mes", D), ("Variación Mensual Positiva", D),
             ("Total Productos", D), ("Difusión (%)", D)],
    needs=("canasta", "dicc"),
)
def _difusion_sv(src):
    clas = _clasificacion(src["dicc"])
    sin_vol = set(clas.loc[clas["Clasificación"] == "Sin volátiles", "Glosa_norm"])
    validos = src["canasta"][src["canasta"]["_es_producto"]]
    return _difusion_de(validos[validos["Glosa_norm"].isin(sin_vol)])


# Nombre corto de cada división del IPC, por código. Las trece divisiones son la
# clasificación COICOP y no cambian; las glosas del INE llegan en mayúsculas y de
# hasta 60 caracteres ("BIENES Y SERVICIOS DIVERSOS", "MUEBLES, ARTÍCULOS PARA EL
# HOGAR Y PARA LA CONSERVACIÓN ORDINARIA DEL HOGAR"), que en el eje X de un
# gráfico de barras se solapan hasta ser ilegibles. Son las mismas abreviaturas
# que usa el operador en la hoja "Divisiones" de su Excel.
_DIVISION_CORTA = {
    1: "Alimentos",         2: "Alcohol y tabaco",  3: "Vestuario",
    4: "Vivienda",          5: "Eq. del hogar",     6: "Salud",
    7: "Transporte",        8: "Comunicaciones",    9: "Rec. y cultura",
    10: "Educación",        11: "Rest. y hoteles",  12: "Bs. y ss. diversos",
    13: "Seguros y ss. financieros",
}


def _nombre_corto(division) -> str:
    try:
        return _DIVISION_CORTA[int(division)]
    except (TypeError, ValueError, KeyError):
        return str(division)


def _divisiones_ultimo_mes(canasta):
    """Las filas de DIVISIÓN del último mes publicado.

    Una división es la fila que trae ``División`` y NO trae ``Grupo``: los niveles
    inferiores (grupo, clase, producto) repiten el código de división.
    """
    ultimo = canasta["Fecha"].max()
    div = canasta[(canasta["Fecha"] == ultimo)
                  & canasta["División"].notna()
                  & canasta["Grupo"].isna()].copy()
    div["División"] = div["División"].astype(int)
    return div


@IPC.dataset(
    ds_id="ipc_difusion_comparada", chart_type="line", segment="ipc",
    name="Difusión: IPC general vs sin volátiles",
    description="Las dos series de difusión en el mismo gráfico: el porcentaje de productos al "
                "alza sobre la canasta completa y sobre la canasta sin volátiles. La brecha "
                "entre ambas dice cuánto de la difusión viene de precios volátiles.",
    unit="%", value_kind="ratio",
    columns=[("Fecha", TS), ("General", D), ("Sin volátiles", D)],
    needs=("canasta", "dicc"),
)
def _difusion_comparada(src):
    """Las dos difusiones juntas: cada transform lee UN parquet, así que si el
    gráfico compara dos series, las dos tienen que viajar en el mismo archivo."""
    general = _difusion(src)[["Fecha", "Difusión (%)"]].rename(
        columns={"Difusión (%)": "General"})
    sv = _difusion_sv(src)[["Fecha", "Difusión (%)"]].rename(
        columns={"Difusión (%)": "Sin volátiles"})
    return general.merge(sv, on="Fecha", how="outer").sort_values("Fecha").reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_divisiones_mes", chart_type="grouped_bar", segment="ipc",
    name="Divisiones del IPC en el último mes",
    description="Las 13 divisiones del último mes publicado con su ponderación, variación e "
                "incidencia mensual. Es el efectivo contra el que se compara la expectativa "
                "del mercado por división.",
    unit="%",
    columns=[("Fecha", TS), ("División", D), ("Nombre", V), ("Ponderación", D),
             ("Variación", D), ("Incidencia", D)],
    needs=("canasta",),
)
def _divisiones_mes(src):
    div = _divisiones_ultimo_mes(src["canasta"])
    div = div[["Fecha", "División", "Glosa", "Ponderación",
               "Variación Mensual (%)", "Incidencia Mensual (%)"]]
    div = div.rename(columns={"Glosa": "Nombre", "Variación Mensual (%)": "Variación",
                              "Incidencia Mensual (%)": "Incidencia"})
    div["Nombre"] = div["División"].map(_nombre_corto)
    return div.sort_values("División").reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_incidencia_12m_division", chart_type="grouped_bar", segment="ipc",
    name="Incidencia 12 meses por división",
    description="Incidencia acumulada en 12 meses de cada división en el último mes publicado, "
                "con su variación 12m y su ponderación.",
    unit="pp",
    columns=[("Fecha", TS), ("División", D), ("Nombre", V), ("Ponderación", D),
             ("Variación 12 Meses (%)", D), ("Incidencia 12 Meses (%)", D)],
    needs=("canasta",),
)
def _incidencia_12m_division(src):
    div = _divisiones_ultimo_mes(src["canasta"])
    div = div[["Fecha", "División", "Glosa", "Ponderación",
               "Variación 12 Meses (%)", "Incidencia 12 Meses (%)"]]
    div = div.rename(columns={"Glosa": "Nombre"})
    div["Nombre"] = div["División"].map(_nombre_corto)
    return div.sort_values("Incidencia 12 Meses (%)", ascending=False).reset_index(drop=True)


def _ipc_general(canasta):
    """La serie mensual del IPC general (Fecha, Año, Mes, Variación Mensual (%)).

    La canasta trae una fila ``Glosa == 'IPC General'`` con el total; es la que usa
    el notebook. Si una descarga del INE no la trajera, se reconstruye sumando las
    incidencias mensuales de los productos, que por definición de incidencia suman
    la variación del general.
    """
    general = canasta[canasta["Glosa_norm"] == "IPC GENERAL"]
    if not general.empty:
        return (general[["Fecha", "Año", "Mes", "Variación Mensual (%)"]]
                .sort_values("Fecha").reset_index(drop=True))
    log.warning("la canasta no trae la fila 'IPC General'; se reconstruye por incidencias")
    validos = canasta[canasta["_es_producto"]]
    out = (validos.groupby(["Fecha", "Año", "Mes"], observed=True)["Incidencia Mensual (%)"]
           .sum().reset_index()
           .rename(columns={"Incidencia Mensual (%)": "Variación Mensual (%)"}))
    out["Variación Mensual (%)"] = out["Variación Mensual (%)"].round(2)
    return out.sort_values("Fecha").reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_variacion_historica_mes", chart_type="grouped_bar", segment="ipc",
    name="Variación histórica del IPC en el mes publicado",
    description="Variación del IPC general en el MISMO mes calendario recién publicado, año por "
                "año, con el promedio histórico y el promedio que excluye 2020-2022 (los años "
                "del shock). Responde si el dato del mes es alto o bajo para ese mes del año.",
    unit="%",
    columns=[("Año", V), ("Variación Mensual (%)", D), ("Promedio Histórico", D),
             ("Promedio Histórico Ex", D)],
    needs=("canasta",),
)
def _variacion_historica_mes(src):
    """Réplica de ``fig_prom`` del notebook, con una diferencia deliberada.

    El notebook elige el mes con ``datetime.now().month - 1``, que depende del día
    en que se corra —y en enero da 0, que no es un mes—. Acá el mes sale del DATO:
    el último publicado en la canasta. Para la corrida normal (se publica el mes
    anterior) es el mismo resultado, pero no se rompe si el informe se rehace
    tarde, ni en el cambio de año.
    """
    serie = _ipc_general(src["canasta"])
    mes = int(serie["Mes"].iloc[-1])
    d = serie[serie["Mes"] == mes].copy()
    if d.empty:
        return None
    d["Promedio Histórico"] = round(float(d["Variación Mensual (%)"].mean()), 2)
    # El promedio "ex" saca 2020-2022: pandemia y shock inflacionario posterior.
    # Es la referencia que el informe usa para decir si el mes fue normal.
    ex = d[(d["Año"] < 2020) | (d["Año"] > 2022)]["Variación Mensual (%)"]
    d["Promedio Histórico Ex"] = round(float(ex.mean()), 2) if not ex.empty else float("nan")
    d["Año"] = d["Año"].astype(int).astype(str)
    return (d[["Año", "Variación Mensual (%)", "Promedio Histórico", "Promedio Histórico Ex"]]
            .reset_index(drop=True))


@IPC.dataset(
    ds_id="ipc_canasta_treemap", chart_type="treemap", segment="ipc",
    name="Canasta del mes por división y grupo",
    description="La canasta del último mes en dos niveles: división (exterior) y grupo de "
                "productos (interior). El ÁREA de cada rectángulo es su ponderación en la "
                "canasta; el COLOR es su variación mensual (rojo sube, azul baja). Réplica del "
                "treemap del informe original, sin bajar al nivel de producto individual (283 "
                "ítems no se leen legibles en un solo gráfico de correo; división+grupo son "
                "13+46 y sí).",
    unit="%", value_kind="ratio",
    columns=[("Division", V), ("Grupo", V), ("Ponderacion", D), ("Variacion", D)],
    needs=("canasta",),
)
def _canasta_treemap(src):
    """Una fila por (división, grupo) del último mes, con su Glosa YA agregada.

    El INE publica el nivel "grupo" con su propia fila (División y Grupo
    presentes, Clase vacía) que TRAE su ponderación y variación ya sumadas —no
    hace falta promediar productos acá, el dato agregado sale directo de la
    canasta tal como la publica el INE.
    """
    c = src["canasta"]
    ultimo = c["Fecha"].max()
    grp = c[(c["Fecha"] == ultimo) & c["División"].notna() & c["Grupo"].notna()
            & c["Clase"].isna()].copy()
    if grp.empty:
        return None
    # Frame nuevo, no rename(): la fila trae DOS columnas que se llamarían
    # "Grupo" si se renombrara sobre el original (el código numérico de grupo,
    # que ya no hace falta, y la Glosa, que es la que sí queremos como "Grupo")
    # — pandas no las fusiona, quedan duplicadas y el parquet sale corrupto.
    out = grp[["Ponderación", "Variación Mensual (%)"]].rename(
        columns={"Ponderación": "Ponderacion", "Variación Mensual (%)": "Variacion"})
    out["Division"] = grp["División"].map(_nombre_corto)
    out["Grupo"] = grp["Glosa"]
    out = out[out["Ponderacion"].notna() & (out["Ponderacion"] > 0)]
    orden = (out.groupby("Division")["Ponderacion"].sum()
             .sort_values(ascending=False).index.tolist())
    out["_o"] = out["Division"].map({d: i for i, d in enumerate(orden)})
    return (out.sort_values(["_o", "Ponderacion"], ascending=[True, False])
            .drop(columns="_o")[["Division", "Grupo", "Ponderacion", "Variacion"]]
            .round(4).reset_index(drop=True))


@IPC.dataset(
    ds_id="ipc_variaciones_mes", chart_type="grouped_bar", segment="ipc",
    name="Variación del mes por agregado analítico",
    description="Variación mensual de cada agregado analítico del BCCh (general, SAE, bienes, "
                "servicios, transables, alimentos, energía…) en el mes recién publicado, con el "
                "mes anterior al lado para leer el cambio.",
    unit="%",
    columns=[("Agregado", V), ("Mes anterior", D), ("Mes actual", D)],
    needs=("analiticos",),
)
def _variaciones_mes(src):
    """Los analíticos TRASPUESTOS: una fila por agregado, dos columnas de mes.

    El parquet de analíticos es ancho por serie y largo por fecha, que es lo
    correcto para las líneas. Este gráfico mira al revés —eje X = los agregados,
    dos barras por agregado— y como cada transform lee un solo parquet, la
    traspuesta se escribe acá en vez de resolverse al dibujar.
    """
    import pandas as pd

    a = src["analiticos"].copy()
    a["Fecha"] = pd.to_datetime(a["Fecha"], errors="coerce")
    a = a.dropna(subset=["Fecha"]).sort_values("Fecha")
    if len(a) < 2:
        return None
    ult, ant = a.iloc[-1], a.iloc[-2]
    agregados = [c for c in a.columns if c != "Fecha"]
    return pd.DataFrame({
        "Agregado": [c.replace("IPC ", "") for c in agregados],
        "Mes anterior": [pd.to_numeric(ant[c], errors="coerce") for c in agregados],
        "Mes actual": [pd.to_numeric(ult[c], errors="coerce") for c in agregados],
    }).dropna(subset=["Mes actual"]).reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_diccionario", chart_type="market_monitor_table", segment="ipc",
    name="Diccionario IPC 2023: clasificación de la canasta",
    description="Clasificación oficial de cada producto de la canasta 2023 (volátil / sin "
                "volátiles) con su ponderación. Fijo mientras rija la canasta 2023.",
    unit="",
    columns=[("Glosa_norm", V), ("Clasificación", V), ("Ponderación 2023", D)],
    needs=("dicc",),
)
def _diccionario(src):
    return _clasificacion(src["dicc"]).reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_expectativas", chart_type="grouped_bar", segment="ipc",
    name="IPC esperado por fuente vs efectivo",
    description="Variación mensual del IPC que esperaba cada fuente de mercado (Seguros, EOF, "
                "EEE, Bloomberg). Lo carga el operador antes de publicar el informe.",
    unit="%",
    columns=[("Fecha", TS), ("Seguros", D), ("EOF", D), ("EEE", D), ("Bloomberg", D)],
    needs=("expectativas",),
)
def _expectativas(src):
    import pandas as pd

    e = src["expectativas"].copy()
    e["Fecha"] = pd.to_datetime(e["Fecha"], errors="coerce")
    e = e.dropna(subset=["Fecha"])
    # A primer día de mes, para que cruce con los analíticos del BCCh.
    e["Fecha"] = e["Fecha"].dt.to_period("M").dt.to_timestamp()
    fuentes = [f for f in ("Seguros", "EOF", "EEE", "Bloomberg") if f in e.columns]
    return (e[["Fecha"] + fuentes].dropna(subset=fuentes, how="all")
            .sort_values("Fecha").reset_index(drop=True))


@IPC.dataset(
    ds_id="ipc_esperado_vs_efectivo", chart_type="grouped_bar", segment="ipc",
    name="IPC esperado por fuente vs efectivo",
    description="Variación mensual del IPC que esperaba cada fuente de mercado (Seguros, EOF, "
                "EEE, Bloomberg) y el dato efectivo del mes, superpuesto. Donde el efectivo cae "
                "fuera de las barras, el mes sorprendió.",
    unit="%",
    columns=[("Fecha", TS), ("Seguros", D), ("EOF", D), ("EEE", D), ("Bloomberg", D),
             ("Efectivo", D)],
    needs=("expectativas", "analiticos"),
)
def _esperado_vs_efectivo(src):
    """El cruce que en el informe original hacía el publicador al vuelo.

    Va PRECALCULADO en el parquet porque cada transform del repo lee un solo
    dataset: si el esperado y el efectivo viven en parquets distintos, no hay
    forma de dibujarlos en el mismo gráfico. El join es por mes (día 1), que es la
    fecha canónica de todo lo mensual del informe.
    """
    import pandas as pd

    e = _expectativas(src)
    efectivo = src["analiticos"][["Fecha", "IPC General"]].copy()
    efectivo["Fecha"] = pd.to_datetime(efectivo["Fecha"], errors="coerce")
    efectivo = efectivo.dropna(subset=["Fecha"])
    efectivo["Fecha"] = efectivo["Fecha"].dt.to_period("M").dt.to_timestamp()
    efectivo = efectivo.rename(columns={"IPC General": "Efectivo"})
    # `how="left"`: el mes en curso ya tiene expectativa cargada pero todavía no
    # tiene dato, y así conserva sus barras con el efectivo vacío.
    out = e.merge(efectivo, on="Fecha", how="left")
    return out.tail(18).reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_incidencias_feedback", chart_type="hist_range", segment="ipc",
    name="Incidencia por división: rango de mercado vs efectivo",
    description="Por división, el rango de incidencia que esperaban las instituciones "
                "(mínimo-máximo), su promedio, y la incidencia efectiva que publicó el INE. "
                "Cuando el efectivo cae fuera del rango, esa división sorprendió.",
    unit="pp", columns=sources.COLS_RANGO, needs=("exp_divisiones", "canasta"),
)
def _incidencias_feedback(src):
    """Mismo esquema de rango que las bandas, con la división como categoría.

    El join es por glosa normalizada (mayúsculas, sin espacios de borde), igual que
    el resto del informe: el Excel del operador escribe los nombres de división a
    mano y no coinciden byte a byte con los del INE.
    """
    exp = _expectativas_divisiones(src)
    ine = _divisiones_ultimo_mes(src["canasta"])[["Glosa", "Incidencia Mensual (%)"]].copy()
    ine["Nombre"] = ine["Glosa"].astype(str).str.strip().str.upper()
    resumen = (exp[["Nombre", "Corto", "Minimo", "Maximo", "Promedio"]]
               .drop_duplicates(subset="Nombre"))
    d = resumen.merge(ine[["Nombre", "Incidencia Mensual (%)"]], on="Nombre", how="left")
    sin_match = int(d["Incidencia Mensual (%)"].isna().sum())
    if sin_match:
        log.warning("ipc_incidencias_feedback - %d división(es) del operador sin match en el INE",
                    sin_match)
    d["Categoria"] = d["Corto"].fillna(d["Nombre"])
    d = d.rename(columns={"Incidencia Mensual (%)": "Actual"})
    # Orden por magnitud de la incidencia efectiva: el gráfico se lee de mayor a
    # menor aporte, que es como lo muestra el informe.
    d = d.sort_values("Actual", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    d["Orden"] = range(len(d))
    return d[[c for c, _ in sources.COLS_RANGO]].round(4)


@IPC.dataset(
    ds_id="ipc_expectativas_divisiones", chart_type="hist_range", segment="ipc",
    name="Incidencia esperada por división y por institución",
    description="Incidencia que cada institución esperaba para cada división, con el rango "
                "(mínimo-máximo) y el promedio del mercado ya calculados.",
    unit="pp",
    columns=[("Nombre", V), ("Corto", V), ("Minimo", D), ("Maximo", D), ("Promedio", D),
             ("Institucion", V), ("Valor", D)],
    needs=("exp_divisiones",),
)
def _expectativas_divisiones(src):
    d = src["exp_divisiones"].copy()
    instituciones = [c for c in d.columns if c not in ("División", "Nombre corto")]
    d = d.dropna(subset=instituciones, how="all")
    d = d.rename(columns={"División": "Nombre", "Nombre corto": "Corto"})
    d["Nombre"] = d["Nombre"].astype(str).str.strip().str.upper()
    d["Minimo"] = d[instituciones].min(axis=1)
    d["Maximo"] = d[instituciones].max(axis=1)
    d["Promedio"] = d[instituciones].mean(axis=1)
    # Formato largo: una fila por (división, institución). Guardarlo ancho
    # obligaría a declarar en el catálogo una columna por institución, y esa lista
    # cambia cuando entra o sale una casa de estudios.
    largo = d.melt(id_vars=["Nombre", "Corto", "Minimo", "Maximo", "Promedio"],
                   value_vars=instituciones, var_name="Institucion", value_name="Valor")
    return largo.dropna(subset=["Valor"]).reset_index(drop=True)


def _hoja_bi(src, key, fecha_col="Fecha"):
    import pandas as pd

    df = src[key].copy()
    if fecha_col in df.columns:
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")
        df = df.dropna(subset=[fecha_col]).sort_values(fecha_col)
    return df.reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_analiticos", chart_type="line", segment="ipc",
    name="IPC analíticos del BCCh",
    description="Variación mensual de los agregados analíticos que publica el BCCh (general, "
                "SAE, transables, bienes, servicios, alimentos, energía, vivienda). Series F074 "
                "que el notebook baja por la API del BCCh y deja en Output IPC BI.xlsx.",
    unit="%",
    columns=[("Fecha", TS), ("IPC General", D), ("IPC SAE", D), ("IPC Transables", D),
             ("IPC No transables", D), ("IPC Frutas y verduras", D), ("IPC Alimentos", D),
             ("IPC Servicios", D), ("IPC Bienes", D), ("IPC Energía", D), ("IPC Vivienda", D),
             ("IPC Servicios menos vivienda", D)],
    needs=("analiticos",),
)
def _analiticos(src):
    return _hoja_bi(src, "analiticos")


@IPC.dataset(
    ds_id="ipc_volatiles", chart_type="line", segment="ipc",
    name="IPC volátiles y sin volátiles (BCCh)",
    description="Series G073 empalmadas por el BCCh. Van aparte de los analíticos porque el "
                "banco no las publica bajo los códigos F074 del resto.",
    unit="%",
    columns=[("Fecha", TS), ("IPC Volátiles", D), ("IPC Sin Volátiles", D)],
    needs=("volatiles",),
)
def _volatiles(src):
    return _hoja_bi(src, "volatiles")


@IPC.dataset(
    ds_id="ipc_grupos", chart_type="grouped_bar", segment="ipc",
    name="Variación del IPC por grupo de volatilidad",
    description="Variación mensual del IPC total, del volátil y del sin volátiles en los "
                "últimos meses.",
    unit="%",
    columns=[("Fecha", TS), ("IPC Volátiles", D), ("IPC Sin Volátiles", D), ("IPC Total", D)],
    needs=("grupos",),
)
def _grupos(src):
    return _hoja_bi(src, "grupos")


# =============================================================================
# CLI
# =============================================================================

def run_cambiario(args, out_dir: Path, stage_dir: Path) -> tuple[int, int]:
    path = Path(args.cambiario_xlsx)
    if not path.exists():
        log.error("No existe %s", path)
        return 0, 0
    src = load_cambiario(path)
    sources.stage_write(src["datos"], stage_dir, "cam_datos_excel")
    merged = sources.stage_merge(stage_dir, propios={"cam_datos_excel": src["datos"]})
    if merged is not None:
        src["datos"] = merged
        log.info("cambiario - tabla ancha: %d columnas (datos_lea manda)", len(merged.columns))
    return sources.write_all(sources.bcp.DATASETS, src, out_dir)


def run_ipc(args, out_dir: Path) -> tuple[int, int]:
    root = Path(args.ipc_dir)
    if not root.is_dir():
        log.error("No existe la carpeta del IPC %s", root)
        return 0, 0
    src = load_ipc(root, cache_fijo=out_dir / _CACHE_FIJO, rebuild_fijo=args.rebuild_fijo)
    return sources.write_all(IPC.datasets, src, out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Excel -> parquets (IPC y Cambiario AM).")
    ap.add_argument("--informe", choices=("ipc", "cambiario", "ambos"), default="ambos")
    ap.add_argument("--out", default=str(sources.PARQUET_DIR), help="Carpeta destino de los parquets")
    ap.add_argument("--stage-dir", default=None,
                    help="Carpeta de intercambio con from_sql.py (default: <out>/_stage)")
    ap.add_argument("--ipc-dir", default=str(_IPC_DIR), help="Carpeta del proyecto IPC")
    ap.add_argument("--rebuild-fijo", action="store_true",
                    help="Vuelve a leer las tres bases CERRADAS del INE (2008/2013/2018) en vez "
                         "de usar ipc_canasta_base_cerrada.parquet. Solo hace falta cuando el "
                         "INE empalma una base nueva o corrige historia.")
    ap.add_argument("--cambiario-xlsx", default=str(_CAM_XLSX), help="Ruta a datos_lea.xlsx")
    ap.add_argument("--emit-catalog", action="store_true",
                    help="Imprime las entradas YAML de catálogo y sale (no necesita datos)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    sources.setup_logging(args.verbose or args.emit_catalog)

    if args.emit_catalog:
        if args.informe == "ipc":
            datasets = IPC.datasets
        elif args.informe == "cambiario":
            datasets = sources.bcp.DATASETS
        else:
            datasets = IPC.datasets + sources.bcp.DATASETS
        print(sources.emit_catalog(datasets))
        return 0

    out_dir = Path(args.out)
    stage_dir = Path(args.stage_dir) if args.stage_dir else out_dir / "_stage"

    written = skipped = 0
    if args.informe in ("ipc", "ambos"):
        w, s = run_ipc(args, out_dir)
        print(f"IPC       - {w} parquet(s)" + (f" - {s} saltado(s)" if s else ""))
        written += w
        skipped += s
    if args.informe in ("cambiario", "ambos"):
        w, s = run_cambiario(args, out_dir, stage_dir)
        print(f"Cambiario - {w} parquet(s)" + (f" - {s} saltado(s)" if s else ""))
        written += w
        skipped += s

    print(f"OK {written} parquet(s) en {out_dir}" + (f" - {skipped} saltado(s)" if skipped else ""))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
