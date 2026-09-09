"""Carga de las tres fuentes del informe IPC.

    1. figuras.json          los gráficos que ya arma el notebook, listos
    2. Output IPC BI.xlsx    las tablas del notebook (analíticos, difusión…)
    3. Expectativas IPC.xlsx lo que llena el operador

Todo entra por `cargar_datasets()`, que devuelve un dict `{clave: DataFrame}`
con lo que haya. Un dataset que falte simplemente no está en el dict; el
builder que lo necesite se saltea y esa tarjeta sale como "Pendiente".
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as cfg

_MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun",
             "jul", "ago", "sep", "oct", "nov", "dic"]
_MESES_LARGO = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def fecha_corta_es(dt, with_time: bool = False) -> str:
    s = f"{dt.day:02d} {_MESES_ES[dt.month - 1]} {dt.year}"
    if with_time:
        s += f" {dt.hour:02d}:{dt.minute:02d}"
    return s


def mes_largo_es(dt) -> str:
    return f"{_MESES_LARGO[dt.month - 1]} {dt.year}"


# ═════════════════════════════════════════════════════════════════════
# 1. LAS FIGURAS DEL NOTEBOOK
# ═════════════════════════════════════════════════════════════════════

def cargar_figuras_notebook(path: Path = cfg.FIGURAS_NOTEBOOK) -> tuple[dict[str, dict], str]:
    """{clave: figura ya serializada} y el último mes ('AAAA-MM').

    Las figuras vienen tal cual las exportó la sección 10 del notebook. Acá
    sólo se les saca lo que en el informe sobra:

    - `width`/`height`: la tarjeta decide el tamaño.
    - `layout.title`: la tarjeta ya lleva el título en su encabezado, y
      repetirlo dentro del gráfico lo achicaba.

    No se recalcula nada: el contrato con el notebook es el JSON.
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    figs = {}
    for clave, info in d["figuras"].items():
        fig = info["fig"]
        layout = fig.setdefault("layout", {})
        layout.pop("width", None)
        layout.pop("height", None)
        layout.pop("title", None)
        layout["autosize"] = True
        figs[clave] = fig
    return figs, d["ultimo_mes"]


# ═════════════════════════════════════════════════════════════════════
# 2. LAS TABLAS DEL NOTEBOOK
# ═════════════════════════════════════════════════════════════════════

def _hoja(path: Path, nombre: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=nombre)
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df = df.sort_values("Fecha").reset_index(drop=True)
    return df


def cargar_tablas_notebook(path: Path = cfg.EXCEL_NOTEBOOK) -> dict[str, pd.DataFrame]:
    """Las hojas del Output IPC BI.xlsx que consume el informe, con claves
    cortas. Agregar una hoja acá la hace visible para figures.py."""
    hojas = {
        "analiticos":  "IPC Analiticos",
        "difusion":    "Variacion Positiva",
        "canasta":     "Canasta Volatiles Ultimo Mes",
        "grupos":      "Variacion IPC por Grupo",
        # Volátiles / sin volátiles va en hoja aparte y no dentro de "IPC
        # Analiticos" porque el BCCh no publica esas dos series bajo los
        # códigos F074 del resto de los analíticos: viven en la familia G073.
        # El notebook las baja en la misma llamada y las deja acá.
        "volatiles":   "IPC Volatiles SV BCCh",
    }
    return {clave: _hoja(path, nombre) for clave, nombre in hojas.items()}


def cargar_divisiones_ine(path: Path = cfg.INE_2023) -> pd.DataFrame:
    """Las 13 divisiones del último mes publicado, con su incidencia efectiva.

    Sale del ipc2023.xlsx del INE directamente (la misma lectura que hace el
    notebook: skiprows=3), no del Consolidado del Output: pesa 8x menos y
    trae exactamente lo mismo para el último mes.
    """
    df = pd.read_excel(path, skiprows=3)
    ultimo = df[df["Año"] == df["Año"].max()]
    ultimo = ultimo[ultimo["Mes"] == ultimo["Mes"].max()]
    div = ultimo[ultimo["División"].notna() & ultimo["Grupo"].isna()].copy()
    div = div[["División", "Glosa", "Ponderación",
               "Variación Mensual (%)", "Incidencia Mensual (%)"]]
    div = div.rename(columns={"Glosa": "Nombre",
                              "Variación Mensual (%)": "Variacion",
                              "Incidencia Mensual (%)": "Incidencia"})
    div["División"] = div["División"].astype(int)
    div["Fecha"] = pd.Timestamp(int(ultimo["Año"].iloc[0]), int(ultimo["Mes"].iloc[0]), 1)
    return div.sort_values("División").reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════
# 3. LO QUE LLENA EL OPERADOR
# ═════════════════════════════════════════════════════════════════════

FUENTES = list(cfg.FUENTES_EXPECTATIVA)   # Seguros, EOF, EEE, Bloomberg


def cargar_expectativas(path: Path = cfg.EXPECTATIVAS) -> dict[str, pd.DataFrame]:
    """Las dos hojas del Excel del operador, si existe.

    - expectativas : Fecha + una columna por fuente, en % mensual
    - divisiones   : División + Nombre corto + una columna por institución (pp)

    Las hojas que no estén o estén vacías no se devuelven — el builder que
    las necesite se saltea solo.
    """
    if not Path(path).exists():
        return {}
    try:
        x = pd.ExcelFile(path)
    except OSError as exc:
        # Pasa si quedó abierto en Excel. Se avisa y sigue: las figuras que
        # dependen de él salen como Pendiente en vez de tumbar la corrida.
        print(f"[AVISO] No pude leer {Path(path).name} ({exc.__class__.__name__}). "
              "¿Está abierto en Excel? Cerralo y volvé a correr.")
        return {}
    out: dict[str, pd.DataFrame] = {}

    if "Esperado vs Efectivo" in x.sheet_names:
        e = pd.read_excel(x, sheet_name="Esperado vs Efectivo")
        e["Fecha"] = pd.to_datetime(e["Fecha"], errors="coerce")
        e = e.dropna(subset=["Fecha"])
        # A primer día de mes, para que cruce con los analíticos del BCCh.
        e["Fecha"] = e["Fecha"].dt.to_period("M").dt.to_timestamp()
        fuentes = [f for f in FUENTES if f in e.columns]
        e = e[["Fecha"] + fuentes].dropna(subset=fuentes, how="all")
        if not e.empty:
            out["expectativas"] = e.sort_values("Fecha").reset_index(drop=True)

    if "Divisiones" in x.sheet_names:
        d = pd.read_excel(x, sheet_name="Divisiones")
        instituciones = [c for c in d.columns if c not in ("División", "Nombre corto")]
        d = d.dropna(subset=instituciones, how="all")
        if not d.empty and instituciones:
            d = d.rename(columns={"División": "Nombre", "Nombre corto": "Corto"})
            d["Nombre"] = d["Nombre"].astype(str).str.strip().str.upper()
            d["Minimo"] = d[instituciones].min(axis=1)
            d["Maximo"] = d[instituciones].max(axis=1)
            d["Promedio"] = d[instituciones].mean(axis=1)
            d.attrs["instituciones"] = instituciones
            out["divisiones"] = d.reset_index(drop=True)

    return out


# ═════════════════════════════════════════════════════════════════════
# TODO JUNTO
# ═════════════════════════════════════════════════════════════════════

def cargar_datasets() -> dict[str, pd.DataFrame]:
    """El dict que consumen figures.BUILDERS y tables.py.

    Claves fijas: analiticos, difusion, canasta, grupos (del notebook);
    ine_divisiones (del INE); y expectativas, divisiones (del operador, si
    están).
    """
    datasets = cargar_tablas_notebook()
    datasets["ine_divisiones"] = cargar_divisiones_ine()
    datasets.update(cargar_expectativas())
    return datasets
