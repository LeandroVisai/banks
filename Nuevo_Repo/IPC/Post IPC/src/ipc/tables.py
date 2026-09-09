"""La portada del IPC: las tarjetas KPI.

Cuatro tarjetas, cada una con su valor del mes y el cambio contra el mes
anterior:

    IPC general           hoja "IPC Analiticos"
    IPC sin volátiles     hoja "IPC Volatiles SV BCCh"  (serie G073 del BCCh)
    Difusión              hoja "Variacion Positiva"
    Sorpresa vs mercado   Excel del operador, columna Bloomberg

Todo va al dict `portada` que consume el template compartido — ver
render.render_html().
"""
from __future__ import annotations

import json

import pandas as pd

from . import data


def _delta(actual: float, anterior: float | None, unidad: str = "pp") -> tuple[str, str]:
    """('+0,3 pp vs mes anterior', 'up'|'down'|'')"""
    if anterior is None or pd.isna(anterior):
        return "", ""
    d = actual - anterior
    if abs(d) < 1e-9:
        return "sin cambio vs mes anterior", ""
    return f"{d:+.1f} {unidad} vs mes anterior", "up" if d > 0 else "down"


def _ultimo_y_anterior(df: pd.DataFrame, columna: str) -> tuple[float, float | None]:
    """El valor del último mes y el del anterior, de una serie con Fecha."""
    d = df.sort_values("Fecha")
    ultimo = float(d[columna].iloc[-1])
    anterior = float(d[columna].iloc[-2]) if len(d) > 1 else None
    return ultimo, anterior


def build_kpis(datasets: dict[str, pd.DataFrame]) -> list[dict]:
    a = datasets["analiticos"].sort_values("Fecha")
    ult = a.iloc[-1]
    kpis = []

    # --- IPC general --- (hoja "IPC Analiticos")
    v, anterior = _ultimo_y_anterior(a, "IPC General")
    sub, cls = _delta(v, anterior)
    kpis.append({"label": "IPC general · var. mensual", "value": f"{v:+.1f}%", "sub": sub, "cls": cls})

    # --- IPC sin volátiles --- (hoja "IPC Volatiles SV BCCh", no la de analíticos)
    # Va en otra hoja porque el BCCh publica esta serie en la familia G073 y no
    # con los códigos F074 del resto de los analíticos. Ver data.cargar_tablas_notebook().
    v, anterior = _ultimo_y_anterior(datasets["volatiles"], "IPC Sin Volátiles")
    sub, cls = _delta(v, anterior)
    kpis.append({"label": "IPC sin volátiles · var. mensual", "value": f"{v:+.1f}%", "sub": sub, "cls": cls})

    if "difusion" in datasets and not datasets["difusion"].empty:
        d = datasets["difusion"].sort_values(["Año", "Mes"])
        v = float(d["Difusión (%)"].iloc[-1])
        anterior = float(d["Difusión (%)"].iloc[-2]) if len(d) > 1 else None
        sub, cls = _delta(v, anterior)
        kpis.append({"label": "Difusión · % productos al alza", "value": f"{v:.1f}%", "sub": sub, "cls": cls})

    # La sorpresa contra el mercado: efectivo menos lo que esperaba Bloomberg
    # ese mes. Sólo si el operador cargó la expectativa de este mes.
    if "expectativas" in datasets:
        e = datasets["expectativas"]
        mes = pd.Timestamp(ult["Fecha"]).to_period("M").to_timestamp()
        fila = e[e["Fecha"] == mes]
        if not fila.empty and "Bloomberg" in fila.columns and pd.notna(fila["Bloomberg"].iloc[0]):
            esperado = float(fila["Bloomberg"].iloc[0])
            sorpresa = float(ult["IPC General"]) - esperado
            cls = "neg" if sorpresa > 0.05 else ("pos" if sorpresa < -0.05 else "")
            kpis.append({"label": "Sorpresa vs mercado · Bloomberg",
                         "value": f"{sorpresa:+.2f} pp",
                         "sub": f"esperaba {esperado:+.1f}%, salió {float(ult['IPC General']):+.1f}%",
                         "cls": cls})
    return kpis


def build_portada(datasets: dict[str, pd.DataFrame]) -> dict:
    return {
        "comentarios": ["Lectura del dato", "Sorpresas y drivers", "Perspectiva"],
        "kpis": build_kpis(datasets),
        "mercado": None,
    }


def fecha_informe(datasets: dict[str, pd.DataFrame]) -> pd.Timestamp:
    return pd.Timestamp(datasets["analiticos"]["Fecha"].max())


# ═════════════════════════════════════════════════════════════════════
# TABLA DE DETALLE (sección final del informe)
# ═════════════════════════════════════════════════════════════════════
# Todos los productos de la canasta con su variación e incidencia. Es la
# tabla que cerraba el dashboard anterior. Va al payload como
# __PAYLOAD__.tablas y ui.js la arma con filtro por clasificación,
# búsqueda y orden por columna.
#
# Cada columna: field (nombre en el DataFrame), label, y opcionales
# num/dec/suffix (formato), signo=False (sin "+"), pill (píldora de
# categoría), buscar (la columna sobre la que busca el cuadro de texto).

_COLS_CANASTA = [
    {"field": "Glosa",                   "label": "Producto", "buscar": True},
    {"field": "Clasificación",           "label": "Clasif.", "pill": True},
    {"field": "Variación Mensual (%)",   "label": "Var. mensual", "num": True, "suffix": "%"},
    {"field": "Incidencia Mensual (%)",  "label": "Inc. mensual", "num": True, "dec": 3, "suffix": " pp"},
    {"field": "Variación 12 Meses (%)",  "label": "Var. 12M", "num": True, "suffix": "%"},
    {"field": "Incidencia 12 Meses (%)", "label": "Inc. 12M", "num": True, "dec": 3, "suffix": " pp"},
    {"field": "Ponderación 2023",        "label": "Ponderación", "num": True, "dec": 2, "suffix": "%", "signo": False},
]


def _tabla(df: pd.DataFrame, columnas: list[dict], *, orden: dict,
           filtro: str = "Clasificación", unidad: str = "productos") -> dict:
    cols = [c for c in columnas if c["field"] in df.columns]
    d = df[[c["field"] for c in cols]]
    # to_json: NaN → null y tipos numpy → nativos, de una.
    filas = json.loads(d.to_json(orient="records", force_ascii=False))
    out = {"columnas": cols, "filas": filas, "orden": orden, "unidad": unidad}
    if filtro in df.columns:
        out["filtro"] = {"field": filtro, "label": "Todos",
                         "opciones": sorted(df[filtro].dropna().unique().tolist())}
    return out


def build_tablas(datasets: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """{id de tabla: tabla}. El id es el que usa el slot en layout.py."""
    tablas = {}
    if "canasta" in datasets and not datasets["canasta"].empty:
        tablas["canasta"] = _tabla(datasets["canasta"], _COLS_CANASTA,
                                   orden={"field": "Ponderación 2023", "dir": "desc"})
    return tablas
