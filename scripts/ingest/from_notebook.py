#!/usr/bin/env python3
"""Corre el notebook de análisis del IPC y convierte su salida en parquets.

Tercer extractor de ``scripts/ingest/``, hermano de ``from_excel.py`` y
``from_sql.py``. Los otros dos leen orígenes quietos (planillas, tablas del DW);
este ejecuta el análisis mensual que ya escribió el DOMA —
``notebooks/01.Analisis IPC.py`` — y se queda con los DataFrames que deja en
memoria.

    python scripts/ingest/from_notebook.py -v
    python scripts/ingest/from_notebook.py --no-excel      # solo parquets
    python scripts/ingest/from_notebook.py --emit-catalog  # YAML, sin correr nada

Correr EN EL SERVIDOR: el notebook necesita el módulo ``Get_Data`` y la API del
BCCh (``bcchapi``, con credenciales), ninguno de los dos disponible fuera.

Por qué ejecutarlo en vez de reimplementarlo
--------------------------------------------
El notebook es la fuente de verdad del análisis del IPC y lo mantiene el equipo
que publica el informe. Reescribir sus 40 KB acá crearía una segunda versión que
se desincroniza en la primera corrección que alguien haga allá. Lo que sí hace
falta es que su salida deje de vivir en un ``.xlsx`` y llegue a
``data_pipeline/parquet/`` como el resto: eso es lo único que agrega este script.

Qué se rescata
--------------
Las series que baja de la **API del BCCh** y no existen en ninguna planilla del
INE ni en el DW:

    ine_ipc          -> ipc_analiticos             F074, los 11 agregados
    variacion_bcch   -> ipc_volatiles              G073 volátiles / sin volátiles
    sv_bcch          -> ipc_bienes_servicios_sv    G073 bienes y servicios SV, desde 2010
    incidencias_grupo-> ipc_grupos                 variación por grupo de volatilidad

``sv_bcch`` es el que más importa: son las dos bandas de percentiles del informe
(bienes y servicios sin volátiles) y el export a Excel del notebook nunca las
incluyó, así que por el camino de ``from_excel.py`` no había forma de obtenerlas.

Relación con from_excel.py
--------------------------
Los cuatro parquets de arriba tienen DOS productores posibles, y escriben el
mismo archivo:

    en el servidor   from_notebook.py   corre el análisis y toma los DataFrames
    fuera            from_excel.py      lee el Output IPC BI.xlsx que quedó

No es duplicación: es el mismo dato por el camino que esté disponible. El
notebook manda cuando se lo puede correr, porque además trae ``sv_bcch``, que al
Excel no llega.

El Excel se sigue escribiendo
-----------------------------
La sección 11 del notebook (el ``ExcelWriter`` de las nueve hojas) está comentada
en el fuente, así que hoy no genera nada. Este script la ejecuta desde afuera,
sobre las variables que el notebook dejó, y deja ``outputs/Output IPC BI.xlsx``
donde el publicador original lo espera. Con ``--no-excel`` se salta ese paso.
"""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources  # noqa: E402

log = logging.getLogger("ingest.notebook")

TS, D, V = sources.TS, sources.D, sources.V

_NOTEBOOK = sources.ROOT / "Nuevo_Repo" / "IPC" / "Post IPC" / "notebooks" / "01.Analisis IPC.py"

# variable del notebook -> hoja del Output IPC BI.xlsx. Es la lista de la sección
# 11 del notebook, que allá está comentada. El orden es el de aquel archivo.
_HOJAS_EXCEL = [
    ("finalDf",                "Consolidado Bases"),
    ("df_resumen_difusion",    "Resumen Difusion IPC"),
    ("variacion_positiva",     "Variacion Positiva"),
    ("ine_ipc",                "IPC Analiticos"),
    ("variacion_bcch",         "IPC Volatiles SV BCCh"),
    ("variacion_positiva_sv",  "Var Positiva Sin Volatiles"),
    ("df_resumen_difusion_sv", "Resumen Difusion SV"),
    ("canasta_ultimo_mes",     "Canasta Volatiles Ultimo Mes"),
    ("incidencias_grupo",      "Variacion IPC por Grupo"),
]


# =============================================================================
# Ejecución del notebook
# =============================================================================

@contextmanager
def _entorno_notebook(directorio: Path):
    """Deja el proceso como lo espera el notebook, y lo devuelve al salir.

    Dos cosas hacen falta:

    - **cwd en ``notebooks/``**: el notebook resuelve sus rutas con
      ``Path.cwd().resolve().parent``, así que corriéndolo desde otro lado
      buscaría las planillas del INE en una carpeta que no existe.
    - **sin ``.show()`` ni ``display()``**: son de Jupyter. ``display`` no existe
      fuera y reventaría con ``NameError`` en la sección 5; ``fig.show()`` abriría
      una pestaña del navegador por cada uno de los once gráficos. Se neutralizan
      los dos y se restauran al terminar, para no dejar plotly parcheado si
      alguien importa este módulo.
    """
    import plotly.graph_objects as go

    cwd = Path.cwd()
    show_original = go.Figure.show
    go.Figure.show = lambda self, *a, **k: self  # noqa: ARG005
    os.chdir(directorio)
    try:
        yield
    finally:
        os.chdir(cwd)
        go.Figure.show = show_original


def ejecutar_notebook(path: Path) -> dict:
    """Corre el notebook y devuelve sus variables globales.

    ``runpy.run_path`` lo ejecuta como si fuera ``__main__`` y devuelve el dict de
    globales, que es exactamente lo que hace falta: los DataFrames ya calculados,
    sin tener que pedirle al notebook que los serialice primero.
    """
    if not path.exists():
        raise FileNotFoundError(f"no encuentro el notebook en {path}")

    faltan = [m for m in ("Get_Data", "bcchapi") if not _importable(m)]
    if faltan:
        raise RuntimeError(
            f"falta {', '.join(faltan)}: este script corre EN EL SERVIDOR, donde viven "
            "el módulo Get_Data del DW y el acceso a la API del BCCh. Fuera de ahí, usá "
            "from_excel.py, que lee el Output IPC BI.xlsx que el notebook ya dejó."
        )

    log.info("ejecutando %s ...", path.name)
    with _entorno_notebook(path.parent):
        globales = runpy.run_path(str(path), init_globals={"display": lambda *a, **k: None})
    log.info("notebook terminado - %d variables en memoria", len(globales))
    return globales


def _importable(nombre: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(nombre) is not None
    except (ImportError, ValueError):
        return False


def escribir_excel(globales: dict, destino: Path) -> Path | None:
    """La sección 11 del notebook, ejecutada desde afuera.

    Una hoja cuya variable el notebook no dejó (porque se cambió el análisis) se
    omite con warning en vez de tumbar el archivo entero.
    """
    import pandas as pd

    hojas = [(nombre, hoja) for nombre, hoja in _HOJAS_EXCEL if nombre in globales]
    faltan = [n for n, _ in _HOJAS_EXCEL if n not in globales]
    if faltan:
        log.warning("Excel - el notebook no dejó %s; esas hojas se omiten", ", ".join(faltan))
    if not hojas:
        log.error("Excel - ninguna de las nueve variables está; no se escribe nada")
        return None

    destino.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(destino) as writer:
        for nombre, hoja in hojas:
            globales[nombre].to_excel(writer, sheet_name=hoja, index=False)
    log.info("Excel - %s (%d hojas)", destino.name, len(hojas))
    return destino


# =============================================================================
# Datasets
# =============================================================================

NB = sources.Registry()


def _con_fecha(frame):
    import pandas as pd

    df = frame.copy()
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df = df.dropna(subset=["Fecha"]).sort_values("Fecha")
    return df.reset_index(drop=True)


@NB.dataset(
    ds_id="ipc_analiticos", chart_type="line", segment="ipc",
    name="IPC analíticos del BCCh",
    description="Variación mensual de los agregados analíticos que publica el BCCh (general, "
                "SAE, transables, bienes, servicios, alimentos, energía, vivienda). Series F074, "
                "bajadas por la API del BCCh en el notebook de análisis.",
    unit="%",
    columns=[("Fecha", TS), ("IPC General", D), ("IPC SAE", D), ("IPC Transables", D),
             ("IPC No transables", D), ("IPC Frutas y verduras", D), ("IPC Alimentos", D),
             ("IPC Servicios", D), ("IPC Bienes", D), ("IPC Energía", D), ("IPC Vivienda", D),
             ("IPC Servicios menos vivienda", D)],
    needs=("ine_ipc",),
)
def _analiticos(src):
    return _con_fecha(src["ine_ipc"])


@NB.dataset(
    ds_id="ipc_volatiles", chart_type="line", segment="ipc",
    name="IPC volátiles y sin volátiles (BCCh)",
    description="Series G073 empalmadas por el BCCh. Van aparte de los analíticos porque el "
                "banco no las publica bajo los códigos F074 del resto.",
    unit="%",
    columns=[("Fecha", TS), ("IPC Volátiles", D), ("IPC Sin Volátiles", D)],
    needs=("variacion_bcch",),
)
def _volatiles(src):
    return _con_fecha(src["variacion_bcch"])


@NB.dataset(
    ds_id="ipc_bienes_servicios_sv", chart_type="line", segment="ipc",
    name="Bienes y servicios sin volátiles (BCCh, desde 2010)",
    description="Variación mensual de bienes y de servicios sin volátiles, series G073 "
                "empalmadas por el BCCh desde 1999. Es la base de las dos bandas de percentiles "
                "del informe: la banda sale de la serie agregada del banco, NO de sumar producto "
                "a producto la canasta del INE.",
    unit="%",
    columns=[("Fecha", TS), ("Año", D), ("Mes", D), ("Bienes SV", D), ("Servicios SV", D)],
    needs=("sv_bcch",),
)
def _bienes_servicios_sv(src):
    return _con_fecha(src["sv_bcch"])


# Ventana de la banda de percentiles de bienes y servicios sin volátiles, y los
# percentiles que la delimitan. Son los del notebook (`ANIO_BASE_SV_*` y el
# `.agg(p10=…, p90=…)` de `banda_percentil_mensual`): 2010-2020 deja fuera la
# pandemia y el shock inflacionario que vino después.
_SV_DESDE, _SV_HASTA = 2010, 2020
_SV_CUANTILES = (0.10, 0.90)


def _banda_sv(src, columna: str):
    sv = src["sv_bcch"]
    faltan = [c for c in ("Año", "Mes", columna) if c not in sv.columns]
    if faltan:
        log.warning("sv_bcch no trae %s; la banda queda sin generar", ", ".join(faltan))
        return None
    return sources.banda_mensual(sv[["Año", "Mes", columna]], columna,
                                 desde=_SV_DESDE, hasta=_SV_HASTA, cuantiles=_SV_CUANTILES)


@NB.dataset(
    ds_id="ipc_banda_bienes_sv", chart_type="hist_range", segment="ipc",
    name="Bienes sin volátiles: el mes contra su banda histórica",
    description=f"Para cada mes calendario, el rango p10-p90 de la variación de bienes sin "
                f"volátiles en {_SV_DESDE}-{_SV_HASTA}, su promedio, y el valor del año en "
                "curso. La banda excluye la pandemia y el shock inflacionario posterior.",
    unit="%", columns=sources.COLS_RANGO, needs=("sv_bcch",),
)
def _banda_bienes_sv(src):
    return _banda_sv(src, "Bienes SV")


@NB.dataset(
    ds_id="ipc_banda_servicios_sv", chart_type="hist_range", segment="ipc",
    name="Servicios sin volátiles: el mes contra su banda histórica",
    description=f"Para cada mes calendario, el rango p10-p90 de la variación de servicios sin "
                f"volátiles en {_SV_DESDE}-{_SV_HASTA}, su promedio, y el valor del año en "
                "curso. La banda excluye la pandemia y el shock inflacionario posterior.",
    unit="%", columns=sources.COLS_RANGO, needs=("sv_bcch",),
)
def _banda_servicios_sv(src):
    return _banda_sv(src, "Servicios SV")


@NB.dataset(
    ds_id="ipc_grupos", chart_type="grouped_bar", segment="ipc",
    name="Variación del IPC por grupo de volatilidad",
    description="Variación mensual del IPC total, del volátil y del sin volátiles en los "
                "últimos meses.",
    unit="%",
    columns=[("Fecha", TS), ("IPC Volátiles", D), ("IPC Sin Volátiles", D), ("IPC Total", D)],
    needs=("incidencias_grupo",),
)
def _grupos(src):
    return _con_fecha(src["incidencias_grupo"])


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Corre el notebook de análisis del IPC y escribe sus parquets.")
    ap.add_argument("--notebook", default=str(_NOTEBOOK), help="Ruta a 01.Analisis IPC.py")
    ap.add_argument("--out", default=str(sources.PARQUET_DIR), help="Carpeta destino de los parquets")
    ap.add_argument("--no-excel", action="store_true",
                    help="No escribir outputs/Output IPC BI.xlsx (solo los parquets)")
    ap.add_argument("--emit-catalog", action="store_true",
                    help="Imprime las entradas YAML de catálogo y sale (no corre el notebook)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    sources.setup_logging(args.verbose or args.emit_catalog)

    if args.emit_catalog:
        print(sources.emit_catalog(NB.datasets))
        return 0

    notebook = Path(args.notebook)
    try:
        globales = ejecutar_notebook(notebook)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2

    if not args.no_excel:
        # El notebook resuelve sus rutas contra la carpeta que contiene notebooks/.
        escribir_excel(globales, notebook.parent.parent / "outputs" / "Output IPC BI.xlsx")

    written, skipped = sources.write_all(NB.datasets, globales, Path(args.out))
    print(f"Notebook  - {written} parquet(s)" + (f" - {skipped} saltado(s)" if skipped else ""))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
