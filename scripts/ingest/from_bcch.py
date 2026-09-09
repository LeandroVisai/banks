#!/usr/bin/env python3
"""API pública del BCCh -> parquets del IPC, SIN correr el notebook.

Cuarto extractor de ``scripts/ingest/``, hermano liviano de ``from_notebook.py``:
mismos seis datasets (``ipc_analiticos``, ``ipc_volatiles``, ``ipc_bienes_servicios_sv``,
``ipc_banda_bienes_sv``, ``ipc_banda_servicios_sv``, ``ipc_grupos``), pero sin ejecutar
``01.Analisis IPC.py``. Le pega directo a la API pública del Banco Central
(``bcchapi``, https://si3.bcentral.cl) con las MISMAS tres consultas que hace el
notebook en su sección 2 — nombres de serie, ventana y post-proceso calcados.

    python scripts/ingest/from_bcch.py --out data_pipeline/parquet
    python scripts/ingest/from_bcch.py --usuario x@y.cl --password ***
    python scripts/ingest/from_bcch.py --emit-catalog     # YAML, sin conectarse

Por qué esto no necesitaba correr el notebook
----------------------------------------------
De las cuatro cosas que hace el notebook, tres YA estaban cubiertas por otro
extractor y solo una de verdad necesitaba la API del BCCh sin alternativa:

    Get_Data (DW)         -> ya extraído a from_sql.py (ipc_ci, ipc_si, ...)
    Excel del INE          -> ya extraído a from_excel.py (canasta, diccionario)
    bcchapi (API pública)  -> ESTE script. Es la única pieza que faltaba, y la
                              API pública no necesita ni Get_Data ni estar en la
                              red del banco: solo ``bcchapi`` + un usuario/clave
                              de https://si3.bcentral.cl (gratis, se pide en la
                              web del BCCh) — mucho más liviano que correr el
                              notebook entero con ``runpy``.

``ipc_analiticos``/``ipc_volatiles`` tienen además un camino alternativo (leer
``outputs/Output IPC BI.xlsx`` si el notebook ya se corrió — ver
``from_excel.py``); pero las BANDAS de bienes/servicios sin volátiles
(``ipc_banda_bienes_sv``/``ipc_banda_servicios_sv``) NO tienen ningún otro
origen: esas series nunca llegan al Excel exportado. Este script es su única
alternativa a correr el notebook completo.

Cero duplicación: reusa los Dataset de from_notebook.py
--------------------------------------------------------
No se reescribe ni un builder. ``from_notebook.NB.datasets`` ya tiene los seis
``Dataset`` (metadata de catálogo + la función que arma cada parquet a partir de
un dict ``src``); ese dict, en el notebook, sale de sus variables globales
(``runpy``), y acá sale de tres llamadas a ``bcchapi`` + una lectura del parquet
de canasta que ya dejó ``from_excel.py``. Mismo contrato de entrada
(``src["analiticos"]``, ``src["volatiles"]``, ``src["sv_bcch"]``,
``src["incidencias_grupo"]``), builders IDÉNTICOS, dos formas de llenarlo.

``ipc_grupos`` cruza dos orígenes
---------------------------------
Es el único de los seis que necesita algo de ``from_excel.py``: la fila "IPC
General" de la canasta consolidada (``ipc_canasta_historica.parquet``), la
misma que usa ``ipc_variacion_historica_mes``. Si ese parquet no está en
``--out`` todavía (from_excel.py no corrió), ``ipc_grupos`` se salta con
warning — el resto de los cinco datasets no lo necesita y se generan igual.

Credenciales
------------
NUNCA hardcodeadas (a diferencia del notebook original, que las trae en texto
plano). Se leen de ``BCCH_API_USER``/``BCCH_API_PASSWORD``, con ``--usuario``/
``--password`` como override explícito. Se piden gratis en
https://si3.bcentral.cl/estadisticas/Principal1/Web_Services.htm.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources  # noqa: E402

log = logging.getLogger("ingest.bcch")

_NB_PATH = Path(__file__).resolve().parent / "from_notebook.py"

# Series y ventanas: EXACTAS a la sección 2 del notebook original
# (Nuevo_Repo/IPC/Post IPC/notebooks/01.Analisis IPC.py). Cualquier cambio ahí
# (una serie nueva, un código distinto) hay que reflejarlo acá a mano: no hay
# forma de importarlo, el notebook no es un módulo.
_DESDE = "2024-01-01"            # ventana de analiticos/volatiles (= DESDE)
_DESDE_HISTORIA = "2010-01-01"   # ventana de sv_bcch (= DESDE_HISTORIA)

_SERIES_ANALITICOS = {
    "F074.IPC.VAR.Z.2023.C.M": "IPC General",
    "F074.IPCSAE.VAR.Z.2023.C.M": "IPC SAE",
    "F074.IPCT.VAR.Z.2023.C.M": "IPC Transables",
    "F074.IPCN.VAR.Z.2023.C.M": "IPC No transables",
    "F074.IPCFV.VAR.Z.2023.C.M": "IPC Frutas y verduras",
    "F074.IPCA.VAR.Z.2023.C.M": "IPC Alimentos",
    "F074.IPCS.VAR.Z.2023.C.M": "IPC Servicios",
    "F074.IPCB.VAR.Z.2023.C.M": "IPC Bienes",
    "F074.IPCE.VAR.Z.2023.C.M": "IPC Energía",
    "F074.IPCVIV.VAR.Z.2023.C.M": "IPC Vivienda",
    "F074.IPCSVIV.VAR.Z.2023.C.M": "IPC Servicios menos vivienda",
}
_SERIES_VOLATILES = {
    "G073.IPCV.VAR.2023.M": "IPC Volátiles",
    "G073.IPCSV.VAR.2023.M": "IPC Sin Volátiles",
}
_SERIES_SV = {
    "G073.IPCBSV.VAR.2023.M": "Bienes SV",
    "G073.IPCSSV.VAR.2023.M": "Servicios SV",
}


def _load_from_notebook():
    """Carga ``from_notebook.py`` por ruta, igual que ``sources._load_bcp()``
    carga ``build_cambiario_parquets.py``. Da acceso a ``NB.datasets`` (los seis
    ``Dataset`` ya declarados, con su catálogo y sus builders) sin reescribir
    nada — ver el docstring del módulo."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("from_notebook", _NB_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - ruta rota
        raise ImportError(f"no pude cargar {_NB_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("from_notebook", module)
    spec.loader.exec_module(module)
    return module


def _cuadro(bde, series: dict[str, str], *, desde: str, hasta: str):
    """Una llamada a ``bcchapi.Siete.cuadro()``, con el mismo post-proceso que
    hace el notebook tras cada una de sus tres consultas (``reset_index`` +
    renombrar el índice a "Fecha"). ``series`` es ``{código: nombre de columna}``
    para no repetir dos listas paralelas como hace el notebook."""
    df = bde.cuadro(series=list(series), nombres=list(series.values()), desde=desde, hasta=hasta)
    return df.reset_index().rename(columns={"index": "Fecha"})


def fetch(usuario: str, password: str, *, desde: str = _DESDE,
          desde_historia: str = _DESDE_HISTORIA) -> dict:
    """Las tres consultas a la API del BCCh, en un dict con las mismas claves
    que espera ``from_notebook.NB`` (``analiticos``/``volatiles``/``sv_bcch``).

    Lazy import de ``bcchapi``: no es una dependencia del resto del repo, solo
    de este script — igual que ``Get_Data`` en ``from_sql.py``.
    """
    import pandas as pd

    try:
        import bcchapi
    except ImportError as exc:
        raise RuntimeError(
            "falta el paquete bcchapi (pip install bcchapi). Es la API pública del "
            "BCCh, no necesita red interna del banco."
        ) from exc

    bde = bcchapi.Siete(usuario, password)
    hasta = pd.Timestamp.today().strftime("%Y-%m-%d")

    out: dict = {}
    log.info("BCCh - analiticos (%d series, %s..%s)", len(_SERIES_ANALITICOS), desde, hasta)
    out["analiticos"] = _cuadro(bde, _SERIES_ANALITICOS, desde=desde, hasta=hasta)

    log.info("BCCh - volatiles (%d series, %s..%s)", len(_SERIES_VOLATILES), desde, hasta)
    out["volatiles"] = _cuadro(bde, _SERIES_VOLATILES, desde=desde, hasta=hasta)

    log.info("BCCh - sv_bcch (%d series, %s..%s)", len(_SERIES_SV), desde_historia, hasta)
    sv = _cuadro(bde, _SERIES_SV, desde=desde_historia, hasta=hasta)
    sv["Fecha"] = pd.to_datetime(sv["Fecha"])
    sv["Año"] = sv["Fecha"].dt.year
    sv["Mes"] = sv["Fecha"].dt.month
    out["sv_bcch"] = sv

    return out


def _incidencias_grupo(volatiles, canasta_parquet: Path):
    """Réplica de ``incidencias_grupo`` del notebook (sección 8): la ventana de
    12 meses de ``volatiles`` cruzada con la fila "IPC General" de la canasta.

    A diferencia de ``analiticos``/``volatiles``/``sv_bcch`` (que salen ENTEROS
    de la API), este cruza con la canasta del INE — que este script no lee, para
    no duplicar 4 planillas que ``from_excel.py`` ya procesó. Lee en cambio el
    parquet que ese script YA dejó: si no está (from_excel.py no corrió antes),
    devuelve ``None`` y ``ipc_grupos`` se salta solo, con warning — los otros
    cinco datasets no lo necesitan.
    """
    import pandas as pd

    if not canasta_parquet.exists():
        log.warning("ipc_grupos - falta %s (correr from_excel.py --informe ipc antes); "
                   "se salta, el resto de los datasets no lo necesita", canasta_parquet.name)
        return None

    canasta = sources.read_parquet(canasta_parquet)
    canasta["Fecha"] = pd.to_datetime(canasta["Fecha"])
    es_general = canasta["Glosa"].fillna("").str.strip().str.upper().eq("IPC GENERAL")
    ipc_general = (canasta.loc[es_general, ["Fecha", "Variación Mensual (%)"]]
                  .rename(columns={"Variación Mensual (%)": "IPC Total"}))
    if ipc_general.empty:
        log.warning("ipc_grupos - %s no trae la fila 'IPC General'; se salta",
                    canasta_parquet.name)
        return None

    v = volatiles.copy()
    v["Fecha"] = pd.to_datetime(v["Fecha"])
    fecha_max = v["Fecha"].max()
    ventana = v[v["Fecha"] >= fecha_max - pd.DateOffset(months=12)]
    out = (ventana.merge(ipc_general, on="Fecha", how="left")
           .round({"IPC Volátiles": 3, "IPC Sin Volátiles": 3, "IPC Total": 3})
           .sort_values("Fecha").reset_index(drop=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="API pública del BCCh -> parquets del IPC.")
    ap.add_argument("--out", default=str(sources.PARQUET_DIR), help="Carpeta destino de los parquets")
    ap.add_argument("--usuario", default=os.getenv("BCCH_API_USER", ""),
                    help="Usuario de si3.bcentral.cl (default: env BCCH_API_USER)")
    ap.add_argument("--password", default=os.getenv("BCCH_API_PASSWORD", ""),
                    help="Clave de si3.bcentral.cl (default: env BCCH_API_PASSWORD)")
    ap.add_argument("--desde", default=_DESDE, help=f"Ventana de analiticos/volatiles (default {_DESDE})")
    ap.add_argument("--desde-historia", default=_DESDE_HISTORIA,
                    help=f"Ventana de sv_bcch, las bandas (default {_DESDE_HISTORIA})")
    ap.add_argument("--emit-catalog", action="store_true",
                    help="Imprime las entradas YAML de catálogo y sale (no se conecta)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    sources.setup_logging(args.verbose or args.emit_catalog)
    nb = _load_from_notebook()

    if args.emit_catalog:
        print(sources.emit_catalog(nb.NB.datasets))
        return 0

    if not args.usuario or not args.password:
        log.error("faltan credenciales de si3.bcentral.cl: --usuario/--password o "
                 "las variables BCCH_API_USER/BCCH_API_PASSWORD. Se piden gratis en "
                 "https://si3.bcentral.cl/estadisticas/Principal1/Web_Services.htm")
        return 2

    out_dir = Path(args.out)
    try:
        src = fetch(args.usuario, args.password, desde=args.desde, desde_historia=args.desde_historia)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2

    grupos = _incidencias_grupo(src["volatiles"], out_dir / "ipc_canasta_historica.parquet")
    if grupos is not None:
        src["incidencias_grupo"] = grupos

    written, skipped = sources.write_all(nb.NB.datasets, src, out_dir)
    print(f"OK {written} parquet(s) en {out_dir}" + (f" - {skipped} saltado(s)" if skipped else ""))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
