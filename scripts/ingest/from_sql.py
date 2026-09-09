#!/usr/bin/env python3
"""Consultas al DW -> parquets, para los DOS informes (IPC y Cambiario AM).

Hermano de ``from_excel.py``: mismo andamiaje (``sources.py``), mismos datasets de
salida, pero el origen es el data warehouse vía el módulo ``Get_Data`` del
servidor. Correr EN EL SERVIDOR, donde ese módulo existe.

    python scripts/ingest/from_sql.py --informe cambiario -v
    python scripts/ingest/from_sql.py --informe ipc -v
    python scripts/ingest/from_sql.py --informe ambos --out data_pipeline/parquet
    python scripts/ingest/from_sql.py --dump-sql            # imprime las queries y sale

Una sola lista de consultas para los dos informes
-------------------------------------------------
Hasta ahora las mismas queries estaban escritas en tres lugares: el ``.sql`` de
documentación del DOMA, el ``data.py`` del publicador y
``build_cambiario_parquets.py``. Acá viven una vez. ``QUERIES`` marca a qué
informe pertenece cada una, así ``--informe ipc`` no dispara las 15 del cambiario
ni al revés.

Dos formas de usar una query
----------------------------
1. **Dataset directo**: la query ES el parquet (``cam_fixing_bancos``,
   ``ipc_seguros_inflacion``). Un builder la toma de ``src[clave]`` y la deja
   lista.

2. **Aporte a la tabla ancha**: la hoja ``Datos`` del ``.xlsm`` original era, en
   sus tres cuartas partes, un volcado del DW hecho con Power Query. Las queries
   marcadas ``wide=True`` la reconstruyen: cada una aporta columnas, se renombran
   al encabezado que esperan los builders del cambiario y se unen por ``Fecha``.
   Así los 32 datasets que leían la hoja ``Datos`` se construyen igual, sin que
   nadie tenga que abrir Excel.

El puente con from_excel.py, y quién manda
------------------------------------------
Varios datasets cruzan una columna de Bloomberg con una del DW (``cam_tot_gs`` =
términos de intercambio + CLP). Este script deja su tabla ancha en
``--stage-dir`` y levanta la que haya dejado ``from_excel.py``, uniéndolas por
``Fecha``. Corran en el orden que corran, el resultado es el mismo: la
precedencia NO depende de quién llegó primero sino de
``sources.PRIORIDAD_CAMBIARIO``, y ahí ``datos_lea.xlsx`` va primero.

Eso es deliberado. ``datos_lea.xlsx`` es la planilla que mantiene el DOMA y a la
que se le van a agregar las columnas que faltan; cuando una llegue, pisa sola a
la versión del DW o a la derivada, sin tocar código.

Lo que hoy no tiene origen
--------------------------
Con ``datos_lea.xlsx`` + estas consultas salen **36 de los 41** datasets. Los
cinco restantes esperan columnas de Bloomberg que el recorte todavía no trae y
que el DW no tiene; quedan sin generar (tarjeta "sin datos" en su lugar del
informe) hasta que aparezcan:

    cam_curva_hga/lma/cl1   "Contrato/Tenor/Price" de HGA, Lma y CL1 (9 columnas;
                            Contrato es el ticker en texto y Tenor una fecha, así
                            que no hay forma de derivarlas de una serie de precios)
    cam_clp_intradia        hoja "Intradía": CLP, DXY y HGA intradía en base 100.
                            OJO: no sale de ``inter_tc_intra``, que solo trae el
                            CLP (Fecha, Hora, TC, Monto) — faltan DXY y cobre.
    cam_carry_trade         hoja "CarryTrade"

Y una que sale, pero degradada: ``cam_clp_ohlc`` se arma con la vela DERIVADA del
intradía (ver ``ohlc_desde_intra``) mientras Bloomberg no aporte "CLP Apertura",
"CLP Max" y "CLP minimo".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources  # noqa: E402

log = logging.getLogger("ingest.sql")

TS, D, V = sources.TS, sources.D, sources.V


class Query:
    """Una consulta al DW.

    - ``informe``: ``ipc`` | ``cambiario`` — para no disparar las de un informe
      cuando se pide el otro.
    - ``wide``: la consulta aporta columnas a la tabla ancha del cambiario (la
      ex hoja ``Datos``) en vez de ser un dataset por sí sola.
    - ``rename``: ``{columna del DW: encabezado del .xlsm}``. Solo se renombra lo
      que exista: si el DW no trae una columna, se omite y el dataset que la
      necesite se saltará con su mensaje, en vez de tumbar la corrida acá.
    """

    def __init__(self, key: str, sql: str, *, informe: str, wide: bool = False,
                 rename: dict[str, str] | None = None, note: str = "") -> None:
        self.key = key
        self.sql = sql
        self.informe = informe
        self.wide = wide
        self.rename = rename or {}
        self.note = note


# =============================================================================
# CAMBIARIO
# =============================================================================
#
# Las siete primeras son datasets directos y se toman VERBATIM de
# build_cambiario_parquets.QUERIES: es la misma consulta que corre hoy en
# producción, así que no se re-escribe acá — se referencia, y si alguien la
# corrige allá esta lista la hereda.

_BCP_Q = sources.bcp.QUERIES

_MONEDAS_XLSM = {
    # bbg_monedas -> encabezado del .xlsm. Reemplaza a los cuatro `select top 30`
    # del .sql de documentación (LATAM / Comparables / Commodities / G10), que
    # difieren entre sí solo por el subconjunto de columnas y por el tope de 30
    # filas. Acá se traen todas juntas y con historia: los paneles base 100 del
    # informe recortan la ventana en la transform (`months`), y con 30 sesiones
    # una ventana de dos meses quedaba corta.
    "Chile": "CLP Cierre",
    "EEUU": "Drivers DXY ",
    "Mexico": "Monedas LATAM MXN",
    "Brasil": "Monedas LATAM BRL",
    "Colombia": "Monedas LATAM COP",
    "Peru": "Monedas LATAM PEN",
    "Argentina": "Monedas LATAM ARS",
    "Rusia": "Monedas Comparables RUBLO RUSO",
    "Sudafrica": "Monedas Comparables RAND SUDAFRICANO",
    "China": "Monedas Comparables RENMINBI CHINO",
    "Hungria": "Monedas Comparables FORINT HUNGARO",
    "Turquia": "Monedas Comparables LIRA TURCA",
    "Korea": "Monedas Comparables WON COREANO",
    "Polonia": "Monedas Comparables ZLOTY POLACO",
    "Filipinas": "Monedas Comparables PESO FILIPINO",
    "Tailandia": "Monedas Comparables BAHT TAILANDÉS",
    "RCheca": "Monedas Comparables CORONA CHECA",
    "Hongkong": "Monedas Comparables DÓLAR HONKONÉS",
    "Singapur": "Monedas Comparables DÓLAR SINGAPUR ",
    "India": "Monedas Comparables RUPIA INDIA",
    "Malasia": "Monedas Comparables RINGGIT MALAYO",
    "Indonesia": "Monedas Comparables RUPIA INDONESA",
    "Bulgaria": "Monedas Comparables LEV BULGARO",
    "Rumania": "Monedas Comparables LEU RUMANO",
    "Noruega": "Monedas Commodities NOK",
    "Australia": "Monedas Commodities AUD",
    "Canada": "Monedas Commodities CAD",
    "NZelanda": "Monedas Commodities NZD",
    "Suecia": "G10 SEK",
    "UK": "G10 GBP",
    "Japon": "G10 JPY",
    "Zeuro": "G10 EUR",
    "Dinamarca": "G10 DKK",
    "Suiza": "G10 CHF",
    # "Monedas Comparables DÓLAR TAIWANES" no tiene columna en bbg_monedas: el
    # panel de monedas comparables se arma sin él (o se agrega la columna al DW).
}

QUERIES: list[Query] = [
    # ── Datasets directos (verbatim de build_cambiario_parquets) ─────────────
    Query("monedas", _BCP_Q["monedas"], informe="cambiario",
          note="variación de monedas en el día"),
    Query("clp_intra", _BCP_Q["clp_intra"], informe="cambiario",
          note="CLP intradía 9:00-14:00, para la volatilidad intradía"),
    Query("fixing", _BCP_Q["fixing"], informe="cambiario",
          note="fixing de la banca por informante y sector"),
    Query("distclp", _BCP_Q["distclp"], informe="cambiario",
          note="histograma del CLP (buckets de 5 pesos)"),
    Query("monedas1m", _BCP_Q["monedas1m"], informe="cambiario",
          note="variación de monedas a 30 días"),
    Query("gammaproxy", _BCP_Q["gammaproxy"], informe="cambiario",
          note="gamma proxy por strike"),
    Query("heatmap", _BCP_Q["heatmap"], informe="cambiario",
          note="gamma proxy por strike y vencimiento"),

    # ── Aportes a la tabla ancha (la ex hoja "Datos" del .xlsm) ──────────────
    Query(
        "clp_tecnico",
        "SELECT c.Fecha, c.CLP, c.[Monto transado], c.MA_10, c.MA_20, c.MA_50, c.MA_100, "
        "c.MA_200, c.BB_UPPER, c.BB_LOWER, f.Valor AS FWD_1M "
        "FROM (SELECT CAST(Fecha AS date) AS Fecha, [FWD Ajus. 30] AS Valor FROM dace.dbo.Base_DMN "
        "WHERE Fecha > '2019-01-01' AND [FWD Ajus. 30] IS NOT NULL) f "
        "RIGHT JOIN (SELECT Fecha, CLP, [Monto transado], "
        "ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),2) AS MA_10, "
        "ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),2) AS MA_20, "
        "ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),2) AS MA_50, "
        "ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 99 PRECEDING AND CURRENT ROW),2) AS MA_100, "
        "ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 199 PRECEDING AND CURRENT ROW),2) AS MA_200, "
        "ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)"
        "+2*STDEV(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),2) AS BB_UPPER, "
        "ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)"
        "-2*STDEV(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),2) AS BB_LOWER "
        "FROM (SELECT CAST(t1.Fecha AS date) AS Fecha, t2.Chile AS CLP, t1.[Monto transado] "
        "FROM dace.dbo.Base_DMN t1 INNER JOIN dace.dbo.bbg_monedas t2 ON t2.Fecha=t1.Fecha "
        "WHERE t1.Fecha>='2019-01-01' AND t1.[Monto transado] IS NOT NULL) A) c "
        "ON f.Fecha=c.Fecha ORDER BY c.Fecha DESC;",
        informe="cambiario", wide=True,
        rename={
            "CLP": "CLP Cierre",
            "Monto transado": "CLP Monto Transado",
            "MA_10": "CLP Media móvil 10 días",
            "MA_20": "CLP Media móvil 20 días",
            "MA_50": "CLP Media móvil 50 días",
            "MA_100": "CLP Media móvil 100 días",
            "MA_200": "CLP Media móvil 200 días",
            "FWD_1M": "Puntos Forward 1M",
        },
        note="CLP, monto, medias móviles, Bollinger y punta forward 1M. Igual que la "
             "query 1 del .sql del DOMA, sin MA_5 (el informe no la dibuja) y desde 2019 "
             "en vez de 2025, que es la historia que piden los gráficos S/R.",
    ),
    Query(
        "fwd_puntas",
        "SELECT CAST(Fecha AS date) AS Fecha, [FWD Ajus. 30] AS [Puntos Forward 1M], "
        "[FWD Ajus. 90] AS [Puntos Forward 3M], [FWD Ajus. 180] AS [Puntos Forward 6M], "
        "[FWD Ajus. 360] AS [Puntos Forward 12M] FROM dace.dbo.Base_DMN "
        "WHERE Fecha > '2019-01-01' ORDER BY Fecha DESC;",
        informe="cambiario", wide=True,
        note="puntas forward 1M/3M/6M/12M. El .sql del DOMA las trae en formato largo "
             "(CROSS APPLY); acá van anchas porque la tabla destino es ancha.",
    ),
    Query(
        "monedas_wide",
        "SELECT * FROM dace.dbo.bbg_monedas WHERE Fecha >= '2019-01-01' ORDER BY Fecha DESC;",
        informe="cambiario", wide=True, rename=_MONEDAS_XLSM,
        note="todas las paridades con historia: CLP, DXY y los cuatro paneles de monedas.",
    ),
    Query(
        "commodities",
        "SELECT * FROM dace.dbo.bbg_commodities WHERE Fecha >= '2019-01-01' ORDER BY Fecha DESC;",
        informe="cambiario", wide=True,
        rename={"Cobre": "Drivers Precio Cobre HGA", "PETRO": "Drivers Petróleo",
                # El precio LME ("Drivers Precio Londres" en el .xlsm) es la única
                # columna que le falta a cam_inventarios_lme y cam_cobre_comex_lme.
                # No sabemos con qué nombre vive en el DW —el .sql del DOMA solo
                # selecciona Cobre y PETRO— así que se prueban los candidatos y el
                # log lista las columnas que llegaron sin mapear, para ajustarlo
                # de una en el servidor.
                "COBRE_LME": "Drivers Precio Londres",
                "Cobre_Londres": "Drivers Precio Londres",
                "LME": "Drivers Precio Londres"},
        note="cobre COMEX, petróleo y (si existe) el precio LME. SELECT * a propósito: "
             "el .sql del DOMA solo pide dos columnas y no sabemos qué más tiene la "
             "tabla. OJO: sin dividir por 100 — el builder de cobre ya convierte de "
             "¢/lb a USD/lb (_HGA_CENTS_PER_LB), y dividir antes daría un precio 100x menor.",
    ),
    Query(
        "pos_nr",
        "SELECT d.Fecha, ROUND(SUM(d.Pos_neta_MM_USD), 0) * -1 AS Pos_neta_MM_USD "
        "FROM dace.Derivados.[Posicion_diaria_(Estadisticas)] d "
        "WHERE d.Sector_contraparte = 'EXTERNO' AND YEAR(d.Fecha) >= 2019 "
        "GROUP BY d.Fecha ORDER BY d.Fecha DESC;",
        informe="cambiario", wide=True,
        rename={"Pos_neta_MM_USD": "No Residentes Posición OffShore"},
        note="posición neta de no residentes. El join con bbg_monedas del .sql original "
             "sobra: el CLP ya entra por monedas_wide.",
    ),
    Query(
        "spc_ois",
        "SELECT * FROM dace.dbo.MIPR_SPC_OIS WHERE Fecha >= '2019-01-01' ORDER BY Fecha DESC;",
        informe="cambiario", wide=True,
        rename={"SPC_1Y": "Misceláneos SPC 1Y", "OIS_1Y": "Misceláneos OIS 1Y",
                "SPC_6M": "Misceláneos SPC 6M", "OIS_6M": "Misceláneos OIS 6M"},
        note="spread SPC-OIS de la tabla de mercado. Los nombres reales de columna se "
             "confirman en el servidor: el rename solo aplica a las que existan.",
    ),

    # =========================================================================
    # IPC
    # =========================================================================
    Query(
        "ipc_ci",
        "SELECT ci.Fecha, ROUND(ci.CI_SPC_2Y, 2) AS CISPC_2y, ROUND(ci.CI_SPC_5Y, 2) AS CISPC_5y, "
        "ROUND(ci.CI_SPC_10Y, 2) AS CISPC_10y FROM dace.dbo.inflation_CI_SPC ci "
        "WHERE ci.Fecha >= DATEADD(MONTH, -12, (SELECT MAX(Fecha) FROM dace.dbo.inflation_CI_SPC)) "
        "ORDER BY ci.Fecha DESC;",
        informe="ipc", note="compensaciones inflacionarias 2y/5y/10y (verbatim del notebook)",
    ),
    Query(
        "ipc_si",
        "SELECT Fecha, SI_12M_continuo, [SI_24M (13 a 24 meses)], SI_dic26 "
        "FROM dace.dbo.inflation_SI "
        "WHERE Fecha >= DATEADD(MONTH, -12, (SELECT MAX(Fecha) FROM dace.dbo.inflation_SI)) "
        "ORDER BY Fecha DESC;",
        informe="ipc", note="seguros de inflación 12M / 1y en 1y / dic-26 (verbatim del notebook)",
    ),
    Query(
        "ipc_si_stairs",
        "SELECT TOP 5 Fecha, septiembre_26, octubre_26, noviembre_26, diciembre_26, enero_27, "
        "febrero_27, marzo_27, abril_27, mayo_27, junio_27 "
        "FROM dace.dbo.inflation_SI_stairs ORDER BY Fecha DESC;",
        informe="ipc",
        note="seguros de inflación por mes de vencimiento. El gráfico muestra la VARIACIÓN en "
             "puntos base entre las dos últimas fechas; la diferencia se calcula en el builder.",
    ),
    Query(
        "ipc_fundamentales",
        "SELECT a.Fecha, a.Chile, b.PETRO FROM dace.dbo.bbg_monedas a "
        "LEFT JOIN dace.dbo.bbg_commodities b ON a.Fecha = b.Fecha "
        "WHERE a.Chile IS NOT NULL AND YEAR(a.Fecha) >= 2025 ORDER BY a.Fecha DESC;",
        informe="ipc", note="CLP y petróleo, los dos fundamentales del gráfico de inflación",
    ),
]


def queries_for(informe: str) -> list[Query]:
    return [q for q in QUERIES if informe in ("ambos", q.informe)]


# =============================================================================
# Datasets del IPC que salen del DW
# =============================================================================

IPC = sources.Registry()


def _con_fecha(frame):
    import pandas as pd

    df = frame.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_compensaciones", chart_type="line", segment="ipc",
    name="Compensaciones inflacionarias SPC",
    description="Compensación inflacionaria implícita en swaps a 2, 5 y 10 años, último año. "
                "Es la medida de expectativas de inflación de largo plazo del informe.",
    unit="%",
    columns=[("Fecha", TS), ("CISPC_2y", D), ("CISPC_5y", D), ("CISPC_10y", D)],
    needs=("ipc_ci",),
)
def _compensaciones(src):
    return _con_fecha(src["ipc_ci"]).dropna()


@IPC.dataset(
    ds_id="ipc_seguros_inflacion", chart_type="line", segment="ipc",
    name="Seguros de inflación",
    description="Inflación esperada a 12 meses, 1 año en 1 año (13 a 24 meses) y al cierre del "
                "año, según seguros de inflación. Último año.",
    unit="%",
    columns=[("Fecha", TS), ("SI_12M_continuo", D), ("SI_24M (13 a 24 meses)", D),
             ("SI_dic26", D)],
    needs=("ipc_si",),
)
def _seguros(src):
    return _con_fecha(src["ipc_si"])


@IPC.dataset(
    ds_id="ipc_variacion_seguros", chart_type="grouped_bar", segment="ipc",
    name="Variación de los seguros de inflación por tenor",
    description="Cambio en puntos base de la inflación implícita en seguros para cada mes de "
                "vencimiento, entre las dos últimas fechas disponibles. Positivo = el mercado "
                "corrigió al alza la inflación esperada de ese mes.",
    unit="pb",
    columns=[("Tenor", V), ("Variacion_pb", D)],
    needs=("ipc_si_stairs",),
)
def _variacion_seguros(src):
    """La diferencia entre las dos últimas fechas, tenor por tenor.

    Réplica del bloque del notebook: se ordena ascendente, se difiere cada
    columna de vencimiento, se pasa a puntos base y se toma la última fila. El
    orden de los tenores es el de la consulta (cronológico), no alfabético — que
    es lo que hace legible el gráfico de barras.
    """
    import pandas as pd

    d = _con_fecha(src["ipc_si_stairs"]).set_index("Fecha").sort_index()
    if len(d) < 2:
        return None
    dif = (d.diff() * 100).iloc[-1]
    return pd.DataFrame({"Tenor": [str(c) for c in d.columns],
                         "Variacion_pb": dif.to_numpy()}).dropna().reset_index(drop=True)


@IPC.dataset(
    ds_id="ipc_fundamentales", chart_type="multi_line_dual", segment="ipc",
    name="Fundamentales de la inflación: CLP y petróleo",
    description="Tipo de cambio y precio del petróleo, los dos fundamentales que el informe usa "
                "para leer la inflación de transables. Doble eje.",
    unit="",
    columns=[("Fecha", TS), ("Chile", D), ("PETRO", D)],
    needs=("ipc_fundamentales",),
)
def _fundamentales(src):
    return _con_fecha(src["ipc_fundamentales"]).dropna()


# =============================================================================
# Ejecución de las consultas
# =============================================================================

def run_queries(qs: list[Query]) -> dict:
    """Corre las consultas vía ``Get_Data``. Una que falle deja fuera solo a SUS
    datasets: el informe conserva el resto de los bloques."""
    try:
        import Get_Data as gd  # type: ignore[import-not-found]
    except ImportError:
        log.error("Get_Data no está disponible: este script corre EN EL SERVIDOR. "
                  "Usá --dump-sql para revisar las consultas sin conectarte.")
        return {}

    out: dict = {}
    for q in qs:
        try:
            out[q.key] = gd.get_data(q.sql)
            log.info("DW - %-14s %d filas", q.key, len(out[q.key]))
        except Exception:
            log.exception("DW - la consulta %s falló", q.key)
    return out


def build_wide(results: dict, qs: list[Query]):
    """Reconstruye la tabla ancha del cambiario (la ex hoja ``Datos``).

    Cada query ``wide`` aporta sus columnas con el encabezado del ``.xlsm``; se
    unen por ``Fecha``. Un rename cuya columna de origen no vino se ignora: el
    dataset que la necesite se saltará solo, con el nombre exacto de lo que falta.
    """
    import pandas as pd

    frames = []
    for q in qs:
        if not q.wide or q.key not in results:
            continue
        df = results[q.key].copy()
        if "Fecha" not in df.columns:
            log.warning("wide - %s no trae columna Fecha; se ignora", q.key)
            continue
        aplicables = {k: v for k, v in q.rename.items() if k in df.columns}
        omitidos = [k for k in q.rename if k not in df.columns]
        if omitidos:
            log.warning("wide - %s: el DW no trajo %s", q.key, ", ".join(omitidos))
        # Las columnas que llegaron y nadie mapeó: se listan para poder completar
        # el `rename` sin tener que abrir la tabla a mano en el servidor. Es como
        # se resuelve el nombre real del precio LME (ver la query `commodities`).
        sin_mapear = [c for c in df.columns if c != "Fecha" and c not in q.rename]
        if sin_mapear:
            log.info("wide - %s: columnas sin mapear -> %s", q.key, ", ".join(map(str, sin_mapear)))
        df = df.rename(columns=aplicables)
        if q.rename:
            # dict.fromkeys y no set: dos claves pueden apuntar al mismo destino
            # (los candidatos del precio LME) y el orden de la hoja importa.
            df = df[["Fecha"] + list(dict.fromkeys(aplicables.values()))]
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        frames.append(df.dropna(subset=["Fecha"]))

    if not frames:
        return None
    out = frames[0]
    for extra in frames[1:]:
        nuevas = ["Fecha"] + [c for c in extra.columns if c != "Fecha" and c not in out.columns]
        out = out.merge(extra[nuevas], on="Fecha", how="outer")
    return out.sort_values("Fecha").reset_index(drop=True)


def ohlc_desde_intra(clp_intra):
    """Apertura / máximo / mínimo diarios del CLP, derivados de los ticks intradía.

    ``cam_clp_ohlc`` (la vela del informe) necesita ``CLP Apertura``, ``CLP Max`` y
    ``CLP minimo``, que en el original venían de Bloomberg por la hoja ``Datos`` y
    hoy no están en ``datos_lea.xlsx``. ``inter_tc_intra`` trae ``Fecha``,
    ``Hora``, ``TC`` y ``Monto``, así que las tres salen de agregar por día:
    apertura = primer tick, máximo y mínimo = extremos de la jornada.

    DOS SALVEDADES, y por eso esto solo se aplica cuando las columnas de Bloomberg
    NO están:

    - la consulta acota a **9:00-14:00**, así que un máximo o mínimo que ocurra
      fuera de esa ventana no se ve; y
    - solo hay intradía desde **2023**, mientras que la vela del informe mira los
      últimos dos meses (le alcanza de sobra, pero la serie histórica queda corta).

    Si en algún momento el recorte de Bloomberg incluye las columnas reales, esas
    mandan: son la jornada completa.
    """
    import pandas as pd

    d = clp_intra.copy()
    fecha, hora, tc = sources.col(d, "Fecha"), sources.col(d, "Hora"), sources.col(d, "TC")
    d[fecha] = pd.to_datetime(d[fecha], errors="coerce")
    d = d.dropna(subset=[fecha, tc]).sort_values([fecha, hora])
    g = d.groupby(fecha)[tc]
    return pd.DataFrame({
        "Fecha": g.first().index,
        "CLP Apertura": g.first().to_numpy(),
        "CLP Max": g.max().to_numpy(),
        "CLP minimo": g.min().to_numpy(),
    }).reset_index(drop=True)


def vela_derivada(results: dict):
    """La vela sacada del intradía, como aporte de última prioridad.

    No decide nada sobre quién gana: devuelve el frame y ``sources.merge_wide``
    lo pone detrás de ``datos_lea.xlsx``. Así, el día que el recorte traiga la
    vela real de Bloomberg, esta pasa a rellenar huecos y nada más.
    """
    if "clp_intra" not in results:
        log.info("cam_clp_ohlc - sin clp_intra; no se deriva la vela")
        return None
    try:
        ohlc = ohlc_desde_intra(results["clp_intra"])
    except Exception:
        log.exception("cam_clp_ohlc - no pude derivar la vela del intradía")
        return None
    log.warning("cam_clp_ohlc - vela derivada de inter_tc_intra (ventana 9:00-14:00, desde 2023): "
                "el máximo sale ~0,7 bajo y el mínimo ~1,0 sobre los reales de Bloomberg "
                "(mediana 0, medido contra la hoja Datos). Para la jornada completa, pedir "
                "CLP Apertura / CLP Max / CLP minimo en datos_lea.xlsx")
    return ohlc


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="DW -> parquets (IPC y Cambiario AM).")
    ap.add_argument("--informe", choices=("ipc", "cambiario", "ambos"), default="ambos")
    ap.add_argument("--out", default=str(sources.PARQUET_DIR), help="Carpeta destino de los parquets")
    ap.add_argument("--stage-dir", default=None,
                    help="Carpeta de intercambio con from_excel.py (default: <out>/_stage)")
    ap.add_argument("--dump-sql", action="store_true",
                    help="Imprime las consultas del informe elegido y sale (no se conecta)")
    ap.add_argument("--emit-catalog", action="store_true",
                    help="Imprime las entradas YAML de los datasets de SQL del IPC y sale")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    sources.setup_logging(args.verbose or args.dump_sql or args.emit_catalog)
    qs = queries_for(args.informe)

    if args.dump_sql:
        for q in qs:
            destino = "tabla ancha" if q.wide else "dataset directo"
            print(f"-- [{q.informe}] {q.key}  ({destino})")
            if q.note:
                print(f"--    {q.note}")
            print(q.sql)
            print()
        print(f"-- {len(qs)} consulta(s)")
        return 0

    if args.emit_catalog:
        print(sources.emit_catalog(IPC.datasets))
        return 0

    out_dir = Path(args.out)
    stage_dir = Path(args.stage_dir) if args.stage_dir else out_dir / "_stage"
    results = run_queries(qs)
    if not results:
        return 2

    written = skipped = 0

    if args.informe in ("ipc", "ambos"):
        w, s = sources.write_all(IPC.datasets, results, out_dir)
        print(f"IPC       - {w} parquet(s)" + (f" - {s} saltado(s)" if s else ""))
        written += w
        skipped += s

    if args.informe in ("cambiario", "ambos"):
        wide = build_wide(results, qs)
        if wide is not None:
            sources.stage_write(wide, stage_dir, "cam_datos_sql")
            propios = {"cam_datos_sql": wide}
            # La vela derivada del intradía va como aporte de MENOR prioridad: si
            # datos_lea.xlsx ya trae la de Bloomberg, esa manda y la derivada solo
            # rellena los huecos que aquella deje (ver sources.PRIORIDAD_CAMBIARIO).
            derivado = vela_derivada(results)
            if derivado is not None:
                sources.stage_write(derivado, stage_dir, "cam_datos_derivado")
                propios["cam_datos_derivado"] = derivado
            wide = sources.stage_merge(stage_dir, propios=propios)
            log.info("cambiario - tabla ancha: %d columnas (datos_lea manda)", len(wide.columns))
            results["datos"] = wide
            log.info("cambiario - tabla combinada DW+stage: %d columnas", len(wide.columns))
        w, s = sources.write_all(sources.bcp.DATASETS, results, out_dir)
        print(f"Cambiario - {w} parquet(s)" + (f" - {s} saltado(s)" if s else ""))
        written += w
        skipped += s

    print(f"OK {written} parquet(s) en {out_dir}" + (f" - {skipped} saltado(s)" if skipped else ""))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
