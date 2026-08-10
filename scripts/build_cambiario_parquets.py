#!/usr/bin/env python3
"""Extracción de los datos del **Informe Cambiario AM** → parquets tidy.

El informe original (`paquete_cambiario/traspaso HTML.txt`) arma sus 41 figuras
leyendo DOS orígenes que hoy NO están en `data_pipeline/parquet/`:

  1. el Excel ``Datos BI Informe Cambiario.xlsm`` (hojas ``Datos``, ``Intradía``
     y ``CarryTrade``), y
  2. siete consultas al DW vía el módulo ``Get_Data`` del servidor.

Este script es el paso 0 del traspaso: recorre esos mismos orígenes —con las
MISMAS queries y las mismas columnas del original— y deja un parquet por
dataset, para que el informe curado (``specs/cambiarioam_spec.py``) lea parquet
como todas las demás familias y no vuelva a tocar ni el Excel ni el DW.

Qué se calcula ACÁ y qué se deja para la transform
--------------------------------------------------
Regla: en el parquet va el dato **crudo por columna**; lo derivado se calcula en
``series_transforms`` para que el gráfico se recalcule solo cuando llegue data
nueva. Se exceptúan tres casos que no son función de una sola columna:

  - conversiones de unidad atadas al origen (cobre ¢/lb → USD/lb, LME USD/t →
    USD/lb): son constantes del dato, no del gráfico;
  - la volatilidad intradía del CLP, que cruza el DW (``inter_tc_intra``) con el
    Excel; y
  - la variación diaria de monedas, que sale de un ``top 10`` del DW.

Bandas de Bollinger, percentiles S/R, niveles 70/30 del RSI, base 100 y el
spread de expectativas TPM se calculan en la transform.

Uso
---
    # en el servidor (Excel + Get_Data disponibles)
    python scripts/build_cambiario_parquets.py
    python scripts/build_cambiario_parquets.py --out data_pipeline/parquet

    # sin DW: solo lo que sale del Excel
    python scripts/build_cambiario_parquets.py --skip-sql

    # sin datos: imprime las entradas de catálogo de los 41 datasets
    python scripts/build_cambiario_parquets.py --emit-catalog

Salida por defecto: ``src/banks_rag/application/reporting/Nuevo Informe/``.
Para producción, ``--out data_pipeline/parquet`` (donde el catálogo los busca) y
después ``python scripts/refresh_catalog_dates.py`` para fijar los `date_range`.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("cambiario")

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _ROOT / "src" / "banks_rag" / "application" / "reporting" / "Nuevo Informe"
_DEFAULT_EXCEL = r"C:\Users\jnunez\Documents\VisualStudio2017\Joaquin\banco\data\raw\Datos BI Informe Cambiario.xlsm"

# Conversiones de unidad del origen (constantes del dato, no del gráfico).
_HGA_CENTS_PER_LB = 100.0     # "Drivers Precio Cobre HGA" viene en ¢/lb
_LME_TON_PER_LB = 2262.64     # "Drivers Precio Londres" viene en USD/t

# ═════════════════════════════════════════════════════════════════════════════
# Consultas al DW — VERBATIM del informe original (traspaso HTML.txt)
# ═════════════════════════════════════════════════════════════════════════════

QUERIES: dict[str, str] = {
    "monedas": "select top 10 * from bbg_monedas order by fecha desc",
    "clp_intra": (
        "select * from inter_tc_intra where year(Fecha) >= 2023 and Hora between "
        "'9:00:00' and '14:00:00' order by fecha desc, hora asc"
    ),
    "fixing": (
        "WITH base AS (SELECT CASE WHEN Nombre_informante = 'BANCOBICE' THEN 'BICE' WHEN "
        "Nombre_informante IN ('BANCOBTGPA','BTGPACTUAL') THEN 'BTG' WHEN Nombre_informante = "
        "'BANCOCONSO' THEN 'Consorcio' WHEN Nombre_informante = 'BANCODECHI' THEN 'Chile' WHEN "
        "Nombre_informante = 'BANCODECRE' THEN 'BCI' WHEN Nombre_informante = 'BANCODELES' THEN "
        "'Estado' WHEN Nombre_informante = 'BANCOFALAB' THEN 'Falabella' WHEN Nombre_informante = "
        "'BANCOINTER' THEN 'Internacional' WHEN Nombre_informante = 'BANCORIPLE' THEN 'Ripley' WHEN "
        "Nombre_informante = 'BANCOSANTA' THEN 'Santander' WHEN Nombre_informante = 'BANCOSECUR' "
        "THEN 'Security' WHEN Nombre_informante = 'CHINACONST' THEN 'China Construction Bank' WHEN "
        "Nombre_informante = 'HSBCBANK' THEN 'HSBC' WHEN Nombre_informante = 'ITAUCORPBA' THEN "
        "'Itaú-Corpbanca' WHEN Nombre_informante = 'JPMORGANCH' THEN 'JP Morgan' WHEN "
        "Nombre_informante = 'SCOTIABANK' THEN 'Scotiabank' ELSE 'Nombre no encontrado' END AS "
        "NombreInformanteNorm, CASE WHEN UPPER(sector_cont) = 'FFMM' OR UPPER(sector_cont) LIKE "
        "'%FFMM%' THEN 'FFMM' WHEN UPPER(sector_cont) = 'AFP' OR UPPER(sector_cont) LIKE '%AFP%' "
        "THEN 'AFP' WHEN UPPER(sector_det_cont) LIKE '%AFP%' THEN 'AFP' WHEN UPPER(sector_det_cont) "
        "LIKE '%FFMM%' THEN 'FFMM' WHEN UPPER(sector_det_cont) LIKE '%PERSONAS%' THEN 'Otros' WHEN "
        "UPPER(sector_det_cont) = 'CORREDORAS_DE_BOLSA' THEN 'CB' WHEN UPPER(sector_det_cont) = "
        "'CIAS_DE_SEGUROS' THEN 'CS' WHEN UPPER(sector_det_cont) = 'OFF_SHORE' THEN 'NR' WHEN "
        "UPPER(sector_det_cont) = 'BANCOS' THEN 'Bancos' WHEN UPPER(sector_det_cont) = 'BCCH' THEN "
        "'BCCh' WHEN UPPER(sector_det_cont) = 'OTROS' THEN 'Otros' WHEN UPPER(sector_det_cont) = "
        "'EMPRESA_REAL' THEN 'Emp_real' WHEN UPPER(sector_det_cont) = 'EMPRESA_FINANCIERA' THEN "
        "'Emp_financiera' ELSE sector_det_cont END AS SectorContNorm, CAST(Fixing AS date) AS "
        "Fixing, Pos_neta FROM dace.dbo.deriv_vencimientos WHERE YEAR(Fecha_ven) >= 2026 AND "
        "YEAR(Fixing) >= 2026 AND Modalidad_pago = 'C' AND instrumento_nom = 'Forward' AND "
        "Nombre_informante NOT IN ('CREDICORPC','EUROAMERIC','LARRAINVIA')) SELECT "
        "NombreInformanteNorm, SectorContNorm, Fixing, SUM(Pos_neta) AS Pos_neta FROM base GROUP BY "
        "NombreInformanteNorm, SectorContNorm, Fixing UNION ALL SELECT 'Total' AS "
        "NombreInformanteNorm, SectorContNorm, Fixing, SUM(Pos_neta) AS Pos_neta FROM base GROUP BY "
        "SectorContNorm, Fixing ORDER BY Fixing DESC;"
    ),
    "distclp": (
        "WITH base AS (SELECT Fecha,Cotizacion FROM mesadineOLTP_.Mercado.Divisas WHERE "
        "Paridad='USD' AND Fecha>=DATEADD(DAY,-400,CAST(GETDATE() AS DATE)) AND Fecha<=CAST(GETDATE"
        "() AS DATE)),stats AS (SELECT AVG(Cotizacion) AS media,STDEV(Cotizacion) AS desviacion "
        "FROM base),normalizado AS (SELECT b.Fecha,b.Cotizacion,(b.Cotizacion-s.media)/s.desviacion"
        " AS z_score,s.media,s.desviacion FROM base b CROSS JOIN stats s),bins AS (SELECT "
        "ROUND(z_score,1) AS z_bin,(z_score*desviacion+media) AS clp_teorico FROM normalizado) "
        "SELECT FLOOR(clp_teorico/5)*5 AS CLP_Bucket,COUNT(*) AS Frecuencia FROM bins WHERE z_bin "
        "BETWEEN -2.5 AND 2.5 GROUP BY FLOOR(clp_teorico/5)*5 ORDER BY CLP_Bucket;"
    ),
    "monedas1m": (
        "WITH base AS (SELECT Fecha,Moneda,Valor,LAG(Valor,30) OVER (PARTITION BY Moneda ORDER BY "
        "Fecha) AS valor_t_30,ROW_NUMBER() OVER (PARTITION BY Moneda ORDER BY Fecha DESC) AS rn "
        "FROM (SELECT Fecha,Moneda,Valor FROM bbg_monedas UNPIVOT (Valor FOR Moneda IN (Argentina,"
        "Australia,Brasil,Bulgaria,Canada,Chile,China,Chinac,Colombia,Dinamarca,EEUU,Filipinas,"
        "Hongkong,Hungria,India,Indonesia,Israel,Japon,Korea,MSCILatam,Malasia,Mexico,NZelanda,"
        "Noruega,Peru,Polonia,RCheca,Rumania,Rusia,Singapur,Sudafrica,Suecia,Suiza,Tailandia,"
        "Turquia,UK,Zeuro)) u WHERE Valor IS NOT NULL AND YEAR(Fecha) >= 2026) t) SELECT Moneda,"
        "Fecha,Valor AS valor_t,valor_t_30,ROUND(((Valor/valor_t_30-1)*100),2) AS variacion_30d "
        "FROM base WHERE rn = 1 ORDER BY Moneda"
    ),
    "gammaproxy": (
        ";WITH base AS (SELECT Strike, FLOOR(Strike) AS Strike_bucket, Nocional, "
        "CAST(Fecha_negociacion AS date) AS fecha, CAST(Vencimiento AS date) AS venc, DATEDIFF(day,"
        " CAST(GETDATE() AS date), CAST(Vencimiento AS date)) AS dte, PERCENTILE_CONT(0.5) WITHIN "
        "GROUP (ORDER BY Strike) OVER (PARTITION BY CAST(Fecha_negociacion AS date)) AS "
        "strike_mediana FROM Inputs.Derivados.SDR_opciones_moneda WHERE Subyacente LIKE '%CLP%' AND"
        " Strike IS NOT NULL AND Strike > 0 AND Nocional > 0 AND Vencimiento >= CAST(GETDATE() AS "
        "date)), gamma_calc AS (SELECT Strike_bucket, MIN(Strike) AS Strike_min, MAX(Strike) AS "
        "Strike_max, SUM(Nocional * EXP(-50 * ABS((Strike * 1.0) / NULLIF(strike_mediana, 0) - 1)) "
        "* CASE WHEN dte > 0 THEN 1.0 / SQRT(dte) ELSE 1.0 END) AS gamma_proxy FROM base WHERE dte "
        "BETWEEN 0 AND 30 GROUP BY Strike_bucket), top20 AS (SELECT TOP 30 Strike_bucket, "
        "Strike_min, Strike_max, gamma_proxy FROM gamma_calc ORDER BY gamma_proxy DESC) SELECT "
        "Strike_bucket, Strike_min, Strike_max, gamma_proxy / 1000000.0 AS gamma_proxy_mm FROM "
        "top20 ORDER BY Strike_bucket ASC;"
    ),
    "heatmap": (
        ";WITH spot AS (SELECT TOP 1 Cotizacion AS spot_value FROM mesadineOLTP_.Mercado.Divisas "
        "WHERE Paridad = 'USD' ORDER BY Fecha DESC), base AS (SELECT Strike, ROUND(Strike,0) AS "
        "Strike_bucket, CONVERT(date, Vencimiento) AS venc, Nocional, CONVERT(date, "
        "Fecha_negociacion) AS fecha, DATEDIFF(day, CONVERT(date, GETDATE()), CONVERT(date, "
        "Vencimiento)) AS dte, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Strike) OVER (PARTITION "
        "BY CONVERT(date, Fecha_negociacion)) AS strike_mediana FROM "
        "Inputs.Derivados.SDR_opciones_moneda WHERE Subyacente LIKE '%CLP%' AND Strike > 0 AND "
        "Nocional > 0 AND Vencimiento >= CONVERT(date, GETDATE())), gamma_calc AS (SELECT "
        "Strike_bucket, venc, SUM(Nocional * EXP(-50 * ABS((Strike*1.0) / NULLIF(strike_mediana,0) "
        "- 1)) * CASE WHEN dte > 0 THEN 1.0 / SQRT(dte) ELSE 1.0 END) / 1000000.0 AS gamma_proxy "
        "FROM base WHERE dte BETWEEN 0 AND 30 GROUP BY Strike_bucket, venc), range_filtered AS "
        "(SELECT g.Strike_bucket, g.venc, g.gamma_proxy FROM gamma_calc g CROSS JOIN spot s WHERE "
        "ABS(g.Strike_bucket - s.spot_value) <= 40), top10 AS (SELECT TOP 10 Strike_bucket FROM "
        "range_filtered GROUP BY Strike_bucket ORDER BY SUM(gamma_proxy) DESC), top_above AS (SELECT"
        " TOP 1 Strike_bucket FROM range_filtered r CROSS JOIN spot s WHERE r.Strike_bucket > "
        "s.spot_value GROUP BY Strike_bucket ORDER BY SUM(gamma_proxy) DESC), final_strikes AS "
        "(SELECT Strike_bucket FROM top10 UNION SELECT Strike_bucket FROM top_above) SELECT "
        "r.Strike_bucket, r.venc, r.gamma_proxy FROM range_filtered r JOIN final_strikes f ON "
        "r.Strike_bucket = f.Strike_bucket ORDER BY r.venc ASC, r.Strike_bucket ASC;"
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# Resolución de columnas del Excel
# ═════════════════════════════════════════════════════════════════════════════
#
# Los encabezados del .xlsm traen espacios de más ("Drivers DXY "), tildes y
# puntos finales ("Terms of Trade."). Buscarlos por igualdad exacta rompe el
# script entero por un espacio; se normaliza (sin tildes, minúsculas, espacios
# colapsados) y, si aun así no aparece, el error dice qué columnas SÍ existen.

def _norm(text: Any) -> str:
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".")


class MissingColumnError(KeyError):
    """Columna del Excel que el informe necesita y la hoja no trae."""


def col(frame, name: str) -> str:
    """Nombre REAL de la columna ``name`` en ``frame`` (tolerante a tildes,
    espacios dobles y punto final). Levanta ``MissingColumnError`` con candidatos."""
    if name in frame.columns:
        return name
    target = _norm(name)
    by_norm = {_norm(c): c for c in frame.columns}
    if target in by_norm:
        return by_norm[target]
    head = target.split()[0] if target.split() else target
    close = [c for n, c in by_norm.items() if head and head in n][:8]
    raise MissingColumnError(
        f"columna {name!r} no está en la hoja; candidatas: {close or list(frame.columns)[:8]}"
    )


def cols(frame, *names: str) -> list[str]:
    return [col(frame, n) for n in names]


def like(frame, fragment: str) -> list[str]:
    """Columnas cuyo nombre CONTIENE ``fragment`` (normalizado), en orden de hoja.
    Réplica de los ``[c for c in df.columns if "Puntos Forward" in c]`` del original."""
    frag = _norm(fragment)
    return [c for c in frame.columns if isinstance(c, str) and frag in _norm(c)]


def pick(frame, mapping: dict[str, str], *, date_col: str | None = None):
    """Sub-DataFrame con las columnas ``{destino: origen}`` resueltas y renombradas.

    Descarta filas sin ninguno de los valores; si ``date_col`` está, además ordena
    por fecha y descarta las filas sin fecha (una fila sin fecha no es graficable)."""
    import pandas as pd

    real = {dst: col(frame, src) for dst, src in mapping.items()}
    out = frame[list(real.values())].copy()
    out.columns = list(real.keys())
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col])
    value_cols = [c for c in out.columns if c != date_col]
    if value_cols:
        out = out.dropna(subset=value_cols, how="all")
    return out.sort_values(date_col).reset_index(drop=True) if date_col else out.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Declaración de datasets
# ═════════════════════════════════════════════════════════════════════════════

class Dataset:
    """Un parquet del informe cambiario: metadata de catálogo + cómo construirlo.

    ``needs`` nombra los orígenes que consume (``datos``/``intradia``/``carry`` del
    Excel, o la clave de una query del DW). El runner salta el dataset —con un
    warning, sin tumbar el resto— cuando alguno de sus orígenes no está
    disponible (p.ej. ``--skip-sql``, o una hoja que el Excel no trae)."""

    def __init__(
        self, ds_id: str, *, chart_type: str, name: str, description: str, unit: str,
        columns: list[tuple[str, str]], needs: tuple[str, ...],
        build: Callable[[dict], Any], segment: str = "cambiarioam",
        value_kind: str = "",
    ) -> None:
        self.id = ds_id
        self.file = f"{ds_id}.parquet"
        self.chart_type = chart_type
        self.name = name
        self.description = description
        self.segment = segment
        self.unit = unit
        self.columns = columns
        self.needs = needs
        self.build = build
        self.value_kind = value_kind

    def catalog_entry(self) -> str:
        """Entrada YAML para ``sql_catalog/parquet_catalog.yaml`` (sin ``date_range``:
        lo fija ``refresh_catalog_dates.py`` desde el parquet real)."""
        lines = [
            f"  - id: {self.id}",
            f"    file: {self.file}",
            f'    chart_type: "{self.chart_type}"',
            f'    name: "{self.name}"',
            "    description: >",
            f"      {self.description}",
            f"    segment: {self.segment}",
            f'    unit: "{self.unit}"',
        ]
        if self.value_kind:
            lines.append(f"    value_kind: {self.value_kind}")
        lines.append("    date_range: []")
        lines.append("    columns:")
        lines += [f"      - {{name: {n}, type: {t}}}" for n, t in self.columns]
        return "\n".join(lines)


DATASETS: list[Dataset] = []


def dataset(**kwargs) -> Callable:
    """Decorador: registra la función como el ``build`` de un ``Dataset``."""
    def wrap(fn: Callable[[dict], Any]) -> Callable[[dict], Any]:
        DATASETS.append(Dataset(build=fn, **kwargs))
        return fn
    return wrap


_TS = "TIMESTAMP"
_D = "DOUBLE"
_V = "VARCHAR"


# ── Resumen: drivers del día ─────────────────────────────────────────────────

@dataset(
    ds_id="cam_drivers_snapshot", chart_type="market_monitor_table",
    name="Snapshot de mercado · drivers del CLP",
    description=(
        "Niveles diarios de los cinco drivers que el Informe Cambiario AM sigue en su tabla de "
        "portada (cobre COMEX, DXY, spread SPC-OIS a 1 año, petróleo Brent y punta forward 30D), "
        "junto al cierre del CLP. La tabla de variación día / semana y el signo de su impacto "
        "sobre el peso se calculan sobre estas columnas."
    ),
    unit="niveles", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("CLP Cierre", _D), ("Cobre", _D), ("DXY", _D),
             ("SPC 1Y", _D), ("OIS 1Y", _D), ("Petróleo", _D), ("Punta FWD 1M", _D)],
)
def _drivers_snapshot(src: dict):
    df = src["datos"]
    return pick(df, {
        "Fecha": "Fecha",
        "CLP Cierre": "CLP Cierre",
        "Cobre": "Drivers Precio Cobre HGA",
        "DXY": "Drivers DXY",
        "SPC 1Y": "Misceláneos SPC 1Y",
        "OIS 1Y": "Misceláneos OIS 1Y",
        "Petróleo": "Drivers Petróleo",
        "Punta FWD 1M": "Puntos Forward 1M",
    }, date_col="Fecha")


@dataset(
    ds_id="cam_monedas_variacion_dia", chart_type="variation_bar",
    name="Variación de monedas en el día",
    description=(
        "Variación porcentual de la última sesión de cada moneda del panel Bloomberg "
        "(bbg_monedas), ordenada de mayor a menor. Es el gráfico de apertura del informe: muestra "
        "de un vistazo qué monedas se apreciaron y cuáles se depreciaron contra el dólar."
    ),
    unit="%", value_kind="return", needs=("monedas",),
    columns=[("Pais", _V), ("Variacion", _D)],
)
def _monedas_variacion_dia(src: dict):
    import pandas as pd

    monedas = src["monedas"].dropna()
    long = pd.melt(monedas, id_vars="Fecha", var_name="Pais", value_name="Valor")
    long = long.sort_values(["Pais", "Fecha"])
    long["Variacion"] = (long.groupby("Pais")["Valor"].pct_change() * 100).round(2)
    last = long.groupby("Pais").tail(1).dropna(subset=["Variacion"])
    return (
        last[["Pais", "Variacion"]]
        .sort_values("Variacion", ascending=False)
        .reset_index(drop=True)
    )


@dataset(
    ds_id="cam_clp_intradia", chart_type="line",
    name="CLP, DXY y cobre intradía (base 100)",
    description=(
        "Evolución intradía del peso, el índice dólar y el cobre llevados a base 100 al inicio de "
        "la ventana, para leer en la misma escala si el movimiento del CLP viene del dólar global, "
        "del cobre o es idiosincrático."
    ),
    unit="Índice base 100", value_kind="level", needs=("intradia",),
    columns=[("Fecha", _TS), ("CLP", _D), ("DXY", _D), ("Cobre", _D)],
)
def _clp_intradia(src: dict):
    return pick(src["intradia"], {
        "Fecha": "Fecha - Hora",
        "CLP": "CLP base 100",
        "DXY": "DXY Base 100",
        "Cobre": "HGA Base 100",
    }, date_col="Fecha")


@dataset(
    ds_id="cam_clp_ohlc", chart_type="candlestick",
    name="Evolución del tipo de cambio (OHLC)",
    description=(
        "Apertura, máximo, mínimo y cierre diarios del USD/CLP más el monto transado de la rueda. "
        "Es la vela del informe: el rango intradía se lee en la mecha y el cuerpo marca si la "
        "sesión cerró sobre o bajo la apertura."
    ),
    unit="CLP/USD", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("Apertura", _D), ("Máximo", _D), ("Mínimo", _D),
             ("Cierre", _D), ("Monto transado", _D)],
)
def _clp_ohlc(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "Apertura": "CLP Apertura",
        "Máximo": "CLP Max",
        "Mínimo": "CLP minimo",
        "Cierre": "CLP Cierre",
        "Monto transado": "CLP Monto Transado",
    }, date_col="Fecha")


# ── Tipo de cambio: análisis técnico ─────────────────────────────────────────

@dataset(
    ds_id="cam_clp_medias_moviles", chart_type="multi_line_dual",
    name="CLP y medias móviles",
    description=(
        "Cierre del USD/CLP contra sus medias móviles (10, 20, 50, 100 y 200 días) y la media "
        "móvil de 5 días del monto transado en el eje derecho. Las de 50 y 200 días son las que "
        "marcan el cruce dorado / de la muerte. Las columnas de media móvil llegan con el nombre "
        "que traiga el Excel (todas las que contienen 'Media móvil'), por eso no van declaradas "
        "acá: la transform toma todas las columnas numéricas del parquet."
    ),
    unit="CLP/USD", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("CLP Cierre", _D), ("Monto transado (MM5d)", _D)],
)
def _clp_medias_moviles(src: dict):
    df = src["datos"]
    ma_cols = like(df, "Media móvil")
    mapping = {"Fecha": "Fecha", "CLP Cierre": "CLP Cierre"}
    mapping.update({c.strip(): c for c in ma_cols})
    out = pick(df, mapping, date_col="Fecha")
    monto = pick(df, {"Fecha": "Fecha", "m": "CLP Monto Transado"}, date_col="Fecha")
    monto["Monto transado (MM5d)"] = monto["m"].rolling(window=5).mean()
    return out.merge(monto[["Fecha", "Monto transado (MM5d)"]], on="Fecha", how="left")


@dataset(
    ds_id="cam_clp_bollinger", chart_type="line",
    name="CLP · bandas de Bollinger",
    description=(
        "Cierre del USD/CLP como insumo de las bandas de Bollinger de 20 días (media ± 2 "
        "desviaciones estándar). Las bandas se calculan en la transform, así se recalculan solas "
        "cuando llega una sesión nueva."
    ),
    unit="CLP/USD", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("CLP Cierre", _D)],
)
def _clp_bollinger(src: dict):
    return pick(src["datos"], {"Fecha": "Fecha", "CLP Cierre": "CLP Cierre"}, date_col="Fecha")


@dataset(
    ds_id="cam_rsi_clp", chart_type="multi_line_dual",
    name="RSI del CLP y de la posición offshore",
    description=(
        "Índices de fuerza relativa (14 días) del peso chileno y de la posición de no residentes, "
        "en ejes separados. Cuando ambos entran juntos en zona de sobrecompra el movimiento del "
        "CLP suele venir acompañado de posicionamiento offshore."
    ),
    unit="RSI", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("RSI Pos. Offshore", _D), ("RSI CLP", _D)],
)
def _rsi_clp(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "RSI Pos. Offshore": "RSI RSI Pos. OffShore",
        "RSI CLP": "RSI RSI CLP",
    }, date_col="Fecha")


@dataset(
    ds_id="cam_clp_sr", chart_type="line",
    name="CLP · soportes y resistencias",
    description=(
        "Cierre del USD/CLP de los últimos doce meses, base de los percentiles P10 a P100 que el "
        "informe usa como soportes y resistencias del régimen reciente."
    ),
    unit="CLP/USD", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("CLP Cierre", _D)],
)
def _clp_sr(src: dict):
    return pick(src["datos"], {"Fecha": "Fecha", "CLP Cierre": "CLP Cierre"}, date_col="Fecha")


@dataset(
    ds_id="cam_clp_volatilidad", chart_type="multi_line_dual",
    name="CLP y volatilidad intradía",
    description=(
        "Cierre del USD/CLP contra la volatilidad intradía del día, medida como desviación "
        "estándar de las variaciones del tipo de cambio entre las 9:00 y las 14:00. Une el dato "
        "tick a tick del DW con el cierre diario del Excel."
    ),
    unit="CLP/USD", value_kind="level", needs=("datos", "clp_intra"),
    columns=[("Fecha", _TS), ("CLP Cierre", _D), ("Volatilidad", _D)],
)
def _clp_volatilidad(src: dict):
    import pandas as pd

    intra = src["clp_intra"].copy()
    intra["Fecha"] = pd.to_datetime(intra[col(intra, "Fecha")], errors="coerce")
    tc = col(intra, "TC")
    intra = intra.dropna(subset=["Fecha"]).sort_values(["Fecha", col(intra, "Hora")])
    intra["Variacion"] = intra.groupby("Fecha")[tc].diff()
    vol = (
        intra.groupby("Fecha")["Variacion"].std().round(3).reset_index()
        .rename(columns={"Variacion": "Volatilidad"})
    )
    # La volatilidad se reporta en porcentaje, como el gráfico original.
    vol["Volatilidad"] = vol["Volatilidad"] * 100
    clp = pick(src["datos"], {"Fecha": "Fecha", "CLP Cierre": "CLP Cierre"}, date_col="Fecha")
    return clp.merge(vol, on="Fecha", how="inner").dropna().reset_index(drop=True)


@dataset(
    ds_id="cam_fixing_bancos", chart_type="stacked_bar",
    name="Fixing de la banca por agente y sector contraparte",
    description=(
        "Posición neta forward que cada banco informante lleva al fixing del día, abierta por "
        "sector de la contraparte (AFP, FFMM, no residentes, empresas, etc.). El total por agente "
        "va superpuesto sobre las barras apiladas."
    ),
    unit="MM USD", value_kind="stock", needs=("fixing",),
    columns=[("Fecha", _TS), ("Informante", _V), ("Sector", _V), ("Pos_neta", _D)],
)
def _fixing_bancos(src: dict):
    import pandas as pd

    fx = src["fixing"].copy()
    fx = fx.rename(columns={
        "NombreInformanteNorm": "Informante",
        "SectorContNorm": "Sector",
        "Fixing": "Fecha",
    })
    fx["Fecha"] = pd.to_datetime(fx["Fecha"], errors="coerce")
    fx["Sector"] = fx["Sector"].fillna("Otros")
    fx = fx.dropna(subset=["Fecha", "Informante"])
    # El original arma el "Total" con un UNION ALL en la query; acá se conserva
    # tal cual viene y solo se agrega por (fecha, agente, sector).
    return (
        fx.groupby(["Fecha", "Informante", "Sector"], as_index=False)["Pos_neta"]
        .sum()
        .sort_values(["Fecha", "Informante"])
        .reset_index(drop=True)
    )


@dataset(
    ds_id="cam_clp_distribucion", chart_type="bar",
    name="Distribución del CLP por tramo de precio",
    description=(
        "Histograma del tipo de cambio teórico de los últimos 400 días agrupado en tramos de 5 "
        "pesos (z-score entre -2,5 y +2,5). Muestra en qué rango el CLP ha pasado más tiempo y, "
        "por lo tanto, dónde el mercado tiene más referencias de precio."
    ),
    unit="Frecuencia (días)", value_kind="level", needs=("distclp",),
    columns=[("Tramo CLP", _D), ("Frecuencia", _D)],
)
def _clp_distribucion(src: dict):
    d = src["distclp"].copy()
    d.columns = ["Tramo CLP", "Frecuencia"]
    return d.sort_values("Tramo CLP").reset_index(drop=True)


# ── No residentes ────────────────────────────────────────────────────────────

@dataset(
    ds_id="cam_pos_no_residentes", chart_type="multi_line_dual",
    name="CLP y posición de no residentes (CLPBODM)",
    description=(
        "Cierre del USD/CLP contra la posición neta forward de no residentes (índice CLPBODM, con "
        "el signo invertido respecto del dato crudo para que un valor más alto sea posición larga "
        "en dólares). Es el gráfico que liga el peso al posicionamiento offshore."
    ),
    unit="CLP/USD", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("CLP Cierre", _D), ("CLPBODM Index", _D)],
)
def _pos_no_residentes(src: dict):
    out = pick(src["datos"], {
        "Fecha": "Fecha",
        "CLP Cierre": "CLP Cierre",
        "CLPBODM Index": "No Residentes Posición OffShore",
    }, date_col="Fecha")
    out["CLPBODM Index"] = out["CLPBODM Index"] * -1
    return out


@dataset(
    ds_id="cam_pos_nr_sr", chart_type="line",
    name="Posición de no residentes · soportes y resistencias",
    description=(
        "Serie del índice CLPBODM (posición offshore, signo invertido) de los últimos doce meses, "
        "base de los percentiles P10 a P100 que acotan cuán extremo está el posicionamiento."
    ),
    unit="MM USD", value_kind="stock", needs=("datos",),
    columns=[("Fecha", _TS), ("CLPBODM Index", _D)],
)
def _pos_nr_sr(src: dict):
    out = pick(src["datos"], {
        "Fecha": "Fecha",
        "CLPBODM Index": "No Residentes Posición OffShore",
    }, date_col="Fecha")
    out["CLPBODM Index"] = out["CLPBODM Index"] * -1
    return out


@dataset(
    ds_id="cam_vol_implicita_1w", chart_type="line",
    name="Volatilidad implícita 1 semana · monedas LATAM",
    description=(
        "Volatilidad implícita de las opciones a una semana del peso chileno, el peso mexicano, el "
        "real brasileño y el peso colombiano. Compara cuánto movimiento está descontando el "
        "mercado en cada moneda de la región."
    ),
    unit="%", value_kind="rate", needs=("datos",),
    columns=[("Fecha", _TS), ("CLP", _D), ("MXN", _D), ("BRL", _D), ("COP", _D)],
)
def _vol_implicita_1w(src: dict):
    base = "Misceláneos Volatilidad implicita Opciones 1W "
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "CLP": base + "CLP", "MXN": base + "MXN",
        "BRL": base + "BRL", "COP": base + "COP",
    }, date_col="Fecha")


@dataset(
    ds_id="cam_fwd_clp", chart_type="multi_line_dual",
    name="CLP y punta forward a 1 mes",
    description=(
        "Cierre del USD/CLP contra la punta forward a 30 días. La punta resume el diferencial de "
        "tasas implícito y suele anticipar presiones de cobertura sobre el spot."
    ),
    unit="CLP/USD", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("CLP Cierre", _D), ("Punta FWD 1M", _D)],
)
def _fwd_clp(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "CLP Cierre": "CLP Cierre",
        "Punta FWD 1M": "Puntos Forward 1M",
    }, date_col="Fecha")


# ── Monedas y carry ──────────────────────────────────────────────────────────

def _currency_builder(mapping: dict[str, str]) -> Callable[[dict], Any]:
    def build(src: dict):
        return pick(src["datos"], {"Fecha": "Fecha", **mapping}, date_col="Fecha")
    return build


_LATAM = {
    "CLP · Peso chileno": "CLP Cierre",
    "MXN · Peso mexicano": "Monedas LATAM MXN",
    "BRL · Real brasileño": "Monedas LATAM BRL",
    "COP · Peso colombiano": "Monedas LATAM COP",
    "PEN · Sol peruano": "Monedas LATAM PEN",
    "ARS · Peso argentino": "Monedas LATAM ARS",
}
_EM = {
    "RUB · Rublo ruso": "Monedas Comparables RUBLO RUSO",
    "ZAR · Rand sudafricano": "Monedas Comparables RAND SUDAFRICANO",
    "CNY · Renminbi chino": "Monedas Comparables RENMINBI CHINO",
    "TRY · Lira turca": "Monedas Comparables LIRA TURCA",
    "KRW · Won coreano": "Monedas Comparables WON COREANO",
    "INR · Rupia india": "Monedas Comparables RUPIA INDIA",
    "PHP · Peso filipino": "Monedas Comparables PESO FILIPINO",
    "THB · Baht tailandés": "Monedas Comparables BAHT TAILANDÉS",
    "MYR · Ringgit malayo": "Monedas Comparables RINGGIT MALAYO",
    "IDR · Rupia indonesia": "Monedas Comparables RUPIA INDONESA",
}
_DEV = {
    "HUF · Forint húngaro": "Monedas Comparables FORINT HUNGARO",
    "PLN · Zloty polaco": "Monedas Comparables ZLOTY POLACO",
    "CZK · Corona checa": "Monedas Comparables CORONA CHECA",
    "HKD · Dólar honkonés": "Monedas Comparables DÓLAR HONKONÉS",
    "SGD · Dólar singapurense": "Monedas Comparables DÓLAR SINGAPUR",
    "TWD · Dólar taiwanés": "Monedas Comparables DÓLAR TAIWANES",
    "BGN · Lev búlgaro": "Monedas Comparables LEV BULGARO",
    "RON · Leu rumano": "Monedas Comparables LEU RUMANO",
}
_G10 = {
    "EUR · Euro": "G10 EUR",
    "GBP · Libra esterlina": "G10 GBP",
    "JPY · Yen japonés": "G10 JPY",
    "CHF · Franco suizo": "G10 CHF",
    "SEK · Corona sueca": "G10 SEK",
    "DKK · Corona danesa": "G10 DKK",
    "NOK · Corona noruega": "Monedas Commodities NOK",
    "AUD · Dólar australiano": "Monedas Commodities AUD",
    "CAD · Dólar canadiense": "Monedas Commodities CAD",
    "NZD · Dólar neozelandés": "Monedas Commodities NZD",
}

for _ds_id, _name, _map, _desc in [
    ("cam_monedas_latam", "Monedas LATAM", _LATAM,
     "Paridades contra el dólar de las principales monedas latinoamericanas más el peso chileno. "
     "La transform las lleva a base 100 al inicio de la ventana para comparar el desempeño "
     "relativo del CLP frente a sus pares regionales."),
    ("cam_monedas_emergentes", "Monedas emergentes", _EM,
     "Paridades contra el dólar de monedas emergentes de Asia, Europa del Este y Sudáfrica. "
     "Llevadas a base 100 permiten separar el movimiento global del dólar del componente "
     "idiosincrático de cada economía."),
    ("cam_monedas_comparables", "Monedas comparables", _DEV,
     "Paridades contra el dólar de las monedas comparables que sigue el informe (Europa central, "
     "Asia desarrollada). Se usan como grupo de control del movimiento del peso."),
    ("cam_monedas_g10", "Monedas G10", _G10,
     "Paridades contra el dólar de las monedas del G10, incluidas las tres monedas commodity "
     "(NOK, AUD, CAD, NZD) que suelen moverse con los términos de intercambio, igual que el CLP."),
]:
    dataset(
        ds_id=_ds_id, chart_type="line", name=_name, description=_desc,
        unit="Índice base 100", value_kind="level", needs=("datos",),
        columns=[("Fecha", _TS)] + [(k, _D) for k in _map],
    )(_currency_builder(_map))


@dataset(
    ds_id="cam_monedas_variacion_30d", chart_type="variation_bar",
    name="Variación de monedas contra el dólar en un mes",
    description=(
        "Variación porcentual a 30 días corridos de cada moneda del panel Bloomberg, ordenada de "
        "la que más se depreció a la que más se apreció. Ubica al peso dentro del movimiento "
        "global del dólar del último mes."
    ),
    unit="%", value_kind="return", needs=("monedas1m",),
    columns=[("Moneda", _V), ("Variacion 30d", _D)],
)
def _monedas_variacion_30d(src: dict):
    m = src["monedas1m"].copy()
    out = m[["Moneda", "variacion_30d"]].rename(columns={"variacion_30d": "Variacion 30d"})
    return out.dropna().sort_values("Variacion 30d").reset_index(drop=True)


@dataset(
    ds_id="cam_carry_trade", chart_type="variation_bar",
    name="Carry trade por país",
    description=(
        "Retorno del carry trade de cada moneda contra el dólar según la planilla del informe. "
        "Un carry positivo premia estar largo en esa moneda y ayuda a explicar los flujos "
        "especulativos hacia o desde el peso."
    ),
    unit="%", value_kind="return", needs=("carry",),
    columns=[("Pais", _V), ("Carry", _D)],
)
def _carry_trade(src: dict):
    c = src["carry"]
    out = pick(c, {"Pais": "País", "Carry": "Value"})
    out["Carry"] = out["Carry"].round(2)
    return out.dropna().reset_index(drop=True)


# ── Drivers: cobre ───────────────────────────────────────────────────────────

@dataset(
    ds_id="cam_cobre_clp", chart_type="multi_line_dual",
    name="Cobre y CLP",
    description=(
        "Precio del cobre COMEX en dólares por libra contra el cierre del USD/CLP. Es la relación "
        "estructural del peso: un cobre más alto mejora los términos de intercambio y tiende a "
        "apreciar la moneda."
    ),
    unit="USD/lb", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("Cobre (USD/lb)", _D), ("CLP Cierre", _D)],
)
def _cobre_clp(src: dict):
    out = pick(src["datos"], {
        "Fecha": "Fecha",
        "Cobre (USD/lb)": "Drivers Precio Cobre HGA",
        "CLP Cierre": "CLP Cierre",
    }, date_col="Fecha")
    out["Cobre (USD/lb)"] = out["Cobre (USD/lb)"] / _HGA_CENTS_PER_LB
    return out


@dataset(
    ds_id="cam_cobre_sr", chart_type="line",
    name="Cobre · soportes y resistencias",
    description=(
        "Precio del cobre COMEX en dólares por libra de los últimos doce meses, base de los "
        "percentiles P10 a P100 que delimitan el rango del régimen reciente."
    ),
    unit="USD/lb", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("Cobre (USD/lb)", _D)],
)
def _cobre_sr(src: dict):
    out = pick(src["datos"], {
        "Fecha": "Fecha", "Cobre (USD/lb)": "Drivers Precio Cobre HGA",
    }, date_col="Fecha")
    out["Cobre (USD/lb)"] = out["Cobre (USD/lb)"] / _HGA_CENTS_PER_LB
    return out


@dataset(
    ds_id="cam_rsi_cobre", chart_type="line",
    name="RSI del cobre",
    description=(
        "Índice de fuerza relativa a 14 días del cobre. Los niveles 70 y 30 marcan sobrecompra y "
        "sobreventa; se dibujan como líneas de referencia en la transform."
    ),
    unit="RSI", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("RSI Cobre", _D)],
)
def _rsi_cobre(src: dict):
    return pick(src["datos"], {"Fecha": "Fecha", "RSI Cobre": "RSI RSI Cobre"}, date_col="Fecha")


@dataset(
    ds_id="cam_inventarios_comex", chart_type="multi_line_dual",
    name="Inventarios COMEX y precio del cobre",
    description=(
        "Inventarios de cobre en bodegas COMEX contra el precio del contrato HGA en dólares por "
        "libra. Inventarios a la baja con precio al alza señalan estrechez física del mercado."
    ),
    unit="Toneladas", value_kind="stock", needs=("datos",),
    columns=[("Fecha", _TS), ("Inventarios COMEX", _D), ("Precio COMEX (USD/lb)", _D)],
)
def _inventarios_comex(src: dict):
    out = pick(src["datos"], {
        "Fecha": "Fecha",
        "Inventarios COMEX": "Drivers Inventarios Comex",
        "Precio COMEX (USD/lb)": "Drivers Precio Cobre HGA",
    }, date_col="Fecha")
    out["Precio COMEX (USD/lb)"] = out["Precio COMEX (USD/lb)"] / _HGA_CENTS_PER_LB
    return out.dropna().reset_index(drop=True)


@dataset(
    ds_id="cam_inventarios_lme", chart_type="multi_line_dual",
    name="Inventarios LME y precio de Londres",
    description=(
        "Inventarios de cobre en bodegas LME contra el precio de Londres convertido a dólares por "
        "libra. Complementa la lectura de COMEX: los dos almacenes suelen moverse en espejo cuando "
        "hay arbitraje entre plazas."
    ),
    unit="Toneladas", value_kind="stock", needs=("datos",),
    columns=[("Fecha", _TS), ("Inventarios LME", _D), ("Precio Londres (USD/lb)", _D)],
)
def _inventarios_lme(src: dict):
    out = pick(src["datos"], {
        "Fecha": "Fecha",
        "Inventarios LME": "Drivers Inv. Cobre",
        "Precio Londres (USD/lb)": "Drivers Precio Londres",
    }, date_col="Fecha")
    out["Precio Londres (USD/lb)"] = out["Precio Londres (USD/lb)"] / _LME_TON_PER_LB
    return out.dropna().reset_index(drop=True)


@dataset(
    ds_id="cam_cobre_comex_lme", chart_type="multi_line_dual",
    name="Cobre COMEX vs Londres y su spread",
    description=(
        "Precio del cobre en COMEX y en Londres, ambos en dólares por libra, con el spread entre "
        "plazas en el eje derecho. Un spread anormalmente ancho anticipa arbitraje físico y "
        "movimientos de inventario entre bodegas."
    ),
    unit="USD/lb", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("COMEX", _D), ("Londres", _D), ("Spread COMEX–Londres", _D)],
)
def _cobre_comex_lme(src: dict):
    out = pick(src["datos"], {
        "Fecha": "Fecha",
        "COMEX": "Drivers Precio Cobre HGA",
        "Londres": "Drivers Precio Londres",
    }, date_col="Fecha").dropna()
    out["COMEX"] = out["COMEX"] / _HGA_CENTS_PER_LB
    out["Londres"] = out["Londres"] / _LME_TON_PER_LB
    out["Spread COMEX–Londres"] = out["COMEX"] - out["Londres"]
    return out.reset_index(drop=True)


# ── Drivers: dólar ───────────────────────────────────────────────────────────

@dataset(
    ds_id="cam_dxy_clp", chart_type="multi_line_dual",
    name="DXY y CLP",
    description=(
        "Índice dólar DXY contra el cierre del USD/CLP. Separa cuánto del movimiento del peso es "
        "fortaleza global del dólar y cuánto es factor local."
    ),
    unit="Índice", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("DXY", _D), ("CLP Cierre", _D)],
)
def _dxy_clp(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha", "DXY": "Drivers DXY", "CLP Cierre": "CLP Cierre",
    }, date_col="Fecha")


@dataset(
    ds_id="cam_dxy_sr", chart_type="line",
    name="DXY · soportes y resistencias",
    description=(
        "Índice dólar DXY de los últimos doce meses, base de los percentiles P10 a P100 que acotan "
        "el rango en que se ha movido el dólar global en el régimen reciente."
    ),
    unit="Índice", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("DXY", _D)],
)
def _dxy_sr(src: dict):
    return pick(src["datos"], {"Fecha": "Fecha", "DXY": "Drivers DXY"}, date_col="Fecha")


@dataset(
    ds_id="cam_rsi_dxy", chart_type="line",
    name="RSI del DXY",
    description=(
        "Índice de fuerza relativa a 14 días del índice dólar. Los niveles 70 y 30 marcan "
        "sobrecompra y sobreventa del dólar global."
    ),
    unit="RSI", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("RSI DXY", _D)],
)
def _rsi_dxy(src: dict):
    return pick(src["datos"], {"Fecha": "Fecha", "RSI DXY": "RSI RSI DXY"}, date_col="Fecha")


# ── Expectativas ─────────────────────────────────────────────────────────────

@dataset(
    ds_id="cam_tpm_spread", chart_type="multi_line_dual",
    name="Expectativas de TPM Chile y Estados Unidos a 1 año",
    description=(
        "Tasa de política monetaria esperada a un año en Chile y en Estados Unidos. La transform "
        "deriva el spread en puntos base y su promedio histórico acumulado: un spread que se "
        "estrecha quita atractivo al carry en pesos."
    ),
    unit="%", value_kind="rate", needs=("datos",),
    columns=[("Fecha", _TS), ("TPM Chile 1Y", _D), ("TPM US 1Y", _D)],
)
def _tpm_spread(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "TPM Chile 1Y": "Expectativas de TPM Expectativas de TPM CL",
        "TPM US 1Y": "Expectativas de TPM Expectativas de TPM US",
    }, date_col="Fecha").dropna().reset_index(drop=True)


@dataset(
    ds_id="cam_puntos_forward", chart_type="line",
    name="Puntas forward del CLP (1M, 3M, 6M y 12M)",
    description=(
        "Puntos forward del peso a uno, tres, seis y doce meses. La pendiente entre plazos resume "
        "el diferencial de tasas que descuenta el mercado y la demanda relativa de cobertura por "
        "tramo."
    ),
    unit="Puntos", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("Fwd 1M", _D), ("Fwd 3M", _D), ("Fwd 6M", _D), ("Fwd 12M", _D)],
)
def _puntos_forward(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "Fwd 1M": "Puntos Forward 1M", "Fwd 3M": "Puntos Forward 3M",
        "Fwd 6M": "Puntos Forward 6M", "Fwd 12M": "Puntos Forward 12M",
    }, date_col="Fecha")


def _curve_builder(contrato: str, tenor: str, price: str) -> Callable[[dict], Any]:
    def build(src: dict):
        import pandas as pd

        df = src["datos"]
        out = df[cols(df, contrato, tenor, price)].copy()
        out.columns = ["Contrato", "Tenor", "Precio"]
        # "Precio" en el Excel real trae alguna celda no numérica suelta (nota,
        # "n/a", espacio) — pandas lee TODA la columna como object (texto +
        # número mezclados) y pyarrow revienta recién al escribir el parquet.
        # Se fuerza a numérico ACÁ, antes del resto del pipeline: lo que no
        # parsea queda NaN y lo saca el dropna() de abajo, en vez de tumbar el
        # dataset entero al final.
        out["Precio"] = pd.to_numeric(out["Precio"], errors="coerce")
        out = out.dropna()
        out["Tenor"] = pd.to_datetime(out["Tenor"], errors="coerce")
        return (
            out.dropna(subset=["Tenor"])
            .sort_values("Tenor")
            .drop_duplicates(subset=["Tenor"])
            .reset_index(drop=True)
        )
    return build


for _ds_id, _name, _args, _unit, _desc in [
    ("cam_curva_hga", "Curva de contratos · cobre COMEX", ("Contrato HGA", "Tenor HGA", "Price HGA"),
     "¢/lb",
     "Precio de cada contrato futuro de cobre COMEX por vencimiento. Una curva en contango "
     "(precios crecientes) indica holgura física; en backwardation, estrechez."),
    ("cam_curva_lma", "Curva de contratos · cobre Londres", ("Contrato Lma", "Tenor Lma", "Price Lma"),
     "USD/t",
     "Precio de cada contrato futuro de cobre en Londres por vencimiento. Se lee junto a la curva "
     "COMEX para detectar desalineamientos entre plazas."),
    ("cam_curva_cl1", "Curva de contratos · petróleo WTI", ("Contrato CL1", "Tenor CL1", "Price CL1"),
     "USD/bbl",
     "Precio de cada contrato futuro de petróleo WTI por vencimiento. El petróleo entra por el "
     "lado importador de los términos de intercambio de Chile."),
]:
    dataset(
        ds_id=_ds_id, chart_type="curve_snapshot", name=_name, description=_desc,
        unit=_unit, value_kind="level", needs=("datos",),
        columns=[("Contrato", _V), ("Tenor", _TS), ("Precio", _D)],
    )(_curve_builder(*_args))


# ── Otros ────────────────────────────────────────────────────────────────────

@dataset(
    ds_id="cam_gamma_proxy", chart_type="bar",
    name="Gamma proxy por strike",
    description=(
        "Proxy de gamma de las opciones sobre CLP con vencimiento dentro de 30 días, agregada por "
        "tramo de strike y ponderada por cercanía al strike mediano y por plazo. Los strikes con "
        "más gamma actúan como imanes o barreras del spot."
    ),
    unit="MM USD", value_kind="stock", needs=("gammaproxy",),
    columns=[("Strike", _D), ("Strike mínimo", _D), ("Strike máximo", _D), ("Gamma", _D)],
)
def _gamma_proxy(src: dict):
    g = src["gammaproxy"].copy()
    g.columns = ["Strike", "Strike mínimo", "Strike máximo", "Gamma"]
    g["Gamma"] = g["Gamma"].round(2)
    return g.sort_values("Strike").reset_index(drop=True)


@dataset(
    ds_id="cam_gamma_heatmap", chart_type="market_monitor_table",
    name="Gamma proxy por strike y vencimiento",
    description=(
        "Misma gamma proxy de las opciones sobre CLP, abierta por strike (filas) y fecha de "
        "vencimiento (columnas), acotada a strikes dentro de 40 pesos del spot. Deja ver en qué "
        "fecha se concentra la exposición de los market makers."
    ),
    unit="MM USD", value_kind="stock", needs=("heatmap",),
    columns=[("Vencimiento", _TS), ("Strike", _D), ("Gamma", _D)],
)
def _gamma_heatmap(src: dict):
    import pandas as pd

    h = src["heatmap"].copy()
    h.columns = ["Strike", "Vencimiento", "Gamma"]
    h["Vencimiento"] = pd.to_datetime(h["Vencimiento"], errors="coerce")
    h = h.dropna(subset=["Vencimiento"])
    return h[["Vencimiento", "Strike", "Gamma"]].reset_index(drop=True)


@dataset(
    ds_id="cam_tasas_implicitas", chart_type="line",
    name="Tasas implícitas en forwards · LATAM",
    description=(
        "Tasa implícita en los contratos forward de Chile, Brasil, Perú, México y Colombia. "
        "Compara el costo de fondeo en dólares que enfrenta cada plaza de la región."
    ),
    unit="%", value_kind="rate", needs=("datos",),
    columns=[("Fecha", _TS), ("Chile", _D), ("Brasil", _D), ("Perú", _D),
             ("México", _D), ("Colombia", _D)],
)
def _tasas_implicitas(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "Chile": "Implicita Chile", "Brasil": "Implicita Brazil",
        "Perú": "Implicita Peru", "México": "Implicita Mexico",
        "Colombia": "Implicita Colombia",
    }, date_col="Fecha")


@dataset(
    ds_id="cam_tot_gs", chart_type="multi_line_dual",
    name="Términos de intercambio (Goldman Sachs) y CLP",
    description=(
        "Índice de términos de intercambio de Goldman Sachs contra el cierre del USD/CLP. Es la "
        "medida de precios relativos de exportación e importación que mejor sigue el fundamento "
        "de mediano plazo del peso."
    ),
    unit="Índice", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("ToT GS", _D), ("CLP Cierre", _D)],
)
def _tot_gs(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "ToT GS": "Términos de intercambio GS Terms of Trade",
        "CLP Cierre": "CLP Cierre",
    }, date_col="Fecha").dropna().reset_index(drop=True)


@dataset(
    ds_id="cam_tot_ct", chart_type="multi_line_dual",
    name="Términos de intercambio (Citi) y CLP",
    description=(
        "Índice de términos de intercambio de Citi contra el cierre del USD/CLP. Se lee junto al "
        "índice de Goldman Sachs: coincidencias entre ambos refuerzan la señal de fundamento."
    ),
    unit="Índice", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("ToT CT", _D), ("CLP Cierre", _D)],
)
def _tot_ct(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha",
        "ToT CT": "Términos de intercambio CT Terms of Trade",
        "CLP Cierre": "CLP Cierre",
    }, date_col="Fecha").dropna().reset_index(drop=True)


@dataset(
    ds_id="cam_tcr", chart_type="line",
    name="Tipo de cambio real",
    description=(
        "Índice de tipo de cambio real del peso chileno. Corrige el tipo de cambio nominal por "
        "inflación relativa y es la referencia de si el peso está caro o barato en términos "
        "estructurales."
    ),
    unit="Índice", value_kind="level", needs=("datos",),
    columns=[("Fecha", _TS), ("TCR", _D)],
)
def _tcr(src: dict):
    return pick(src["datos"], {
        "Fecha": "Fecha", "TCR": "Tipo de cambio real CLP",
    }, date_col="Fecha").dropna().reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Carga de orígenes
# ═════════════════════════════════════════════════════════════════════════════

def load_excel(path: Path) -> dict:
    """Las tres hojas del ``.xlsm``. Una hoja que falte NO aborta: los datasets
    que dependen de ella se saltan con warning y el resto se construye igual."""
    import pandas as pd

    out: dict[str, Any] = {}
    datos = pd.read_excel(path, sheet_name="Datos", skiprows=3)
    # Renombres del original: la primera columna es la fecha y la tercera el
    # mínimo del día, ambas sin encabezado utilizable en la planilla.
    datos = datos.rename(columns={datos.columns[0]: "Fecha", datos.columns[2]: "CLP minimo"})
    datos["Fecha"] = pd.to_datetime(datos["Fecha"], errors="coerce")
    out["datos"] = datos.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    log.info("Excel · hoja Datos: %d filas x %d columnas", len(out["datos"]), len(out["datos"].columns))

    for key, sheet in (("intradia", "Intradía"), ("carry", "CarryTrade")):
        try:
            out[key] = pd.read_excel(path, sheet_name=sheet)
            log.info("Excel · hoja %s: %d filas", sheet, len(out[key]))
        except Exception as exc:  # hoja ausente o corrupta
            log.warning("Excel · hoja %s no disponible (%s)", sheet, exc)
    return out


_CACHE_DF_FILE = "df1.xlsx"
_CACHE_CLP_INTRA_FILES = ("clp_intra1.xlsx", "clp_intra2.xlsx", "clp_intra3.xlsx", "clp_intra4.xlsx")
_CACHE_EXTRAS_FILE = "Excel_cache2.xlsx"
# Hoja de Excel_cache2.xlsx → clave interna de ``sources`` (mismo nombre que la
# variable del script original, salvo "carrytrade"→"carry" para que coincida con
# ``needs=("carry",)`` del dataset ``cam_carry_trade``).
_CACHE_SHEET_TO_KEY = {
    "carrytrade": "carry",
    "distclp": "distclp",
    "fixing": "fixing",
    "gammaproxy": "gammaproxy",
    "heatmap": "heatmap",
    "intradia": "intradia",
    "monedas": "monedas",
    "monedas1m": "monedas1m",
}


def _read_dumped_excel(path: Path):
    """Lee un ``.xlsx`` volcado con ``DataFrame.to_excel()`` (índice incluido):
    la primera columna es el índice posicional sin encabezado útil y se descarta."""
    import pandas as pd

    df = pd.read_excel(path)
    first = df.columns[0]
    if str(first).lower().startswith("unnamed"):
        df = df.drop(columns=[first])
    return df


def load_from_cache(cache_dir: Path) -> dict:
    """Carga los DataFrames ya extraídos A MANO en ``cache_dir`` — reemplaza
    ``load_excel``/``load_sql`` cuando el Excel real y el DW no están disponibles
    acá, pero alguien ya corrió el script original (o equivalente) y volcó sus
    variables a ``.xlsx``. Formato esperado (nombres FIJOS):

    - ``df1.xlsx``          → ``df``, la tabla ancha con drivers/RSI/medias
      móviles YA calculados (las transforms ``cam_*`` los leen tal cual, no
      recalculan nada).
    - ``clp_intraN.xlsx`` (N=1..4) → ``clp_intra`` partido en 4 archivos por
      tamaño (>400k filas cada uno); se concatena preservando el orden.
    - ``Excel_cache2.xlsx`` → un dataset por HOJA, con el nombre de la variable
      original del script (``_CACHE_SHEET_TO_KEY``).

    Un archivo u hoja que falte no aborta: los datasets que dependen de esa
    fuente se saltan con warning y el resto se construye igual."""
    import pandas as pd

    out: dict[str, Any] = {}

    df_path = cache_dir / _CACHE_DF_FILE
    if df_path.exists():
        datos = _read_dumped_excel(df_path)
        datos["Fecha"] = pd.to_datetime(datos["Fecha"], errors="coerce")
        out["datos"] = datos.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
        log.info("cache · %s: %d filas x %d columnas", _CACHE_DF_FILE, len(out["datos"]), len(out["datos"].columns))
    else:
        log.warning("cache · falta %s", _CACHE_DF_FILE)

    intra_parts = []
    for name in _CACHE_CLP_INTRA_FILES:
        path = cache_dir / name
        if not path.exists():
            log.warning("cache · falta %s (clp_intra queda incompleto)", name)
            continue
        intra_parts.append(_read_dumped_excel(path))
    if intra_parts:
        clp_intra = pd.concat(intra_parts, ignore_index=True)
        clp_intra["Fecha"] = pd.to_datetime(clp_intra["Fecha"], errors="coerce")
        out["clp_intra"] = clp_intra
        log.info("cache · clp_intra (%d/%d partes): %d filas",
                 len(intra_parts), len(_CACHE_CLP_INTRA_FILES), len(clp_intra))

    extras_path = cache_dir / _CACHE_EXTRAS_FILE
    if extras_path.exists():
        xls = pd.ExcelFile(extras_path)
        for sheet, key in _CACHE_SHEET_TO_KEY.items():
            if sheet not in xls.sheet_names:
                log.warning("cache · %s no trae la hoja %r", _CACHE_EXTRAS_FILE, sheet)
                continue
            out[key] = pd.read_excel(xls, sheet_name=sheet)
            log.info("cache · %s[%s] -> %r: %d filas", _CACHE_EXTRAS_FILE, sheet, key, len(out[key]))
    else:
        log.warning("cache · falta %s", _CACHE_EXTRAS_FILE)

    return out


def load_sql() -> dict:
    """Las siete consultas del informe vía ``Get_Data`` (módulo del servidor).

    Una query que falle deja fuera solo a sus datasets: el informe conserva el
    resto de bloques y los faltantes salen como tarjeta "sin datos"."""
    try:
        import Get_Data as gd  # type: ignore[import-not-found]
    except ImportError:
        log.warning("Get_Data no está disponible: se omiten los datasets del DW")
        return {}

    out: dict[str, Any] = {}
    for key, sql in QUERIES.items():
        try:
            out[key] = gd.get_data(sql)
            log.info("DW · %s: %d filas", key, len(out[key]))
        except Exception:
            log.exception("DW · la consulta %s falló", key)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def _coerce_declared_numeric(frame: Any, ds: Dataset) -> Any:
    """Fuerza a numérico las columnas que el catálogo declara DOUBLE pero pandas
    leyó como ``object`` — texto y número mezclados en la misma columna del
    Excel real (una nota, un "n/a", una celda vacía leída distinto). Sin esto
    ``to_parquet`` revienta con ``ArrowTypeError`` recién al escribir, tarde
    para saber cuál de los ~40 datasets fue. Usa el TIPO DECLARADO en
    ``ds.columns`` (no adivina por el nombre): una columna de texto genuina
    (``Contrato``, ``Sector``) nunca se toca. Lo que no parsea queda NaN — se
    pierden esas filas puntuales, no la columna ni el dataset entero."""
    import pandas as pd

    for name, decl_type in ds.columns:
        if decl_type != _D or name not in frame.columns or frame[name].dtype != object:
            continue
        before = frame[name].notna().sum()
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        dropped = before - frame[name].notna().sum()
        if dropped:
            log.warning("%-28s columna %r: %d valor(es) no numérico(s) descartado(s)",
                       ds.id, name, dropped)
    return frame


def build_all(sources: dict, out_dir: Path) -> tuple[int, int]:
    """Construye y escribe todos los datasets cuyos orígenes estén disponibles.
    Devuelve ``(escritos, saltados)``. Un dataset que falle —al construirse O AL
    ESCRIBIRSE— se salta con log y NO tumba la corrida: sin esto, un solo Excel
    con una celda sucia dejaba sin generar los otros ~40 datasets que estaban
    bien."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for ds in DATASETS:
        missing = [n for n in ds.needs if n not in sources]
        if missing:
            log.warning("%-28s saltado — falta el origen %s", ds.id, ", ".join(missing))
            skipped += 1
            continue
        try:
            frame = ds.build(sources)
        except MissingColumnError as exc:
            log.warning("%-28s saltado — %s", ds.id, exc)
            skipped += 1
            continue
        except Exception:
            log.exception("%-28s falló al construirse", ds.id)
            skipped += 1
            continue
        if frame is None or frame.empty:
            log.warning("%-28s saltado — sin filas", ds.id)
            skipped += 1
            continue
        frame = _coerce_declared_numeric(frame, ds)
        dst = out_dir / ds.file
        try:
            frame.to_parquet(dst, index=False)
        except Exception:
            log.exception("%-28s falló al escribir el parquet (revisá tipos de columna)", ds.id)
            skipped += 1
            continue
        log.info("%-28s -> %s (%d filas x %d cols)", ds.id, dst.name, len(frame), len(frame.columns))
        written += 1
    return written, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Parquets del Informe Cambiario AM (Excel + DW).")
    ap.add_argument("--excel", default=_DEFAULT_EXCEL, help="Ruta al Datos BI Informe Cambiario.xlsm")
    ap.add_argument("--out", default=str(_DEFAULT_OUT), help="Carpeta destino de los parquets")
    ap.add_argument("--skip-sql", action="store_true", help="No consultar el DW (solo Excel)")
    ap.add_argument("--skip-excel", action="store_true", help="No leer el Excel (solo DW)")
    ap.add_argument(
        "--from-cache", default=None, metavar="DIR",
        help="Carpeta con df1.xlsx + clp_intraN.xlsx + Excel_cache2.xlsx (datos ya "
             "extraídos a mano); reemplaza --excel/--skip-sql por completo.",
    )
    ap.add_argument("--emit-catalog", action="store_true",
                    help="Imprime las entradas YAML de catálogo y sale (no necesita datos)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose or args.emit_catalog else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    if args.emit_catalog:
        print("\n".join(ds.catalog_entry() for ds in DATASETS))
        return 0

    sources: dict[str, Any] = {}
    if args.from_cache:
        cache_dir = Path(args.from_cache)
        if not cache_dir.is_dir():
            log.error("No existe la carpeta de caché %s", cache_dir)
            return 2
        sources.update(load_from_cache(cache_dir))
    else:
        if not args.skip_excel:
            excel = Path(args.excel)
            if not excel.exists():
                log.error("No existe el Excel %s (usá --excel o --skip-excel)", excel)
                return 2
            sources.update(load_excel(excel))
        if not args.skip_sql:
            sources.update(load_sql())

    written, skipped = build_all(sources, Path(args.out))
    print(f"OK {written} parquet(s) en {args.out}" + (f" · {skipped} saltado(s)" if skipped else ""))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
