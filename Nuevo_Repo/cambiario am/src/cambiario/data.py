from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config as cfg

_MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun",
             "jul", "ago", "sep", "oct", "nov", "dic"]

def fecha_corta_es(dt, with_time: bool = False) -> str:
    s = f"{dt.day:02d} {_MESES_ES[dt.month - 1]} {dt.year}"
    if with_time:
        s += f" {dt.hour:02d}:{dt.minute:02d}"
    return s


# ═════════════════════════════════════════════════════════════════════
# EXCEL LOCAL
# ═════════════════════════════════════════════════════════════════════

def resolve_data_path() -> Path:
    if cfg.DATA_PATH.exists():
        return cfg.DATA_PATH
    raise FileNotFoundError(f"No se encontró el Excel en {cfg.DATA_PATH}")

#==================== Carga de data ======================================

def load_data(path: Path | str) -> pd.DataFrame:
    """Carga la hoja 'Datos' del Excel del Informe Cambiario AM."""
    df = pd.read_excel(path, sheet_name="Datos", skiprows=3)
    df = df.rename(columns={df.columns[0]: "Fecha", df.columns[2]: "CLP minimo"})
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha").reset_index(drop=True)

def load_intradia(path: Path | str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Intradía").sort_values("Fecha - Hora")

def load_carrytrade(path: Path | str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="CarryTrade", skiprows=0)

def last_year(frame: pd.DataFrame, fecha_col: str = 'Fecha') -> pd.DataFrame:
    max_date = frame[fecha_col].max()
    return frame[frame[fecha_col] >= max_date - pd.DateOffset(years=1)].copy()

def candlestick_window(df: pd.DataFrame, months: int = 2) -> pd.DataFrame:
    candle = df.iloc[:, 0:7].copy()
    candle['Fecha'] = pd.to_datetime(candle['Fecha'])
    candle = candle[candle['Fecha'] >= candle['Fecha'].max() - pd.DateOffset(months=months)]
    return candle.reset_index(drop=True)

def support_resistance(series: pd.Series, lo: float = 25, hi: float = 75) -> tuple[float, float]:
    lo_val, hi_val = np.percentile(series.dropna(), [lo, hi])
    return lo_val, hi_val

def percentile_summary(
    df: pd.DataFrame,
    fecha_col: str,
    value_col: str,
    *,
    transform: Callable[[pd.Series], pd.Series] | None = None,
    output_col: str | None = None,
    years: int = 1,
) -> dict:
    output_col = output_col or value_col
    windowed = last_year(df[[fecha_col, value_col]].dropna(subset=[value_col]), fecha_col) \
        if years == 1 else \
        df[df[fecha_col] >= df[fecha_col].max() - pd.DateOffset(years=years)][[fecha_col, value_col]] \
            .dropna(subset=[value_col]).copy()

    if transform is not None:
        windowed[output_col] = transform(windowed[value_col])
    elif output_col != value_col:
        windowed[output_col] = windowed[value_col]

    p10, p25, p50, p75, p90, p100 = np.percentile(windowed[output_col], [10, 25, 50, 75, 90, 100])
    return {
        "data": windowed,
        "output_col": output_col,
        "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90, "p100": p100,
        "min_date": windowed[fecha_col].min(),
        "max_date": windowed[fecha_col].max(),
    }


# ═════════════════════════════════════════════════════════════════════
# SQL 

_gd = None


def _get_data_module():
    global _gd
    if _gd is None:
        import sys
        if cfg.GET_DATA_MODULE_PATH not in sys.path:
            sys.path.append(cfg.GET_DATA_MODULE_PATH)
        import Get_Data as gd
        _gd = gd
    return _gd


def _query(sql: str) -> pd.DataFrame:
    return _get_data_module().get_data(sql)


def load_monedas() -> pd.DataFrame:
    return _query("SELECT TOP 5 Fecha, Chile AS CLP, Mexico AS MXN, Brasil AS BRL, Colombia AS COP, Peru AS PEN, Sudafrica AS ZAR, China AS CNY, Korea AS KRW, India AS INR, Filipinas AS PHP, Hungria AS HUF FROM DACE.dbo.bbg_monedas WHERE Chile IS NOT NULL ORDER BY Fecha DESC;")


def load_clp_intra() -> pd.DataFrame:
    return _query(
        "select * from inter_tc_intra where year(Fecha) >= 2023 and Hora between "
        "'9:00:00' and '14:00:00' order by fecha desc, hora asc"
    )


def load_fixing() -> pd.DataFrame:
    return _query(
        "WITH base AS (SELECT CASE WHEN Nombre_informante = 'BANCOBICE' THEN 'BICE' WHEN Nombre_informante IN "
        "('BANCOBTGPA','BTGPACTUAL') THEN 'BTG' WHEN Nombre_informante = 'BANCOCONSO' THEN 'Consorcio' WHEN "
        "Nombre_informante = 'BANCODECHI' THEN 'Chile' WHEN Nombre_informante = 'BANCODECRE' THEN 'BCI' WHEN "
        "Nombre_informante = 'BANCODELES' THEN 'Estado' WHEN Nombre_informante = 'BANCOFALAB' THEN 'Falabella' "
        "WHEN Nombre_informante = 'BANCOINTER' THEN 'Internacional' WHEN Nombre_informante = 'BANCORIPLE' THEN "
        "'Ripley' WHEN Nombre_informante = 'BANCOSANTA' THEN 'Santander' WHEN Nombre_informante = 'BANCOSECUR' "
        "THEN 'Security' WHEN Nombre_informante = 'CHINACONST' THEN 'China Construction Bank' WHEN "
        "Nombre_informante = 'HSBCBANK' THEN 'HSBC' WHEN Nombre_informante = 'ITAUCORPBA' THEN 'Itaú-Corpbanca' "
        "WHEN Nombre_informante = 'JPMORGANCH' THEN 'JP Morgan' WHEN Nombre_informante = 'SCOTIABANK' THEN "
        "'Scotiabank' ELSE 'Nombre no encontrado' END AS NombreInformanteNorm, CASE WHEN UPPER(sector_cont) = "
        "'FFMM' OR UPPER(sector_cont) LIKE '%FFMM%' THEN 'FFMM' WHEN UPPER(sector_cont) = 'AFP' OR "
        "UPPER(sector_cont) LIKE '%AFP%' THEN 'AFP' WHEN UPPER(sector_det_cont) LIKE '%AFP%' THEN 'AFP' WHEN "
        "UPPER(sector_det_cont) LIKE '%FFMM%' THEN 'FFMM' WHEN UPPER(sector_det_cont) LIKE '%PERSONAS%' THEN "
        "'Otros' WHEN UPPER(sector_det_cont) = 'CORREDORAS_DE_BOLSA' THEN 'CB' WHEN UPPER(sector_det_cont) = "
        "'CIAS_DE_SEGUROS' THEN 'CS' WHEN UPPER(sector_det_cont) = 'OFF_SHORE' THEN 'NR' WHEN "
        "UPPER(sector_det_cont) = 'BANCOS' THEN 'Bancos' WHEN UPPER(sector_det_cont) = 'BCCH' THEN 'BCCh' WHEN "
        "UPPER(sector_det_cont) = 'OTROS' THEN 'Otros' WHEN UPPER(sector_det_cont) = 'EMPRESA_REAL' THEN "
        "'Emp_real' WHEN UPPER(sector_det_cont) = 'EMPRESA_FINANCIERA' THEN 'Emp_financiera' ELSE "
        "sector_det_cont END AS SectorContNorm, CAST(Fixing AS date) AS Fixing, Pos_neta FROM "
        "dace.dbo.deriv_vencimientos WHERE YEAR(Fecha_ven) >= 2026 AND YEAR(Fixing) >= 2026 AND "
        "Modalidad_pago = 'C' AND instrumento_nom = 'Forward' AND Nombre_informante NOT IN "
        "('CREDICORPC','EUROAMERIC','LARRAINVIA')) SELECT NombreInformanteNorm, SectorContNorm, Fixing, "
        "SUM(Pos_neta) AS Pos_neta FROM base GROUP BY NombreInformanteNorm, SectorContNorm, Fixing UNION ALL "
        "SELECT 'Total' AS NombreInformanteNorm, SectorContNorm, Fixing, SUM(Pos_neta) AS Pos_neta FROM base "
        "GROUP BY SectorContNorm, Fixing ORDER BY Fixing DESC;"
    )


def load_distclp() -> pd.DataFrame:
    return _query(
        "WITH base AS (SELECT Fecha,Cotizacion FROM mesadineOLTP_.Mercado.Divisas WHERE Paridad='USD' AND "
        "Fecha>=DATEADD(DAY,-400,CAST(GETDATE() AS DATE)) AND Fecha<=CAST(GETDATE() AS DATE)),stats AS "
        "(SELECT AVG(Cotizacion) AS media,STDEV(Cotizacion) AS desviacion FROM base),normalizado AS "
        "(SELECT b.Fecha,b.Cotizacion,(b.Cotizacion-s.media)/s.desviacion AS z_score,s.media,s.desviacion FROM "
        "base b CROSS JOIN stats s),bins AS (SELECT ROUND(z_score,1) AS z_bin,(z_score*desviacion+media) AS "
        "clp_teorico FROM normalizado) SELECT FLOOR(clp_teorico/5)*5 AS CLP_Bucket,COUNT(*) AS Frecuencia FROM "
        "bins WHERE z_bin BETWEEN -2.5 AND 2.5 GROUP BY FLOOR(clp_teorico/5)*5 ORDER BY CLP_Bucket;"
    )


def load_monedas1m() -> pd.DataFrame:
    return _query(
        "WITH base AS (SELECT Fecha,Moneda,Valor,LAG(Valor,30) OVER (PARTITION BY Moneda ORDER BY Fecha) AS "
        "valor_t_30,ROW_NUMBER() OVER (PARTITION BY Moneda ORDER BY Fecha DESC) AS rn FROM (SELECT "
        "Fecha,Moneda,Valor FROM bbg_monedas UNPIVOT (Valor FOR Moneda IN "
        "(Argentina,Australia,Brasil,Bulgaria,Canada,Chile,China,Chinac,Colombia,Dinamarca,EEUU,Filipinas,"
        "Hongkong,Hungria,India,Indonesia,Israel,Japon,Korea,MSCILatam,Malasia,Mexico,NZelanda,Noruega,Peru,"
        "Polonia,RCheca,Rumania,Rusia,Singapur,Sudafrica,Suecia,Suiza,Tailandia,Turquia,UK,Zeuro)) u WHERE "
        "Valor IS NOT NULL AND YEAR(Fecha) >= 2026) t) SELECT Moneda,Fecha,Valor AS valor_t,valor_t_30,"
        "ROUND(((Valor/valor_t_30-1)*100),2) AS variacion_30d FROM base WHERE rn = 1 ORDER BY Moneda"
    ).sort_values('variacion_30d')


def load_gammaproxy() -> pd.DataFrame:
    return _query(
        ";WITH base AS (SELECT Strike, FLOOR(Strike) AS Strike_bucket, Nocional, CAST(Fecha_negociacion AS "
        "date) AS fecha, CAST(Vencimiento AS date) AS venc, DATEDIFF(day, CAST(GETDATE() AS date), "
        "CAST(Vencimiento AS date)) AS dte, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Strike) OVER "
        "(PARTITION BY CAST(Fecha_negociacion AS date)) AS strike_mediana FROM "
        "Inputs.Derivados.SDR_opciones_moneda WHERE Subyacente LIKE '%CLP%' AND Strike IS NOT NULL AND "
        "Strike > 0 AND Nocional > 0 AND Vencimiento >= CAST(GETDATE() AS date)), gamma_calc AS (SELECT "
        "Strike_bucket, MIN(Strike) AS Strike_min, MAX(Strike) AS Strike_max, SUM(Nocional * EXP(-50 * "
        "ABS((Strike * 1.0) / NULLIF(strike_mediana, 0) - 1)) * CASE WHEN dte > 0 THEN 1.0 / SQRT(dte) ELSE "
        "1.0 END) AS gamma_proxy FROM base WHERE dte BETWEEN 0 AND 30 GROUP BY Strike_bucket), top20 AS "
        "(SELECT TOP 30 Strike_bucket, Strike_min, Strike_max, gamma_proxy FROM gamma_calc ORDER BY "
        "gamma_proxy DESC) SELECT Strike_bucket, Strike_min, Strike_max, gamma_proxy / 1000000.0 AS "
        "gamma_proxy_mm FROM top20 ORDER BY Strike_bucket ASC;"
    ).round(2)


def load_heatmap() -> pd.DataFrame:
    return _query(
        ";WITH spot AS (SELECT TOP 1 Cotizacion AS spot_value FROM mesadineOLTP_.Mercado.Divisas WHERE "
        "Paridad = 'USD' ORDER BY Fecha DESC), base AS (SELECT Strike, ROUND(Strike,0) AS Strike_bucket, "
        "CONVERT(date, Vencimiento) AS venc, Nocional, CONVERT(date, Fecha_negociacion) AS fecha, "
        "DATEDIFF(day, CONVERT(date, GETDATE()), CONVERT(date, Vencimiento)) AS dte, PERCENTILE_CONT(0.5) "
        "WITHIN GROUP (ORDER BY Strike) OVER (PARTITION BY CONVERT(date, Fecha_negociacion)) AS "
        "strike_mediana FROM Inputs.Derivados.SDR_opciones_moneda WHERE Subyacente LIKE '%CLP%' AND Strike "
        "> 0 AND Nocional > 0 AND Vencimiento >= CONVERT(date, GETDATE())), gamma_calc AS (SELECT "
        "Strike_bucket, venc, SUM(Nocional * EXP(-50 * ABS((Strike*1.0) / NULLIF(strike_mediana,0) - 1)) * "
        "CASE WHEN dte > 0 THEN 1.0 / SQRT(dte) ELSE 1.0 END) / 1000000.0 AS gamma_proxy FROM base WHERE "
        "dte BETWEEN 0 AND 30 GROUP BY Strike_bucket, venc), range_filtered AS (SELECT g.Strike_bucket, "
        "g.venc, g.gamma_proxy FROM gamma_calc g CROSS JOIN spot s WHERE ABS(g.Strike_bucket - s.spot_value) "
        "<= 40), top10 AS (SELECT TOP 10 Strike_bucket FROM range_filtered GROUP BY Strike_bucket ORDER BY "
        "SUM(gamma_proxy) DESC), top_above AS (SELECT TOP 1 Strike_bucket FROM range_filtered r CROSS JOIN "
        "spot s WHERE r.Strike_bucket > s.spot_value GROUP BY Strike_bucket ORDER BY SUM(gamma_proxy) DESC), "
        "final_strikes AS (SELECT Strike_bucket FROM top10 UNION SELECT Strike_bucket FROM top_above) SELECT "
        "r.Strike_bucket, r.venc, r.gamma_proxy FROM range_filtered r JOIN final_strikes f ON "
        "r.Strike_bucket = f.Strike_bucket ORDER BY r.venc ASC, r.Strike_bucket ASC;"
    ).round(2)
