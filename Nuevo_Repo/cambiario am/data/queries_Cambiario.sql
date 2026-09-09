--QUERYS Informe Cambiario AM:
-- Compilación en SQL para los gráficos .

-- CLP y BB / MM / Punta FWD 1m (fuera 3 grafos)
SELECT c.Fecha,c.CLP,c.[Monto transado],c.MA_5,c.MA_10,c.MA_20,c.MA_50,c.MA_100,c.MA_200,c.BB_STD_20,c.BB_UPPER,c.BB_LOWER,f.Valor AS FWD_1M FROM (SELECT CAST(Fecha AS date) AS Fecha,[FWD Ajus. 30] AS Valor FROM dace.dbo.Base_DMN WHERE Fecha > '2025-01-01' AND [FWD Ajus. 30] IS NOT NULL) f RIGHT JOIN (SELECT Fecha,CLP,[Monto transado],ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),2) AS MA_5,ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),2) AS MA_10,ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),2) AS MA_20,ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),2) AS MA_50,ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 99 PRECEDING AND CURRENT ROW),2) AS MA_100,ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 199 PRECEDING AND CURRENT ROW),2) AS MA_200,ROUND(STDEV(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),2) AS BB_STD_20,ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)+2*STDEV(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),2) AS BB_UPPER,ROUND(AVG(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)-2*STDEV(CLP) OVER (ORDER BY Fecha ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),2) AS BB_LOWER FROM (SELECT CAST(t1.Fecha AS date) AS Fecha,t2.Chile AS CLP,t1.[Monto transado] FROM dace.dbo.Base_DMN t1 INNER JOIN dace.dbo.bbg_monedas t2 ON t2.Fecha=t1.Fecha WHERE t1.Fecha>='2019-01-01' AND t1.[Monto transado] IS NOT NULL) A) c ON f.Fecha=c.Fecha ORDER BY c.Fecha DESC;

-- Puntas FWD 
SELECT Fecha, Serie, Valor FROM (SELECT CAST(Fecha AS date) AS Fecha, [FWD Ajus. 30] AS Fwd_1M, [FWD Ajus. 90] AS Fwd_3M, [FWD Ajus. 180] AS Fwd_6M, [FWD Ajus. 360] AS Fwd_12M FROM dace.dbo.Base_DMN WHERE Fecha > '2025-01-01') src CROSS APPLY (VALUES ('1M', Fwd_1M), ('3M', Fwd_3M), ('6M', Fwd_6M), ('12M', Fwd_12M)) unpvt(Serie, Valor) WHERE Valor IS NOT NULL ORDER BY Fecha DESC;

-- Drivers (2 grafos)
SELECT CAST(t1.Fecha AS date)  AS Fecha, ROUND((t3.COBRE/100),2) AS Cobre, t2.EEUU AS DXY FROM dace.dbo.Base_DMN t1 INNER JOIN dace.dbo.bbg_monedas t2 ON t2.Fecha = t1.Fecha LEFT JOIN dace.dbo.bbg_commodities t3 ON t3.Fecha = t1.Fecha WHERE t1.Fecha >= '2019-01-01' AND t3.COBRE is not null ORDER BY Fecha Desc;
SELECT Fecha, Chile, EEUU from bbg_monedas where Chile is not null and EEUU is not null order by Fecha desc 

-- Variación monedas comparables 1d
WITH base AS (SELECT Fecha,Moneda,Valor,LAG(Valor,1) OVER (PARTITION BY Moneda ORDER BY Fecha) AS valor_t_1,ROW_NUMBER() OVER (PARTITION BY Moneda ORDER BY Fecha DESC) AS rn FROM (SELECT Fecha,Moneda,Valor FROM bbg_monedas UNPIVOT (Valor FOR Moneda IN (Argentina,Australia,Brasil,Bulgaria,Canada,Chile,China,Chinac,Colombia,Dinamarca,EEUU,Filipinas,Hongkong,Hungria,India,Indonesia,Israel,Japon,Korea,MSCILatam,Malasia,Mexico,NZelanda,Noruega,Peru,Polonia,RCheca,Rumania,Rusia,Singapur,Sudafrica,Suecia,Suiza,Tailandia,Turquia,UK,Zeuro)) u WHERE Valor IS NOT NULL AND YEAR(Fecha) >= 2026) t) SELECT Moneda,Fecha,Valor AS valor_t,valor_t_1,ROUND(((Valor/valor_t_1-1)*100),2) AS variacion_1d FROM base WHERE rn = 1 ORDER BY Moneda;

-- Variacion Monedas 1 Mes
WITH base AS (SELECT Fecha,Moneda,Valor,LAG(Valor,30) OVER (PARTITION BY Moneda ORDER BY Fecha) AS valor_t_30,ROW_NUMBER() OVER (PARTITION BY Moneda ORDER BY Fecha DESC) AS rn FROM (SELECT Fecha,Moneda,Valor FROM bbg_monedas UNPIVOT (Valor FOR Moneda IN (Argentina,Australia,Brasil,Bulgaria,Canada,Chile,China,Chinac,Colombia,Dinamarca,EEUU,Filipinas,Hongkong,Hungria,India,Indonesia,Israel,Japon,Korea,MSCILatam,Malasia,Mexico,NZelanda,Noruega,Peru,Polonia,RCheca,Rumania,Rusia,Singapur,Sudafrica,Suecia,Suiza,Tailandia,Turquia,UK,Zeuro)) u WHERE Valor IS NOT NULL AND YEAR(Fecha) >= 2026) t) SELECT Moneda,Fecha,Valor AS valor_t,valor_t_30,ROUND(((Valor/valor_t_30-1)*100),2) AS variacion_30d FROM base WHERE rn = 1 ORDER BY Moneda;

-- Fixing de la banca
WITH base AS (SELECT CASE WHEN Nombre_informante = 'BANCOBICE' THEN 'BICE' WHEN Nombre_informante IN ('BANCOBTGPA','BTGPACTUAL') THEN 'BTG' WHEN Nombre_informante = 'BANCOCONSO' THEN 'Consorcio' WHEN Nombre_informante = 'BANCODECHI' THEN 'Chile' WHEN Nombre_informante = 'BANCODECRE' THEN 'BCI' WHEN Nombre_informante = 'BANCODELES' THEN 'Estado' WHEN Nombre_informante = 'BANCOFALAB' THEN 'Falabella' WHEN Nombre_informante = 'BANCOINTER' THEN 'Internacional' WHEN Nombre_informante = 'BANCORIPLE' THEN 'Ripley' WHEN Nombre_informante = 'BANCOSANTA' THEN 'Santander' WHEN Nombre_informante = 'BANCOSECUR' THEN 'Security' WHEN Nombre_informante = 'CHINACONST' THEN 'China Construction Bank' WHEN Nombre_informante = 'HSBCBANK' THEN 'HSBC' WHEN Nombre_informante = 'ITAUCORPBA' THEN 'Itaú-Corpbanca' WHEN Nombre_informante = 'JPMORGANCH' THEN 'JP Morgan' WHEN Nombre_informante = 'SCOTIABANK' THEN 'Scotiabank' ELSE 'Nombre no encontrado' END AS NombreInformanteNorm, CASE WHEN UPPER(sector_cont) = 'FFMM' OR UPPER(sector_cont) LIKE '%FFMM%' THEN 'FFMM' WHEN UPPER(sector_cont) = 'AFP' OR UPPER(sector_cont) LIKE '%AFP%' THEN 'AFP' WHEN UPPER(sector_det_cont) LIKE '%AFP%' THEN 'AFP' WHEN UPPER(sector_det_cont) LIKE '%FFMM%' THEN 'FFMM' WHEN UPPER(sector_det_cont) LIKE '%PERSONAS%' THEN 'Otros' WHEN UPPER(sector_det_cont) = 'CORREDORAS_DE_BOLSA' THEN 'CB' WHEN UPPER(sector_det_cont) = 'CIAS_DE_SEGUROS' THEN 'CS' WHEN UPPER(sector_det_cont) = 'OFF_SHORE' THEN 'NR' WHEN UPPER(sector_det_cont) = 'BANCOS' THEN 'Bancos' WHEN UPPER(sector_det_cont) = 'BCCH' THEN 'BCCh' WHEN UPPER(sector_det_cont) = 'OTROS' THEN 'Otros' WHEN UPPER(sector_det_cont) = 'EMPRESA_REAL' THEN 'Emp_real' WHEN UPPER(sector_det_cont) = 'EMPRESA_FINANCIERA' THEN 'Emp_financiera' ELSE sector_det_cont END AS SectorContNorm, CAST(Fixing AS date) AS Fixing, Pos_neta FROM dace.dbo.deriv_vencimientos WHERE YEAR(Fecha_ven) >= 2026 AND YEAR(Fixing) >= 2026 AND Modalidad_pago = 'C' AND instrumento_nom = 'Forward' AND Nombre_informante NOT IN ('CREDICORPC','EUROAMERIC','LARRAINVIA')) SELECT NombreInformanteNorm, SectorContNorm, Fixing, SUM(Pos_neta) AS Pos_neta FROM base GROUP BY NombreInformanteNorm, SectorContNorm, Fixing UNION ALL SELECT 'Total' AS NombreInformanteNorm, SectorContNorm, Fixing, SUM(Pos_neta) AS Pos_neta FROM base GROUP BY SectorContNorm, Fixing ORDER BY Fixing DESC;

--Posicion NR (t-2)vs CLP
SELECT d.Fecha, ROUND(SUM(d.Pos_neta_MM_USD), 0) * -1 AS Pos_neta_MM_USD, b.Chile FROM dace.Derivados.[Posicion_diaria_(Estadisticas)] d INNER JOIN bbg_monedas b ON d.Fecha = b.Fecha WHERE d.Sector_contraparte = 'EXTERNO' AND YEAR(d.Fecha) >= 2025 AND b.Chile IS NOT NULL GROUP BY d.Fecha, b.Chile ORDER BY d.Fecha DESC;

-- Monedas LATAM:
select top 30 Fecha, Chile, Colombia, Mexico, Brasil, Peru from bbg_monedas  where Chile is not null and Colombia is not null and Mexico is not null and Brasil is not null  and Peru is not null order by fecha desc;

-- Monedas Comparables
select top 30 Fecha, Chile, Hungria, Korea, Polonia, RCheca as 'Republica Checa', Sudafrica  from bbg_monedas where Chile is not null and Hungria is not null and Korea is not null and Polonia is not null and RCheca is not null order by fecha desc

-- Monedas Commodities 
select top 30 Fecha, Chile, Noruega, Australia, Canada, NZelanda from bbg_monedas where Chile is not null and Noruega is not null and Australia is not null and Canada is not null and NZelanda is not null order by Fecha desc;

-- Monedas G10
select top 30 Fecha, Chile, Suecia, UK, Japon, Zeuro, Dinamarca, Suiza from bbg_monedas where Chile is not null and Suecia is not null and UK is not null and Japon is not null and Zeuro is not null and Dinamarca is not null and Suiza is not null order by Fecha desc;

-- Gamma Proxy
WITH base AS (SELECT Strike, FLOOR(Strike) AS Strike_bucket, Nocional, CAST(Fecha_negociacion AS date) AS fecha, CAST(Vencimiento AS date) AS venc, DATEDIFF(day, CAST(GETDATE() AS date), CAST(Vencimiento AS date)) AS dte, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Strike) OVER (PARTITION BY CAST(Fecha_negociacion AS date)) AS strike_mediana FROM Inputs.Derivados.SDR_opciones_moneda WHERE Subyacente LIKE '%CLP%' AND Strike IS NOT NULL AND Strike > 0 AND Nocional > 0 AND Vencimiento >= CAST(GETDATE() AS date)), gamma_calc AS (SELECT Strike_bucket, MIN(Strike) AS Strike_min, MAX(Strike) AS Strike_max, SUM(Nocional * EXP(-50 * ABS((Strike * 1.0) / NULLIF(strike_mediana, 0) - 1)) * CASE WHEN dte > 0 THEN 1.0 / SQRT(dte) ELSE 1.0 END) AS gamma_proxy FROM base WHERE dte BETWEEN 0 AND 30 GROUP BY Strike_bucket), top20 AS (SELECT TOP 30 Strike_bucket, Strike_min, Strike_max, gamma_proxy FROM gamma_calc ORDER BY gamma_proxy DESC) SELECT Strike_bucket, Strike_min, Strike_max, gamma_proxy / 1000000.0 AS gamma_proxy_mm FROM top20 ORDER BY Strike_bucket ASC;

-- Heatmap gamma proxy
WITH spot AS (SELECT TOP 1 Cotizacion AS spot_value FROM mesadineOLTP_.Mercado.Divisas WHERE Paridad = 'USD' ORDER BY Fecha DESC), base AS (SELECT Strike, ROUND(Strike,0) AS Strike_bucket, CONVERT(date, Vencimiento) AS venc, Nocional, CONVERT(date, Fecha_negociacion) AS fecha, DATEDIFF(day, CONVERT(date, GETDATE()), CONVERT(date, Vencimiento)) AS dte, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Strike) OVER (PARTITION BY CONVERT(date, Fecha_negociacion)) AS strike_mediana FROM Inputs.Derivados.SDR_opciones_moneda WHERE Subyacente LIKE '%CLP%' AND Strike > 0 AND Nocional > 0 AND Vencimiento >= CONVERT(date, GETDATE())), gamma_calc AS (SELECT Strike_bucket, venc, SUM(Nocional * EXP(-50 * ABS((Strike*1.0) / NULLIF(strike_mediana,0) - 1)) * CASE WHEN dte > 0 THEN 1.0 / SQRT(dte) ELSE 1.0 END) / 1000000.0 AS gamma_proxy FROM base WHERE dte BETWEEN 0 AND 30 GROUP BY Strike_bucket, venc), range_filtered AS (SELECT g.Strike_bucket, g.venc, g.gamma_proxy FROM gamma_calc g CROSS JOIN spot s WHERE ABS(g.Strike_bucket - s.spot_value) <= 40), top10 AS (SELECT TOP 10 Strike_bucket FROM range_filtered GROUP BY Strike_bucket ORDER BY SUM(gamma_proxy) DESC), top_above AS (SELECT TOP 1 Strike_bucket FROM range_filtered r CROSS JOIN spot s WHERE r.Strike_bucket > s.spot_value GROUP BY Strike_bucket ORDER BY SUM(gamma_proxy) DESC), final_strikes AS (SELECT Strike_bucket FROM top10 UNION SELECT Strike_bucket FROM top_above) SELECT r.Strike_bucket, r.venc, r.gamma_proxy FROM range_filtered r JOIN final_strikes f ON r.Strike_bucket = f.Strike_bucket ORDER BY r.venc ASC, r.Strike_bucket ASC;
 
 -- SoS base DAP
 select Fecha, [Spread On Shore Base DAP 1M], [Spread On Shore Base DAP 3M], [Spread On Shore Base DAP 6M], [Spread On Shore Base DAP 1M]  from base_DMN order by fecha desc;

 -- CLP vs VOl
 SELECT v.Fecha, ROUND(v.Volatilidad * 100, 3) AS Volatilidad, b.Chile FROM vol_TC_daily v INNER JOIN bbg_monedas b ON v.Fecha = b.Fecha WHERE b.Chile IS NOT NULL ORDER BY v.Fecha DESC;

 -- Histograma CLP
 WITH base AS (SELECT Fecha,Cotizacion FROM mesadineOLTP_.Mercado.Divisas WHERE Paridad='USD' AND Fecha>=DATEADD(DAY,-400,CAST(GETDATE() AS DATE)) AND Fecha<=CAST(GETDATE() AS DATE)),stats AS (SELECT AVG(Cotizacion) AS media,STDEV(Cotizacion) AS desviacion FROM base),normalizado AS (SELECT b.Fecha,b.Cotizacion,(b.Cotizacion-s.media)/s.desviacion AS z_score,s.media,s.desviacion FROM base b CROSS JOIN stats s),bins AS (SELECT ROUND(z_score,1) AS z_bin,(z_score*desviacion+media) AS clp_teorico FROM normalizado) SELECT FLOOR(clp_teorico/5)*5 AS CLP_Bucket,COUNT(*) AS Frecuencia FROM bins WHERE z_bin BETWEEN -2.5 AND 2.5 GROUP BY FLOOR(clp_teorico/5)*5 ORDER BY CLP_Bucket;

 -- Datos tabla

select Fecha, Cobre, PETRO
from bbg_commodities
order by fecha desc 

select * from MIPR_SPC_OIS order by fecha desc


