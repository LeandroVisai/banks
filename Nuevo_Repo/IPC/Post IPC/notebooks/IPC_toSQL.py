# %% [markdown]
# # IPC → SQL (carga de canastas)
# 
# ### Código SOLO para cargar las canastas de IPC, mergearlas y subirlas a SQL
# 
# Además y en caso de, tener acceso a los analíticos del IPC a través de la API del BCCh.
# De paso, genera las tablas finales a través de ETL, para poder consumir de manera directa.
# 
# **Solo corre en el banco**: necesita el fileserver (`RUTA_PROYECTO`/`RUTA_INFLACION`), el módulo interno `Get_Data` y el SQL Server `srvdatscience06`. No forma parte de la corrida mensual automática (`run_mensual.ps1`) del informe — es un paso aparte, manual, que sube la canasta consolidada (2008–hoy) a `Practica_BCCh.inf_Master_CanastasIPC`.
# 
# Distinto del notebook de análisis (`01.Analisis IPC.ipynb`): ese usa las bases 2013/2018/2023 para los gráficos del informe; este usa 2008/2013/2018/2023 para la tabla histórica en SQL.

# %%
import sys
sys.path.append(r"d:\DOMA-DACE\isepulveda\Production\Modulos\Datos")

# %%
import os
import pandas as pd
import numpy as np
import polars as pl
import bcchapi
from datetime import date
from tqdm import tqdm
from sqlalchemy import create_engine, inspect
import Get_Data as gd
import urllib
from urllib import parse

# %%
RUTA_PROYECTO = r"\\fileserver\VOLM\GMN\DOMA\JNUNEZ\ds06\banco\reports\ipc ver2"
RUTA_INFLACION = r"\\fileserver\VOLM\GMN\DOMA\JNUNEZ\ds06\banco\reports\ipc ver2\data"

USUARIO = "bmontenegro@bcentral.cl"
CONTRASENA = "APIDS06"

DESDE = "2024-01-01"
HASTA = str(date.today())

# %%
# Configuración de carga a SQL
SERVER = "srvdatscience06"
DATABASE = "Practica_BCCh"
DRIVER = "ODBC Driver 17 for SQL Server"

conn_str = urllib.parse.quote_plus(
    f"DRIVER={{{DRIVER}}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection=yes;"
)

engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={conn_str}",
    fast_executemany=True
)


def cargar_tabla(df: pd.DataFrame, table: str, key_col: str = None) -> None:
    """
    Carga a SQL. 3 casos según el estado de la tabla (se sube a DATABASE fijada arriba):
      - Tabla no existe      -> CREATE (reemplaza)
      - Existe, sin key_col  -> APPEND directo, sin dedup
      - Existe, con key_col  -> compara key_col contra lo ya cargado en SQL y solo
                                 sube las filas cuyo key_col todavía no está (upsert simple
                                 a nivel de key_col, no de fila -- para inf_Master_CanastasIPC
                                 el key_col es 'Fecha': un mes ya cargado no se vuelve a tocar,
                                 un mes nuevo se agrega completo).
    """
    inspector = inspect(engine)
    tabla_existe = inspector.has_table(table)

    if not tabla_existe:
        df.to_sql(table, con=engine, index=False, if_exists="replace")
        print(f"\n[CREADA] {table} - {len(df)} registros")
        return

    if key_col is None:
        df.to_sql(table, con=engine, index=False, if_exists="append")
        print(f"\n[APPEND] {table} - {len(df)} filas")
        return

    existentes = pd.read_sql(f"SELECT {key_col} FROM {table}", con=engine)
    df_nuevo = df[~df[key_col].isin(existentes[key_col])]

    if df_nuevo.empty:
        print(f"\n[OK] {table} - sin cambios")
        return

    df_nuevo.to_sql(table, con=engine, index=False, if_exists="append")
    print(f"\n[UPDATE] {table} - {len(df_nuevo)} nuevas filas")

# %%
# Carga de datos
carga_pbar = tqdm(total=4, desc="Cargando datos", unit="fuente")

df08 = pd.read_excel(f"{RUTA_INFLACION}/ipc2008.xlsx", skiprows=2)
carga_pbar.update(1)

df13 = pd.read_excel(f"{RUTA_INFLACION}/ipc2013.xlsx", skiprows=2)
carga_pbar.update(1)

df18 = pd.read_excel(f"{RUTA_INFLACION}/ipc2018.xlsx", skiprows=3)
carga_pbar.update(1)

df23 = pd.read_excel(f"{RUTA_INFLACION}/ipc2023.xlsx", skiprows=3)
carga_pbar.update(1)
carga_pbar.close()

# %% [markdown]
# # Carga de las bases consolidadas a SQL
# 
# Crea la base histórica desde 2008 hasta el presente de las canastas de IPC.

# %%
# df08 y df13 traen los encabezados con " ( % )"; df18/df23 ya vienen limpios.
RENAME_COLUMNAS = {
    "Variación Mensual ( % )":    "Variación Mensual (%)",
    "Variación Acumulada ( % )":  "Variación Acumulada (%)",
    "Variación 12 Meses ( % )":   "Variación 12 Meses (%)",
    "Incidencia Mensual ( % )":   "Incidencia Mensual (%)",
    "Incidencia Acumulada ( % )": "Incidencia Acumulada (%)",
    "Incidencia 12 Meses ( % )":  "Incidencia 12 Meses (%)",
}
df08 = df08.rename(columns=RENAME_COLUMNAS)
df13 = df13.rename(columns=RENAME_COLUMNAS)

finalDf = (
    pd.concat([df08, df23, df18, df13], ignore_index=True)
    .sort_values(by=['Año', 'Mes'], ascending=[False, False])
    .reset_index(drop=True)
)

finalDf['Fecha'] = pd.to_datetime(finalDf['Año'].astype(str)+'-'+finalDf['Mes'].astype(str)+'-01').dt.date
fecha = finalDf.pop('Fecha')
finalDf.insert(0, 'Fecha', fecha)

for nombre, df in [('2023', df23), ('2018', df18), ('2013', df13), ('2008', df08)]:
    print(f"=== Base {nombre}: {df.shape[0]} filas, {df.shape[1]} columnas ===")
    faltantes = [c for c in ('Año', 'Mes', 'Glosa') if c not in df.columns]
    # Control de errores en caso que existan algunos datos faltantes según las glosas, año y mes.
    if faltantes:
        print(f"  Aviso: columnas clave ausentes: {faltantes}")
    else:
        nulos = df[['Año', 'Mes']].isna().sum()
        if nulos.sum():
            print(f"  Aviso: nulos en Año/Mes -> {nulos.to_dict()}")
    display(df.head(1))

cargar_tabla(finalDf, 'inf_Master_CanastasIPC', key_col='Fecha')


