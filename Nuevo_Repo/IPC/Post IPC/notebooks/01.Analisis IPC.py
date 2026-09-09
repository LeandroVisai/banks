# %% [markdown]
# # Análisis IPC — DOMA
# 
# **Contexto:** análisis mensual del IPC para el informe DOMA. Consolida las bases 2013/2018/2023 del INE, calcula la difusión inflacionaria (general y sin volátiles), clasifica la canasta según el diccionario IPC 2023 y descarga los analíticos del BCCh.
# 
# **Objetivo:** generar las tablas y gráficos del informe y el Excel de salida para BI.

# %% [markdown]
# ## 1. Configuración

# %%
import sys
sys.path.append(r"d:\DOMA-DACE\isepulveda\Production\Modulos\Datos")
import Get_Data as gd

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import bcchapi
from datetime import date
from pathlib import Path
from tqdm import tqdm

# Rutas relativas a la carpeta del propio notebook 
RUTA_PROYECTO = str(Path.cwd().resolve().parent)
RUTA_INFLACION = f"{RUTA_PROYECTO}/data"

USUARIO = "joaquin.nunez@usach.cl"
CONTRASENA = "Joaquin06"  # BCCh API

DESDE = "2024-01-01"
DESDE_HISTORIA = "2010-01-01"
HASTA = date.today()

# Estilo del informe: identidad visual del banco. Se aplica al final de cada figura
# con fig.update_layout(**ESTILO_INFORME), así pisa el template que traiga cada una.
#
# La leyenda va SIEMPRE abajo y centrada: arriba se comía el alto útil del gráfico y
# en las tarjetas del informe chocaba con el título de la tarjeta. Se define acá y no
# en cada figura para que las siete queden iguales; como este update_layout corre
# último, pisa los legend=dict(y=1.02) que traen algunas celdas.
#
# El margen inferior acompaña a la leyenda: sin él Plotly la dibuja encima de las
# etiquetas del eje X. Los 110 no son al azar: en una tarjeta de media columna la
# leyenda de las bandas (una entrada por año) se envuelve a dos filas, y con menos
# margen quedaba justo al borde — al agregarse un año más se cortaba.
# El ancho/alto no va acá: el informe los quita para que las figuras sean
# responsivas dentro de su tarjeta.
ESTILO_INFORME = dict(
    font=dict(family="Calibri", color="#44546A"),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                font=dict(size=10), tracegroupgap=2),
    margin=dict(l=55, r=25, t=30, b=110),
)

# %% [markdown]
# ## 2. Ingestión y Carga de Datos
# 
# 

# %%
bde = bcchapi.Siete(USUARIO, CONTRASENA)

carga_pbar = tqdm(total=6, desc="Cargando datos", unit="fuente")

df13 = pd.read_excel(f"{RUTA_INFLACION}/ipc2013.xlsx", skiprows=2)
carga_pbar.update(1)

df18 = pd.read_excel(f"{RUTA_INFLACION}/ipc2018.xlsx", skiprows=3)
carga_pbar.update(1)

df23 = pd.read_excel(f"{RUTA_INFLACION}/ipc2023.xlsx", skiprows=3)
carga_pbar.update(1)

ine_ipc = bde.cuadro(
    series = [  "F074.IPC.VAR.Z.2023.C.M",
                "F074.IPCSAE.VAR.Z.2023.C.M",
                "F074.IPCT.VAR.Z.2023.C.M",
                "F074.IPCN.VAR.Z.2023.C.M",
                "F074.IPCFV.VAR.Z.2023.C.M",
                "F074.IPCA.VAR.Z.2023.C.M",
                "F074.IPCS.VAR.Z.2023.C.M",
                "F074.IPCB.VAR.Z.2023.C.M",
                "F074.IPCE.VAR.Z.2023.C.M",
                "F074.IPCVIV.VAR.Z.2023.C.M",
                "F074.IPCSVIV.VAR.Z.2023.C.M"],
    nombres=["IPC General",
             "IPC SAE",
             "IPC Transables",
             "IPC No transables",
             "IPC Frutas y verduras",
             "IPC Alimentos",
             "IPC Servicios",
             "IPC Bienes",
             "IPC Energía",
             "IPC Vivienda",
             "IPC Servicios menos vivienda"],
    desde = DESDE, hasta = HASTA
).reset_index().rename(columns={'index':'Fecha'})
carga_pbar.update(1)

# IPC Volátiles / Sin Volátiles (BCCh, familia G073 "IPC Analíticos"): series
# empalmadas oficiales, no algo que convenga recalcular producto a producto.
variacion_bcch = bde.cuadro(
    series=["G073.IPCV.VAR.2023.M", "G073.IPCSV.VAR.2023.M"],
    nombres=["IPC Volátiles", "IPC Sin Volátiles"],
    desde=DESDE, hasta=HASTA
).reset_index().rename(columns={'index': 'Fecha'})
carga_pbar.update(1)

# Bienes y servicios sin volátiles, misma familia G073 y también empalmadas por el
# BCCh desde 1999. Es la lógica del informe Excel del banco: la banda sale de la
# serie agregada, no de sumar productos uno a uno.
sv_bcch = bde.cuadro(
    series=["G073.IPCBSV.VAR.2023.M", "G073.IPCSSV.VAR.2023.M"],
    nombres=["Bienes SV", "Servicios SV"],
    desde=DESDE_HISTORIA, hasta=HASTA
).reset_index().rename(columns={'index': 'Fecha'})
sv_bcch['Fecha'] = pd.to_datetime(sv_bcch['Fecha'])
sv_bcch['Año'] = sv_bcch['Fecha'].dt.year
sv_bcch['Mes'] = sv_bcch['Fecha'].dt.month
carga_pbar.update(1)
carga_pbar.close()


# %% [markdown]
# ## 3. Limpieza y consolidación

# %%
# Estandarizar los nombres de las columnas de df13
df13 = df13.rename(columns={
    "Variación Mensual ( % )":    "Variación Mensual (%)",
    "Variación Acumulada ( % )":  "Variación Acumulada (%)",
    "Variación 12 Meses ( % )":   "Variación 12 Meses (%)",
    "Incidencia Mensual ( % )":   "Incidencia Mensual (%)",
    "Incidencia Acumulada ( % )": "Incidencia Acumulada (%)",
    "Incidencia 12 Meses ( % )":  "Incidencia 12 Meses (%)",
})
df13.head(0)

# %%
# Concatenar, ordenar y resetear índice
finalDf = (
    pd.concat([df23, df18, df13], ignore_index=True)
    .sort_values(by=['Año', 'Mes'], ascending=[False, False])
    .reset_index(drop=True)
)
finalDf['Fecha'] = pd.to_datetime(finalDf['Año'].astype(str)+'-'+finalDf['Mes'].astype(str)+'-01')


# %% [markdown]
# ## 4. Análisis: difusión inflacionaria del IPC general

# %%
# --- Filtrar productos válidos ---
finalDf_validos = finalDf[
    finalDf['Producto'].notna() &
    (finalDf['Producto'] != '') &
    (finalDf['Producto'] != 0)
].copy()

# --- Calcular difusión mensual ---
finalDf_validos['Variación Mensual Positiva'] = finalDf_validos['Variación Mensual (%)'] > 0

variacion_positiva = (
    finalDf_validos
    .groupby(['Año', 'Mes'], observed=True)['Variación Mensual Positiva']
    .sum()
    .reset_index()
)
total_productos = (
    finalDf_validos
    .groupby(['Año', 'Mes'], observed=True)
    .size()
    .reset_index(name='Total Productos')
)

variacion_positiva = variacion_positiva.merge(total_productos, on=['Año', 'Mes'])
variacion_positiva['Difusión (%)'] = (
    variacion_positiva['Variación Mensual Positiva'] / variacion_positiva['Total Productos'] * 100
).round(2)

# --- Años de referencia: se detectan desde los datos, no se hardcodean ---
ANIO_ACTUAL = int(variacion_positiva['Año'].max())
ANIO_ANTERIOR = ANIO_ACTUAL - 1
ANIO_MIN = int(variacion_positiva['Año'].min())

# --- Banda histórica: ventana FIJA (excluye a propósito 2022+, el shock inflacionario
# post-pandemia, para que la banda refleje comportamiento "normal") en vez de "todos los
# años hasta el anterior". El informe de referencia (Excel del banco) usa 2010-2021;
# nuestra data propia por producto solo alcanza ANIO_MIN, así que el inicio queda en lo
# que haya disponible (hoy 2014) en vez de fingir 2010.
ANIO_BASE_INICIO = max(ANIO_MIN, 2010)
ANIO_BASE_FIN = 2021

variacion_filtrada = variacion_positiva[
    variacion_positiva['Año'].between(ANIO_BASE_INICIO, ANIO_BASE_FIN)
]

df_resumen_difusion = (
    variacion_filtrada
    .groupby('Mes', observed=True)['Difusión (%)']
    .agg(['min', 'max', 'mean'])
    .reset_index()
    .rename(columns={
        'min':  'Difusión Mínima Histórica (%)',
        'max':  'Difusión Máxima Histórica (%)',
        'mean': 'Difusión Promedio Histórica (%)',
    })
)
df_resumen_difusion['Difusión Promedio Histórica (%)'] = df_resumen_difusion['Difusión Promedio Histórica (%)'].round(2)
df_resumen_difusion['Año Base Inicio'] = ANIO_BASE_INICIO
df_resumen_difusion['Año Base Fin'] = ANIO_BASE_FIN

# --- Años a graficar como líneas individuales: desde 2023, saltando 2022 (shock) ---
# igual que el informe de referencia. Se leen directamente de variacion_positiva
# (formato largo) en la celda del gráfico, así el esquema de esta tabla no crece
# cada año que pasa.
ANIOS_LINEA_DIFUSION = [a for a in range(2024, ANIO_ACTUAL + 1) if a != 2022]

# %% [markdown]
# ## 5. Diccionario IPC 2023: clasificación de la canasta
# 
# Dummies del diccionario oficial: sin volátiles, bienes y servicios. Con el join canasta–diccionario se analiza el último mes por grupo de volatilidad.

# %%
# --- Cargar diccionario IPC 2023 y definir listas de productos ---
dic_ipc = pd.read_excel(
    f"{RUTA_PROYECTO}/data/diccionario ipc 2023.xlsx"
)

# Columnas relevantes (por posición, independiente de encoding)
_col_nombre  = dic_ipc.columns[2]   # PRODUCTO IPC 2023=100
_col_sinvol  = dic_ipc.columns[15]  # IPC SIN VOLATILES
_col_bienes  = dic_ipc.columns[9]   # BIENES

# Listas de nombres (uppercase, sin espacios extras)
nombres_sinvol    = set(
    dic_ipc[dic_ipc[_col_sinvol] == 1][_col_nombre]
    .dropna().str.strip().str.upper()
)

# Normalizar Glosa en finalDf_validos para el matching
finalDf_validos['Glosa_norm'] = finalDf_validos['Glosa'].str.strip().str.upper()

print(f'Productos IPC Sin Volatiles  : {len(nombres_sinvol)}')


# %%
# --- Canasta último mes: volátiles vs sin volátiles (variaciones e incidencias) ---
# El diccionario trae dummies IPC VOLÁTILES / IPC SIN VOLÁTILES por producto;
# las últimas filas del Excel son totales (conteo y ponderación), se filtran por código.
_col_codigo = dic_ipc.columns[1]   # CÓDIGO IPC 2023=100
_col_pond   = dic_ipc.columns[4]   # POND 2023=100
_col_vol    = dic_ipc.columns[16]  # IPC VOLÁTILES

clasificacion = (
    dic_ipc[dic_ipc[_col_codigo].notna()]
    .assign(
        Glosa_norm=lambda d: d[_col_nombre].str.strip().str.upper(),
        Clasificación=lambda d: np.where(d[_col_vol] == 1, 'Volátil', 'Sin volátiles'),
    )
    .rename(columns={_col_pond: 'Ponderación 2023'})
    [['Glosa_norm', 'Clasificación', 'Ponderación 2023']]
)

# Join con la canasta en el último mes disponible
ULTIMO_MES = finalDf_validos['Fecha'].max()
cols_analisis = ['Glosa', 'Glosa_norm',
                 'Variación Mensual (%)', 'Incidencia Mensual (%)',
                 'Variación 12 Meses (%)', 'Incidencia 12 Meses (%)']

canasta_ultimo_mes = (
    finalDf_validos.loc[finalDf_validos['Fecha'] == ULTIMO_MES, cols_analisis]
    .merge(clasificacion, on='Glosa_norm', how='left')
)

sin_match = canasta_ultimo_mes['Clasificación'].isna().sum()
if sin_match:
    print(f"Aviso: {sin_match} productos del último mes sin match en el diccionario:")
    display(canasta_ultimo_mes.loc[canasta_ultimo_mes['Clasificación'].isna(), 'Glosa'])

canasta_ultimo_mes = (
    canasta_ultimo_mes
    .drop(columns='Glosa_norm')
    .sort_values('Incidencia Mensual (%)', ascending=False)
    .reset_index(drop=True)
)
canasta_ultimo_mes.insert(0, 'Fecha', ULTIMO_MES)

# %% [markdown]
# ## 6. Difusión por subcategoría: IPC sin volátiles

# %%
# --- Funcion reutilizable para calcular difusion de una subcategoria ---
def calcular_difusion_subcat(
    df_validos, nombres_set, etiqueta,
    anio_actual=ANIO_ACTUAL, anio_anterior=ANIO_ANTERIOR, anio_min=ANIO_MIN,
    ponderada=False,
):
    """
    df_validos : finalDf_validos con columna Glosa_norm
    nombres_set: set de nombres de productos de la subcategoria
    etiqueta   : nombre descriptivo para prints
    anio_actual, anio_anterior, anio_min : por defecto usan los años detectados
                 automáticamente desde variacion_positiva (ver celda anterior)
    ponderada  : si True, la difusión pondera cada producto por su peso en la
                 canasta (columna Ponderación) en vez de conteo simple
    Retorna: (variacion_positiva_df, df_resumen_df)
    """
    sub = df_validos[df_validos['Glosa_norm'].isin(nombres_set)].copy()

    print(f'\n=== {etiqueta} ===')
    print('Productos unicos por año:')
    print(sub.groupby('Año', observed=True)['Glosa_norm'].nunique())

    # Variaciones positivas (ponderadas por peso en canasta si ponderada=True)
    if ponderada:
        sub['Variación Mensual Positiva'] = sub['Ponderación'] * (sub['Variación Mensual (%)'] > 0)
        var_pos  = sub.groupby(['Año', 'Mes'], observed=True)['Variación Mensual Positiva'].sum().reset_index()
        tot_prod = sub.groupby(['Año', 'Mes'], observed=True)['Ponderación'].sum().reset_index(name='Total Productos')
    else:
        sub['Variación Mensual Positiva'] = sub['Variación Mensual (%)'] > 0
        var_pos  = sub.groupby(['Año', 'Mes'], observed=True)['Variación Mensual Positiva'].sum().reset_index()
        tot_prod = sub.groupby(['Año', 'Mes'], observed=True).size().reset_index(name='Total Productos')
    var_pos  = var_pos.merge(tot_prod, on=['Año', 'Mes'])
    var_pos['Difusión (%)'] = (
        var_pos['Variación Mensual Positiva'] / var_pos['Total Productos'] * 100
    ).round(2)

    print('\nUltimas filas de variacion positiva:')
    display(var_pos.tail(10))

    # Resumen historico: todos los años completos hasta el año anterior
    filt = var_pos[var_pos['Año'].between(anio_min, anio_anterior)]
    resumen = (
        filt.groupby('Mes', observed=True)['Difusión (%)']
        .agg(['min', 'max', 'mean'])
        .reset_index()
        .rename(columns={
            'min':  'Difusión Mínima Histórica (%)',
            'max':  'Difusión Máxima Histórica (%)',
            'mean': 'Difusión Promedio Histórica (%)',
        })
    )
    resumen['Difusión Promedio Histórica (%)'] = resumen['Difusión Promedio Histórica (%)'].round(2)

    # Trayectorias año actual y año anterior (nombres de columna fijos, sin el número de año)
    datos_actual = var_pos[var_pos['Año'] == anio_actual][['Mes', 'Difusión (%)']].rename(
        columns={'Difusión (%)': 'Difusión Año Actual (%)'}
    )
    datos_anterior = var_pos[var_pos['Año'] == anio_anterior][['Mes', 'Difusión (%)']].rename(
        columns={'Difusión (%)': 'Difusión Año Anterior (%)'}
    )
    resumen = resumen.merge(datos_actual, on='Mes', how='left').merge(datos_anterior, on='Mes', how='left')
    resumen['Año Actual'] = anio_actual
    resumen['Año Anterior'] = anio_anterior

    print('\nResumen historico por mes:')
    display(resumen)

    return var_pos, resumen

# %%
# DIFUSION IPC SIN VOLATILES
variacion_positiva_sv, df_resumen_difusion_sv = calcular_difusion_subcat(
    finalDf_validos, nombres_sinvol, 'IPC Sin Volatiles'
)

# %% [markdown]
# ## 7. Visualización: difusión IPC general

# %%
# --- Difusión IPC general: gráfico Plotly (banda histórica fija + una línea por año) ---
MESES = {
    1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
}

fig_difusion_general = go.Figure()

# Borde superior del rango — invisible, solo ancla el fill
fig_difusion_general.add_trace(go.Scatter(
    x=df_resumen_difusion['Mes'], y=df_resumen_difusion['Difusión Máxima Histórica (%)'],
    mode='lines', line=dict(width=0),
    showlegend=False, hoverinfo='skip', name='_max'
))

# Banda histórica (fill hacia la traza anterior)
fig_difusion_general.add_trace(go.Scatter(
    x=df_resumen_difusion['Mes'], y=df_resumen_difusion['Difusión Mínima Histórica (%)'],
    mode='lines', line=dict(width=0),
    fill='tonexty', fillcolor='rgba(173,216,230,0.5)',
    name=f'Rango {ANIO_BASE_INICIO}-{ANIO_BASE_FIN} (%)',
    customdata=df_resumen_difusion[['Difusión Mínima Histórica (%)', 'Difusión Máxima Histórica (%)']].values,
    hovertemplate='Rango: %{customdata[0]:.1f}% – %{customdata[1]:.1f}%<extra></extra>'
))

# Promedio histórico
fig_difusion_general.add_trace(go.Scatter(
    x=df_resumen_difusion['Mes'], y=df_resumen_difusion['Difusión Promedio Histórica (%)'],
    mode='lines+markers',
    marker=dict(symbol='circle', size=8, color='orange'),
    line=dict(color='orange', width=2),
    name=f'Promedio {ANIO_BASE_INICIO}-{ANIO_BASE_FIN} (%)',
    hovertemplate='Promedio: %{y:.1f}%<extra></extra>'
))

# Una línea por año (2023 en adelante, salta 2022): degradé gris-verde para los años
# pasados, el año en curso siempre destaca en verde sólido y trazo grueso.
_colores_linea = px.colors.sequential.Greens[2:2 + len(ANIOS_LINEA_DIFUSION)]
for i, anio in enumerate(ANIOS_LINEA_DIFUSION):
    datos_anio = variacion_positiva[variacion_positiva['Año'] == anio][['Mes', 'Difusión (%)']]
    es_actual = anio == ANIO_ACTUAL
    color = '#2ca02c' if es_actual else _colores_linea[i]
    fig_difusion_general.add_trace(go.Scatter(
        x=datos_anio['Mes'], y=datos_anio['Difusión (%)'],
        mode='lines+markers',
        marker=dict(symbol='square' if es_actual else 'diamond', size=9 if es_actual else 6, color=color),
        line=dict(color=color, width=3 if es_actual else 1.5, dash=None if es_actual else 'dot'),
        name=f'Difusión {anio} (%)',
        hovertemplate=f'{anio}: ' + '%{y:.1f}%<extra></extra>'
    ))

fig_difusion_general.update_layout(
    title=dict(
        text=f'Difusión Mensual de Variación Positiva del IPC ({ANIO_BASE_INICIO}-{ANIO_ACTUAL})',
        font=dict(size=14, family='Arial', color='#222')
    ),
    xaxis=dict(
        title='Mes', tickmode='array',
        tickvals=list(MESES.keys()),
        ticktext=list(MESES.values())
    ),
    yaxis=dict(title='Difusión (%)', range=[0, 100]),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    hovermode='x unified',
    plot_bgcolor='white',
    width=950, height=520,
    template='plotly_white'
)


fig_difusion_general.update_xaxes(showgrid=False, gridcolor='rgba(0,0,0,0.1)')
fig_difusion_general.update_yaxes(showgrid=False, gridcolor='rgba(0,0,0,0.1)', range=[25, 75])

fig_difusion_general.update_layout(**ESTILO_INFORME)
fig_difusion_general.show()


# %% [markdown]
# ## 8. IPC por grupo y división
# 
# Variación mensual del IPC volátiles/sin volátiles/total (series oficiales del BCCh vía API, mismo mecanismo que el resto de los analíticos), y descomposición del último mes por división y grupo de la canasta (treemap + incidencia 12 meses, data propia). Solo base 2023.

# %%
# --- Variación mensual IPC: volátiles, sin volátiles y total ---
# "IPC Volátiles"/"IPC Sin Volátiles": series oficiales del BCCh (variacion_bcch,
# sección 2), igual mecanismo que ine_ipc para el resto de los analíticos. "IPC
# Total" no viene del BCCh: es la fila agregada 'IPC General' que el propio INE
# publica dentro de finalDf (Producto vacío, por eso quedó fuera de finalDf_validos).
# grupos_canasta y base23 se calculan aquí porque el treemap de la celda siguiente
# también los reutiliza.
grupos_canasta = (
    dic_ipc[dic_ipc[_col_codigo].notna()]
    .assign(
        Glosa_norm=lambda d: d[_col_nombre].str.strip().str.upper(),
        **{'Bien/Servicio': lambda d: np.where(d[_col_bienes] == 1, 'Bienes', 'Servicios')},
    )
    [['Glosa_norm', 'Bien/Servicio']]
)

base23 = (
    finalDf_validos[finalDf_validos['Fecha'] >= '2024-01-01']
    .merge(clasificacion, on='Glosa_norm', how='left')
    .merge(grupos_canasta, on='Glosa_norm', how='left')
)

mask_ipc = finalDf['Glosa'].fillna('').str.strip().str.upper().eq('IPC GENERAL')
ipc_general = (
    finalDf.loc[mask_ipc & (finalDf['Fecha'] >= '2024-01-01'), ['Fecha', 'Variación Mensual (%)']]
    .rename(columns={'Variación Mensual (%)': 'IPC Total'})
)

#Ventana de 12 meses
fecha_max = variacion_bcch['Fecha'].max()
incidencias_grupo = (
    variacion_bcch.loc[variacion_bcch['Fecha'] >= fecha_max - pd.DateOffset(months = 12) ]
    .merge(ipc_general, on='Fecha', how='left')
    .round({'IPC Volátiles': 3, 'IPC Sin Volátiles': 3, 'IPC Total': 3})
    .sort_values('Fecha')
    .reset_index(drop=True)
)

fig_incidencias_grupo = go.Figure()
fig_incidencias_grupo.add_trace(go.Bar(
    x=incidencias_grupo['Fecha'], y=incidencias_grupo['IPC Volátiles'],
    name='IPC volátiles', marker_color='#d62728',
    hovertemplate='IPC volátiles: %{y:.2f}%<extra></extra>',
))
fig_incidencias_grupo.add_trace(go.Bar(
    x=incidencias_grupo['Fecha'], y=incidencias_grupo['IPC Sin Volátiles'],
    name='IPC sin volátiles', marker_color='#1f77b4',
    hovertemplate='IPC sin volátiles: %{y:.2f}%<extra></extra>',
))
fig_incidencias_grupo.add_trace(go.Scatter(
    x=incidencias_grupo['Fecha'], y=incidencias_grupo['IPC Total'],
    mode='lines+markers', name='IPC total (publicado INE)',
    line=dict(color='black', width=2), marker=dict(size=6),
    hovertemplate='IPC total: %{y:.2f}%<extra></extra>',
))
fig_incidencias_grupo.update_layout(
    barmode='overlay',
    title='Variación mensual IPC: volátiles, sin volátiles y total',
    yaxis_title='Variación mensual (%)',
    legend=dict(orientation='h', yanchor='bottom', y=-.5, xanchor='left', x=.5), width=950, height=460, template='plotly_white',
)
fig_incidencias_grupo.update_layout(**ESTILO_INFORME)
fig_incidencias_grupo.show()

# %%
# --- Último mes: descomposición por división (treemap + incidencias 12 meses) ---
# Nombres de división y grupo desde las filas AGREGADAS de df23:
#   - fila de división: División set, Grupo NaN
#   - fila de grupo:    Grupo set, Clase NaN
#
# OJO con la llave de 'Grupo': el INE numera los grupos de forma CORRELATIVA DENTRO
# de cada división, no global. Hay 46 grupos repartidos en solo 8 números distintos,
# así que el Grupo 1 existe en las 13 divisiones (Alimentos, Vestuario, Arriendo,
# Seguros, ...). Cruzar solo por 'Grupo' genera un producto cartesiano que replica
# cada producto 13 veces y lo cuelga de divisiones ajenas. La llave es el par
# ('División', 'Grupo'); los validate='many_to_one' de abajo lo dejan explícito y
# hacen fallar el merge en vez de inflar la tabla en silencio.
divisiones = (
    df23.loc[df23['División'].notna() & df23['Grupo'].isna(), ['División', 'Glosa']]
    .drop_duplicates()
    .assign(**{'División Nombre': lambda d: d['Glosa'].str.strip().str.title()})
    [['División', 'División Nombre']]
)
grupos = (
    df23.loc[df23['Grupo'].notna() & df23['Clase'].isna(), ['División', 'Grupo', 'Glosa']]
    .drop_duplicates()
    .assign(**{'Grupo Nombre': lambda d: d['Glosa'].str.strip().str.title()})
    [['División', 'Grupo', 'Grupo Nombre']]
)

_canasta_mes = base23[base23['Fecha'] == ULTIMO_MES]
ultimo_mes_div = (
    _canasta_mes
    .merge(divisiones, on='División', how='left', validate='many_to_one')
    .merge(grupos, on=['División', 'Grupo'], how='left', validate='many_to_one')
)

# Control: el merge no debe crear ni perder filas, y las ponderaciones deben sumar
# la canasta completa. Si esto se rompe, el treemap y las incidencias por división
# quedan mal y no es evidente a simple vista.
assert len(ultimo_mes_div) == len(_canasta_mes), (
    f'el merge alteró el número de filas: {len(_canasta_mes)} -> {len(ultimo_mes_div)}')
_pond = ultimo_mes_div['Ponderación'].sum()
if abs(_pond - 100) > 0.5:
    print(f'Aviso: la ponderación no suma ~100% ({_pond:.2f}%), revisar cobertura de la canasta')

_sin_nombre = ultimo_mes_div['Grupo Nombre'].isna().sum() + ultimo_mes_div['División Nombre'].isna().sum()
if _sin_nombre:
    print(f'Aviso: {_sin_nombre} producto(s) sin nombre de división o grupo')

# Fallbacks para no dejar nodos vacíos en el treemap si faltara algún nombre agregado
ultimo_mes_div['División Nombre'] = ultimo_mes_div['División Nombre'].fillna(
    'División ' + ultimo_mes_div['División'].astype(str))
ultimo_mes_div['Grupo Nombre'] = ultimo_mes_div['Grupo Nombre'].fillna(
    'División ' + ultimo_mes_div['División'].astype(str)
    + ' · Grupo ' + ultimo_mes_div['Grupo'].astype(str))

# Treemap: jerarquía IPC General -> División -> Grupo -> Producto.
treemap_df = ultimo_mes_div[ultimo_mes_div['Ponderación'] > 0].copy()
# Agregar IPC General como nivel superior
treemap_df['IPC General'] = 'IPC General'
_cap = float(np.ceil(np.nanpercentile(treemap_df['Variación Mensual (%)'].abs(), 90)))
_cap = max(_cap, 1.0)

fig_treemap = px.treemap(
    treemap_df,
    path=['IPC General', 'División Nombre', 'Grupo Nombre', 'Glosa'],
    values='Ponderación',
    color='Variación Mensual (%)', color_continuous_scale='RdBu_r',
    color_continuous_midpoint=0, range_color=[-_cap, _cap],
    hover_data={'Ponderación': ':.2f',
                'Variación Mensual (%)': ':.2f',
                'Incidencia Mensual (%)': ':.3f'},
    title=(f'Canasta IPC por división, grupo y producto — {ULTIMO_MES:%B %Y}<br>')
)
fig_treemap.update_layout(width=950, height=600, template='plotly_white',
                          coloraxis_colorbar_title='Var. mens.<br>(%)')
fig_treemap.update_layout(**ESTILO_INFORME)
fig_treemap.show()

# Incidencias 12 meses acumuladas por división (barra divergente)
inc12_division = (
    ultimo_mes_div.groupby('División Nombre')['Incidencia 12 Meses (%)']
    .sum(min_count=1).round(3).sort_values()
)
fig_inc12_division = go.Figure(go.Bar(
    x=inc12_division.values, y=inc12_division.index, orientation='h',
    marker_color=np.where(inc12_division.values >= 0, '#d62728', '#1f77b4'),
    hovertemplate='%{y}: %{x:.2f} pp<extra></extra>',
))
fig_inc12_division.update_layout(
    title=f'Incidencia 12 meses por división — {ULTIMO_MES:%B %Y}',
    xaxis_title='Puntos porcentuales', width=950, height=520, template='plotly_white',
)
fig_inc12_division.update_layout(**ESTILO_INFORME)
fig_inc12_division.show()
fig_treemap.write_html("IPC Agosto.html")

# %% [markdown]
# ## 9. IPC bienes/servicios sin volátiles: banda de percentiles
# 
# Variación mensual de las series analíticas del BCCh (`G073.IPCBSV` y `G073.IPCSSV`, empalmadas desde 1999) contra su banda 10-90% por mes calendario en 2010-2020. Misma lógica que los gráficos N°6 y N°7 del informe Excel del banco.
# 

# %%
# --- IPC bienes/servicios sin volátiles: variación mensual con banda de percentiles ---
# Igual que el informe Excel del banco: la variación sale de la serie agregada del
# BCCh (empalmada desde 1999), no de promediar productos. Ventana 2010-2020 como allá.
ANIO_BASE_SV_INICIO, ANIO_BASE_SV_FIN = 2010, 2020
ANIOS_LINEA_SV = ANIOS_LINEA_DIFUSION

def serie_sv(nombre):
    return sv_bcch[['Año', 'Mes', nombre]].rename(columns={nombre: 'Variación Mensual (%)'})

var_bienes_sv = serie_sv('Bienes SV')
var_servicios_sv = serie_sv('Servicios SV')

def banda_percentil_mensual(df_var):
    base = df_var[df_var['Año'].between(ANIO_BASE_SV_INICIO, ANIO_BASE_SV_FIN)]
    return (
        base.groupby('Mes', observed=True)['Variación Mensual (%)']
        .agg(p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90), promedio='mean')
        .reset_index()
    )

def fig_banda_sv(df_var, titulo):
    banda = banda_percentil_mensual(df_var)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=banda['Mes'], y=banda['p90'], mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=banda['Mes'], y=banda['p10'], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(173,216,230,0.5)',
        name=f'Rango 10-90% ({ANIO_BASE_SV_INICIO}-{ANIO_BASE_SV_FIN})',
        customdata=banda[['p10']].values,
        hovertemplate='Rango: %{customdata[0]:.2f}% – %{y:.2f}%<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=banda['Mes'], y=banda['promedio'], mode='lines+markers',
        marker=dict(symbol='circle', size=7, color='orange'), line=dict(color='orange', width=2),
        name=f'Promedio {ANIO_BASE_SV_INICIO}-{ANIO_BASE_SV_FIN}',
        hovertemplate='Promedio: %{y:.2f}%<extra></extra>',
    ))
    colores = px.colors.sequential.Greens[2:2 + len(ANIOS_LINEA_SV)]
    for i, anio in enumerate(ANIOS_LINEA_SV):
        es_actual = anio == ANIO_ACTUAL
        color = '#2ca02c' if es_actual else colores[i]
        d = df_var[df_var['Año'] == anio][['Mes', 'Variación Mensual (%)']]
        fig.add_trace(go.Scatter(
            x=d['Mes'], y=d['Variación Mensual (%)'], mode='lines+markers',
            marker=dict(symbol='square' if es_actual else 'diamond', size=8 if es_actual else 6, color=color),
            line=dict(color=color, width=3 if es_actual else 1.5, dash=None if es_actual else 'dot'),
            name=str(anio),
            hovertemplate=f'{anio}: ' + '%{y:.2f}%<extra></extra>',
        ))
    fig.update_layout(
        title=titulo,
        xaxis=dict(title='Mes', tickmode='array', tickvals=list(MESES.keys()), ticktext=list(MESES.values())),
        yaxis=dict(title='Variación mensual (%)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        hovermode='x unified', width=950, height=460, template='plotly_white',
    )
    return fig

fig_bienes_sv = fig_banda_sv(var_bienes_sv, 'IPC bienes sin volátiles (variación mensual, %)')
fig_bienes_sv.update_layout(**ESTILO_INFORME)
fig_bienes_sv.show()

fig_servicios_sv = fig_banda_sv(var_servicios_sv, 'IPC servicios sin volátiles (variación mensual, %)')
fig_servicios_sv.update_layout(**ESTILO_INFORME)
fig_servicios_sv.show()


# %%
## Para las variaciones de todos los meses
from datetime import datetime

#Para buscar el IPC actual
_mes_actual = datetime.now().month-1
#Para el nombre del mes
mes_nombre = MESES[_mes_actual]

#Variaciones de IPC del mismo mes, histórico
variaciones_mes = (
    finalDf
    .query("Mes == @_mes_actual and Glosa == 'IPC General'")
    .drop(columns = {'División', 'Grupo', 'Clase', 'Subclase', 'Producto','Ponderación', 'Índice','Variación Acumulada (%)', 'Variación 12 Meses (%)','Incidencia Mensual (%)', 'Incidencia Acumulada (%)',
       'Incidencia 12 Meses (%)', 'Fecha'})
)

#Para los promedios históricos
promedio_meses = variaciones_mes['Variación Mensual (%)'].mean().round(2)
#Promedio excluyendo 2020 a 2022
promedio_meses_ex = (variaciones_mes
                     .query("Año < 2020 or Año > 2022")['Variación Mensual (%)']
                     .mean().round(2))

variaciones_mes['Promedio Histórico'] = promedio_meses
variaciones_mes['Promedio Histórico Ex'] = promedio_meses_ex

display(variaciones_mes)
fig_prom = go.Figure()
fig_prom.add_trace(
    go.Bar(
        x = variaciones_mes['Año'], y =variaciones_mes['Variación Mensual (%)'],
        marker = dict(color = "#031e31"), name = 'Variación mensual',texttemplate = '%{y:.2f}', textposition = 'outside'
        # texttemplate y no text=: la columna es float32 y el .round(2) igual
        # arrastra la cola (0.1 -> 0.10000000149). Formateando en el navegador
        # las etiquetas salen siempre con 2 decimales.
    )
)
fig_prom.add_hline(y = promedio_meses, line_dash = 'dot', line_color = 'gold', annotation_text = f'Promedio histórico: {promedio_meses}%', annotation_position = 'top right')
fig_prom.add_hline(y = promedio_meses_ex, line_dash = 'dot', line_color = 'red', annotation_text = f'Promedio histórico ex: {promedio_meses_ex}%', annotation_position = 'top right')
fig_prom.update_layout(template = 'plotly_white', title = f'Variación IPC histórica del mes de {mes_nombre}', width = 1200, height = 600)
fig_prom.update_layout(**ESTILO_INFORME)



# %% [markdown]
# ### Gráficos desde SQL
# 
# - Compensaciones inflacionarias
# 
# - Seguros de Inflación

# %%
#Grafico de compensaciones inflacionarias
ci = gd.get_data("""
SELECT
    ci.Fecha,
    ROUND(ci.CI_SPC_2Y, 2)  AS CISPC_2y,
    ROUND(ci.CI_SPC_5Y, 2)  AS CISPC_5y,
    ROUND(ci.CI_SPC_10Y, 2) AS CISPC_10y
FROM dace.dbo.inflation_CI_SPC ci
WHERE ci.Fecha >= DATEADD(
    MONTH,
    -12,
    (SELECT MAX(Fecha) FROM dace.dbo.inflation_CI_SPC)
)
ORDER BY ci.Fecha DESC;
""").dropna()


#Para las expectativas de inflación
si = gd.get_data("""
SELECT
    Fecha,
    SI_12M_continuo,
    [SI_24M (13 a 24 meses)],
    SI_dic26
FROM dace.dbo.inflation_SI
WHERE Fecha >= DATEADD(MONTH, -12, (SELECT MAX(Fecha) FROM dace.dbo.inflation_SI))
ORDER BY Fecha DESC;
""")

#Grafico de Compensaciones inflacionarias
fig_ci = go.Figure()
fig_ci.add_trace(go.Scatter(
    x = ci['Fecha'], y = ci['CISPC_2y'], name = 'CI SCP 2y', line = dict(color = 'navy')))
fig_ci.add_trace(go.Scatter(
    x = ci['Fecha'], y  =ci['CISPC_5y'], name = 'CI SPC 5y', line = dict(color = 'gray')))
fig_ci.add_trace(go.Scatter(
    x = ci['Fecha'], y  =ci['CISPC_10y'], name = 'CI SPC 10y', line = dict(color = "#9E840F")))
fig_ci.update_layout(**ESTILO_INFORME, title = 'Expectativas de Inflación de Largo plazo', width = 1200, height = 600).show()

#De seguros de inflación
fig_si = go.Figure() 
fig_si.add_trace(go.Scatter(
    x = si['Fecha'], y = si['SI_12M_continuo'], name = 'Inflación 12 meses', line = dict(color = 'gray')))
fig_si.add_trace(go.Scatter(
    x = si['Fecha'], y = si['SI_24M (13 a 24 meses)'], name = '1y en 1y', line = dict(color = '#9E840F')))
fig_si.add_trace(go.Scatter(
    x = si['Fecha'], y = si['SI_dic26'], name = 'Diciembre 2026',  line = dict(color = 'red')))
fig_si.update_layout(**ESTILO_INFORME, title = 'Expectativas de Inflación', width = 1200, height = 600).show()

# %% [markdown]
# ### Variación de Seguros inflacionarios y de expectativas de Inflación
# 

# %%
var_si = gd.get_data("""SELECT TOP 5
    Fecha,
    septiembre_26,
    octubre_26,
    noviembre_26,
    diciembre_26,
    enero_27,
    febrero_27,
    marzo_27,
    abril_27,
    mayo_27,
    junio_27
FROM dace.dbo.inflation_SI_stairs
ORDER BY Fecha desc;""").set_index('Fecha').sort_index(ascending = True)

for col in var_si.columns:
    var_si[f"{col}_dif_pb"] = var_si[col].diff() * 100
var_si = var_si.sort_values('Fecha', ascending = False).head(2)

var_si_dif = var_si.loc[:,var_si.columns.str.contains("_dif_pb")].head(1).T
display(var_si_dif.columns)
display(var_si_dif)

graf = var_si_dif.reset_index()

graf.columns = ["tenor", "valor"]

graf["tenor"] = (
graf["tenor"]
    .str.replace("_dif_pb", "", regex=False)
    .str.replace("septiembre", "sep")
    .str.replace("octubre", "oct")
    .str.replace("noviembre", "nov")
    .str.replace("diciembre", "dic")
    .str.replace("enero", "ene")
    .str.replace("febrero", "feb")
    .str.replace("marzo", "mar")
    .str.replace("abril", "abr")
    .str.replace("mayo", "may")
    .str.replace("junio", "jun")
    .str.replace("_", "-")
)

figvariacion = go.Figure()

figvariacion.add_trace(
go.Bar(
    x=graf["tenor"],
    y=graf["valor"],
    text=graf["valor"],
    textposition="inside",
    marker_color="#0A2A6B",
    texttemplate = '%{y:.0f}',
    textfont=dict(
        color="white",
        size=12
    )))

figvariacion.update_layout(**ESTILO_INFORME)
figvariacion.add_hline(y=0,line_width=1,line_color="lightgray")

figvariacion.show()

# %% [markdown]
# ### Fundamentales: Petróleo y Tipo de Cambio

# %%
# Tipo de Cambio VS Petróleo

fund = gd.get_data("""
select 
	a.Fecha, 
	a.Chile, 
	b.PETRO
from bbg_monedas a
LEFT JOIN bbg_commodities b
	ON a.Fecha = b.Fecha
Where a.Chile is not null
and year(a.Fecha) >= 2025
order by Fecha desc 
""").dropna()

#Gráfico
figfund = go.Figure()
figfund.add_trace(go.Scatter(
    x = fund['Fecha'], y = fund['Chile'],
    name = 'CLP'
))
figfund.add_trace(go.Scatter(
    x = fund['Fecha'], y = fund['PETRO'],
    name = 'Petróleo',
    yaxis = "y2"
))
figfund.update_layout(
    title = 'Fundamentales: CLP y Petróleo',
    yaxis = dict(title = 'ClP'),
    yaxis2 = dict(
        title = 'USD/lb',
        overlaying = "y",
        side = 'right'
    )
)
figfund.update_layout(**ESTILO_INFORME)

# %% [markdown]
# ## 10. Gráficos del informe
# 
# Registro único de las figuras que se publican en el dashboard. Es la **única fuente de verdad**: `scripts/build_dashboard.py` las lee por clave desde `outputs/figuras.json`, que se escribe acá.
# 
# Para agregar un gráfico basta con crearlo más arriba y sumar una línea a este dict — no hay que tocar el script ni la plantilla HTML.

# %%
# --- Registro de figuras del informe (ÚNICA fuente de verdad) ---
# El orden de este dict es el orden en que salen los gráficos en el dashboard.
# 'ancho': 'full' ocupa las dos columnas de la grilla, 'medio' ocupa una sola.
# El título NO va acá: vive en el layout.title de cada figura, que es donde se edita.
# Para agregar un gráfico: crearlo arriba como fig_<algo> y agregar una línea acá.
FIGURAS = {
    'variacion_historica_mes': {'fig': fig_prom,               'ancho': 'full',  'seccion': 9},
    'difusion_general':        {'fig': fig_difusion_general,  'ancho': 'full',  'seccion': 7},
    'variacion_ipc_grupo':     {'fig': fig_incidencias_grupo, 'ancho': 'full',  'seccion': 9},
    'treemap_canasta':         {'fig': fig_treemap,           'ancho': 'full',  'seccion': 9},
    'incidencia_12m_division': {'fig': fig_inc12_division,    'ancho': 'full', 'seccion': 9},
    'bienes_sv_banda':         {'fig': fig_bienes_sv,         'ancho': 'full',  'seccion': 10},
    'servicios_sv_banda':      {'fig': fig_servicios_sv,      'ancho': 'full',  'seccion': 10},
    'fundamentales_inf':       {'fig': figfund,               'ancho': 'full', 'seccion': 10},
    'expectativas_si':         {'fig': fig_si,                'ancho': 'full', 'seccion': 10},
    'expectativas_ci':         {'fig': fig_ci,                'ancho': 'full', 'seccion': 10},
    'variacion_si':            {'fig': figvariacion,          'ancho': 'full', 'seccion': 10}
}

# --- Exportar las figuras a JSON para el dashboard ---
# Este archivo es el contrato con scripts/build_dashboard.py: se leen POR CLAVE, así que
# agregar o mover un gráfico acá ya no puede desalinear los demás.
# OJO con la serialización: fig.to_plotly_json() deja arrays numpy adentro y json.dump
# revienta con el treemap y con inc12_division (su marker_color viene de np.where).
# fig.to_json() usa el encoder propio de plotly, que sí los serializa.
import json

salida_figuras = {
    'ultimo_mes': ULTIMO_MES.strftime('%Y-%m'),
    'figuras': {
        clave: {'ancho': info['ancho'], 'fig': json.loads(info['fig'].to_json())}
        for clave, info in FIGURAS.items()
    },
}

ruta_figuras = f"{RUTA_PROYECTO}/outputs/figuras.json"
Path(f"{RUTA_PROYECTO}/outputs").mkdir(exist_ok=True)
with open(ruta_figuras, 'w', encoding='utf-8') as f:
    json.dump(salida_figuras, f, ensure_ascii=False)

print(f"{len(FIGURAS)} figuras exportadas a {ruta_figuras}")
for clave, info in FIGURAS.items():
    print(f"  [Sec. {info['seccion']:>2}] {clave:<25} ({info['ancho']})")

# %% [markdown]
# ## 11. Exportación
# 
# Subida de todas las tablas a SQL

# %%
# ruta = f"{RUTA_PROYECTO}/outputs/Output IPC BI.xlsx"

# tasks = [
#     (finalDf,                'Consolidado Bases'      ,     False),
#     (df_resumen_difusion,    'Resumen Difusion IPC',         False),
#     (variacion_positiva,     'Variacion Positiva',           False),
#     (ine_ipc,                'IPC Analiticos',                False),
#     (variacion_bcch,         'IPC Volatiles SV BCCh',        False),
#     (variacion_positiva_sv,  'Var Positiva Sin Volatiles',   False),
#     (df_resumen_difusion_sv, 'Resumen Difusion SV',          False),
#     (canasta_ultimo_mes,     'Canasta Volatiles Ultimo Mes', False),
#     (incidencias_grupo,      'Variacion IPC por Grupo',      False),
# ]

# with pd.ExcelWriter(ruta) as writer:
#     for df, name, idx in tqdm(tasks, desc="Escribiendo Excel"):
#         df.to_excel(writer, sheet_name=name, index=idx)

# print(f"\rGuardado de datos completado exitosamente! Archivo guardado en {ruta}")



