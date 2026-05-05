# %%
import base64
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
import os
import plotly.express as px
import sys 
sys.path.append(r"D:\GOM\DACE\Nacho\Modulos")
import Get_Data as gd

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from plotly.subplots import make_subplots


brand_palette = [
    "#BF9C69",  # beige
    "#001730",  # azul oscuro
    "#000000",  # negro
    "#C00000",  # rojo
    "#70AD47",  # verde
    "#FFC000",  # amarillo
    "#7F7F7F",  # gris
    "#57257D",  # púrpura
    "#ED7D31",  # naranjo
    "#4472C4",  # azul
    "#44546A",  # azul grisáceo
]

def img_to_base64(img_path):
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        print(f"Error loading image: {img_path}")
        return ""

def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'


# %%
USD = gd.get_data("select Fecha,Cotizacion from mesadineOLTP_.mercado.divisas where Paridad = 'USD'")

# %%
def fechas(df,Variable):
    fecha_referencia = pd.to_datetime(df.index.max()).date()
    if Variable == "Today":
        fecha = fecha_referencia
        return str(fecha),Variable
    elif Variable == "1d":
        fecha = fecha_referencia- timedelta(days=1)
        return str(fecha),Variable
    elif Variable == "1w":
        fecha = fecha_referencia- timedelta(days=7)
        return str(fecha),Variable
    elif Variable == "2w":
        fecha = fecha_referencia- timedelta(days=14)
        return str(fecha),Variable
    elif Variable == "MTD":
        fecha = date(fecha_referencia.year, fecha_referencia.month, 1)
        return str(fecha),Variable
    elif Variable == "1M":
        fecha = fecha_referencia- timedelta(weeks=4)
        return str(fecha),Variable
    elif Variable == "YTD":
        fecha = date(fecha_referencia.year, 1, 1)
        return str(fecha),Variable
    elif Variable == "RPM-1":
        fecha = fecha_ultima_rpm
        return str(fecha),Variable
    elif Variable == "RPM-2":
        fecha = fecha_antepenultima_rpm
        return str(fecha),Variable
    else:
        return print(f'Variable no encontrada {Variable}')


# %%

Primer4 = gd.get_data('select Fecha, [SPREAD DS 30], [SPREAD DS 90],[SPREAD DS 180],[SPREAD DS 360],[SPREAD PS 30], [SPREAD PS 90],[SPREAD PS 180],[SPREAD PS 360] from dbo.Base_DMN')
spread1 = Primer4.rename(columns = {'SPREAD DS 30':'Spread DAP-Swap 1M',
                                   'SPREAD DS 90':'Spread DAP-Swap 3M',
                                   'SPREAD DS 180':'Spread DAP-Swap 6M',
                                   'SPREAD DS 360':'Spread DAP-Swap 12M',
                                   'SPREAD PS 30':'Spread Prime-Swap 1M',
                                   'SPREAD PS 90':'Spread Prime-Swap 3M',
                                   'SPREAD PS 180':'Spread Prime-Swap 6M',
                                   'SPREAD PS 360':'Spread Prime-Swap 12M'}).set_index('Fecha').sort_index().query('Fecha>="2019-01-01"')
fig_spread_dapswap1m = (spread1[['Spread DAP-Swap 1M', 'Spread Prime-Swap 1M']]*100).pipe(px.line,template='plotly_white',color_discrete_sequence=brand_palette,title='Spread DAP_SWAP 1M (bps)')
fig_spread_dapswap3m = (spread1[['Spread DAP-Swap 3M', 'Spread Prime-Swap 3M']]*100).pipe(px.line,template='plotly_white',color_discrete_sequence=brand_palette,title='Spread DAP_SWAP 3M (bps)')
fig_spread_dapswap6m = (spread1[['Spread DAP-Swap 6M', 'Spread Prime-Swap 6M']]*100).pipe(px.line,template='plotly_white',color_discrete_sequence=brand_palette,title='Spread DAP_SWAP 6M (bps)')
fig_spread_dapswap12m = (spread1[['Spread DAP-Swap 12M', 'Spread Prime-Swap 12M']]*100).pipe(px.line,template='plotly_white',color_discrete_sequence=brand_palette,title='Spread DAP_SWAP 12M (bps)')


fig_spread_dapswap1m

# %%
fig_spread_dapswap3m

# %%
fig_spread_dapswap6m

# %%
fig_spread_dapswap12m

# %%

#PDBC PENDIENTE

pdbc_bolsa = gd.get_data('select Fecha, [PDBC BOLSA 7], [PDBC BOLSA 30], [PDBC BOLSA 90], [PDBC BOLSA 180],  [PDBC BOLSA 360] from dbo.Base_DMN').query('Fecha>="2019-01-01"')
fig_pdbc_7D = pdbc_bolsa.rename(columns={'PDBC BOLSA 7':'7D','PDBC BOLSA 30':'1M', 'PDBC BOLSA 90':'3M', 'PDBC BOLSA 180':'6M', 'PDBC BOLSA 360':'1Y'}).set_index('Fecha').sort_index()[['7D']].pipe(px.line,template='plotly_white',color_discrete_sequence=brand_palette,title='Tasas PDBC Mercado Secundario')
fig_pdbc_7D


# %%
#TIB + Monto interbancario
TIB = gd.get_data('select * from mesadineOLTP_.Interbancario.Transacciones')
tib_fig = (TIB.sort_values('Fecha')
                .assign(Fecha = lambda df: pd.to_datetime(df.Fecha))
                .assign(Ponderado_monto = lambda df: df.Monto/df.groupby('Fecha').Monto.transform(sum))
                .assign(Tasa_ponderada = lambda df: (df.Ponderado_monto*df.Tasa).groupby(df.Fecha).transform(sum), 
                        Monto_transado = lambda df: df.groupby(df.Fecha).Monto.transform(sum))
                .drop_duplicates('Fecha')[['Fecha','Tasa_ponderada','Monto_transado']]).set_index('Fecha').query('Fecha>="2020-01-01"')



figTIB= make_subplots(specs= [[{"secondary_y": True }]])
barTIB = tib_fig['Monto_transado'].pipe(px.bar,color_discrete_sequence = brand_palette)

for tr in barTIB.data:
    figTIB.add_trace(tr, secondary_y=True)

fig_TIB_Montotransado = (figTIB.add_trace(go.Scatter(
    x=tib_fig.index,
    y=tib_fig["Tasa_ponderada"],
    name="TIB",
    mode="lines",
    line=dict(color="#C00000",width=3)),secondary_y=False).update_layout(title="TIB vs Monto Transado",template="plotly_white",hovermode="x unified",barmode="overlay",bargap=0.25
                ).update_yaxes(title_text="Tasa",secondary_y=False).update_yaxes(title_text="Monto",secondary_y=True).update_xaxes(title_text="Fecha"))

fig_TIB_Montotransado

# %%
# Pendiente spred6m DAP UF PRIME UF (No existe)

# %%


# %%
dap_prime_uf = gd.get_data('select Fecha, [Prime UF90],[Prime UF360], [DAP UF 90],[DAP UF 360] from dbo.Base_DMN').assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).query('Fecha>="2019-01-01"').sort_values('Fecha')
spc_UF = gd.get_data('select Fecha, SPC_1Y, SPC_3M from dbo.bbg_spc_uf').assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).query('Fecha>="2019-01-01"').sort_values('Fecha')
spread_UF = dap_prime_uf.merge(spc_UF,how='inner',on='Fecha').assign(Spread_PrimeSwap_UF1Y = lambda df: (df['Prime UF360']-df['SPC_1Y'])*100,Spread_DAPSwap_UF1Y = lambda df: (df['DAP UF 360']-df['SPC_1Y'])*100 )[['Fecha','Spread_PrimeSwap_UF1Y','Spread_DAPSwap_UF1Y']].set_index('Fecha')
fig_spread_dapprimeufswap = spread_UF.pipe(px.line,color='variable',labels={"Fecha":"Fecha",'value':"bp","variable":""},
        template='plotly_white',color_discrete_sequence=brand_palette,title='Spread DAP-Swap y Prime-Swap 12M (bps)').update_layout(
            legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=15,color="#353232")
            ),
            margin=dict(l=40,r=20,t=30,b=120)
        )


fig_spread_dapprimeufswap

# %% [markdown]
# .add_annotation(
#         x=0.5, y=-0.2, xref="paper", yref="paper",
#         text=texto_multilinea,
#         showarrow=False,
#         xanchor="center", yanchor="top",
#         font=dict(size=15, color="#353232")).update_layout(margin = dict(l=40,r=20,t=30,b=120+len(bloques)))

# %%
#Liquidez MN

# %%
# TABLA Avance encaje.... Pendiente



# %%
#gd.get_data('select * from Outputs.Bancos.Ratio_Cobertura_MN_30d_Historico')

# %%
gd.get_data('select * from Outputs.Bancos.Ratio_Cobertura_MX_30d_Historico')

# %%
#Liquidez MN vs Concentración de liquidez
gd.get_data('select * from Outputs.Bancos.Ratio_Cobertura_MX_30d_Historico')

# %%
#Ratio Liquidez obligaciones 30d
#LCR

# %%
#Niveles de liquidez y necesidades del periodo

#gd.get_data('select * from Outputs.Bancos.Liquidez_diaria_MN')


# %%
#Operaciones de liquidez mn(CC,FPD,PFL,Colocaciones y captaciones intbancarias,RT)
fig_Operaciones_de_liquidez = gd.get_data('select * from Outputs.Bancos.Liquidez_diaria_MN').fillna(0).set_index('Fecha').groupby('Fecha').sum()[['CC','FPD','FPL','Int_col','Int_cap','RT$']].pipe(px.area,color_discrete_sequence=brand_palette,template='plotly_white',title='Operaciones de liquidez mn(CC,FPD,PFL,Colocaciones y captaciones intbancarias,RT)')
fig_Operaciones_de_liquidez

# %%
#Liquidez MX

# %%
SoS = (gd.get_data('select Fecha, [Spread On Shore Base DAP 1M],[Spread On Shore Base DAP 3M],[Spread On Shore Base DAP 6M],[Spread On Shore Base DAP 12M] from dbo.Base_DMN where Fecha >= getdate()-700 order by Fecha desc')
              .rename(columns={'Spread On Shore Base DAP 1M':'SoS 1M','Spread On Shore Base DAP 3M':'SoS 3M','Spread On Shore Base DAP 6M':'SoS 6M','Spread On Shore Base DAP 12M':'SoS 12M'})
              
)

Tado = (gd.get_data('select Fecha, [Tasa Tado 1M],[Tasa Tado 3M],[Tasa Tado 6M],[Tasa Tado 12M] from dbo.Base_DMN where Fecha >= getdate()-700 order by Fecha desc')  
        #      .rename(columns={'Spread On Shore Base DAP 1M':'SoS 1M','Spread On Shore Base DAP 3M':'SoS 3M','Spread On Shore Base DAP 6M':'SoS 6M','Spread On Shore Base DAP 12M':'SoS 12M'})

)

Sofr = (gd.get_data('select Fecha, [SOFR 1M],[SOFR 3M],[SOFR 6M],[SOFR 12M] from dbo.Base_DMN where Fecha >= getdate()-700 order by Fecha desc')  

)

Prime = (gd.get_data('select Fecha, [Prime USD30],[Prime USD90],[Prime USD180],[Prime USD360] from dbo.Base_DMN where Fecha >= getdate()-700 order by Fecha desc')  

)

DAP_USD = (gd.get_data('select Fecha, [DAP US$ 30],[DAP US$ 90],[DAP US$ 180],[DAP US$ 360] from dbo.Base_DMN where Fecha >= getdate()-700 order by Fecha desc')  
)

tadosofr = Tado.merge(Sofr).assign(Spread_TADO_1M = lambda df: df['Tasa Tado 1M']-df['SOFR 1M'],Spread_TADO_3M = lambda df: df['Tasa Tado 3M']-df['SOFR 3M'],Spread_TADO_6M = lambda df: df['Tasa Tado 6M']-df['SOFR 6M'], Spread_TADO_12M = lambda df: df['Tasa Tado 12M']-df['SOFR 12M'] )
primesofr = Prime.merge(Sofr).assign(Spread_Prime_1M = lambda df: df['Prime USD30']-df['SOFR 1M'],Spread_Prime_3M = lambda df: df['Prime USD90']-df['SOFR 3M'],Spread_Prime_6M = lambda df: df['Prime USD180']-df['SOFR 6M'], Spread_Prime_12M = lambda df: df['Prime USD360']-df['SOFR 12M'] )
dapsofr = DAP_USD.merge(Sofr).assign(Spread_DAP_US_1M = lambda df: df['DAP US$ 30']-df['SOFR 1M'],Spread_DAP_US_3M= lambda df: df['DAP US$ 90']-df['SOFR 3M'],Spread_DAP_US_6M = lambda df: df['DAP US$ 180']-df['SOFR 6M'], Spread_DAP_US_12M = lambda df: df['DAP US$ 360']-df['SOFR 12M'] )

# %%
#Spread on shore, spread tado sofr, spread prime usd sofr, spread dap sofr 1M
Fig_Liquidez_mx_spreads_1M = SoS[['Fecha','SoS 1M']].merge(tadosofr[['Fecha','Spread_TADO_1M']]).merge(primesofr[['Fecha','Spread_Prime_1M']]).merge(dapsofr[['Fecha','Spread_DAP_US_1M']]).set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title='Spread Liquidez MX')
Fig_Liquidez_mx_spreads_1M

# %%
#Spread on shore, spread tado sofr, spread prime usd sofr, spread dap sofr 3M
Fig_Liquidez_mx_spreads_3M = SoS[['Fecha','SoS 3M']].merge(tadosofr[['Fecha','Spread_TADO_3M']]).merge(primesofr[['Fecha','Spread_Prime_3M']]).merge(dapsofr[['Fecha','Spread_DAP_US_3M']]).set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title='Spread Liquidez MX')
Fig_Liquidez_mx_spreads_3M



# %%
#Spread on shore, spread tado sofr, spread prime usd sofr, spread dap sofr 6M
Fig_Liquidez_mx_spreads_6M = SoS[['Fecha','SoS 6M']].merge(tadosofr[['Fecha','Spread_TADO_6M']]).merge(primesofr[['Fecha','Spread_Prime_6M']]).merge(dapsofr[['Fecha','Spread_DAP_US_6M']]).set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title='Spread Liquidez MX')
Fig_Liquidez_mx_spreads_6M


# %%
#Spread on shore, spread tado sofr, spread prime usd sofr, spread dap sofr 12M
Fig_Liquidez_mx_spreads_12M = SoS[['Fecha','SoS 12M']].merge(tadosofr[['Fecha','Spread_TADO_12M']]).merge(primesofr[['Fecha','Spread_Prime_12M']]).merge(dapsofr[['Fecha','Spread_DAP_US_12M']]).set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title='Spread Liquidez MX')
Fig_Liquidez_mx_spreads_12M

# %%
#Liq MX vs concentracion de liquidez
Liquidez_mx = gd.get_data("with c0 as (SELECT RIM.Fecha, codigoInstitucion, [Saldo BCCh en USD] as BCCh, [Saldo bancos corresponsales en exterior] as Corresponsales FROM MesadineOLTP_.StockHistorico.RIM RIM where cast(RIM.Fecha as date) in (select Fecha from (select Fecha, ROW_NUMBER() over (order by fecha desc) as ID from MesadineOLTP_.dbo.CalendarioHabil where esferiado=0 and esFinSemana=0 and Fecha <= dateadd(day, 0, cast(SYSDATETIME() as date)))D where ID <= 280)) select c0.Fecha, registroAlternativo as Institucion, BCCh, Corresponsales, (BCCh + Corresponsales) as Total, IIF(registroAlternativo = 'BICE', IHH.IHH, Null) as IHH from c0 left join mesadineOLTP_.Soma.participantesCodigoBCCh sp on sp.codigoInstitucion = c0.codigoInstitucion left join mesadineOLTP_.Soma.Participantes spx on sp.ParticipantesID = spx.ParticipantesID left join (select Fecha, (sum(([Saldo BCCh en USD] + [Saldo bancos corresponsales en exterior]) * ([Saldo BCCh en USD] + [Saldo bancos corresponsales en exterior])))/(sum(([Saldo BCCh en USD] + [Saldo bancos corresponsales en exterior])) * sum(([Saldo BCCh en USD] + [Saldo bancos corresponsales en exterior])))*100 as IHH from MesadineOLTP_.StockHistorico.RIM where [Saldo BCCh en USD] <>0 and [Saldo bancos corresponsales en exterior] <> 0 and codigoInstitucion not in ('BCECCLR0', 'BCECCLRM') group by Fecha)IHH on C0.Fecha = IHH.Fecha WHERE (spx.registroAlternativo In ('CHILE','ESTADO','BICE','SANTANDER','ITAÚ-CORPBANCA','SCOTIABANK','BCI', 'BTG','CONSORCIO','INTERNACIONAL','SECURITY','CHINA CONSTRUCTION BANK','HSBC','JP MORGAN','RIPLEY','FALABELLA','Bank of China','Tanner')) order by c0.Fecha desc")
parametro = gd.get_data('select * from Outputs.Bancos.Ratio_Cobertura_MX_30d_Historico')[['Banco','Categoria']].drop_duplicates(subset=['Banco','Categoria']).rename(columns={'Banco':'Institucion'})
parametro
base = parametro.copy()
base['Institucion'] ='Itaú-Corpbanca'
base['Categoria'] = "Sistémico"
sistematic = pd.concat([parametro,base.drop_duplicates()],ignore_index=True)
IHH_fig = Liquidez_mx.fillna(0)[['Fecha','IHH']].groupby('Fecha').sum().reset_index().assign(Fecha=lambda df: pd.to_datetime(df.Fecha))

fig_liq_concentracion = (Liquidez_mx.fillna(0)
                            .assign(Fecha = lambda df: pd.to_datetime(df.Fecha))
                            .sort_values('Fecha')
                            .merge(sistematic,how='inner',on='Institucion').groupby(['Fecha','Categoria']).Total.sum().reset_index()
                            .pivot(index='Fecha',columns='Categoria',values='Total')
                            .assign(Total_liq = lambda df: df['No Sistémico']+ df['Sistémico'])
                            .reset_index()
                            .merge(IHH_fig,how='inner',on='Fecha').set_index('Fecha')
)

area = fig_liq_concentracion[['Sistémico','No Sistémico']].pipe(px.area,color_discrete_sequence=brand_palette,template='plotly_white',title='Liquidez MX vs concentración de liquidez').add_scatter(x=fig_liq_concentracion.index,
        y=fig_liq_concentracion['Total_liq'],
        name='Total_liq',
        mode="lines",
        line=dict(color="#C08F46",width=3)
        )
       #.update_layout(title="Curva BTP y BTU 5Y",template="plotly_white",hovermode="x unified",barmode="overlay",bargap=0.25
        #).update_yaxes(title_text="BTU %",secondary_y=False,range=[1.5,4.0]).update_yaxes(title_text="BTP %",secondary_y=True,range=[4.0,6.5]).update_xaxes(title_text="Fecha"))

figliq = make_subplots(specs= [[{"secondary_y": True }]])
for tr in area.data:
    figliq.add_trace(tr, secondary_y=True)

fig_LiquidezMX_IHH = figliq.add_trace(go.Scatter(
        x=fig_liq_concentracion.index,
        y=fig_liq_concentracion['IHH'],
        name='IHH',
        mode="lines",
        line=dict(color="#FF0000",width=3)),secondary_y=False).update_layout(title="Liquidez MX vs concentración de liquidez",template="plotly_white",hovermode="x unified",barmode="overlay",bargap=0.25
        ).update_yaxes(title_text="Liquidez",secondary_y=True,range=[0,15000]).update_yaxes(title_text="Concentracion de liquidez",secondary_y=False,range=[10,25])

fig_LiquidezMX_IHH


# %%
#Ratio liquidez/obligaciones 30d
RatioLO = (gd.get_data('select * from Outputs.Bancos.Ratio_Cobertura_MX_30d_Historico')
        .assign(Fecha=lambda df: pd.to_datetime(df.Fecha)
            ,Ratio_liquidez_obligaciones = lambda df: 
                df.groupby(['Fecha','Categoria']).Liquidez.transform(sum)/(df.groupby(['Fecha','Categoria']).DAP.transform(sum) + df.groupby(['Fecha','Categoria']).Creditos_externos.transform(sum))*0.75 ,
             Ratio_liquidez_obligaciones_t = lambda df:  df.groupby('Fecha').Liquidez.transform(sum)/(df.groupby('Fecha').DAP.transform(sum) + df.groupby('Fecha').Creditos_externos.transform(sum))*0.75)
        .drop_duplicates(subset= ['Fecha','Categoria']).sort_values('Fecha')

)

Total_RatioLiquidezObligaciones30d = RatioLO[['Fecha','Ratio_liquidez_obligaciones_t']].drop_duplicates('Fecha')

fig_RatioLiquidezObligaciones = (RatioLO[['Fecha','Categoria','Ratio_liquidez_obligaciones']]
          .pivot(index='Fecha', columns='Categoria',values='Ratio_liquidez_obligaciones')
          .reset_index()
          .merge(Total_RatioLiquidezObligaciones30d,how='inner',on='Fecha')
          .rename(columns={'Ratio_liquidez_obligaciones_t':'Total'})
          .set_index('Fecha')
          .pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title='Ratio Liquidez/Obligaciones 30d').update_yaxes(title_text="Proporción",range=[0,3.5])
)

fig_RatioLiquidezObligaciones

# %%
#Pagina 2
#Perfil de vencimientos de la banca PENDIENTE NO TIENE PROCESO

# %%
#Liquidez proyectada Proceso en Excel


# %%
#Flujos de capital  No se almacena

# %%
#LCR NSFR Liquidez MX

map = (gd.get_data('select * from mesadineOLTP_.SBIF.Institucion'))
maip = (map[['Codigo','Institucion']]
            .rename(columns={'Institucion':'Banco','Codigo':'Institucion'})
            .reset_index().pivot(index='index',columns='Banco',values='Institucion')
            .rename(columns = {'Tanner Banco Digital':'Tanner','Banco Estado':'Estado','Banco BICE':'BICE','Banco Chile':'Chile','Banco Internacional':'Internacional','Banco BTG':'BTG','Banco Santander':'Santander','Banco Security':'Security','Itaú-Corpbanca':'Itaú','Banco Falabella': 'Falabella','Banco Consorcio':'Consorcio','Banco Ripley':'Ripley'})
            .unstack()
            .reset_index()
            .dropna()
            .rename(columns={0:'Institucion'})
            .drop(columns='index')
            .merge((gd.get_data('select * from Outputs.Bancos.Ratio_Cobertura_MX_30d_Historico')[['Banco','Categoria']]),how='inner',on='Banco')
)

#LCR,NSFR
Query_LCRMX = gd.get_data('select * from mesadineOLTP_.c49.indicadoresMonitoreo where nivelConsolidacion = 1')


LCRNSFR_MX = (Query_LCRMX
            .merge(maip,how='inner',on='Institucion')
            .assign(Fecha=lambda df: pd.to_datetime(df.Fecha))[['Fecha','fuenteFinanciamientoEstable','financiamientoEstableRequerido','activosLiquidos','egresosNetos','Categoria']]
            .set_index('Fecha')
            .groupby(['Fecha','Categoria'])
            .sum().reset_index()
            .assign(NSFR = lambda df: df.groupby(['Fecha','Categoria']).fuenteFinanciamientoEstable.transform(sum)/df.groupby(['Fecha','Categoria']).financiamientoEstableRequerido.transform(sum),
                    LCR = lambda df: df.groupby(['Fecha','Categoria']).activosLiquidos.transform(sum)/df.groupby(['Fecha','Categoria']).egresosNetos.transform(sum))
            .set_index('Fecha'))


NSFRMX = LCRNSFR_MX[['NSFR','Categoria']].reset_index().pivot(index='Fecha',columns='Categoria',values='NSFR').rename(columns={'No Sistémico':'NSFR_No_Sistemico','Sistémico':'NSFR_Sistemico'} )
LCRMX = LCRNSFR_MX[['LCR','Categoria']].reset_index().pivot(index='Fecha',columns='Categoria',values='LCR').rename(columns={'No Sistémico':'LCR_No_Sistemico','Sistémico':'LCR_Sistemico'} )

TotalMX = (LCRNSFR_MX.groupby('Fecha').sum().reset_index()[['Fecha','fuenteFinanciamientoEstable','financiamientoEstableRequerido','activosLiquidos','egresosNetos','Categoria']].drop_duplicates('Fecha')
            .assign(NSFR_Total = lambda df: df.fuenteFinanciamientoEstable/df.financiamientoEstableRequerido,
                    LCR_Total = lambda df: df.activosLiquidos/df.egresosNetos)
            .set_index('Fecha')[['NSFR_Total','LCR_Total']].reset_index()
            .merge(NSFRMX,how='inner',on='Fecha')
            .merge(LCRMX,how='inner',on='Fecha')
            .set_index('Fecha')            
)

# %%

Total_MX_NSFR = TotalMX[['NSFR_Total','NSFR_No_Sistemico','NSFR_Sistemico']].query('Fecha>="2020-01-01"').pipe(px.line,template = 'plotly_white',color_discrete_sequence=brand_palette,title= 'NSFR MX')
Total_MX_NSFR

# %%

Total_MX_LCR = TotalMX[['LCR_Total','LCR_No_Sistemico','LCR_Sistemico']].query('Fecha>="2020-02-01"').pipe(px.line,template = 'plotly_white',color_discrete_sequence=brand_palette,title= 'LCR MX')
Total_MX_LCR

# %% [markdown]
# ##MERCADO RF

# %%
##Tasas 5 años btp btu evolucion
figbtpbtu5 = (gd.get_data('select Fecha, [BONOS $ 5Y],[BONOS UF 5Y] from dbo.Base_DMN')
              .sort_values('Fecha')
              .rename(columns={'BONOS $ 5Y':'BTP 5Y','BONOS UF 5Y':'BTU 5Y'})
              .set_index('Fecha')).query("Fecha>='2023-01-01'")

b= ["#001730"]  # azul oscuro]
fig1_bonos = make_subplots(specs= [[{"secondary_y": True }]])
line = figbtpbtu5['BTP 5Y'].pipe(px.line,color_discrete_sequence=b)
for tr in line.data:
    fig1_bonos.add_trace(tr, secondary_y=True)

fig_btptu5 = (fig1_bonos.add_trace(go.Scatter(
        x=figbtpbtu5.index,
        y=figbtpbtu5['BTU 5Y'],
        name='BTU 5Y',
        mode="lines",
        line=dict(color="#BF9C69",width=3)
        ),secondary_y=False).update_layout(title="Curva BTP y BTU 5Y",template="plotly_white",hovermode="x unified",barmode="overlay",bargap=0.25
        ).update_yaxes(title_text="BTU %",secondary_y=False,range=[1.5,6.0]).update_yaxes(title_text="BTP %",secondary_y=True,range=[4.0,7]).update_xaxes(title_text="Fecha"))

fig_btptu5

# %%
#CURVA BTP Todos los tenor

figbtp= (gd.get_data('select Fecha, [BONOS $ 1Y],[BONOS $ 2Y],[BONOS $ 5Y],[BONOS $ 7Y],[BONOS $ 10Y],[BONOS $ 15Y],[BONOS $ 20Y],[BONOS $ 25Y],[BONOS $ 30Y] from dbo.Base_DMN where Fecha >= getdate()-700 order by Fecha desc')
              # .rename(columns={'BONOS $ 5Y':'BTP 5Y','BONOS UF 5Y':'BTU 5Y'})
              .set_index('Fecha'))

figbtu= (gd.get_data('select Fecha, [BONOS UF 2Y],[BONOS UF 5Y],[BONOS UF 7Y],[BONOS UF 10Y],[BONOS UF 15Y],[BONOS UF 20Y],[BONOS UF 25Y],[BONOS UF 30Y] from dbo.Base_DMN where Fecha >= getdate()-700 order by Fecha desc')
              # .rename(columns={'BONOS $ 5Y':'BTP 5Y','BONOS UF 5Y':'BTU 5Y'})
              .set_index('Fecha'))


hoy = fechas(figbtp,'Today')[0]
d1 = fechas(figbtp,"1d")[0]
d = str(figbtp.query('Fecha >= @d1').index.min())
ytd = fechas(figbtp,'YTD')[0]
y = str(figbtp.query('Fecha >= @ytd').index.min())
week = fechas(figbtp,'1w')[0]
w = str(figbtp.query('Fecha >= @week').index.min())
week2 = fechas(figbtp,"2w")[0]
w2 = str(figbtp.query('Fecha >= @week2').index.min())

month = fechas(figbtp,"1M")[0]
m1 = str(figbtp.query('Fecha >= @month').index.min())

fig_btp_curva = (figbtp.query('Fecha in @hoy or Fecha in @y or Fecha in @w or Fecha in @w2 or Fecha in @m1 or Fecha in @d')
    .sort_index()
    .rename(columns = {'BONOS $ 1Y':1,'BONOS $ 2Y':2,'BONOS $ 5Y':5,'BONOS $ 7Y':7,'BONOS $ 10Y':10,'BONOS $ 15Y':15,'BONOS $ 20Y':20,'BONOS $ 25Y':25,'BONOS $ 30Y':30})
    .stack().reset_index().rename(columns={'level_1':'Plazo',0:'tasa'}).pivot(columns='Fecha',index='Plazo',values='tasa').rename(columns={pd.Timestamp(y):'YTD',pd.Timestamp(m1):'1M',pd.Timestamp(w2):'2W',pd.Timestamp(w):'1W',pd.Timestamp(d1):'t-1',pd.Timestamp(hoy):'Hoy'})
    .pipe(px.line,template="plotly_white",color_discrete_sequence= brand_palette,title='Curva BTP %').update_yaxes(title_text="%",secondary_y=False,range=[4.0,6.0])
)

fig_btp_curva

# %%
USTtable = gd.get_data('select t1.Fecha, t1.EEUU as UST10Y, t2.EEUU as UST2Y, t3.EEUU AS UST5Y, t4.EEUU AS UST20Y, t5.EEUU AS UST30Y from DACE.dbo.bbg_rfi10_gen t1 INNER JOIN DACE.dbo.bbg_rfi2_gen t2 on t1.Fecha = t2. Fecha INNER JOIN DACE.DBO.bbg_rfi5_gen t3 on t1.Fecha = t3. Fecha INNER JOIN DACE.DBO.bbg_rfi20_gen t4 on t1.Fecha = t4. Fecha INNER JOIN DACE.DBO.bbg_rfi30_gen t5 on t1.Fecha = t5. Fecha Order by t1.Fecha desc')

fig_btp_ust_510y   = (figbtp[['BONOS $ 5Y','BONOS $ 10Y']]
        .reset_index()
        .assign(Fecha = lambda df: pd.to_datetime(df.Fecha))
        .merge(USTtable[['Fecha','UST5Y','UST10Y']].assign(Fecha = lambda df: pd.to_datetime(df.Fecha)),how='inner',on='Fecha')
        .assign(Spread_BTPUST_5Y = lambda df: (df['BONOS $ 5Y'] - df['UST5Y'])*100)
        .assign(Spread_BTPUST_10Y = lambda df: (df['BONOS $ 10Y'] - df['UST10Y'])*100)
        .dropna()
        .set_index('Fecha')
        [['Spread_BTPUST_5Y','Spread_BTPUST_10Y']]
        .pipe(px.line,color_discrete_sequence=brand_palette,title="Spread BTP-UST 5Y y 10Y (bp)",template='plotly_white')
    )

fig_btp_ust_510y

# %%
## Pendiente 2-10 Pendiente 5-10
fig_pendiente_btp_btu = (figbtp[['BONOS $ 2Y','BONOS $ 5Y','BONOS $ 10Y']]
    .reset_index()
    .merge(figbtu[['BONOS UF 2Y','BONOS UF 5Y','BONOS UF 10Y']].reset_index(),how='inner',on='Fecha')
    .set_index('Fecha')
    .dropna()
    .query('Fecha>="2025-01-01"')
    .assign(Pendiente_BTP_2_10 = lambda df: df['BONOS $ 10Y']-df['BONOS $ 2Y'], 
            Pendiente_BTP_5_10 = lambda df: df['BONOS $ 10Y']-df['BONOS $ 5Y'],
            Pendiente_BTU_2_10 = lambda df: df['BONOS UF 10Y']-df['BONOS UF 2Y'], 
            Pendiente_BTU_5_10 = lambda df: df['BONOS UF 10Y']-df['BONOS UF 5Y'],
            )
   [['Pendiente_BTP_2_10','Pendiente_BTP_5_10','Pendiente_BTU_2_10','Pendiente_BTU_5_10']].pipe(px.line,color_discrete_sequence=brand_palette,title="Pendiente BTP/BTU 10-5 y 10-2",template='plotly_white')
     )

fig_pendiente_btp_btu


# %%


##Tasas 10 años btp btu evolucion
figbtpbtu10 = (gd.get_data('select Fecha, [BONOS $ 10Y],[BONOS UF 10Y] from dbo.Base_DMN')
               .sort_values('Fecha')
              .rename(columns={'BONOS $ 10Y':'BTP 10Y','BONOS UF 10Y':'BTU 10Y'})
              .set_index('Fecha')).query("Fecha>='2023-05-01'")

b= ["#001730"]  # azul oscuro]
fig2_bonos = make_subplots(specs= [[{"secondary_y": True }]])
line = figbtpbtu10['BTP 10Y'].pipe(px.line,color_discrete_sequence=b)
for tr in line.data:
    fig2_bonos.add_trace(tr, secondary_y=True)

fig_curva_btptu10 = (fig2_bonos.add_trace(go.Scatter(
                    x=figbtpbtu10.index,
                    y=figbtpbtu10['BTU 10Y'],
                    name='BTU 10Y',
                    mode="lines",
                    line=dict(color="#BF9C69",width=3)
                    ),secondary_y=False).update_layout(title="Curva BTP y BTU 10Y",template="plotly_white",hovermode="x unified",barmode="overlay",bargap=0.25
                    ).update_yaxes(title_text="BTU %",secondary_y=False,range=[1.5,4.0]).update_yaxes(title_text="BTP %",secondary_y=True,range=[4.0,7.0]).update_xaxes(title_text="Fecha"))

fig_curva_btptu10

# %%
#CURVA BTU Todos los tenor



hoy = fechas(figbtu,'Today')[0]
d1 = fechas(figbtu,"1d")[0]
d = str(figbtu.query('Fecha >= @d1').index.min())

ytd = fechas(figbtu,'YTD')[0]
y = str(figbtu.query('Fecha >= @ytd').index.min())

week = fechas(figbtu,'1w')[0]
w = str(figbtu.query('Fecha >= @week').index.min())

week2 = fechas(figbtu,"2w")[0]
w2 = str(figbtu.query('Fecha >= @week2').index.min())

month = fechas(figbtu,"1M")[0]
m1 = str(figbtu.query('Fecha >= @month').index.min())

fig_btu_curva = (figbtu.query('Fecha in @hoy or Fecha in @y or Fecha in @w or Fecha in @w2 or Fecha in @m1 or Fecha in @d')
    .sort_index()
    .rename(columns = {'BONOS UF 1Y':1,'BONOS UF 2Y':2,'BONOS UF 5Y':5,'BONOS UF 7Y':7,'BONOS UF 10Y':10,'BONOS UF 15Y':15,'BONOS UF 20Y':20,'BONOS UF 25Y':25,'BONOS UF 30Y':30})
    .stack().reset_index().rename(columns={'level_1':'Plazo',0:'tasa'}).pivot(columns='Fecha',index='Plazo',values='tasa').rename(columns={pd.Timestamp(y):'YTD',pd.Timestamp(m1):'1M',pd.Timestamp(w2):'2W',pd.Timestamp(w):'1W',pd.Timestamp(d1):'t-1',pd.Timestamp(hoy):'Hoy'})
    .pipe(px.line,template="plotly_white",color_discrete_sequence= brand_palette,title='Curva BTU %').update_yaxes(title_text="%",secondary_y=False,range=[1.5,2.5])
)
fig_btu_curva

# %%
SPCtable = gd.get_data('select Fecha, [SPC_5Y], [SPC_10Y] from dbo.bbg_spc_clp where Fecha >= getdate()-700 order by Fecha desc')

fig_spread_btp_spc_510y = (figbtp[['BONOS $ 5Y','BONOS $ 10Y']]
                        .reset_index()
                        .assign(Fecha = lambda df: pd.to_datetime(df.Fecha))
                        .merge(SPCtable.assign(Fecha = lambda df: pd.to_datetime(df.Fecha)),how='inner',on='Fecha')
                        .assign(Spread_BTPSPC_5Y = lambda df: (df['BONOS $ 5Y'] - df['SPC_5Y'])*100)
                        .assign(Spread_BTPSPC_10Y = lambda df: (df['BONOS $ 10Y'] - df['SPC_10Y'])*100)
                        .dropna()
                        .set_index('Fecha')
                        [['Spread_BTPSPC_5Y','Spread_BTPSPC_10Y']]
                        .pipe(px.line,template="plotly_white",color_discrete_sequence=brand_palette,title="Spread BTP-SPC 5Y y 10Y (bp)")
)

fig_spread_btp_spc_510y

# %%
##Pendiente Tabla Ultimas emisiiones tgr monto emitido,ofertado,bid to cover,spreadvsreferencia etc.

# %%
#Mercado RF
#querys bonos
Bolsa_bonos = gd.get_data("select Instrumento, Monto, Fecha, Familia, TIR from MesadineOLTP_.BDS.RTRN").assign(Fecha= lambda df: pd.to_datetime(df.Fecha).dt.strftime('%Y-%m-%d'))
map_bonos = gd.get_data("select Fecha, [Nemotecnico], [TipoSVS] from MesadineOLTP_.DCV.Valorizacion WHERE TipoSVS in('BE','BB','BU')")
mapping_bonos = map_bonos.drop_duplicates(subset = ['Fecha','Nemotecnico'])[['Nemotecnico','TipoSVS']].rename(columns={'Nemotecnico':'Instrumento'})
Dolar = gd.get_data("select * from mesadineOLTP_.mercado.divisas where Paridad = 'USD'")[['Fecha','Cotizacion']].assign(Fecha=lambda df:pd.to_datetime(df.Fecha)).sort_values('Fecha').query('Fecha>"2019-01-01"')

M1 = mapping_bonos.query('TipoSVS == "BE"').Instrumento.to_list()
M2 = mapping_bonos.query('TipoSVS == "BB" or TipoSVS == "BU"').Instrumento.to_list()

Bolsa_BC = Bolsa_bonos.query('Instrumento in @M1').groupby('Fecha').Monto.sum().reset_index()
Bolsa_BB = Bolsa_bonos.query('Instrumento in @M2').groupby('Fecha').Monto.sum().reset_index()


# %%
vol_btp = Bolsa_bonos.assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).query("Instrumento.str.contains('BTP')").groupby('Fecha').TIR.std().reset_index().rename(columns={'TIR':'Volatilidad BTP'})
vol_btu = Bolsa_bonos.assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).query("Instrumento.str.contains('BTU')").groupby('Fecha').TIR.std().reset_index().rename(columns={'TIR':'Volatilidad BTU'})

fig_vol_btpbtu = vol_btp.merge(vol_btu,how='inner',on='Fecha').set_index('Fecha').pipe(px.scatter,color_discrete_sequence=brand_palette,template='plotly_white',title='Volatilidad diaria BTP/BTU (%)')

# %%
#Spread bancarios vs spc falta procesar 
#Spread bancarios vs bonos falta procesar
#Spread corporativos vs spc falta procesar

# %%
#Volatilidad tasas BB
Bolsa_BB_volatilidad = Bolsa_bonos.query('Instrumento in @M2')[['Fecha','TIR']]
Bolsa_BB_volatilidad['Volatilidad'] = pd.to_numeric(Bolsa_BB_volatilidad['TIR'])
Fig_Volatilidad_Tasas_BB = Bolsa_BB_volatilidad.groupby('Fecha').Volatilidad.std().reset_index().set_index('Fecha').pipe(px.scatter,color_discrete_sequence=brand_palette,template='plotly_white',title='Volatilidad Tasas BB')
Fig_Volatilidad_Tasas_BB

# %%
#Montos transados BTP BTU

Bolsa_btp = Bolsa_bonos.assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).query("Instrumento.str.contains('BTP')").groupby('Fecha').Monto.sum().reset_index().merge(Dolar,how='inner',on='Fecha').assign(Monto_BTP= lambda df: df.Monto/df.Cotizacion).drop(columns={'Cotizacion','Monto'})
Bolsa_btu = Bolsa_bonos.assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).query("Instrumento.str.contains('BTU')").groupby('Fecha').Monto.sum().reset_index().merge(Dolar,how='inner',on='Fecha').assign(Monto_BTU= lambda df: df.Monto/df.Cotizacion).drop(columns={'Cotizacion','Monto'})
Fig_Monto_btpbtu = Bolsa_btu.merge(Bolsa_btp,how='inner',on='Fecha').set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title='Monto transado BTP/BTU (Mill $USD)')
Fig_Monto_btpbtu

# %%
#Montos tansados BB
Bolsa_BB_usd = Bolsa_BB.assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).merge(Dolar,how='inner',on='Fecha').assign(Monto_USD= lambda df: df.Monto/df.Cotizacion).drop(columns={'Cotizacion','Monto'})

Fig_Monto_BB = Bolsa_BB_usd.set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title='Monto transado BB (Mill $USD)')
Fig_Monto_BB

# %%
#Montos transados BC
Bolsa_BC_usd = Bolsa_BC.assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).merge(Dolar,how='inner',on='Fecha').assign(Monto_USD= lambda df: df.Monto/df.Cotizacion).drop(columns={'Cotizacion','Monto'})
Fig_Monto_BC = Bolsa_BC_usd.set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title='Monto transado BC (Mill $USD)')
Fig_Monto_BC

# %% [markdown]
# ## Mercado FX

# %%
Cambiario = gd.get_data("select t1.Fecha,t2.Chile as CLP, t1.[Monto transado], t3.COBRE, t2.EEUU as DXY, t2.Zeuro, t2.Japon, t2.Suiza, t2.Mexico as MXN, t2.Brasil AS BRL, t2.UK, t2.Colombia as COP, t2.Peru AS PEN, t2.Turquia, t2.Korea as KRW, t2.RCheca as CZK, t2.Filipinas, t2.Polonia AS PLN, t2.Australia AS AUD, t2.Canada, t2.NZelanda AS NZD, t2.Sudafrica, t1.[MON. COMPARABLES] AS Mon_Comparables, t1.[MON. COMMODITIES] as Mon_Commodities, t1.[MON. LATAM] as Mon_Latam, t1.[DIF. Tasas EEUU 1Y] from dbo.Base_DMN t1 INNER JOIN dbo.bbg_monedas t2 ON t2.Fecha = t1.Fecha LEFT JOIN dbo.bbg_commodities t3 ON t3.Fecha = t1.Fecha").sort_values('Fecha').query('Fecha>="2022-01-01"')

fig1_cambiario = Cambiario[['Fecha','CLP','Monto transado']].set_index('Fecha')
fig2_cambiario = Cambiario[['Fecha','COBRE','DXY']].set_index('Fecha').dropna()
fig3_cambiario = Cambiario[['Fecha','CLP','DXY','MXN','BRL','COP','PEN','KRW','CZK','PLN','AUD','NZD']].set_index('Fecha').dropna()

# %%
fig = make_subplots(specs= [[{"secondary_y": True }]])
bar = fig1_cambiario['Monto transado'].pipe(px.bar,color_discrete_sequence=brand_palette)

for tr in bar.data:
    fig.add_trace(tr, secondary_y=True)

Fig_CLP_Monto = (fig.add_trace(go.Scatter(
    x=fig1_cambiario.index,
    y=fig1_cambiario["CLP"],
    name="CLP",
    mode="lines",
    line=dict(color="#C00000",width=3)),secondary_y=False).update_layout(title="CLP vs Monto Transado",template="plotly_white",hovermode="x unified",barmode="overlay",bargap=0.25
                ).update_yaxes(title_text="CLP",secondary_y=False).update_yaxes(title_text="US$ Millones",secondary_y=True).update_xaxes(title_text="Fecha"))

Fig_CLP_Monto


# %%
b= ["#001730"]  # azul oscuro]
fig2 = make_subplots(specs= [[{"secondary_y": True }]])
line = fig2_cambiario['COBRE'].pipe(px.line,color_discrete_sequence=b)
for tr in line.data:
    fig2.add_trace(tr, secondary_y=True)

fig_cobre_dxy = fig2.add_trace(go.Scatter(
    x=fig2_cambiario.index,
    y=fig2_cambiario['DXY'],
    name='DXY',
    mode="lines",
    line=dict(color="#C00000",width=3)
),secondary_y=False).update_layout(title="COBRE vs DXY",template="plotly_white",hovermode="x unified",barmode="overlay",bargap=0.25
).update_yaxes(title_text="DXY",secondary_y=False).update_yaxes(title_text="Cobre",autorange='reversed',secondary_y=True).update_xaxes(title_text="Fecha")

# %%
hoy = fechas(fig3_cambiario,'Today')[0]
ytd = fechas(fig3_cambiario,'YTD')[0]
y = str(fig3_cambiario.query('Fecha >= @ytd').index.min())
week = fechas(fig3_cambiario,'1w')[0]
w = str(fig3_cambiario.query('Fecha >= @week').index.min())

Variacion_moneda_5 = fig3_cambiario.query('Fecha in @hoy or Fecha in @w').sort_index().pct_change().dropna().unstack().reset_index().rename(columns = {'level_0':'Moneda',0:'var_pct'}).sort_values('var_pct')
Variacion_moneda_ytd = fig3_cambiario.query('Fecha in @hoy or Fecha in @y').sort_index().pct_change().dropna().unstack().reset_index().rename(columns = {'level_0':'Moneda',0:'var_pct'}).sort_values('var_pct')

base_color = ["#000000"]
clp_color = ["#BF9C69"]  # beige

order_w = Variacion_moneda_5.sort_values('var_pct').Moneda.to_list()
order_y = Variacion_moneda_ytd.sort_values('var_pct').Moneda.to_list()

Fig_varmoneda5 = (px.bar(Variacion_moneda_5,x='var_pct',y='Moneda',orientation='h',text=Variacion_moneda_5['var_pct'].map(lambda v: f'{v:.1%}'),color=Variacion_moneda_5['Moneda'].eq('CLP').map({True:'CLP',False:'Resto'}),color_discrete_map={'CLP':clp_color,'Resto':base_color}
                       ).add_vline(x=0,line_width=1,line_color='#96A0AA'
                        ).update_traces(textposition='outside',cliponaxis=False
                        ).update_layout(title='Variacion 1w Moneda(porcentaje)',template='plotly_white',bargap=0.2,showlegend=False,margin=dict(l=80,r=40,t=60,b=60)
                        ).update_xaxes(title= '% Negativo: Apreciación - % Positivo: Depreciación',tickformat= ".1%",zeroline=False,range=[-0.04,0.04])).update_yaxes(categoryorder='array',categoryarray=order_w)

Fig_varmoneda1y = (px.bar(Variacion_moneda_ytd,x='var_pct',y='Moneda',orientation='h',text=Variacion_moneda_ytd['var_pct'].map(lambda v: f'{v:.1%}'),color=Variacion_moneda_ytd['Moneda'].eq('CLP').map({True:'CLP',False:'Resto'}),color_discrete_map={'CLP':clp_color,'Resto':base_color}
                       ).add_vline(x=0,line_width=1,line_color='#96A0AA'
                        ).update_traces(textposition='outside',cliponaxis=False
                        ).update_layout(title='Variacion YTD Moneda (porcentaje)',template='plotly_white',bargap=0.2,showlegend=False,margin=dict(l=80,r=40,t=60,b=60)
                        ).update_xaxes(title= '% Negativo: Apreciación - % Positivo: Depreciación',tickformat= ".1%",zeroline=False,range=[-0.2,0.2])).update_yaxes(categoryorder='array',categoryarray=order_y)


Fig_varmoneda5


# %%
Fig_varmoneda1y

# %%
#Transado datatec Spot Derivados (promedio vs Act)
gd.get_data('select * from DACE.dbo.inter_TC_intra').sort_values('Fecha')

# %%
#Preguntar CLP conformacion de precios 1 + top20 Volatilidades
#Max me dio algunas querys necesito confirmar con la cata
Fig_conformacion_precios = gd.get_data('select Fecha, [Gap/Price] from Inputs.Spot.Datatec_bid_ask_diario').assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).rename(columns={'Gap/Price':'Promedio puntas'}).sort_values('Fecha').set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title= 'Promedio de amplitud de puntas (%)')
Fig_conformacion_precios

# %%
#Preguntar CLP conformacion de precios 2 + top20 Promedio
#Max me dio algunas querys necesito confirmar con la cata
Fig_conformacion_precios2 = gd.get_data('select Fecha, Volatilidad from Inputs.Spot.Datatec_volatilidad_diaria').assign(Fecha=lambda df: pd.to_datetime(df.Fecha),Volatilidad_diaria=lambda df: df.Volatilidad*1000).sort_values('Fecha').set_index('Fecha').query('Fecha>"2021-01-01"')['Volatilidad_diaria'].pipe(px.scatter,template='plotly_white',color_discrete_sequence=brand_palette,title='Volatilidad del precio de transacciones (%) X 1000' ).update_yaxes(range=[0,0.5])
Fig_conformacion_precios2

# %%
#RSI CLP ####Descartar no tenemos nada en SQL, tendriamos que crear un nuevo proceso

# %%
#Punto fwd 1 3 6 12 meses
Fig_fwdp = (gd.get_data('select Fecha, [FWD Ajus. 30], [FWD Ajus. 90], [FWD Ajus. 180], [FWD Ajus. 360] from Base_DMN')
            .query('Fecha>="2019-01-01"')
            .sort_values('Fecha')
            .set_index('Fecha')
            .query('Fecha>"2025-01-01"')
            .rename(columns = {'FWD Ajus. 30':'Fwd point 1M','FWD Ajus. 90':'Fwd point 3M', 'FWD Ajus. 180':'Fwd point 6M','FWD Ajus. 360':'Fwd point 12M'})
            .pipe(px.line,color_discrete_sequence=brand_palette,title="Fwd Point (pesos)", template='plotly_white')
)
Fig_fwdp

# %%
#Diferencial de tasas con EEUU

#Spread TPM IMP - FED IMP 3 6 9 12 24
#Tasa impliciata en el fwd 1 3 6 12 meses ##
spread_TPMFED_IMP = gd.get_data("select Fecha, [spread_MIPR_3M1D],[spread_MIPR_6M1D], [spread_MIPR_9M1D], [spread_MIPR_12M1D], [spread_MIPR_24M1D] from DACE.dbo.zero_curve_MIPR").sort_values('Fecha').assign(Fecha=lambda df: pd.to_datetime(df.Fecha))
spread_TPMFED_IMP

#Spread Swap clp - OIS 3 6 9 12 24
df_tasa_ois = gd.get_data("select Fecha, [ois3m], [ois6m], [ois9m], [ois12m],[ois24m] from DACE.dbo.bbg_OIS_SOFR").sort_values('Fecha').assign(Fecha=lambda df: pd.to_datetime(df.Fecha))
df_tasa_spc = gd.get_data("select Fecha, [SPC_3M], [SPC_6M], [SPC_9M], [SPC_1Y], [SPC_2Y]  from DACE.dbo.bbg_spc_clp").sort_values('Fecha').assign(Fecha=lambda df: pd.to_datetime(df.Fecha))


Spread_swap_ois = df_tasa_ois.merge(df_tasa_spc,how='inner',on='Fecha').assign(spread_SPC_OIS_3M = lambda df: (df.SPC_3M-df.ois3m)*100,
                                                                               spread_SPC_OIS_6M = lambda df: (df.SPC_6M-df.ois6m)*100,
                                                                               spread_SPC_OIS_9M = lambda df: (df.SPC_9M-df.ois9m)*100,
                                                                               spread_SPC_OIS_12M = lambda df: (df.SPC_1Y-df.ois12m)*100,
                                                                               spread_SPC_OIS_24M = lambda df: (df.SPC_2Y-df.ois24m)*100,
                                                                               )[['Fecha','spread_SPC_OIS_3M','spread_SPC_OIS_6M','spread_SPC_OIS_9M','spread_SPC_OIS_12M','spread_SPC_OIS_24M']].dropna()

# %%
Fig_spread_USA = Spread_swap_ois.merge(spread_TPMFED_IMP,how='inner',on='Fecha').set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white')
Fig_spread_USA

# %%
Fig_spread_swap_ios = Spread_swap_ois.set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title = 'Spread Swap clp - OIS')
Fig_spread_swap_ios

# %%
Fig_spread_tpm_fed_imp = spread_TPMFED_IMP.set_index('Fecha').pipe(px.line,color_discrete_sequence=brand_palette,template='plotly_white',title = 'Spread TPM IMP - FED IMP')
Fig_spread_tpm_fed_imp

# %%
#Flujos cambiarios
#spot ndf preguntar a diego
flujo_cambiario_spot = gd.get_data("""
WITH s1 AS (  -- SAS: 2024-01-01 <= Fecha < 2026-01-01
    SELECT
        CAST(Fecha AS date) AS Fecha,
        CASE
            WHEN Sector_contraparte LIKE 'AFP%'  THEN 'AFP'
            WHEN Sector_contraparte LIKE 'FFMM%' THEN 'FFMM'
            WHEN Sector_contraparte IN ('CONT FALCOM','CONT HMC','CONT PICTON','CONT FRONTAL TRUST','CONT PRUDENTIAL') THEN 'FFMM'
            WHEN Sector_contraparte LIKE '%PERSONAS%' THEN 'OTROS'
            WHEN Sector_contraparte = 'CORREDORA DE BOLSA' THEN 'CB'
            WHEN Sector_contraparte = 'CIA SEGUROS'      THEN 'CS'
            WHEN Sector_contraparte = 'EMPRESA FINANCIERA' THEN 'EMPRESA_FINANCIERA'
            WHEN Sector_contraparte = 'EMPRESA REAL'       THEN 'EMPRESA_REAL'
            WHEN Sector_contraparte = 'EMPRESA MINERA'     THEN 'MINERA'
            WHEN Sector_contraparte IN ('EUROCLEAR','NO RESIDENTES') THEN 'NR'
            ELSE Sector_contraparte
        END AS Grupo_Sector,
        Afecto_a_derivados,
        Monto_MM_USD
    FROM spot.[Agregada_por_sector_(SAS)]
    WHERE CAST(Fecha AS date) >= '2024-01-01'
      AND CAST(Fecha AS date) <  '2026-01-01'
),
s2 AS (  -- SICAM: desde 2026-01-01
    SELECT
        CAST(Fecha AS date) AS Fecha,
        CASE
            WHEN Sector_contraparte = 'Empresas' THEN
                CASE
                    WHEN Sector_detalle = 'EMPRESA FINANCIERAS' THEN 'EMPRESA_FINANCIERA'
                    WHEN Sector_detalle = 'EMPRESA MINERA'      THEN 'MINERA'
                    WHEN Sector_detalle IN ('EMPRESAS AGRUPADAS OP VARIAS','EMPRESAS SECTOR REAL','PERSONA NATURAL','POR CLASIFICA')
                                                             THEN 'EMPRESA_REAL'
                    ELSE 'EMPRESA_REAL'
                END
            ELSE
                CASE Sector_contraparte
                    WHEN 'Corredora De Bolsa' THEN 'CB'
                    WHEN 'Tgr'                 THEN 'TGR'
                    WHEN 'No Residentes'       THEN 'NR'
                    WHEN 'Bancos'              THEN 'BANCOS'
                    WHEN 'Bcch'                THEN 'BCCH'
                    WHEN 'Ffmm'                THEN 'FFMM'
                    WHEN 'Otros'               THEN 'OTROS'
                    WHEN 'Cia Seguros'         THEN 'CS'
                    WHEN 'Afp'                 THEN 'AFP'
                    ELSE Sector_contraparte
                END
        END AS Grupo_Sector,
        Afecto_derivado AS Afecto_a_derivados,
        CAST(Monto AS decimal(38,6)) AS Monto_MM_USD  -- ya en MM USD
    FROM [Inputs].[Spot].[Temp_SICAM_Estadisticas]
    WHERE CAST(Fecha AS date) >= '2026-01-01'
),
u AS (
    SELECT Fecha, Grupo_Sector, Monto_MM_USD, Afecto_a_derivados FROM s1
    UNION ALL
    SELECT Fecha, Grupo_Sector, Monto_MM_USD, Afecto_a_derivados FROM s2
)
SELECT
    Fecha,
    Grupo_Sector,
    SUM(Monto_MM_USD) AS Monto_MM_USD,
    Afecto_a_derivados
FROM u
GROUP BY Fecha, Grupo_Sector, Afecto_a_derivados
ORDER BY Fecha DESC, Grupo_Sector ASC;""" )

flujo_cambiario_forward = gd.get_data("""
WITH datos AS (
    SELECT 
        Fecha, nombre_cont, sector_cont, sector_det_cont, dias, plazo, instrumento_nom, 
        Monto_, Modalidad_pago, [Tipo contrato] = 'Suscripcion'
    FROM DACE.dbo.deriv_suscripciones
    WHERE Fecha <  CAST(GETDATE() AS date)                           -- ayer a 23:59:59 implícito
      AND Fecha >= DATEADD(day, -200, CAST(GETDATE() AS date))       -- últimos 200 días

    UNION ALL

    SELECT 
        Fecha_ven AS Fecha, nombre_cont, sector_cont, sector_det_cont, dias, plazo, instrumento_nom, 
        Pos_neta AS Monto_, Modalidad_pago, [Tipo contrato] = 'Vencimiento'
    FROM DACE.dbo.deriv_vencimientos
    WHERE Fecha_ven <  CAST(GETDATE() AS date)
      AND Fecha_ven >= DATEADD(day, -200, CAST(GETDATE() AS date))
),
base AS (
    SELECT
        CAST(Fecha AS date) AS Fecha,
        sector_det_cont,
        sector_cont,
        instrumento_nom,
        Modalidad_pago,
        plazo,
        dias,
        Monto_,
        Grupo_Sector =
            CASE 
                -- 1) Priorizar sector_cont para forzar FFMM (y AFP si quieres)
                WHEN UPPER(sector_cont) = 'FFMM'  OR UPPER(sector_cont) LIKE '%FFMM%' THEN 'FFMM'
                WHEN UPPER(sector_cont) = 'AFP'   OR UPPER(sector_cont) LIKE '%AFP%'  THEN 'AFP'

                -- 2) Si no vino por sector_cont, usar sector_det_cont
                WHEN UPPER(sector_det_cont) LIKE '%AFP%'       THEN 'AFP'
                WHEN UPPER(sector_det_cont) LIKE '%FFMM%'      THEN 'FFMM'
                WHEN UPPER(sector_det_cont) LIKE '%PERSONAS%'  THEN 'OTROS'
                WHEN UPPER(sector_det_cont) = 'CORREDORAS_DE_BOLSA' THEN 'CB'
                WHEN UPPER(sector_det_cont) = 'CIAS_DE_SEGUROS'     THEN 'CS'
                WHEN UPPER(sector_det_cont) = 'OFF_SHORE'           THEN 'NR'
                ELSE sector_det_cont
            END
    FROM datos
)
SELECT
    Fecha,
    sector_det_cont,
    sector_cont,
    Grupo_Sector,
    SUM(Monto_) AS Monto,
    instrumento_nom,
    Modalidad_pago,
    plazo,
    dias
FROM base
GROUP BY
    Fecha,
    sector_det_cont,
    sector_cont,
    Grupo_Sector,
    instrumento_nom,
    Modalidad_pago,
    plazo,
    dias
ORDER BY Fecha DESC;""")

# %%

ultima_semana_fecha = fechas(flujo_cambiario_forward.set_index('Fecha'),"1w")[0]
hoy = fechas(flujo_cambiario_forward.set_index('Fecha'),"Today")[0]
ultima_dossemanas_fecha = fechas(flujo_cambiario_forward.set_index('Fecha'),"2w")[0]


Flujo_cambiario_1w  = (

                    flujo_cambiario_spot
                                        .assign(Fecha = lambda df: pd.to_datetime(df.Fecha))
                                        .query('Fecha>=@ultima_semana_fecha and Fecha <= @hoy')
                                        .set_index('Fecha')
                                        .groupby('Grupo_Sector')
                                        .Monto_MM_USD.sum()
                                        .reset_index()
                                        .rename(columns={'Monto_MM_USD':'Spot'})
                        .merge((flujo_cambiario_forward
                                                    .assign(Fecha = lambda df: pd.to_datetime(df.Fecha))
                                                    .query('Fecha>=@ultima_semana_fecha and Fecha <= @hoy')
                                                    .set_index('Fecha')
                                                    .groupby('Grupo_Sector')
                                                    .Monto.sum()
                                                    .reset_index()
                                                    .rename(columns={'Monto':'Forward'})
                                ),how='left',on='Grupo_Sector')
                        .fillna(0)
                        .set_index('Grupo_Sector')
                        .assign(Spot_derivado = lambda df: df.Spot+df.Forward)
                        .query('Grupo_Sector != "BANCOS" and Grupo_Sector != "Real" ')
)



Flujo_cambiario_2w  = (

                    flujo_cambiario_spot
                                        .assign(Fecha = lambda df: pd.to_datetime(df.Fecha))
                                        .query('Fecha>=@ultima_dossemanas_fecha and Fecha <= @hoy')
                                        .set_index('Fecha')
                                        .groupby('Grupo_Sector')
                                        .Monto_MM_USD.sum()
                                        .reset_index()
                                        .rename(columns={'Monto_MM_USD':'Spot'})
                        .merge((flujo_cambiario_forward
                                                    .assign(Fecha = lambda df: pd.to_datetime(df.Fecha))
                                                    .query('Fecha>=@ultima_dossemanas_fecha and Fecha <= @hoy')
                                                    .set_index('Fecha')
                                                    .groupby('Grupo_Sector')
                                                    .Monto.sum()
                                                    .reset_index()
                                                    .rename(columns={'Monto':'Forward'})
                                ),how='left',on='Grupo_Sector')
                        .fillna(0)
                        .set_index('Grupo_Sector')
                        .assign(Spot_derivado = lambda df: df.Spot+df.Forward)
                        .query('Grupo_Sector != "BANCOS" and Grupo_Sector != "Real" ')
)




# %%
Fig_Flujo_cambiario2w = Flujo_cambiario_2w.sort_values('Spot_derivado',ascending=False)[['Spot','Forward']].pipe(px.bar,color_discrete_sequence=brand_palette,title=f"Flujo Cambiario desde {hoy} - 14 días (US$ Mill.)",template='plotly_white').add_scatter(x=Flujo_cambiario_2w.index,y=Flujo_cambiario_2w['Spot_derivado'],mode='markers',marker=dict(size=10,color='red'),name='Spot + Derivados')
Fig_Flujo_cambiario2w

# %%
Fig_Flujo_cambiario1w = Flujo_cambiario_1w.sort_values('Spot_derivado',ascending=False)[['Spot','Forward']].pipe(px.bar,color_discrete_sequence=brand_palette,title=f"Flujo Cambiario desde {hoy} - 7 días (US$ Mill.)",template='plotly_white').add_scatter(x=Flujo_cambiario_1w.index,y=Flujo_cambiario_1w['Spot_derivado'],mode='markers',marker=dict(size=10,color='red'),name='Spot + Derivados')
Fig_Flujo_cambiario1w

# %%

brand_palette1 = [
     "#4472C4",  # azul
   
    "#C00000",  # rojo
    "#70AD47",  # verde
    "#FFC000",  # amarillo
     "#ED7D31",  # naranjo
    "#57257D",  # púrpura
    "#FF0DFF",  # gris
     "#001730",  # azul oscuro
    "#44546A",  # azul grisáceo
    "#000000",  # negro
    "#BF9C69",  # beige
]


# %%
#Fixing de la banca
tabla_fixingbanca = gd.get_data("SELECT CASE WHEN Nombre_informante = 'BANCOBICE' THEN 'BICE' WHEN Nombre_informante = 'BANCOBTGPA' THEN 'BTG PACTUAL' WHEN Nombre_informante = 'BTGPACTUAL' THEN 'BTG PACTUAL' WHEN Nombre_informante = 'BANCOCONSO' THEN 'BANCO CONSORCIO' WHEN Nombre_informante = 'BANCODECHI' THEN 'BANCO DE CHILE' WHEN Nombre_informante = 'BANCODECRE' THEN 'BCI'		WHEN Nombre_informante = 'BANCODELES' THEN 'BANCO ESTADO' WHEN Nombre_informante = 'BANCOFALAB' THEN 'BANCO FALABELLA' WHEN Nombre_informante = 'BANCOINTER' THEN 'BANCO INTERNACIONAL' WHEN Nombre_informante = 'BANCORIPLE' THEN 'BANCO RIPLEY' WHEN Nombre_informante = 'BANCOSANTA' THEN 'BANCO SANTANDER'  WHEN Nombre_informante = 'BANCOSECUR' THEN 'BANCO SECURITY'		WHEN Nombre_informante = 'CHINACONST' THEN 'CHINA CONSTRUCTION BANK'		WHEN Nombre_informante = 'CREDICORPC' THEN 'BANCO CREDICORP'		WHEN Nombre_informante = 'EUROAMERIC' THEN 'EUROAMERICA' 		WHEN Nombre_informante = 'HSBCBANK' THEN 'HSBC BANK'		WHEN Nombre_informante = 'ITAUCORPBA' THEN 'BANCO ITAU CORPBANCA' WHEN Nombre_informante = 'JPMORGANCH' THEN 'JP MORGAN'		WHEN Nombre_informante = 'LARRAINVIA' THEN 'LARRAIN VIAL' WHEN Nombre_informante = 'SCOTIABANK' THEN 'BANCO SCOTIABANK' ELSE 'Nombre no encontrado' END AS NombreInformanteNorm,	CASE WHEN sector_cont = 'AFP' THEN 'AFP'		WHEN sector_cont = 'BANCOS' THEN 'Bancos' WHEN sector_cont = 'BCCH' THEN 'BCCh'		WHEN sector_cont = 'CIAS_DE_SEGUROS' THEN 'Seguros' WHEN sector_cont = 'CORREDORAS_DE_BOLSA' THEN 'Corredoras de Bolsa'		WHEN sector_cont = 'EMPRESAS' THEN 'Empresas'		WHEN sector_cont = 'FFMM' THEN 'Fondos Mutuos'		WHEN sector_cont = 'GOBIERNO' THEN 'Gobierno'		WHEN sector_cont = 'OFF_SHORE' THEN 'NR' ELSE sector_cont END AS SectorContNorm, * FROM deriv_vencimientos	WHERE YEAR(Fecha_ven) >= 2026 AND YEAR(Fixing) >= 2026 AND Modalidad_pago = 'C' 	AND instrumento_nom = 'Forward' ORDER BY Fecha_ven DESC")

hoy = date.today()

fig_fixingbanca = (tabla_fixingbanca.query('Fixing == @hoy')[['NombreInformanteNorm','sector_cont','Pos_neta']]
                .assign(Informante = lambda df: df.NombreInformanteNorm)
                .drop(columns='NombreInformanteNorm')
                .groupby(['Informante','sector_cont'])
                .Pos_neta.sum()
                .reset_index() #.sector_cont.unique()
                .pivot(index='Informante',columns='sector_cont',values='Pos_neta')
                .assign(Neto = lambda df: df.sum(axis=1))
                .sort_values('Neto',ascending=False)
                .rename(columns={'OFF_SHORE':'NR'})
                .fillna(0))

f = fig_fixingbanca.reset_index().sum().to_frame().T
f['Informante'] = ['Total']
t = pd.concat([fig_fixingbanca,f.set_index('Informante')],ignore_index=False).apply(pd.to_numeric)

Fig_fixingbanca_sector = (t
    .drop(columns='Neto')
    .pipe(px.bar,color_discrete_sequence=brand_palette1,title="Fixing de la banca", template='plotly_white')
    .add_scatter(x=t.index,y=t['Neto'],mode='markers',marker=dict(size=10,color='red'),name='Neto')       
)

Fig_fixingbanca_sector

# %%

ayer = hoy - timedelta(days=1)
w1 = hoy - timedelta(weeks=1)
w2 = hoy - timedelta(weeks=2)
m1 = hoy - timedelta(weeks = 4)


def funcion_fixing(Temporalidad):
    Fecha_fix = Temporalidad
    df = (tabla_fixingbanca.query('Fixing == @Fecha_fix')[['NombreInformanteNorm','sector_cont','Pos_neta']]
                .assign(Informante = lambda df: df.NombreInformanteNorm)
                .drop(columns='NombreInformanteNorm')).groupby('sector_cont').Pos_neta.sum().reset_index().assign(Fecha=lambda df: Fecha_fix).pivot(index='Fecha',columns='sector_cont',values='Pos_neta')
    return df


Fixinghoy = funcion_fixing(hoy)
Fixingayer = funcion_fixing(ayer)
Fixingw1 = funcion_fixing(w1)
Fixingw2 = funcion_fixing(w2)
Fixingm1 = funcion_fixing(m1)


Fixingporfecha = pd.concat([Fixinghoy,Fixingayer,Fixingw1,Fixingw2,Fixingm1]).assign(Neto = lambda df: df.sum(axis=1))



Fig_fixingporfecha = (Fixingporfecha
    .drop(columns='Neto')
    .pipe(px.bar,color_discrete_sequence=brand_palette1,title="Fixing de la banca", template='plotly_white')
    .add_scatter(x=Fixingporfecha.index,y=Fixingporfecha['Neto'],mode='markers',marker=dict(size=10,color='red'),name='Neto')       
)

Fig_fixingporfecha

# %% [markdown]
# Estructura por portafolios

# %%
Allocation_internacionallocal = (gd.get_data('select * from dbo.tbl_cartera_afp_mensual_agg').query('fondo == "TOTAL" and afp=="TOTAL"').sort_values('fecha').groupby(['fecha','pais']).mmus.sum().reset_index().assign(weight= lambda df: df.mmus/df.groupby('fecha').mmus.transform(sum)))
AllocationilF = Allocation_internacionallocal.pivot(index='fecha',columns='pais',values='mmus').sum(axis=1).reset_index().rename(columns={0:'AUM'}).merge(Allocation_internacionallocal.pivot(index='fecha',columns='pais',values='weight').reset_index(),how='inner',on='fecha').set_index('fecha')

# %%


fig = make_subplots(specs= [[{"secondary_y": True }]])
bar_Allocation = AllocationilF['AUM'].pipe(px.bar,color_discrete_sequence=brand_palette)

for tr in bar_Allocation.data:
    fig.add_trace(tr, secondary_y=False)


Fig_AllocationInt_Nac = (fig.add_trace(go.Scatter(
    x=AllocationilF.index,
    y=AllocationilF["Nacional"],
    name="Nacional",
    mode="lines",
    line=dict(color="#C00000",width=3)),secondary_y=True).add_trace(go.Scatter(
    x=AllocationilF.index,
    y=AllocationilF["Extranjero"],
    name="Extranjero",
    mode="lines",
    line=dict(color="#0004FF",width=3)),secondary_y=True)    
    .update_layout(title="Allocation internacional vs local (porcentaje; US$ Mill.)",template="plotly_white",hovermode="x unified",barmode="overlay",bargap=0.25
                ).update_yaxes(title_text="US$ Millones",secondary_y=False,range=[0,300000]).update_yaxes(title_text="Porcentaje",secondary_y=True,range=[0.35,0.65]).update_xaxes(title_text="Fecha"))

Fig_AllocationInt_Nac


# %%
retornos_afp = (gd.get_data("select * from tbl_cuota_afp where fecha>'2023-01-01'")
                .sort_values('fecha').assign(weight=lambda df: df.patrimonio/df.groupby(['fecha','fondo']).patrimonio.transform(sum),
                                            retorno_por_afp = lambda df:  df.weight*df.retornos_cuota)
                .groupby(['fondo','fecha'])
                .retorno_por_afp.sum()
                .reset_index()
                .query('fecha>="2026-03-01" and fecha<="2026-03-19"')
                #.groupby('fondo')
                #.retorno_por_afp.sum()                
                )

# %%
df_Atributtion = (gd.get_data('select * from tbl_performance_attribution_afp_sep')
    .sort_values('fecha').set_index(['fecha','fondo'])
    .assign(Total=lambda df: df.sum(axis=1))
    .reset_index()
    .merge(retornos_afp,how='inner',on=['fondo','fecha'])
    .assign(Residuo=lambda df: df.retorno_por_afp-df.Total)) #.query('fondo=="A"').set_index(['fecha','fondo']).cumsum().query('fecha==fecha.max()').reset_index()



FondoA = df_Atributtion.query('fondo=="A"').set_index(['fecha','fondo']).cumsum().query('fecha==fecha.max()').reset_index()
FondoB = df_Atributtion.query('fondo=="B"').set_index(['fecha','fondo']).cumsum().query('fecha==fecha.max()').reset_index()
FondoC = df_Atributtion.query('fondo=="C"').set_index(['fecha','fondo']).cumsum().query('fecha==fecha.max()').reset_index()
FondoD = df_Atributtion.query('fondo=="D"').set_index(['fecha','fondo']).cumsum().query('fecha==fecha.max()').reset_index()
FondoE = df_Atributtion.query('fondo=="E"').set_index(['fecha','fondo']).cumsum().query('fecha==fecha.max()').reset_index()

atributtion_mtd = pd.concat([FondoA,FondoB,FondoC,FondoD,FondoE]).drop(columns= 'fecha').drop(columns={'Total'}).set_index('fondo') #.stack().reset_index().pivot(index='level_1',columns='fondo',values=0).reset_index().rename(columns={'level_1':'Attribution'})
Fig_attribution = atributtion_mtd.drop(columns='retorno_por_afp').pipe(px.bar,color_discrete_sequence=brand_palette,title='Attribution Fondos por clase de activos en marzo (porcentaje)',template='plotly_white').add_scatter(x=atributtion_mtd.index,y=atributtion_mtd['retorno_por_afp'],mode='markers',marker=dict(size=10,color='red'),name='Total')

# %%

Fig_attribution

# %%
fig_dv01_spc = gd.get_data('select * from tbl_mtm_swap_afp_proyeccion').sort_values('fecha')[['fecha','fondo','pais','dv01','moneda']].groupby(['fecha','moneda']).dv01.sum().reset_index().query('fecha>"2019-10-01"').pivot(index='fecha',columns='moneda',values='dv01').rename(columns={'NO':'CLP'}).pipe(px.line,title='DV01 SPC por moneda proyectado (US$ Mill.)',template='plotly_white')

# %%
pseudo_dolar = 940
m1 = date.today() - timedelta(weeks=4)

Fig_Movimientos_fondos = (gd.get_data("select fecha,flujo,fondo from tbl_cuota_afp where fecha>'2023-01-01'")
        .groupby(['fecha','fondo'])
        .flujo.sum().reset_index()
        .assign(flujos_usd = lambda df:  df.flujo/(1000000*940))
        .drop(columns='flujo')
        .pivot(index='fecha',columns='fondo',values='flujos_usd')
        .query('fecha>=@m1')
        .pipe(px.bar,template='plotly_white',color_discrete_sequence=brand_palette1, title='Traspaso de fondos de FP (US$ Mill.)')
)
Fig_Movimientos_fondos

# %%
Spot = gd.get_data(""" 
-- AFP 2024–2025 (formato original SAS)
SELECT
    CAST(Fecha AS date) AS Fecha,
    REPLACE(Sector_contraparte, 'AFP  PLANVITAL', 'AFP PLANVITAL') AS Sector_contraparte,
    Monto_MM_USD,
    Afecto_a_derivados
FROM spot.[Agregada_por_sector_(SAS)]
WHERE Fecha >= '2024-01-01'
  AND Fecha <= '2025-12-31'
  AND Sector_contraparte LIKE 'AFP %'

UNION ALL

-- AFP 2026+ (SICAM, mapeado y AGRUPADO por día, AFP y afecto)
SELECT
    s.Fecha,
    s.Sector_contraparte,
    SUM(s.Monto) AS Monto_MM_USD,
    s.Afecto_derivado AS Afecto_a_derivados
FROM (
    SELECT
        CAST(Fecha AS date) AS Fecha,
        CASE Sector_detalle
            WHEN 'Capital'   THEN 'AFP CAPITAL'
            WHEN 'Cuprum'    THEN 'AFP CUPRUM'
            WHEN 'Habitat'   THEN 'AFP HABITAT'
            WHEN 'Modelo'    THEN 'AFP MODELO'
            WHEN 'Planvital' THEN 'AFP PLANVITAL'
            WHEN 'Provida'   THEN 'AFP PROVIDA'
            WHEN 'Uno'       THEN 'AFP UNO'
        END AS Sector_contraparte,
        Monto,
        Afecto_derivado
    FROM [Inputs].[Spot].[Temp_SICAM_Estadisticas]
    WHERE Fecha >= '2026-01-01'
      AND Sector_contraparte = 'AFP'
      AND Sector_detalle IN ('Capital','Cuprum','Habitat','Modelo','Planvital','Provida','Uno')
) s
GROUP BY s.Fecha, s.Sector_contraparte, s.Afecto_derivado
ORDER BY Fecha DESC, Sector_contraparte;
""")

Forward = gd.get_data(""" 
SELECT
    CAST(Fecha AS DATE) AS Fecha,
    sector_det_cont,
    sector_cont,
    CASE 
        WHEN sector_det_cont LIKE '%AFP%' THEN 'AFP'
        WHEN sector_det_cont LIKE '%FFMM%' THEN 'FFMM'
        WHEN sector_det_cont LIKE '%PERSONAS%' THEN 'OTROS'
        WHEN sector_det_cont = 'CORREDORAS_DE_BOLSA' THEN 'CB'
        WHEN sector_det_cont = 'CIAS_DE_SEGUROS' THEN 'CS'
        WHEN sector_det_cont = 'OFF_SHORE' THEN 'NR'
        ELSE sector_det_cont
    END AS Grupo_Sector,
    SUM(Monto_) AS Monto,
    instrumento_nom,
    Modalidad_pago,
    plazo,
    dias
FROM (
    SELECT 
        Fecha, nombre_cont, sector_cont, sector_det_cont, dias, plazo, instrumento_nom, 
        Monto_, Modalidad_pago, [Tipo contrato] = 'Suscripcion'
    FROM DACE.dbo.deriv_suscripciones
    WHERE Fecha <= GETDATE() - 1 AND Fecha > GETDATE() - 400

    UNION ALL

    SELECT 
        Fecha_ven AS Fecha, nombre_cont, sector_cont, sector_det_cont, dias, plazo, instrumento_nom, 
        Pos_neta AS Monto_, Modalidad_pago, [Tipo contrato] = 'Vencimiento'
    FROM DACE.dbo.deriv_vencimientos
    WHERE Fecha_ven <= GETDATE() - 1 AND Fecha_ven > GETDATE() - 400
) datos
GROUP BY 
    CAST(Fecha AS DATE),
    sector_det_cont,
    sector_cont,
    instrumento_nom,
    Modalidad_pago,
    plazo,
    dias,
    CASE 
        WHEN sector_det_cont LIKE '%AFP%' THEN 'AFP'
        WHEN sector_det_cont LIKE '%FFMM%' THEN 'FFMM'
        WHEN sector_det_cont LIKE '%PERSONAS%' THEN 'OTROS'
        WHEN sector_det_cont = 'CORREDORAS_DE_BOLSA' THEN 'CB'
        WHEN sector_det_cont = 'CIAS_DE_SEGUROS' THEN 'CS'
        WHEN sector_det_cont = 'OFF_SHORE' THEN 'NR'
        ELSE sector_det_cont
    END;
""")

# %%
PosicionDerivados = Forward.query('sector_cont == "AFP" and Grupo_Sector=="AFP"').groupby('Fecha').Monto.sum().reset_index().assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).rename(columns={'Monto':'Derivados'})
Posicionspot = Spot.groupby('Fecha').Monto_MM_USD.sum().reset_index().assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).rename(columns={'Monto_MM_USD':'Spot'})
Posicionspotderivados = PosicionDerivados.merge(Posicionspot,how='inner',on='Fecha').assign(Neto=lambda df: df.Derivados+df.Spot).set_index('Fecha').query('Fecha>="2025-11-01"').cumsum()


# %%
fig_spot_derivados_afp = Posicionspotderivados[['Derivados','Spot']].pipe(px.area,template='plotly_white',color_discrete_sequence=brand_palette, title='Posición spot y derivados AFP (US$ Mill. ; Acu. 01nov)').add_scatter(x=Posicionspotderivados.index, y=Posicionspotderivados['Neto'],name='Neto',mode="lines",line=dict(color="#524D5A",width=5))

# %%
MtM = gd.get_data('select fecha,fondo,moneda,pais,mtm from tbl_mtm_swap_afp_proyeccion').groupby(['fecha','fondo']).mtm.sum().reset_index().pivot(index='fecha',columns='fondo',values='mtm').query('fecha>="2020-06-01"').pipe(px.line,template='plotly_white',color_discrete_sequence=brand_palette)
MtM

# %%
###3 MTM SWAP

# %%
#Posicion RFL
RFL = gd.get_data("""select fecha,nemotecnico,instrumento_largo,plazo,nominales_pesos_ultimo from [view_stock_dcv] where fecha>=DATEADD(day,-40,getdate()) and institucion='AFP y AFC' and (instrumento_largo = 'Bono Banco' or instrumento_largo='Bono Tesoreria General de la Republica' or instrumento_largo = 'Bono TGR en $' or instrumento_largo = 'Debenture' or instrumento_largo= 'Deposito a Plazo Fijo' or instrumento_largo = 'Pagare Descontable del Banco Central') order by fecha desc""")


# %%
fecha_max = str(RFL.fecha.max())

RFL_Mat = (RFL
    .assign(fecha=lambda df: pd.to_datetime(df.fecha)).query('fecha in @fecha_max')
    .assign(Maturity = lambda df: df.plazo/365)   # .assign(Maturity = lambda df: (pd.to_datetime(df.fecha)+pd.to_timedelta(df['plazo'],unit='D')))
    .assign(MM_USD = lambda df: df.nominales_pesos_ultimo/(950*1000000))
)

RFL_Mat['Bucket'] = np.select(
    [
        RFL_Mat['Maturity'] <= 2,
        (RFL_Mat['Maturity'] > 2 ) & (RFL_Mat['Maturity'] < 6),
        (RFL_Mat['Maturity'] >= 6)  & (RFL_Mat['Maturity'] < 10),
        RFL_Mat['Maturity'] >= 10,],
        ["Menor 2Y","Entre 3-5Y", "Entre 6-9Y", "Mayor 10Y"],default="Mayor 10Y")

orden = ["Menor 2Y","Entre 3-5Y", "Entre 6-9Y", "Mayor 10Y"]
RFL_Mat['Bucket'] = pd.Categorical(RFL_Mat["Bucket"], categories=orden,ordered=True)
posicion_rfl_max = (RFL_Mat[['nemotecnico','plazo','instrumento_largo','Bucket','MM_USD']].rename(columns={'instrumento_largo':'Instrumento'})
                    .groupby(['Instrumento','Bucket'])
                    .MM_USD.sum().reset_index()
                    .pivot(index='Bucket',columns='Instrumento',values='MM_USD')
                    .rename(columns = {'Deposito a Plazo Fijo':'DAP','Pagare Descontable del Banco Central':'PDBC','Bono TGR en $':'BTP','Bono Tesoreria General de la Republica':'BTU'})
                    .fillna(0)
                    .sort_index()
)

Fig_posicion_rfl_max = posicion_rfl_max.pipe(px.bar,template='plotly_white',color_discrete_sequence=brand_palette1,title = 'Posicion RFL AFP')

Fig_posicion_rfl_max

# %%
#Variacion dcv afp t-5

fecha_t5 = str(RFL.fecha.max()-timedelta(weeks=1))

RFL_Mat = (RFL
    .assign(fecha=lambda df: pd.to_datetime(df.fecha)).query('fecha in @fecha_t5')
    .assign(Maturity = lambda df: df.plazo/365)   # .assign(Maturity = lambda df: (pd.to_datetime(df.fecha)+pd.to_timedelta(df['plazo'],unit='D')))
    .assign(MM_USD = lambda df: df.nominales_pesos_ultimo/(950*1000000))
)

RFL_Mat['Bucket'] = np.select(
    [
        RFL_Mat['Maturity'] <= 2,
        (RFL_Mat['Maturity'] > 2 ) & (RFL_Mat['Maturity'] < 6),
        (RFL_Mat['Maturity'] >= 6)  & (RFL_Mat['Maturity'] < 10),
        RFL_Mat['Maturity'] >= 10,],
        ["Menor 2Y","Entre 3-5Y", "Entre 6-9Y", "Mayor 10Y"],default="Mayor 10Y")

orden = ["Menor 2Y","Entre 3-5Y", "Entre 6-9Y", "Mayor 10Y"]
RFL_Mat['Bucket'] = pd.Categorical(RFL_Mat["Bucket"], categories=orden,ordered=True)

posicion_rfl_t1w = (RFL_Mat[['nemotecnico','plazo','instrumento_largo','Bucket','MM_USD']].rename(columns={'instrumento_largo':'Instrumento'})
                    .groupby(['Instrumento','Bucket'])
                    .MM_USD.sum().reset_index()
                    .pivot(index='Bucket',columns='Instrumento',values='MM_USD')
                    .rename(columns = {'Deposito a Plazo Fijo':'DAP','Pagare Descontable del Banco Central':'PDBC','Bono TGR en $':'BTP','Bono Tesoreria General de la Republica':'BTU'})
                    .fillna(0)
                    .sort_index())

Fig_posicion_rfl_t1w = posicion_rfl_t1w.pipe(px.bar,template='plotly_white',color_discrete_sequence=brand_palette1)

Variacion_DCV_RFL = posicion_rfl_max - posicion_rfl_t1w

Fig_variacion_dcv_afp = Variacion_DCV_RFL.pipe(px.bar,template='plotly_white',color_discrete_sequence=brand_palette1,title='Variación Semanal DCV (US$ Mill.)')
Fig_variacion_dcv_afp

# %%
Spot.Sector_contraparte.unique() #.query('Sector_contraparte == "AFP"')

# %%
# Flujo Cambiario
Var5 = str(Spot.Fecha.max() - timedelta(weeks=1))


Forward_afp = (Forward.query('sector_cont == "AFP" and Grupo_Sector == "AFP"')
                                    .assign(Fecha= lambda df: pd.to_datetime(df.Fecha))
                                    .query("Fecha >= @Var5 and Fecha <= '2026-03-23'")
                                    .groupby('sector_det_cont').Monto.sum().reset_index().rename(columns={'sector_det_cont': 'Sector_contraparte'})
                                    )

Forward_afp['Sector_contraparte'] = (Forward_afp['Sector_contraparte'].str.replace("AFP_","",regex=False))


Spot_afp = (Spot.assign(Fecha=lambda df: pd.to_datetime(df.Fecha))
                     .query("Fecha >= @Var5 and Fecha <= '2026-03-23'")
                     .groupby('Sector_contraparte')
                     .Monto_MM_USD.sum().reset_index()
                     )

Spot_afp['Sector_contraparte'] = (Spot_afp['Sector_contraparte'].str.replace("AFP ","",regex=False))

 
Cambiario_por_afp = (Spot_afp
                     .merge(Forward_afp, how='inner',on='Sector_contraparte')
                    .rename(columns={'Monto_MM_USD':'Spot','Monto':'Forward'})
                    .assign(Spot_derivados = lambda df: df.Spot+df.Forward)
                    .set_index('Sector_contraparte')
                    .sort_values('Spot_derivados',ascending=False)
)
fig_cambiario_afp = Cambiario_por_afp[['Spot','Forward']].pipe(px.bar,template='plotly_white',color_discrete_sequence=brand_palette,title='Flujos cambiarios Variacion 1w (US$ Mill)').add_scatter(x=Cambiario_por_afp.index,y=Cambiario_por_afp['Spot_derivados'],mode='markers',marker=dict(size=10,color='red'),name='Spot+Derivados')


# %%
#Stock por tipo de fondo


Fig_stock_fondo = gd.get_data("select * from tbl_cartera_afp_mensual_agg where fecha>'01-01-2010' and afp='TOTAL' order by fecha desc").groupby(['fecha','fondo']).mmus.sum().reset_index().pivot(index='fecha',columns='fondo',values='mmus').drop(columns='TOTAL').pipe(px.area,template='plotly_white',color_discrete_sequence=brand_palette1,title='Stock Fondos (US$ Mill.)')
Fig_stock_fondo

# %%
# Flujos fondos

Flujos_FFMM = gd.get_data("select [fecha], [tipo_fondo],[patrimonio_efectivo],[patrimonio_neto], [flujo], [dc_moneda],[cuotas_circulacion], [patrimonio_efectivo]/[cuotas_circulacion] as valor_cuota from DACE.dbo.tbl_cuota_ffmm where Fecha >= '2019-01-01' and dc_moneda = '$$'")
Fig_flujos_FFMM = (Flujos_FFMM
    .assign(flujos_usd = lambda df: df.flujo/950/1000000,fecha = lambda df: pd.to_datetime(df.fecha))
    .groupby(['fecha','tipo_fondo']).flujos_usd.sum().reset_index()
    .query('fecha <= "2026-03-24" and fecha > "2026-03-17"')
    .groupby('tipo_fondo')
    .flujos_usd.sum().reset_index()
    .set_index('tipo_fondo').pipe(px.bar,template='plotly_white',color_discrete_sequence=brand_palette,title='Flujos semanales por fond (US$ Mill.)')
)
Fig_flujos_FFMM

# %%
#Evolucion de descomposicion fondo t6
df_allocation_t6 =  gd.get_data("select EOMONTH([Fecha_informacion]) as Fecha, [filter_intrs] as Tipo_instrumento, [Tipo_fondo], [valorizacion_cierre] as Monto_MCLP from DACE.dbo.tbl_cartera_ffmm_mensual_agg where [Fecha_informacion] >= '2019-01-01'")


Fig_allocation_t6 = (df_allocation_t6
            .assign(Fecha=lambda df: pd.to_datetime(df.Fecha))
            .groupby(['Fecha','Tipo_instrumento','Tipo_fondo'])
            .Monto_MCLP.sum().reset_index().assign(Monto_USD=lambda df: df.Monto_MCLP/(1000*950))
            .query('Tipo_fondo==6 and Tipo_instrumento != "CFM" and Tipo_instrumento != "Otro_RV"')
            .pivot(index='Fecha',columns='Tipo_instrumento',values='Monto_USD')
            .pipe(px.area,template='plotly_white',color_discrete_sequence=brand_palette,title='Allocation T6 (Mill US$.)')
        )

# %%
DCV_FFMM = gd.get_data("""
SELECT T.Fecha, T.Tipo_Instrumento, T.Tramo, sum(T.Nominales_CLP) as Nominales_CLP
FROM (SELECT Fecha,
	   CASE WHEN Instrumento in ('BONO BANCO', 'BONO SUB', 'B BANCO DG') THEN 'BB'
	   WHEN Instrumento in ('BCU') THEN 'BCU'
	   WHEN Instrumento in ('DEBENTURES') THEN 'BE'
	   WHEN Instrumento in ('BONO TGR$') THEN 'BTP'
	   WHEN Instrumento in ('BONO TGR') THEN 'BTU'
	   WHEN Instrumento in ('DPF') THEN 'DAP'
	   WHEN Instrumento in ('PDBC') THEN 'PDBC'
	   WHEN Instrumento in ('LH', 'BR', 'BonoMINVU', 'BONO FETGR', 'TITSEC', 'BONO HIP', 'BCA', 'EFECOMERCI') THEN 'Otro'
	   ELSE 'No Clasificado' END AS Tipo_Instrumento,
       Nemotecnico,
	   CASE WHEN YEAR(DATEADD(DAY, avg(plazo), Fecha)) - YEAR(Fecha) <= 2 THEN 'Menor a 2Y'
	   WHEN YEAR(DATEADD(DAY, avg(plazo), Fecha)) - YEAR(Fecha) <= 5 THEN 'Entre 3 y 5Y'
	   WHEN YEAR(DATEADD(DAY, avg(plazo), Fecha)) - YEAR(Fecha) <= 9 THEN 'Entre 6 y 9Y'
	   WHEN YEAR(DATEADD(DAY, avg(plazo), Fecha)) - YEAR(Fecha) <= 12 THEN 'Entre 10 y 12Y'
	   ELSE '>12Y' END AS Tramo,
	   sum(Nominales_pesos_ultimo) as Nominales_CLP
FROM [DACE].[dbo].[view_stock_dcv]
WHERE fecha >= '2024-08-30' and Instrumento in ('BONO TGR', 'DPF', 'DEBENTURES', 'PDBC', 'BONO BANCO', 'BONO SUB', 'BONO TGR$', 'B BANCO DG')
and Institucion in ('Fondos Mutuos', 'Administradora Fondos')
GROUP BY Fecha, Institucion, Instrumento, Nemotecnico) T
GROUP BY T.Fecha, T.Tipo_Instrumento, T.Tramo
ORDER BY T.Fecha DESC
            """)

# %%
fig_dcv_composicion_ffmm = DCV_FFMM.assign(mmusd = lambda df: df.Nominales_CLP/1000000/950, Fecha = lambda df: pd.to_datetime(df.Fecha)).query('Fecha == "2026-03-24"').groupby('Tipo_Instrumento').mmusd.sum().reset_index().pipe(px.pie,names='Tipo_Instrumento',values='mmusd',color_discrete_sequence=brand_palette1,title='Composición portafolio DCV (porcentaje)')

# %%
df_dap_pdbc = gd.get_data("""
SELECT fecha,
	   instrumento_largo,
	   sum(nominales_pesos_ultimo) as nominales_pesos_ultimo
FROM view_stock_dcv
WHERE fecha >= '2023-12-30' and institucion in ('Fondos Mutuos','Administradora Fondos') and
instrumento_largo in ('Deposito a Plazo Fijo',	'Pagare Descontable del Banco Central')
GROUP BY fecha, instrumento_largo
""").assign(fecha=lambda df: pd.to_datetime(df.fecha))

# %%
FIG_DAP_PDBC_FFMM = df_dap_pdbc.assign(mmus = lambda df: df.nominales_pesos_ultimo/1000000/950).pivot(index='fecha', columns='instrumento_largo',values='mmus').rename(columns={'Deposito a Plazo Fijo': 'DAP', 'Pagare Descontable del Banco Central':'PDBC'}).pipe(px.line,template='plotly_white',color_discrete_sequence = brand_palette, title='Stock DAP y PDBC(Acum US$ Mill.)')
FIG_DAP_PDBC_FFMM

# %%
Fig_Flujos_spot_FFMM = gd.get_data("""

WITH s1 AS (  -- SAS: 2024-01-01 <= Fecha < 2026-01-01
    SELECT
        CAST(Fecha AS date) AS Fecha,
        CASE
            WHEN Sector_contraparte LIKE 'AFP%'  THEN 'AFP'
            WHEN Sector_contraparte LIKE 'FFMM%' THEN 'FFMM'
            WHEN Sector_contraparte IN ('CONT FALCOM','CONT HMC','CONT PICTON','CONT FRONTAL TRUST','CONT PRUDENTIAL') THEN 'FFMM'
            WHEN Sector_contraparte LIKE '%PERSONAS%' THEN 'OTROS'
            WHEN Sector_contraparte = 'CORREDORA DE BOLSA' THEN 'CB'
            WHEN Sector_contraparte = 'CIA SEGUROS'      THEN 'CS'
            WHEN Sector_contraparte = 'EMPRESA FINANCIERA' THEN 'EMPRESA_FINANCIERA'
            WHEN Sector_contraparte = 'EMPRESA REAL'       THEN 'EMPRESA_REAL'
            WHEN Sector_contraparte = 'EMPRESA MINERA'     THEN 'MINERA'
            WHEN Sector_contraparte IN ('EUROCLEAR','NO RESIDENTES') THEN 'NR'
            ELSE Sector_contraparte
        END AS Grupo_Sector,
        Afecto_a_derivados,
        Monto_MM_USD
    FROM spot.[Agregada_por_sector_(SAS)]
    WHERE CAST(Fecha AS date) >= '2019-01-01'
      AND CAST(Fecha AS date) <  '2026-01-01'
),
s2 AS (  -- SICAM: desde 2026-01-01
    SELECT
        CAST(Fecha AS date) AS Fecha,
        CASE
            WHEN Sector_contraparte = 'Empresas' THEN
                CASE
                    WHEN Sector_detalle = 'EMPRESA FINANCIERAS' THEN 'EMPRESA_FINANCIERA'
                    WHEN Sector_detalle = 'EMPRESA MINERA'      THEN 'MINERA'
                    WHEN Sector_detalle IN ('EMPRESAS AGRUPADAS OP VARIAS','EMPRESAS SECTOR REAL','PERSONA NATURAL','POR CLASIFICA')
                                                             THEN 'EMPRESA_REAL'
                    ELSE 'EMPRESA_REAL'
                END
            ELSE
                CASE Sector_contraparte
                    WHEN 'Corredora De Bolsa' THEN 'CB'
                    WHEN 'Tgr'                 THEN 'TGR'
                    WHEN 'No Residentes'       THEN 'NR'
                    WHEN 'Bancos'              THEN 'BANCOS'
                    WHEN 'Bcch'                THEN 'BCCH'
                    WHEN 'Ffmm'                THEN 'FFMM'
                    WHEN 'Otros'               THEN 'OTROS'
                    WHEN 'Cia Seguros'         THEN 'CS'
                    WHEN 'Afp'                 THEN 'AFP'
                    ELSE Sector_contraparte
                END
        END AS Grupo_Sector,
        Afecto_derivado AS Afecto_a_derivados,
        CAST(Monto AS decimal(38,6)) AS Monto_MM_USD  -- ya en MM USD
    FROM [Inputs].[Spot].[Temp_SICAM_Estadisticas]
    WHERE CAST(Fecha AS date) >= '2026-01-01'
),
u AS (
    SELECT Fecha, Grupo_Sector, Monto_MM_USD, Afecto_a_derivados FROM s1
    UNION ALL
    SELECT Fecha, Grupo_Sector, Monto_MM_USD, Afecto_a_derivados FROM s2
)
SELECT
    Fecha,
    SUM(Monto_MM_USD) AS Monto_MMUSD
FROM u
WHERE Grupo_Sector = 'FFMM'
GROUP BY Fecha, Grupo_Sector
ORDER BY Fecha DESC, Grupo_Sector ASC;


""").assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).set_index('Fecha').sort_index().query('Fecha>"2022-12-29"').rename(columns={'Monto_MMUSD':'Spot'}).cumsum().pipe(px.line,template='plotly_white',title='Flujo spot acumulados (US$ Mill.)')

# %%
Fig_Flujos_spot_FFMM

# %%
#gd.get_data("""select * FROM [Inputs].[Spot].[Temp_SICAM_Estadisticas] where Fecha >= '2025-01-01'""").Sector_contraparte.unique() #Flujos de fondos #

# %%
dv01_FFMM = gd.get_data("""
SELECT EOMONTH([Fecha_informacion]) as Fecha, [1], [2], [3], [4], [5], [6], [7], [8]
FROM
(SELECT [fecha_informacion]
      ,[tipo_fondo]
      ,[dv01]/950/1000 as [dv01]
FROM [DACE].[dbo].[tbl_cartera_ffmm_mensual_agg]
WHERE fecha_informacion >= '2019-01-01' and filtro_aplica = 'Aplica') as T
PIVOT (SUM([dv01]) for [tipo_fondo] in ([1], [2], [3], [4], [5], [6], [7], [8])) as PVT
ORDER BY Fecha ASC

""")

fig_dv01_ffmm = dv01_FFMM[['1','2','3','6']].pipe(px.line,title='DV01 por fondo',template='plotly_white',color_discrete_sequence=brand_palette)

# %%
Base_duracion_FFMM = gd.get_data("""SELECT EOMONTH([Fecha_informacion]) as Fecha, [tipo_fondo], [BB], [BCP], [BCU], [BE], [BTP], [BTU], [BU], [DPC], [DPL], [Otro_RF]+[Otro_RV] as Otro, [PDBC]
FROM
(SELECT [fecha_informacion]
      ,[filter_intrs]
      ,[tipo_fondo]
      ,[contribucion_duracion]
FROM [DACE].[dbo].[tbl_cartera_ffmm_mensual_agg]
WHERE filtro_aplica = 'Aplica' and fecha_informacion >= '2019-01-01') as T
PIVOT (SUM(Contribucion_duracion) for Filter_intrs in ([BB], [BCP], [BCU], [BE], [BTP], [BTU], [BU], [DPC], [DPL], [Otro_RF], [Otro_RV], [PDBC])) as PVT
ORDER BY Fecha, tipo_fondo ASC
""").fillna(0).set_index(['Fecha','tipo_fondo']).sum(axis=1).reset_index().rename(columns={0:'Duración'}).pivot(index='Fecha',columns='tipo_fondo',values='Duración')[[1,2]]

# %%

line = Base_duracion_FFMM.rename(columns={1:'Dur T1'})['Dur T1'].pipe(px.line,color_discrete_sequence=["#060F8B"],template='plotly_white',title='Duración por tipo de fondo (Años)')
figdur = make_subplots(specs= [[{"secondary_y": True }]])

for tr in line.data:
    figdur.add_trace(tr, secondary_y=True)

Fig_duracion_ffmm = figdur.add_trace(go.Scatter(
        x=Base_duracion_FFMM.index,
        y=Base_duracion_FFMM[2],
        name='Dur T2',
        mode="lines",
        line=dict(color="#FF5100",width=2)),secondary_y=False).update_layout(template='plotly_white',title='Duración por tipo de fondo (Años)',hovermode="x unified",barmode="overlay",bargap=0.25
        ).update_yaxes(title_text="Dur T1",secondary_y=True,range=[0,0.3]).update_yaxes(title_text="Dur T2",secondary_y=False,range=[0.6,1])

Fig_duracion_ffmm


# %%
posicion_NR = gd.get_data("""

WITH base AS (
    SELECT
        Fecha,
        CASE
            WHEN Plazo IN (N'1 a 7', N'8 a 30', N'31 a 90') THEN N'1 a 90 días'
            WHEN Plazo IN (N'91 a 180', N'181 a 360')        THEN N'91 a 360 días'
            WHEN Plazo IN (N'361 a 720', N'2 y 5 años', N'5 y 10 años', N'> 10 años')
                                                           THEN N'Mayor a 360 días'
            ELSE N'Descartar/No clasificado'
        END AS Plazo_agrupado,
        TRY_CONVERT(decimal(18,3),
            REPLACE(CAST(Pos_neta_MM_USD AS nvarchar(50)), ',', '.')
        ) AS Pos_neta_MM_USD
    FROM derivados.[Posicion_diaria_(Estadisticas)]
    WHERE Sector_contraparte = 'EXTERNO'
)
SELECT
    Fecha,
    SUM(CASE WHEN Plazo_agrupado = N'1 a 90 días'       THEN Pos_neta_MM_USD * -1 ELSE 0 END) AS [1 a 90 días],
    SUM(CASE WHEN Plazo_agrupado = N'91 a 360 días'     THEN Pos_neta_MM_USD * -1 ELSE 0 END) AS [91 a 360 días],
    SUM(CASE WHEN Plazo_agrupado = N'Mayor a 360 días'  THEN Pos_neta_MM_USD * -1 ELSE 0 END) AS [Mayor a 360 días]
FROM base
WHERE Plazo_agrupado <> N'Descartar/No clasificado' and Fecha >= '2010-01-01'
GROUP BY Fecha
ORDER BY Fecha DESC;
            
           
 """).assign(Fecha = lambda df: pd.to_datetime(df.Fecha), Neto = lambda df: df['1 a 90 días']+df['91 a 360 días']+df['Mayor a 360 días']).set_index('Fecha') 
 
 


# %%
Fig_posicion_NR_derivados = posicion_NR.drop(columns= {'Neto'}).pipe(px.area,template='plotly_white',color_discrete_sequence= ["#A2A4A7", '#BF9C69', '#001730'],title='Posicion cambiaria histórica NR (US$ Mill.)').add_scatter(x=posicion_NR.index,
        y=posicion_NR['Neto'],
        name='Neto',
        mode="lines",
        line=dict(color="#C00000",width=3)
        )

Fig_posicion_NR_derivados

# %%
PosicionNR_SWAP = gd.get_data("select * from [Inputs].[Tasas].[SPC_posicion_diaria_(Estadisticas)] where Sector_contraparte = 'Externo'").assign(Fecha=lambda df: pd.to_datetime(df.Fecha))

# %%
PosicionNR_SWAP["Plazos D"] = np.select(
    [
        PosicionNR_SWAP['Plazo'] == '3M',
        PosicionNR_SWAP['Plazo'] == '6M',
        PosicionNR_SWAP['Plazo'] == '9M',
        PosicionNR_SWAP['Plazo'] == '12M',
        PosicionNR_SWAP['Plazo'] == '18M',
        PosicionNR_SWAP['Plazo'] == '2Y',
        PosicionNR_SWAP['Plazo'] == '5Y',
        PosicionNR_SWAP['Plazo'] == '> 10Y'
    ],[

        "Menor =90d",
        "Entre 90d y 360d",
        "Entre 90d y 360d",
        "Mayor 360d y Menor 2y",
        "Mayor 360d y Menor 2y",
        "Mayor 360d y Menor 2y",
        "Mayor 2y",
        "Mayor 2y"
    ], default= "Mayor 2y")

orden = ["Menor =90d","Entre 90d y 360d", "Mayor 360d y Menor 2y", "Mayor 2y"]
PosicionNR_SWAP['Plazos D'] = pd.Categorical(PosicionNR_SWAP["Plazos D"], categories=orden,ordered=True)

Fig_PosicionNR = PosicionNR_SWAP.groupby(['Fecha','Plazos D']).Pos_neta_MMM_CLP.sum().reset_index().assign(Posicion_neta_MM_USD = lambda df: df.Pos_neta_MMM_CLP*-1/(950)*1000).pivot(index='Fecha',columns='Plazos D',values='Posicion_neta_MM_USD').pipe(px.bar,template='plotly_white',color_discrete_sequence= ["#C00000", '#BF9C69',"#A2A4A7", '#001730'],title='Posicion NR en SPC nominal (US$ Mill.)')
Fig_PosicionNR

# %%
Flujo_spot_NR = gd.get_data("""

SELECT
    CAST(Fecha AS date) AS Fecha,
    'NR' AS Grupo_Sector,
    SUM(Monto_MM_USD) AS Monto_MM_USD,
    Afecto_a_derivados
FROM spot.[Agregada_por_sector_(SAS)]
WHERE Fecha < '2026-01-01'
  AND Sector_contraparte IN ('EUROCLEAR','NO RESIDENTES')
GROUP BY CAST(Fecha AS date), Afecto_a_derivados

UNION ALL

SELECT
    CAST(Fecha AS date) AS Fecha,
    'NR' AS Grupo_Sector,
    SUM(Monto) AS Monto_MM_USD,
    Afecto_derivado AS Afecto_a_derivados
FROM [Inputs].[Spot].[Temp_SICAM_Estadisticas]
WHERE Fecha >= '2026-01-01'
  AND Sector_contraparte IN ('EUROCLEAR','NO RESIDENTES')
GROUP BY CAST(Fecha AS date), Afecto_derivado
ORDER BY Fecha DESC;



            
            """).sort_values('Fecha').groupby(['Fecha','Afecto_a_derivados']).Monto_MM_USD.sum().reset_index().assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).query('Fecha>="2025-01-02"').pivot(index='Fecha',columns='Afecto_a_derivados',values='Monto_MM_USD').rename(columns={1:'Afecto',0:'No Afecto'}).assign(Neto = lambda df: df.Afecto+df['No Afecto'])


fig_flujo_spot_nr = Flujo_spot_NR[['No Afecto','Afecto']].cumsum().pipe(px.bar,template='plotly_white',color_discrete_sequence=brand_palette,title= 'NR: Flujos Spot (US$ Mill ; Acu. desde ene.25)').add_scatter(x=Flujo_spot_NR.index,
        y=Flujo_spot_NR['Neto'].cumsum(),
        name='Neto',
        mode="lines",
        line=dict(color="#C00000",width=3)
        )

# %%
Posicion_RFL_NR = gd.get_data("""
select fecha, instrumento_largo, plazo, nominales_pesos_ultimo from [view_stock_dcv]
where fecha >='2024-12-30' and institucion in ('Mandantes', 'Depositos de Valores') and instrumento_largo ='Bono TGR en $'
order by fecha asc  
            """).assign(plazo_y = lambda df: df.plazo/365)


Posicion_RFL_NR['Bucket'] = np.select(
    [
        Posicion_RFL_NR ['plazo_y'] <= 2,
        (Posicion_RFL_NR ['plazo_y'] > 2 ) & (Posicion_RFL_NR['plazo_y'] < 6),
        (Posicion_RFL_NR ['plazo_y'] >= 6)  & (Posicion_RFL_NR ['plazo_y'] < 10),
        Posicion_RFL_NR ['plazo_y'] >= 10,],
        ["Menor 2Y","Entre 3-5Y", "Entre 6-9Y", "Mayor 10Y"],default="Mayor 10Y")

orden = ["Menor 2Y","Entre 3-5Y", "Entre 6-9Y", "Mayor 10Y"]
Posicion_RFL_NR['Bucket'] = pd.Categorical(Posicion_RFL_NR["Bucket"], categories=orden,ordered=True)
Fig_posicion_RFL_NR= Posicion_RFL_NR[['fecha','Bucket','nominales_pesos_ultimo']].groupby(['fecha','Bucket']).nominales_pesos_ultimo.sum().reset_index().assign(mmusd=lambda df: df.nominales_pesos_ultimo/(1000000*950)).pivot(index='fecha',columns='Bucket',values='mmusd').pipe(px.bar, color_discrete_sequence=brand_palette,template='plotly_white',title='Posicion RFL No Residente')


# %%
Fig_posicion_RFL_NR ###Preguntar como quiere este. 


# %%
Variacion_RFL_DCV = gd.get_data("""
            
            
select fecha, instrumento_largo, sum(nominales_pesos_ultimo) as 'monto nominal'
from [view_stock_dcv]
where fecha >='2024-12-30' and institucion in ('Mandantes', 'Depositos de Valores') and instrumento_largo ='Bono TGR en $'
group by fecha, instrumento_largo
order by fecha asc

            
            
            """).assign(fecha=lambda df: pd.to_datetime(df.fecha), mmusd=lambda df: df['monto nominal']/(1000000*950)).sort_values('fecha').query('fecha >= "2025-01-02"')

# %%
min = Variacion_RFL_DCV.query('fecha == fecha.min()').mmusd

Fig_variacion_RFLDCV = Variacion_RFL_DCV.assign(Variacion = lambda df: df.mmusd-int(min)).set_index('fecha')['Variacion'].pipe(px.line, title='NR: Var.Stock BTP en DCV (US$ Mill. ; YtD)',template='plotly_white')
Fig_variacion_RFLDCV

# %%


# %%
#Stock bonos bancarios

# %%
cruce_p40 = gd.get_data("""SELECT Fecha,
       Institucion = registroAlternativo,
	   TRIM(Nemotecnicos) as Nemotecnico,
	   nominalActual
FROM mesadineOLTP_.p40.instrumentosDeuda p40
 
left join  mesadineOLTP_.Soma.participantesCodigoSBIF cod on cod.codigoSBIF = p40.Institucion
left join  mesadineOLTP_.Soma.Participantes part on part.ParticipantesID = cod.ParticipantesID
 
WHERE condicionInstrumento = 2 and Fecha>=DATEADD(day,-1500,getdate())
GROUP BY Fecha, registroAlternativo, Emisor, Nemotecnicos, nominalActual
ORDER BY Fecha DESC
""")

map_bonos_BANCARIOS = gd.get_data("select * from MesadineOLTP_.DCV.Valorizacion WHERE TipoSVS in('BB')")


# %%
mapping_bonos_bancario = map_bonos_BANCARIOS.sort_values('Fecha').Nemotecnico.unique()
nemo_banco = cruce_p40.query('Nemotecnico in @mapping_bonos_bancario').rename(columns = {'Nemotecnico':'nemotecnico','Fecha':'fecha'}) 
Stock_bonos_bancarios = gd.get_data("""select fecha, nemotecnico,instrumento,moneda from [view_stock_dcv] where fecha>=DATEADD(day,-1500,getdate()) and institucion = 'Banco' order by fecha desc""").merge(nemo_banco,how='inner',on=['fecha','nemotecnico'])

# %%
ultima_fecha = Stock_bonos_bancarios.fecha.max()
Dolar_uf = gd.get_data("select * from mesadineOLTP_.mercado.divisas where Paridad = 'USD' or Paridad = 'UF'")[['Fecha','Paridad','Cotizacion']].assign(Fecha=lambda df:pd.to_datetime(df.Fecha)).sort_values('Fecha').query("Fecha == @ultima_fecha")
USD = float(Dolar_uf.query('Paridad == "USD"').Cotizacion)



# %%
ultima_semana = ultima_fecha-timedelta(weeks=1)

# %%
fig_stock_bonos_bancarios = Stock_bonos_bancarios.merge(Dolar_uf[['Paridad','Cotizacion']].rename(columns= {'Paridad':'moneda'}),how='left',on='moneda').fillna(1).assign(nominalMMUSD = lambda df: (df.nominalActual*df.Cotizacion)/1000000/USD).groupby(['fecha','Institucion']).nominalMMUSD.sum().reset_index().pivot(index='fecha',columns='Institucion',values='nominalMMUSD').fillna(0).cumsum().pipe(px.area,template='plotly_white',title='Stock Bonos Bancarios por Banco') # .query('fecha == @ultima_fecha or fecha == @ultima_semana')

# %%
#Emisiones DAP
DAP = gd.get_data("""select * from [view_stock_dcv] where fecha>=DATEADD(day,-1500,getdate()) and (instrumento_largo= 'Deposito a Plazo Fijo') and institucion = 'Banco' order by fecha desc""")


nemo_banco = cruce_p40.rename(columns = {'Nemotecnico':'nemotecnico','Fecha':'fecha'})
DAP_EMISIONES = DAP.merge(nemo_banco,how='inner',on=['fecha','nemotecnico'])

Fig_dap_emisiones = DAP_EMISIONES.merge(Dolar_uf[['Paridad','Cotizacion']].rename(columns= {'Paridad':'moneda'}),how='left',on='moneda').fillna(1).assign(nominalMMUSD = lambda df: (df.nominalActual*df.Cotizacion)/1000000/USD).groupby(['fecha','Institucion']).nominalMMUSD.sum().reset_index().pivot(index='fecha',columns='Institucion',values='nominalMMUSD').fillna(0).pipe(px.bar,template='plotly_white',title='DAP por Banco (MM USD)')

Fig_dap_emisiones

# %%
#Vencimientos en el corto plazo

# %%
#Emisiones y vtos Afuera 

# %%
#CSV
#Posicion RFL
posicion_cvs= gd.get_data("""select fecha,nemotecnico,instrumento_largo,plazo,nominales_pesos_ultimo from [view_stock_dcv] where institucion='Compania de Seguro' and fecha>='2019-01-01'  order by fecha desc""").assign(plazo_y = lambda df: df.plazo/365)


posicion_cvs['Bucket'] = np.select(
    [
        posicion_cvs['plazo_y'] <= 2,
        (posicion_cvs['plazo_y'] > 2 ) & (posicion_cvs['plazo_y'] < 6),
        (posicion_cvs['plazo_y'] >= 6)  & (posicion_cvs['plazo_y'] < 10),
        posicion_cvs['plazo_y'] >= 10,],
        ["Menor 2Y","Entre 3-5Y", "Entre 6-9Y", "Mayor 10Y"],default="Mayor 10Y")

orden = ["Menor 2Y","Entre 3-5Y", "Entre 6-9Y", "Mayor 10Y"]
posicion_cvs['Bucket'] = pd.Categorical(posicion_cvs["Bucket"], categories=orden,ordered=True)
Fig_posicion_cvs= posicion_cvs[['fecha','Bucket','nominales_pesos_ultimo']].groupby(['fecha','Bucket']).nominales_pesos_ultimo.sum().reset_index().assign(mmusd=lambda df: df.nominales_pesos_ultimo/(1000000*950)).pivot(index='fecha',columns='Bucket',values='mmusd').pipe(px.area, color_discrete_sequence=brand_palette,template='plotly_white',title='Stock DCV')
Fig_posicion_cvs

# %%
Fig_posicion_cvs1= posicion_cvs[['fecha','Bucket','nominales_pesos_ultimo','instrumento_largo']].groupby(['fecha','instrumento_largo']).nominales_pesos_ultimo.sum().reset_index().assign(mmusd=lambda df: df.nominales_pesos_ultimo/(1000000*950)).pivot(index='fecha',columns='instrumento_largo',values='mmusd').pipe(px.bar, color_discrete_sequence=brand_palette,template='plotly_white',title='Stock DCV')
Fig_posicion_cvs1


# %%
#Flujos DCV solo restar con stocks, pero ns cual stock quiere cata. (PREGUNTAR)

# %%
#Flujos cambiarios

Flujo_cambiario_csv  = gd.get_data("""

WITH base AS (
    SELECT
        Fecha,
        CASE
            WHEN Plazo IN (N'1 a 7', N'8 a 30', N'31 a 90') THEN N'1 a 90 días'
            WHEN Plazo IN (N'91 a 180', N'181 a 360')        THEN N'91 a 360 días'
            WHEN Plazo IN (N'361 a 720', N'2 y 5 años', N'5 y 10 años', N'> 10 años')
                                                           THEN N'Mayor a 360 días'
            ELSE N'Descartar/No clasificado'
        END AS Plazo_agrupado,
        TRY_CONVERT(decimal(18,3),
            REPLACE(CAST(Pos_neta_MM_USD AS nvarchar(50)), ',', '.')
        ) AS Pos_neta_MM_USD
    FROM derivados.[Posicion_diaria_(Estadisticas)]
    WHERE Sector_contraparte = 'SEGUROS'
)
SELECT
    Fecha,
    SUM(CASE WHEN Plazo_agrupado = N'1 a 90 días'       THEN Pos_neta_MM_USD * -1 ELSE 0 END) AS [1 a 90 días],
    SUM(CASE WHEN Plazo_agrupado = N'91 a 360 días'     THEN Pos_neta_MM_USD * -1 ELSE 0 END) AS [91 a 360 días],
    SUM(CASE WHEN Plazo_agrupado = N'Mayor a 360 días'  THEN Pos_neta_MM_USD * -1 ELSE 0 END) AS [Mayor a 360 días]
FROM base
WHERE Plazo_agrupado <> N'Descartar/No clasificado' and Fecha >= '2010-01-01'
GROUP BY Fecha
ORDER BY Fecha DESC;
            
           
 """).assign(Fecha = lambda df: pd.to_datetime(df.Fecha), Neto = lambda df: df['1 a 90 días']+df['91 a 360 días']+df['Mayor a 360 días']).set_index('Fecha') 
 

# %%
Fig_posicion_csv_cambiario = Flujo_cambiario_csv.drop(columns= {'Neto'}).pipe(px.area,template='plotly_white',color_discrete_sequence= ["#A2A4A7", '#BF9C69', '#001730'],title='Posicion cambiaria histórica CSV (US$ Mill.)').add_scatter(x=Flujo_cambiario_csv.index,
        y=Flujo_cambiario_csv['Neto'],
        name='Neto',
        mode="lines",
        line=dict(color="#C00000",width=3)
        )

Fig_posicion_csv_cambiario

# %%
Flujo_spot_CSV = gd.get_data("""

SELECT
    CAST(Fecha AS date) AS Fecha,
    'NR' AS Grupo_Sector,
    SUM(Monto_MM_USD) AS Monto_MM_USD,
    Afecto_a_derivados
FROM spot.[Agregada_por_sector_(SAS)]
WHERE Fecha < '2026-01-01'
  AND Sector_contraparte IN ('CIA SEGUROS')
GROUP BY CAST(Fecha AS date), Afecto_a_derivados

UNION ALL

SELECT
    CAST(Fecha AS date) AS Fecha,
    'NR' AS Grupo_Sector,
    SUM(Monto) AS Monto_MM_USD,
    Afecto_derivado AS Afecto_a_derivados
FROM [Inputs].[Spot].[Temp_SICAM_Estadisticas]
WHERE Fecha >= '2026-01-01'
  AND Sector_contraparte IN ('CIA SEGUROS')
GROUP BY CAST(Fecha AS date), Afecto_derivado
ORDER BY Fecha DESC;



            
            """).sort_values('Fecha').groupby(['Fecha','Afecto_a_derivados']).Monto_MM_USD.sum().reset_index().assign(Fecha=lambda df: pd.to_datetime(df.Fecha)).query('Fecha>="2025-01-02"').pivot(index='Fecha',columns='Afecto_a_derivados',values='Monto_MM_USD').rename(columns={1:'Afecto',0:'No Afecto'}).assign(Neto = lambda df: df.Afecto+df['No Afecto'])


fig_flujo_spot_csv = Flujo_spot_CSV[['No Afecto','Afecto']].cumsum().pipe(px.bar,template='plotly_white',color_discrete_sequence=brand_palette,title= 'CSV: Flujos Spot (US$ Mill ; Acu. desde ene.25)').add_scatter(x=Flujo_spot_NR.index,
        y=Flujo_spot_NR['Neto'].cumsum(),
        name='Neto',
        mode="lines",
        line=dict(color="#C00000",width=3)
        )

# %%
#Duracion de cartera RFL

Cruce_duracion_rfl = gd.get_data("""
select fecha, nemotecnico,instrumento_largo, plazo, nominales_pesos_ultimo from [view_stock_dcv]
where fecha >='2019-01-01' and institucion in ('Compania de Seguro') and instrumento_largo ='Bono TGR en $'
order by fecha asc  
            """).assign(w_duration = lambda df: df.nominales_pesos_ultimo/df.groupby('fecha').nominales_pesos_ultimo.transform(sum))
Base_duraciones_csv = gd.get_data("""select Fecha, Nemotecnico, Tipo, duracionModificadaACT365 from mesadineOLTP_.DCV.Valorizacion where Fecha>='2019-01-01' and Tipo = 'BTP'  order by fecha desc""").query('Fecha != "2023-05-26" or Fecha != "2023-07-05" or Fecha != "2023-07-06"')
fig_duracion_rfl_csv = Base_duraciones_csv.rename(columns={'Nemotecnico':'nemotecnico','Fecha':'fecha'}).merge(Cruce_duracion_rfl,how='inner',on=['nemotecnico','fecha']).assign(Duracion_Cartera = lambda df: df.duracionModificadaACT365*df.w_duration).groupby('fecha').Duracion_Cartera.sum().reset_index().assign(fecha = lambda df: pd.to_datetime(df.fecha)).query("fecha>='2024-01-01'").set_index('fecha').pipe(px.line,template='plotly_white',title='Duración carter RFL CSV')

# %%

# ============================================================
# DASHBOARD HTML GENERATION
# ============================================================

from pathlib import Path
import dashboard_builder as db

# --- Logos ---
gemma_dir = Path(__file__).resolve().parent / "gemma"
logos = db.build_logos(gemma_dir)

# --- Definicion de secciones ---
# Cada seccion: (titulo_seccion, [(titulo_card, fig_o_None), ...])
# fig_o_None = None genera card PENDIENTE

# ===================== ESTRUCTURA POR MERCADO =====================

S1_Mercado_Monetario = [
    ("Spread DAP-Swap 1M (bps)", fig_spread_dapswap1m),
    ("Spread DAP-Swap 3M (bps)", fig_spread_dapswap3m),
    ("Spread DAP-Swap 6M (bps)", fig_spread_dapswap6m),
    ("Spread DAP-Swap 12M (bps)", fig_spread_dapswap12m),
    ("Tasas PDBC Mercado Secundario", fig_pdbc_7D),
    ("TIB vs Monto Transado", fig_TIB_Montotransado),
    ("Spread DAP/Prime-Swap UF 12M (bps)", fig_spread_dapprimeufswap),
    ("Spread DAP UF 6M", None),
]

S2_Liquidez_MN = [
    ("Operaciones de Liquidez MN", fig_Operaciones_de_liquidez),
    ("Liquidez MN vs Concentracion", None),
    ("Ratio Liquidez/Obligaciones 30d MN", None),
    ("LCR MN", None),
]

S3_Liquidez_MX = [
    ("Spread Liquidez MX 1M", Fig_Liquidez_mx_spreads_1M),
    ("Spread Liquidez MX 3M", Fig_Liquidez_mx_spreads_3M),
    ("Spread Liquidez MX 6M", Fig_Liquidez_mx_spreads_6M),
    ("Spread Liquidez MX 12M", Fig_Liquidez_mx_spreads_12M),
    ("Liquidez MX vs Concentracion IHH", fig_LiquidezMX_IHH),
    ("Ratio Liquidez/Obligaciones 30d", fig_RatioLiquidezObligaciones),
    ("NSFR MX", Total_MX_NSFR),
    ("LCR MX", Total_MX_LCR),
]

S4_Mercado_RF_Tasas = [
    ("Curva BTP y BTU 5Y", fig_btptu5),
    ("Curva BTP %", fig_btp_curva),
    ("Spread BTP-UST 5Y y 10Y (bp)", fig_btp_ust_510y),
    ("Pendiente BTP/BTU 10-5 y 10-2", fig_pendiente_btp_btu),
    ("Curva BTP y BTU 10Y", fig_curva_btptu10),
    ("Curva BTU %", fig_btu_curva),
    ("Spread BTP-SPC 5Y y 10Y (bp)", fig_spread_btp_spc_510y),
    ("Tabla Emisiones TGR", None),
]

S5_Mercado_RF_Volumenes = [
    ("Volatilidad diaria BTP/BTU (%)", fig_vol_btpbtu),
    ("Volatilidad Tasas BB", Fig_Volatilidad_Tasas_BB),
    ("Monto transado BTP/BTU (Mill USD)", Fig_Monto_btpbtu),
    ("Monto transado BB (Mill USD)", Fig_Monto_BB),
    ("Monto transado BC (Mill USD)", Fig_Monto_BC),
    ("Spread Bancarios vs SPC", None),
    ("Spread Bancarios vs Bonos", None),
    ("Spread Corporativos vs SPC", None),
]

S6_Mercado_FX = [
    ("CLP vs Monto Transado", Fig_CLP_Monto),
    ("COBRE vs DXY", fig_cobre_dxy),
    ("Variacion 1w Moneda", Fig_varmoneda5),
    ("Variacion YTD Moneda", Fig_varmoneda1y),
    ("Promedio Amplitud de Puntas (%)", Fig_conformacion_precios),
    ("Volatilidad Precio Transacciones", Fig_conformacion_precios2),
    ("Fwd Point (pesos)", Fig_fwdp),
    ("RSI CLP", None),
]

S7_Mercado_FX_Diferencial = [
    ("Spread Swap CLP - OIS", Fig_spread_swap_ios),
    ("Spread TPM IMP - FED IMP", Fig_spread_tpm_fed_imp),
    ("Flujo Cambiario 14 dias (US$ Mill.)", Fig_Flujo_cambiario2w),
    ("Flujo Cambiario 7 dias (US$ Mill.)", Fig_Flujo_cambiario1w),
    ("Fixing de la banca (sector)", Fig_fixingbanca_sector),
    ("Fixing de la banca (por fecha)", Fig_fixingporfecha),
    ("Transado Datatec", None),
    ("Tasa Implicita Forward", None),
]

# =================== ESTRUCTURA POR PORTAFOLIOS ===================

S8_AFP_Allocation = [
    ("Allocation internacional vs local", Fig_AllocationInt_Nac),
    ("Attribution Fondos por clase de activos", Fig_attribution),
    ("DV01 SPC por moneda proyectado", fig_dv01_spc),
    ("Traspaso de fondos de FP", Fig_Movimientos_fondos),
]

S9_AFP_Derivados = [
    ("Posicion spot y derivados AFP", fig_spot_derivados_afp),
    ("MtM Swap AFP", MtM),
    ("AFP Derivados", None),
    ("AFP Derivados", None),
]

S10_AFP_RFL = [
    ("Posicion RFL AFP", Fig_posicion_rfl_max),
    ("Variacion Semanal DCV (US$ Mill.)", Fig_variacion_dcv_afp),
    ("Flujos cambiarios Variacion 1w (US$ Mill)", fig_cambiario_afp),
    ("Stock Fondos (US$ Mill.)", Fig_stock_fondo),
]

S11_FFMM = [
    ("Flujos semanales por fondo", Fig_flujos_FFMM),
    ("Allocation T6 (Mill US$.)", Fig_allocation_t6),
    ("Composicion portafolio DCV", fig_dcv_composicion_ffmm),
    ("Stock DAP y PDBC (Acum US$ Mill.)", FIG_DAP_PDBC_FFMM),
]

S12_FFMM_Cont = [
    ("Flujo spot acumulados (US$ Mill.)", Fig_Flujos_spot_FFMM),
    ("DV01 por fondo", fig_dv01_ffmm),
    ("Duracion por tipo de fondo (Anos)", Fig_duracion_ffmm),
    ("FFMM (pendiente)", None),
]

S13_No_Residentes = [
    ("Posicion cambiaria historica NR", Fig_posicion_NR_derivados),
    ("Posicion NR en SPC nominal (US$ Mill.)", Fig_PosicionNR),
    ("NR: Flujos Spot (US$ Mill)", fig_flujo_spot_nr),
    ("Posicion RFL No Residente", Fig_posicion_RFL_NR),
]

S14_No_Residentes_Cont = [
    ("NR: Var.Stock BTP en DCV (US$ Mill.)", Fig_variacion_RFLDCV),
    ("NR (pendiente 1)", None),
    ("NR (pendiente 2)", None),
    ("NR (pendiente 3)", None),
]

S15_Bancos = [
    ("Stock Bonos Bancarios por Banco", fig_stock_bonos_bancarios),
    ("DAP por Banco (MM USD)", Fig_dap_emisiones),
    ("Vencimientos corto plazo", None),
    ("Emisiones y vtos afuera", None),
]

S16_CSV = [
    ("Stock DCV (por plazo)", Fig_posicion_cvs),
    ("Stock DCV (por instrumento)", Fig_posicion_cvs1),
    ("Posicion cambiaria historica CSV", Fig_posicion_csv_cambiario),
    ("CSV: Flujos Spot (US$ Mill)", fig_flujo_spot_csv),
    ("Duracion cartera RFL CSV", fig_duracion_rfl_csv),
    ("Flujos DCV", None),
]

# --- Construir secciones HTML ---
all_sections = []

# Divider: Estructura por Mercado
all_sections.append(db.section_divider("Estructura por Mercado", divider_id="mercado"))

sections_mercado = [
    ("Mercado Monetario", S1_Mercado_Monetario, "s1"),
    ("Liquidez MN", S2_Liquidez_MN, "s2"),
    ("Liquidez MX", S3_Liquidez_MX, "s3"),
    ("Mercado RF - Tasas", S4_Mercado_RF_Tasas, "s4"),
    ("Mercado RF - Volumenes y Volatilidad", S5_Mercado_RF_Volumenes, "s5"),
    ("Mercado FX", S6_Mercado_FX, "s6"),
    ("Mercado FX - Diferencial de Tasas y Fixing", S7_Mercado_FX_Diferencial, "s7"),
]

for title, card_defs, sid in sections_mercado:
    all_sections.append(db.build_section(title, card_defs, section_id=sid))

# Divider: Estructura por Portafolios
all_sections.append(db.section_divider("Estructura por Portafolios", divider_id="portafolios"))

sections_portafolios = [
    ("AFP - Allocation y Attribution", S8_AFP_Allocation, "s8"),
    ("AFP - Derivados", S9_AFP_Derivados, "s9"),
    ("AFP - RFL y Flujos Cambiarios", S10_AFP_RFL, "s10"),
    ("FFMM", S11_FFMM, "s11"),
    ("FFMM - Continuacion", S12_FFMM_Cont, "s12"),
    ("No Residentes", S13_No_Residentes, "s13"),
    ("No Residentes - Continuacion", S14_No_Residentes_Cont, "s14"),
    ("Bancos", S15_Bancos, "s15"),
    ("CSV (Companias de Seguros)", S16_CSV, "s16"),
]

for title, card_defs, sid in sections_portafolios:
    all_sections.append(db.build_section(title, card_defs, section_id=sid))

# --- Navegacion ---
nav_items = [
    ("Mercado", "mercado"),
    ("Monetario", "s1"),
    ("Liq. MN", "s2"),
    ("Liq. MX", "s3"),
    ("RF Tasas", "s4"),
    ("RF Vol.", "s5"),
    ("FX", "s6"),
    ("FX Dif.", "s7"),
    ("Portafolios", "portafolios"),
    ("AFP Alloc.", "s8"),
    ("AFP Deriv.", "s9"),
    ("AFP RFL", "s10"),
    ("FFMM", "s11"),
    ("FFMM Cont.", "s12"),
    ("NR", "s13"),
    ("NR Cont.", "s14"),
    ("Bancos", "s15"),
    ("CSV", "s16"),
]

# --- Generar HTML ---
full_html = db.build_dashboard_html(
    sections=all_sections,
    logos=logos,
    title="Monitor DACE",
    current_date=datetime.now().strftime("%Y-%m-%d"),
    nav_items=nav_items,
)

# --- Guardar ---
OUTPUT_DIR = Path(r"D:/GOM/DACE/Leandro/HTML Dashboard")
output_file = OUTPUT_DIR / f"monitor_dashboard_{datetime.now().strftime('%Y%m%d')}.html"
db.save_and_open(full_html, output_file)

