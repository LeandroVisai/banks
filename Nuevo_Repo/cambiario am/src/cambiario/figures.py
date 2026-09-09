from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as cfg
from . import data



def _add_pct_labels(fig: go.Figure, x_end, levels: list[tuple[float, str, str]]) -> None:
    """Tag de texto (P10, P25, …) junto al extremo de cada línea de percentil.

    x_end debe venir como string ISO, no pd.Timestamp: a diferencia de un
    array de datos de una traza (que Plotly normaliza a datetime64 al
    asignarlo), `add_annotation(x=...)` guarda el valor tal cual — un
    Timestamp crudo ahí sobrevive hasta kaleido, que serializa con orjson
    directo y no sabe qué hacer con él ("Type is not JSON serializable:
    Timestamp"). Mismo problema que xaxis_range en email._clip_for_email."""
    for level, color, label in levels:
        fig.add_annotation(
            x=x_end, y=level, text=f'<b>{label}</b>',
            showarrow=False, xref='x', yref='y',
            xanchor='left', yanchor='middle', xshift=6,
            font=dict(size=9.5, color=color, family='Barlow, Arial, sans-serif'),
            bgcolor='rgba(255,255,255,0.80)', borderpad=1,
        )


_PCT_WIDTH = {'P50': 1.8, 'P100': 1.2}


def _pct_lines(fig: go.Figure, summary: dict, value_fmt: str = ',.0f') -> None:
    """Agrega las 6 líneas P10..P100 + labels a una figura S/R, desde un percentile_summary()."""
    # .isoformat(), no Timestamp crudo — ver nota en _add_pct_labels.
    min_dt, max_dt = summary['min_date'].isoformat(), summary['max_date'].isoformat()
    levels = [
        (summary['p10'],  cfg.PCT_P10,  'P10',  'dash'),
        (summary['p25'],  cfg.PCT_P25,  'P25',  'dot'),
        (summary['p50'],  cfg.PCT_P50,  'P50',  'dash'),
        (summary['p75'],  cfg.PCT_P75,  'P75',  'dot'),
        (summary['p90'],  cfg.PCT_P90,  'P90',  'dash'),
        (summary['p100'], cfg.PCT_P100, 'P100', 'longdash'),
    ]
    fig.add_traces([
        go.Scatter(x=[min_dt, max_dt], y=[lvl, lvl],
                   name=f'{lbl}{" " if lbl == "P100" else "  "}· {format(lvl, value_fmt)}',
                   mode='lines', line=dict(color=color, width=_PCT_WIDTH.get(lbl, 1.3), dash=dash),
                   showlegend=False)
        for lvl, color, lbl, dash in levels
    ])
    _add_pct_labels(fig, max_dt, [(lvl, color, lbl) for lvl, color, lbl, _ in levels])


# ── CLP Candlestick (últimos 2 meses) ────────────────────────────────
def build_clp_candlestick(df: pd.DataFrame) -> go.Figure:
    candle = data.candlestick_window(df, months=2)
    fig = go.Figure(data=[go.Candlestick(
        x=candle['Fecha'],
        open=candle['CLP Apertura'], high=candle['CLP Max'],
        low=candle['CLP minimo'],   close=candle['CLP Cierre'],
        increasing=dict(line=dict(color='#27ae60'), fillcolor='#27ae60'),
        decreasing=dict(line=dict(color='#e74c3c'), fillcolor='#e74c3c'),
        name='USD/CLP', whiskerwidth=0.4,
    )])
    fig.update_layout(
        **cfg.LAYOUT_BASE,
        xaxis=dict(**cfg.GRID, rangeslider=dict(visible=False)),
        yaxis=dict(**cfg.GRID, title='CLP', fixedrange=False),
    )
    return fig


# ── Cobre vs CLP ─────────────────────────────────────────────────────
def build_clp_cobre(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df['Fecha'], y=(df['Drivers Precio Cobre HGA'] / 100)*-1,
        name='Cobre (neg)', line=dict(color='#b5551b', width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['Fecha'], y=df['CLP Cierre'],
        name='CLP Cierre', line=dict(color=cfg.NAVY, width=1.5),
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='Cobre · USD/lb'),
        yaxis2=dict(**cfg.GRID, title='CLP', overlaying='y', side='right'),
    )
    return fig


# ── DXY vs CLP ───────────────────────────────────────────────────────
def build_clp_dxy(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df['Fecha'], y=df['Drivers DXY '],
        name='DXY', line=dict(color='#e67e22', width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['Fecha'], y=df['CLP Cierre'],
        name='CLP Cierre', line=dict(color=cfg.NAVY, width=1.5),
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='DXY'),
        yaxis2=dict(**cfg.GRID, title='CLP', overlaying='y', side='right'),
    )
    return fig


# ── Variación monedas del día (SQL: monedas) ──────────────────────────
def build_variacion_dia(monedas: pd.DataFrame) -> go.Figure:
    monedas2 = pd.melt(monedas, id_vars='Fecha', var_name='Pais', value_name='Valor Moneda').sort_values(['Fecha', 'Pais'])
    monedas2['Variacion pct (%)'] = (monedas2.groupby('Pais')['Valor Moneda'].pct_change() * 100).round(2)
    last = monedas2.sort_values(['Pais', 'Fecha']).groupby('Pais').tail(1)
    last['color'] = last['Variacion pct (%)'].apply(lambda x: 'Depreciación' if x > 0 else 'Apreciación')
    last = last.sort_values('Variacion pct (%)', ascending=False)
    fig = px.bar(
    data_frame=last, x='Pais', y='Variacion pct (%)', color='color', color_discrete_map={
        'Depreciación':"#a80b0b",
        'Apreciación': '#2ca02c'
    },
    template='plotly_white',
    )
    fig.add_annotation(
        x = .2, y = 1, xref = 'paper', yref= 'y', text = 'Depreciación', showarrow= False, font = dict(color = 'red')
    )
    fig.add_annotation(
        x = .8, y = -1, xref = 'paper', yref= 'y', text = 'Apreciación', showarrow= False, font = dict(color = 'green')
    )
    fig.update_layout(showlegend = False)
    return fig


# ── Intradía (Excel: hoja Intradía) ───────────────────────────────────
def build_clp_intradia(intradia: pd.DataFrame) -> go.Figure:
    min = intradia['Fecha - Hora'].min()
    max = intradia['Fecha - Hora'].max()
    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=intradia['Fecha - Hora'], y=intradia['CLP base 100'],
                   name='CLP Base 100', line=dict(color=cfg.NAVY, width=1.5)),
        go.Scatter(x=intradia['Fecha - Hora'], y=intradia['DXY Base 100'],
                   name='DXY Base 100', line=dict(color='#e67e22', width=1.5)),
        go.Scatter(x=intradia['Fecha - Hora'], y=intradia['HGA Base 100'],
                   name='Cobre Base 100', line=dict(color='#b5551b', width=1.5)),
    ])
    fig.update_layout(
        **cfg.LAYOUT_BASE,
        xaxis=dict(**cfg.GRID, tickangle=0, range = [min, max]),
        yaxis=dict(**cfg.GRID, title='Índice Base 100'),
    )
    return fig


# ── Bollinger CLP (20d) ───────────────────────────────────────────────
def build_clp_bollinger(df: pd.DataFrame, period: int = 20) -> go.Figure:
    clp_bb = df[['Fecha', 'CLP Cierre']].copy()
    clp_bb['BB_MA']    = clp_bb['CLP Cierre'].rolling(period).mean()
    clp_bb['BB_STD']   = clp_bb['CLP Cierre'].rolling(period).std()
    clp_bb['BB_Upper'] = clp_bb['BB_MA'] + 2 * clp_bb['BB_STD']
    clp_bb['BB_Lower'] = clp_bb['BB_MA'] - 2 * clp_bb['BB_STD']
    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=clp_bb['Fecha'], y=clp_bb['BB_Upper'], name='Banda superior',
                   line=dict(color=cfg.GOLD, width=1, dash='dot'), showlegend=True),
        go.Scatter(x=clp_bb['Fecha'], y=clp_bb['BB_Lower'], name='Banda inferior',
                   line=dict(color=cfg.GOLD, width=1, dash='dot'),
                   fill='tonexty', fillcolor='rgba(133,117,25,0.07)'),
        go.Scatter(x=clp_bb['Fecha'], y=clp_bb['BB_MA'], name=f'MA {period}d',
                   line=dict(color=cfg.MAROON, width=1, dash='dash')),
        go.Scatter(x=clp_bb['Fecha'], y=clp_bb['CLP Cierre'], name='CLP Cierre',
                   line=dict(color=cfg.NAVY, width=1.8)),
    ])
    fig.update_layout(**cfg.LAYOUT_BASE, xaxis=cfg.data_select, yaxis=dict(**cfg.GRID, title='CLP'))
    return fig


# ── Medias móviles CLP ───────────────────────────────────────────────
def _ma_period(col: str) -> int | None:
    for token in col.split():
        if token.isdigit():
            return int(token)
    return None


_MA_HIGHLIGHT = {50: cfg.GOLD, 200: cfg.MAROON}
_MA_MUTED     = {10: "#c3ccd6", 20: "#8fa0b3", 100: "#4d6178"}


def build_clp_medias_moviles(df: pd.DataFrame) -> go.Figure:
    ma_cols = [c for c in df.columns if isinstance(c, str) and "Media móvil" in c]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for col in ["CLP Cierre"] + ma_cols:
        if col == "CLP Cierre":
            style = dict(color=cfg.NAVY, width=1.6)
        else:
            period = _ma_period(col)
            if period in _MA_HIGHLIGHT:
                style = dict(color=_MA_HIGHLIGHT[period], width=2.2)
            else:
                style = dict(color=_MA_MUTED.get(period, "#9aa2b1"), width=0.8)
        fig.add_trace(go.Scatter(
            x=df["Fecha"], y=df[col],
            name=col,
            line=dict(**style, shape="spline", smoothing=0.8),
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["Fecha"], y=df["CLP Monto Transado"].rolling(window=5).mean(),
        name="Media Móvil 5d Monto Transado",
        fill="tozeroy",
        fillcolor="rgba(128,128,128,0.15)",
        line=dict(color="rgba(128,128,128,0.45)", width=0, shape="spline", smoothing=0.8),
        hovertemplate="<b>Media Móvil 5d Monto Transado</b><br>%{x|%Y-%m-%d}: %{y:,.0f}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        **cfg.LAYOUT_BASE,
        xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title="CLP"),
        yaxis2=dict(**cfg.GRID, title="Monto Transado", overlaying="y", side="right",
                    range=[500, df["CLP Monto Transado"].max() * 1.2]),
    )
    return fig


# ── S/R CLP ────────────────────────────────────────────────────────────
def build_clp_sr(df: pd.DataFrame) -> go.Figure:
    summary = data.percentile_summary(df, 'Fecha', 'CLP Cierre')
    clp_sr = summary['data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clp_sr['Fecha'], y=clp_sr['CLP Cierre'],
                              name='CLP Cierre', line=dict(color=cfg.NAVY, width=1.8)))
    _pct_lines(fig, summary)
    fig.update_layout(
        **cfg.SR_LAYOUT,
        xaxis=dict(**cfg.GRID, rangeslider=dict(visible=False)),
        yaxis=dict(**cfg.GRID, title='CLP', fixedrange=False)
    )
    return fig


# ── RSI CLP ──────────────────────────────────────────────────────────
def build_clp_rsi(df: pd.DataFrame) -> go.Figure:
    rsi_start = df['RSI RSI Pos. OffShore'].first_valid_index()
    rsi_start_date = df.loc[rsi_start, 'Fecha']
    rsi_df = df[df['Fecha'] >= rsi_start_date].copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=rsi_df['Fecha'], y=rsi_df['RSI RSI Pos. OffShore'],
                              name='RSI Pos. Offshore', line=dict(color=cfg.MAROON, width=1.5)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=rsi_df['Fecha'], y=rsi_df['RSI RSI CLP'],
                              name='RSI CLP', line=dict(color=cfg.NAVY, width=1.5)),
                  secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='RSI Pos. Offshore'),
        yaxis2=dict(**cfg.GRID, title='RSI CLP', overlaying='y', side='right'),
    )
    return fig


# ── Distribución CLP (SQL: distclp) ───────────────────────────────────
def build_clp_dist(distclp: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        data_frame=distclp, x='CLP_Bucket', y='Frecuencia',
        template='plotly_white', color_discrete_sequence=["#001730"],
    )
    fig.update_xaxes(title='Bucket CLP', dtick=5)
    fig.update_layout(**cfg.LAYOUT_BASE, xaxis=dict(**cfg.GRID, tickangle=-40))
    return fig


# ── S/R Cobre ────────────────────────────────────────────────────────
def build_cobre_sr(df: pd.DataFrame) -> go.Figure:
    summary = data.percentile_summary(df, 'Fecha', 'Drivers Precio Cobre HGA')
    cobre_sr = summary['data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cobre_sr['Fecha'], y=cobre_sr['Drivers Precio Cobre HGA'],
                              name='Cobre (USD/lb)', line=dict(color='#b5551b', width=1.8)))
    _pct_lines(fig, summary, value_fmt='.2f')
    fig.update_layout(
        **cfg.SR_LAYOUT,
        xaxis=dict(**cfg.GRID, rangeslider=dict(visible=False)),
        yaxis=dict(**cfg.GRID, title='Cobre · USD/lb', fixedrange=False)
    )
    return fig


# ── S/R DXY ──────────────────────────────────────────────────────────
def build_dxy_sr(df: pd.DataFrame) -> go.Figure:
    summary = data.percentile_summary(df, 'Fecha', 'Drivers DXY ')
    dxy_sr = summary['data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dxy_sr['Fecha'], y=dxy_sr['Drivers DXY '],
                              name='DXY', line=dict(color='#e67e22', width=1.8)))
    _pct_lines(fig, summary, value_fmt='.2f')
    fig.update_layout(
        **cfg.SR_LAYOUT,
        xaxis=dict(**cfg.GRID, rangeslider=dict(visible=False)),
        yaxis=dict(**cfg.GRID, title='DXY', fixedrange=False)
    )
    return fig


# ── RSI DXY (14d) con niveles 70/30 ────────────────────────────────────
def build_rsi_dxy(df: pd.DataFrame) -> go.Figure:
    rsi_dx = df[['Fecha', 'RSI RSI DXY']].dropna(subset=['RSI RSI DXY']).sort_values('Fecha')
    # .isoformat(), no Timestamp crudo — ver nota en _add_pct_labels.
    mn_dx, mx_dx = rsi_dx['Fecha'].min().isoformat(), rsi_dx['Fecha'].max().isoformat()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rsi_dx['Fecha'], y=rsi_dx['RSI RSI DXY'],
        name='RSI DXY (14d)',
        line=dict(color='#e67e22', width=1.8),
        hovertemplate='<b>RSI DXY</b><br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>',
    ))
    for level, name, color in [(70, 'Sobrecompra · 70', cfg.MAROON), (30, 'Sobreventa · 30', cfg.TEAL)]:
        fig.add_trace(go.Scatter(
            x=[mn_dx, mx_dx], y=[level, level],
            name=name, mode='lines',
            line=dict(color=color, width=1.2, dash='dash'),
            hoverinfo='skip',
        ))
    fig.update_layout(
        **cfg.LAYOUT_BASE,
        xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='RSI', range=[0, 100]),
    )
    return fig


# ── Volatilidad implícita 1w de todos los países ───────────────────────
def build_vol_1w(df: pd.DataFrame) -> go.Figure:
    vol_cfg = [
        ('Misceláneos Volatilidad implicita Opciones 1W CLP', 'Vol 1W · CLP', cfg.NAVY),
        ('Misceláneos Volatilidad implicita Opciones 1W MXN', 'Vol 1W · MXN', cfg.MAROON),
        ('Misceláneos Volatilidad implicita Opciones 1W BRL', 'Vol 1W · BRL', '#2ECC40'),
        ('Misceláneos Volatilidad implicita Opciones 1W COP', 'Vol 1W · COP', cfg.BLUE),
    ]
    fig = go.Figure()
    for col, label, color in vol_cfg:
        s = df[['Fecha', col]].dropna(subset=[col])
        fig.add_trace(go.Scatter(
            x=s['Fecha'], y=s[col],
            name=label,
            line=dict(color=color, width=1.5),
            hovertemplate=f'<b>{label}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}}%<extra></extra>',
        ))
    fig.update_layout(
        **cfg.LAYOUT_BASE,
        xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='Vol. Implícita (%)'),
    )
    return fig


# ── Monedas base 100 ──────────────────────────────────────────────────
def fig_base100_dynamic(df: pd.DataFrame, fecha_col: str, series_cfg: list[tuple[str, str, str]]) -> go.Figure:
    """Gráfico de monedas base 100, con customdata (valores crudos) por serie.

    Todas las series se rebasan a 100 desde una fecha de inicio común (la
    última entre las primeras-fechas-válidas de cada serie), para que la
    comparación "Todo" arranque a la par. El customdata permite que el JS
    del dashboard (rebaseChart) recalcule la base 100 al cambiar de período
    sin volver a tocar Python.
    """
    all_starts = [
        df[[fecha_col, col]].dropna(subset=[col])[fecha_col].min()
        for col, _, _ in series_cfg
        if len(df[[fecha_col, col]].dropna(subset=[col])) > 0
    ]
    min_date = max(all_starts) if all_starts else df[fecha_col].min()

    fig = go.Figure()
    for col, label, color in series_cfg:
        s = df[[fecha_col, col]].dropna(subset=[col]).sort_values(fecha_col)
        sf = s[s[fecha_col] >= min_date]
        if len(sf) == 0:
            continue
        base = sf[col].iloc[0]
        fig.add_trace(go.Scatter(
            x=sf[fecha_col], y=(sf[col] / base * 100).round(3),
            customdata=sf[col].round(4).tolist(),
            name=label,
            line=dict(color=color, width=1.5),
            hovertemplate=f'<b>{label}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>',
        ))

    fig.update_layout(
        **cfg.LAYOUT_BASE,
        xaxis=dict(**cfg.GRID, rangeslider=dict(visible=False)),
        yaxis=dict(**cfg.GRID, title='Índice Base 100'),
    )
    return fig


_G10_CFG = [
    ('CLP Cierre',         'CLP · Peso chileno',    '#000080'),
    ('G10 EUR',                 'EUR · Euro',              '#0074D9'),
    ('G10 GBP',                 'GBP · Libra esterlina',   '#9b59b6'),
    ('G10 JPY',                 'JPY · Yen japonés',       '#e74c3c'),
    ('G10 CHF',                 'CHF · Franco suizo',      '#e67e22'),
    ('G10 SEK',                 'SEK · Corona sueca',      '#1abc9c'),
    ('G10 DKK',                 'DKK · Corona danesa',     None),
    ('Monedas Commodities NOK', 'NOK · Corona noruega',    '#2ecc71'),
    ('Monedas Commodities AUD', 'AUD · Dólar australiano', '#3498db'),
    ('Monedas Commodities CAD', 'CAD · Dólar canadiense',  None),
    ('Monedas Commodities NZD', 'NZD · Dólar neozelandés', '#f39c12'),
]

_LATAM_CFG = [
    ('CLP Cierre',         'CLP · Peso chileno',    '#000080'),
    ('Monedas LATAM MXN',  'MXN · Peso mexicano',   None),
    ('Monedas LATAM BRL',  'BRL · Real brasileño',  '#2ECC40'),
    ('Monedas LATAM COP',  'COP · Peso colombiano', None),
    ('Monedas LATAM PEN',  'PEN · Sol peruano',     '#FF851B'),
]

_EM_CFG = [
    ('CLP Cierre',         'CLP · Peso chileno',    '#000080'),
    ('Monedas Comparables RAND SUDAFRICANO', 'ZAR · Rand sudafricano',  '#f39c12'),
    ('Monedas Comparables RENMINBI CHINO',   'CNY · Renminbi chino',    '#c0392b'),
    ('Monedas Comparables WON COREANO',      'KRW · Won coreano',       '#27ae60'),
    ('Monedas Comparables RUPIA INDIA',      'INR · Rupia india',       '#2980b9'),
    ('Monedas Comparables PESO FILIPINO',    'PHP · Peso filipino',     '#16a085'),
    ('Monedas Comparables FORINT HUNGARO',   'HUF · Forint húngaro',     '#2980b9'),
]

_DEV_CFG = [
    ('CLP Cierre',         'CLP · Peso chileno',    '#000080'),
    ('Monedas Comparables ZLOTY POLACO',     'PLN · Zloty polaco',        '#8e44ad'),
    ('Monedas Comparables CORONA CHECA',     'CZK · Corona checa',        '#27ae60'),
    ('Monedas Comparables DÓLAR SINGAPUR ',  'SGD · Dólar singapurense',  '#c0392b'),
    ('Monedas Comparables LEV BULGARO',      'BGN · Lev búlgaro',         '#d35400'),
    ('Monedas Comparables LEU RUMANO',       'RON · Leu rumano',          '#7f8c8d'),
]


def _resolve_colors(series_cfg: list[tuple[str, str, str | None]], fallback: str) -> list[tuple[str, str, str]]:
    return [(col, label, color or fallback) for col, label, color in series_cfg]


def build_fig_g10(df: pd.DataFrame) -> go.Figure:
    return fig_base100_dynamic(df, 'Fecha', _resolve_colors(_G10_CFG, cfg.TEAL))


def build_fig_latam(df: pd.DataFrame) -> go.Figure:
    return fig_base100_dynamic(df, 'Fecha', _resolve_colors(_LATAM_CFG, cfg.MAROON))


def build_fig_emerg(df: pd.DataFrame) -> go.Figure:
    return fig_base100_dynamic(df, 'Fecha', _EM_CFG)


def build_fig_comparables(df: pd.DataFrame) -> go.Figure:
    return fig_base100_dynamic(df, 'Fecha', _DEV_CFG)


# ── RSI Cobre (14d), niveles fijos 70/30 ──────────────────────────────
def build_rsi_cobre(df: pd.DataFrame) -> go.Figure:
    rsi_cu = df[['Fecha', 'RSI RSI Cobre']].dropna(subset=['RSI RSI Cobre']).copy()
    # .isoformat(), no Timestamp crudo — ver nota en _add_pct_labels.
    mn_cu, mx_cu = rsi_cu['Fecha'].min().isoformat(), rsi_cu['Fecha'].max().isoformat()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rsi_cu['Fecha'], y=rsi_cu['RSI RSI Cobre'],
        name='RSI Cobre (14d)', line=dict(color='#b5551b', width=1.8),
        hovertemplate='<b>RSI Cobre</b><br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>'
    ))
    for lvl, nm, col in [(70, 'Sobrecompra · 70', cfg.RED), (30, 'Sobreventa · 30', cfg.TEAL)]:
        fig.add_trace(go.Scatter(
            x=[mn_cu, mx_cu], y=[lvl, lvl], name=nm, mode='lines',
            line=dict(color=col, width=1.2, dash='dash'), hoverinfo='skip',
        ))
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='RSI'),
    )
    return fig


# ── Cobre COMEX (HGA) vs Londres (LME) + spread ──────────────────────
def build_cobre_ny_spread(df: pd.DataFrame) -> go.Figure:
    cu_cols = ['Drivers Precio Cobre HGA', 'Drivers Precio Londres']
    cobre_ny = df[['Fecha'] + cu_cols].dropna(subset=cu_cols).copy()
    cobre_ny['HGA_usdlb']     = cobre_ny['Drivers Precio Cobre HGA'] / 100
    cobre_ny['Londres_usdlb'] = cobre_ny['Drivers Precio Londres'] / 2262.64
    cobre_ny['Spread_usdlb']  = cobre_ny['HGA_usdlb'] - cobre_ny['Londres_usdlb']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=cobre_ny['Fecha'], y=cobre_ny['HGA_usdlb'],
        name='HGA (COMEX)', mode='lines',
        line=dict(color='#b5551b', width=1.5),
        hovertemplate='<b>COMEX</b><br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=cobre_ny['Fecha'], y=cobre_ny['Londres_usdlb'],
        name='Londres (LME)', mode='lines',
        line=dict(color=cfg.NAVY, width=1.5),
        hovertemplate='<b>Londres</b><br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=cobre_ny['Fecha'], y=cobre_ny['Spread_usdlb'],
        name='Spread HGA–Londres', mode='markers',
        marker=dict(color=cfg.GOLD, size=4, opacity=0.8),
        hovertemplate='<b>Spread</b><br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>',
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='Cobre (USD/lb)'),
        yaxis2=dict(**cfg.GRID, title='Spread (USD/lb)', overlaying='y', side='right'),
    )
    return fig


# ── Términos de intercambio GS vs CLP ─────────────────────────────────
def build_tot_gs(df: pd.DataFrame) -> go.Figure:
    tot_gs_df = df[['Fecha',
                    'Términos de intercambio GS Terms of Trade',
                    'CLP Cierre']].dropna(subset=['Términos de intercambio GS Terms of Trade']).copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=tot_gs_df['Fecha'], y=tot_gs_df['Términos de intercambio GS Terms of Trade'],
        name='ToT GS', line=dict(color=cfg.GOLD, width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=tot_gs_df['Fecha'], y=tot_gs_df['CLP Cierre'],
        name='CLP', line=dict(color=cfg.NAVY, width=1.5),
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='Términos de Intercambio GS'),
        yaxis2=dict(**cfg.GRID, title='CLP', overlaying='y', side='right'),
    )
    return fig


# ── Términos de intercambio CT vs CLP ─────────────────────────────────
def build_tot_ct(df: pd.DataFrame) -> go.Figure:
    tot_ct_df = df[['Fecha',
                    'Términos de intercambio CT Terms of Trade.',
                    'CLP Cierre']].dropna(subset=['Términos de intercambio CT Terms of Trade.']).copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=tot_ct_df['Fecha'], y=tot_ct_df['Términos de intercambio CT Terms of Trade.'],
        name='ToT CT', line=dict(color=cfg.MAROON, width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=tot_ct_df['Fecha'], y=tot_ct_df['CLP Cierre'],
        name='CLP', line=dict(color=cfg.NAVY, width=1.5),
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='Términos de Intercambio CT'),
        yaxis2=dict(**cfg.GRID, title='CLP', overlaying='y', side='right'),
    )
    return fig


# ── Punta FWD (1M) vs CLP ───────────────────────────────────────────
def build_fwd_puntas(df: pd.DataFrame) -> go.Figure:
    fwd = df[['Fecha', 'Puntos Forward 1M', 'CLP Cierre']].sort_values('Fecha', ascending=False)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=fwd['Fecha'], y=fwd['CLP Cierre'],
        name='CLP', line=dict(color=cfg.NAVY, width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=fwd['Fecha'], y=fwd['Puntos Forward 1M'],
        name='Punta FWD 1M', line=dict(color=cfg.GOLD, width=1.2)
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='CLP'),
        yaxis2=dict(**cfg.GRID, title='Punta FWD', overlaying='y', side='right'),
    )
    return fig


# ── Puntas FWD Chile (1M/3M/6M/12M, último año) ───────────────────────
def build_fwd_puntas12m(df: pd.DataFrame) -> go.Figure:
    fwd_cols = [c for c in df.columns if isinstance(c, str) and "Puntos Forward" in c]
    datafwd = df[['Fecha'] + fwd_cols].reset_index().sort_values('Fecha', ascending=False)
    fechapivot = df['Fecha'].max() - pd.Timedelta(days=365)
    datafwd = datafwd[datafwd['Fecha'] >= fechapivot]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datafwd['Fecha'], y=datafwd['Puntos Forward 1M'], name='Puntos FWD 1M', line=dict(color=cfg.NAVY)))
    fig.add_trace(go.Scatter(x=datafwd['Fecha'], y=datafwd['Puntos Forward 3M'], name='Puntos FWD 3M', line=dict(color=cfg.GOLD)))
    fig.add_trace(go.Scatter(x=datafwd['Fecha'], y=datafwd['Puntos Forward 6M'], name='Puntos FWD 6M', line=dict(color=cfg.RED)))
    fig.add_trace(go.Scatter(x=datafwd['Fecha'], y=datafwd['Puntos Forward 12M'], name='Puntos FWD 12M', line=dict(color=cfg.GRAY)))
    fig.update_layout(**cfg.LAYOUT_BASE, xaxis=dict(**cfg.GRID), yaxis=dict(**cfg.GRID))
    return fig


# ── Spread TPM US vs Chile a 1Y ───────────────────────────────────────
def build_tpm_spread(df: pd.DataFrame) -> go.Figure:
    tpm_cols = ['Expectativas de TPM Expectativas de TPM US',
                'Expectativas de TPM Expectativas de TPM CL']
    tpm_df = df[['Fecha'] + tpm_cols].dropna(subset=tpm_cols).copy()
    tpm_df['Spread_bp']  = (tpm_df[tpm_cols[1]] - tpm_df[tpm_cols[0]]) * 100
    tpm_df['Spread_avg'] = tpm_df['Spread_bp'].expanding().mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=tpm_df['Fecha'], y=tpm_df[tpm_cols[1]],
        name='TPM Chile 1Y', line=dict(color=cfg.NAVY, width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=tpm_df['Fecha'], y=tpm_df[tpm_cols[0]],
        name='TPM US 1Y', line=dict(color=cfg.MAROON, width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=tpm_df['Fecha'], y=tpm_df['Spread_bp'],
        name='Spread CL–US (pb)', mode='markers',
        marker=dict(color=cfg.GOLD, size=4, opacity=0.7),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=tpm_df['Fecha'], y=tpm_df['Spread_avg'],
        name='Promedio histórico', mode='lines',
        line=dict(color=cfg.TEAL, width=1.5, dash='dash'),
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='TPM esperado (%)'),
        yaxis2=dict(**cfg.GRID, title='Spread (pb)', overlaying='y', side='right'),
    )
    return fig


# ── Tipo de Cambio Real (TCR) ─────────────────────────────────────────
def build_tcr(df: pd.DataFrame) -> go.Figure:
    tcr_df = df[['Fecha', 'Tipo de cambio real CLP']].dropna(subset=['Tipo de cambio real CLP']).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tcr_df['Fecha'], y=tcr_df['Tipo de cambio real CLP'],
        name='TCR', line=dict(color=cfg.NAVY, width=1.8),
        hovertemplate='<b>TCR</b><br>%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>',
    ))
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='Tipo de Cambio Real'),
    )
    return fig


# ── S/R Posición No Residentes (CLPBODM Index) ─────────────────────────
def build_pos_nr_sr(df: pd.DataFrame) -> go.Figure:
    summary = data.percentile_summary(
        df, 'Fecha', 'No Residentes Posición OffShore',
        transform=lambda s: s * -1, output_col='CLPBODM Index',
    )
    nr_sr = summary['data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nr_sr['Fecha'], y=nr_sr['CLPBODM Index'],
                              name='CLPBODM Index', line=dict(color=cfg.GOLD, width=1.8)))
    _pct_lines(fig, summary)
    fig.update_layout(
        **cfg.SR_LAYOUT,
        xaxis=dict(**cfg.GRID, rangeslider=dict(visible=False)),
        yaxis=dict(**cfg.GRID, title='Pos. No Residentes (Millones de dólares)', fixedrange=False)
    )
    return fig


# ── Volatilidad intradía CLP (SQL: clp_intra) ─────────────────────────
def build_vol_clp(df: pd.DataFrame, clp_intra: pd.DataFrame) -> go.Figure:
    clp_intra = clp_intra.copy()
    clp_intra['Variacion'] = clp_intra.groupby('Fecha')['TC'].diff()
    clp_intra_vol = clp_intra.groupby('Fecha')['Variacion'].std().round(3).reset_index()
    clp_intra_vol.rename(columns={'Variacion': 'Volatilidad'}, inplace=True)

    df_clpintra = clp_intra.merge(clp_intra_vol, on='Fecha', how='left')
    df_clpintra = df_clpintra.groupby('Fecha')['Volatilidad'].first().reset_index()

    clp = df.iloc[:, 0:7].copy().sort_values('Fecha', ascending=False)
    clp_vol = clp.merge(df_clpintra, on='Fecha', how='left').dropna()
    clp_vol = clp_vol[clp_vol['Fecha'] >= pd.Timestamp('2026-01-01')]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=clp_vol['Fecha'], y=clp_vol['CLP Cierre'],
        name='CLP', line=dict(color=cfg.NAVY, width=1.5),
        hovertemplate='<b>CLP</b><br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=clp_vol['Fecha'], y=clp_vol['Volatilidad'] * 100,
        name='Volatilidad (%)', mode='lines',
        hovertemplate='<b>Volatilidad (%)</b><br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>',
        line=dict(color=cfg.GOLD, width=1, dash=None)
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='CLP'),
        yaxis2=dict(**cfg.GRID, title='Volatilidad (%)', overlaying='y', side='right'),
    )
    return fig


# ── Pos. No Residentes vs CLP ─────────────────────────────────────────
def build_clp_nr(df: pd.DataFrame) -> go.Figure:
    derivados = df[['Fecha', 'CLP Cierre', 'No Residentes Posición OffShore']].dropna().sort_values('Fecha', ascending=False)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=derivados['Fecha'], y=derivados['CLP Cierre'],
        name='CLP', line=dict(color=cfg.NAVY, width=1.5),
        hovertemplate='<b>CLP (%)</b><br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=derivados['Fecha'], y=derivados['No Residentes Posición OffShore'] * -1,
        name='CLPBODM Index', line=dict(color=cfg.GOLD, width=1.5),
        hovertemplate='<b>Pos. No Residentes (%)</b><br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>',
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='CLP'),
        yaxis2=dict(**cfg.GRID, title='Millones de dólares', overlaying='y', side='right'),
    )
    return fig


# ── Carry Trade (Excel: hoja CarryTrade) ──────────────────────────────
def build_carry_trade(carrytrade: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=carrytrade['País'], y=carrytrade['Value'].round(2),
        marker=dict(color=['#2ca02c' if v >= 0 else '#d62728' for v in carrytrade['Value']]),
        text=carrytrade['Value'].round(2), textposition='outside',
        hovertemplate='%{x}<br>%{y}<extra></extra>'
    ))
    fig.update_layout(**cfg.LAYOUT_BASE, yaxis=dict(range=[carrytrade['Value'].min() * 1.25, carrytrade['Value'].max() * 1.25]))
    return fig


# ── Variación monedas 1 mes (SQL: monedas1m) ──────────────────────────
def build_variacion_1mes(monedas1m: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monedas1m['Moneda'], y=monedas1m['variacion_30d'],
        marker=dict(color=['#2ca02c' if v >= 0 else '#d62728' for v in monedas1m['variacion_30d']])
    ))
    fig.update_layout(**cfg.LAYOUT_BASE, xaxis=cfg.data_select)
    fig.add_annotation(
        text=f"Variación entre {monedas1m['Fecha'].iloc[0]} y {monedas1m['Fecha'].iloc[0] - pd.Timedelta(days=30)}",
        xref='paper', yref='paper', x=0, y=-1.02,
        xanchor='left', yanchor='bottom', showarrow=False, font=dict(size=10),
    )
    return fig


# ── Curvas de contratos: Cobre COMEX (HGA) / Londres (LMA) / WTI (CL1) ─
def _build_curva(df: pd.DataFrame, contrato_col: str, tenor_col: str, price_col: str,
                  name: str, color: str, hover_unit: str, hover_fmt: str = '.2f') -> go.Figure:
    d = (
        df[[contrato_col, tenor_col, price_col]]
        .dropna()
        .assign(**{tenor_col: lambda x: pd.to_datetime(x[tenor_col])})
        .sort_values(tenor_col)
        .drop_duplicates(subset=[tenor_col])
        .reset_index(drop=True)
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d[tenor_col], y=d[price_col],
        mode='lines+markers',
        name=name,
        line=dict(color=color, width=2),
        marker=dict(color=color, size=5, symbol='circle'),
        customdata=d[contrato_col],
        hovertemplate=f'<b>%{{customdata}}</b><br>Vto.: %{{x|%b %Y}}<br>Price: %{{y:{hover_fmt}}} {hover_unit}<extra></extra>',
    ))
    fig.update_layout(
        **cfg.LAYOUT_BASE,
        xaxis=dict(**cfg.GRID, title='Vencimiento', tickformat='%b %Y'),
        yaxis=dict(**cfg.GRID, title=hover_unit),
    )
    return fig


def build_contrato_hga(df: pd.DataFrame) -> go.Figure:
    return _build_curva(df, 'Contrato HGA', 'Tenor HGA', 'Price HGA', 'HGA · COMEX', '#b5551b', '¢/lb')


def build_contrato_lma(df: pd.DataFrame) -> go.Figure:
    return _build_curva(df, 'Contrato Lma', 'Tenor Lma', 'Price Lma', 'LMA · Londres', cfg.NAVY, 'USD/t', ',.2f')


def build_contrato_cl1(df: pd.DataFrame) -> go.Figure:
    return _build_curva(df, 'Contrato CL1', 'Tenor CL1', 'Price CL1', 'CL1 · WTI', '#1a5276', 'USD/bbl')


# ── Gamma Proxy (SQL: gammaproxy) ─────────────────────────────────────
def build_gamma_plot(gammaproxy: pd.DataFrame) -> go.Figure:
    gammaproxy = gammaproxy.copy()
    gammaproxy['label'] = (
        gammaproxy['Strike_bucket'].astype(int).astype(str) +
        ' (' +
        gammaproxy['Strike_min'].astype(int).astype(str) +
        ' - ' +
        gammaproxy['Strike_max'].astype(int).astype(str) +
        ')'
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gammaproxy['gamma_proxy_mm'].round(2), y=gammaproxy['label'], orientation='h',
        marker=dict(color=cfg.NAVY), width=.7,
    ))
    fig.update_layout(
        **cfg.LAYOUT_BASE, height=600,
        yaxis=dict(type='category', categoryorder='array', categoryarray=gammaproxy.sort_values('Strike_bucket')['label']),
    )
    return fig


# ── Inventarios COMEX + Precio HGA ───────────────────────────────────
def build_inv_comex(df: pd.DataFrame) -> go.Figure:
    inv_comex = df[['Fecha', 'Drivers Inventarios Comex', 'Drivers Precio Cobre HGA']].dropna()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=inv_comex['Fecha'], y=inv_comex['Drivers Inventarios Comex'],
        name='Inventarios COMEX',
        marker=dict(color='rgba(181,85,27,0.45)', line=dict(width=0)),
        hovertemplate='<b>Inventarios COMEX</b><br>%{x|%Y-%m-%d}: %{y:,.0f} t<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=inv_comex['Fecha'], y=inv_comex['Drivers Precio Cobre HGA'] / 100,
        name='COMEX',
        mode='lines',
        line=dict(color='#b5551b', width=1.5),
        hovertemplate='<b>COMEX</b><br>%{x|%Y-%m-%d}: %{y:.3f} USD/lb<extra></extra>',
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='Inventarios (t)'),
        yaxis2=dict(**cfg.GRID, title='Precio (USD/lb)', overlaying='y', side='right'),
        bargap=0,
    )
    return fig


# ── Inventarios LME + Precio Londres ─────────────────────────────────
def build_inv_lme(df: pd.DataFrame) -> go.Figure:
    inv_lme = df[['Fecha', 'Drivers Inv. Cobre', 'Drivers Precio Londres']].dropna()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=inv_lme['Fecha'], y=inv_lme['Drivers Inv. Cobre'],
        name='Inventarios LME',
        marker=dict(color='rgba(0,23,48,0.45)', line=dict(width=0)),
        hovertemplate='<b>Inventarios LME</b><br>%{x|%Y-%m-%d}: %{y:,.0f} t<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=inv_lme['Fecha'], y=inv_lme['Drivers Precio Londres'] / 2262.64,
        name='LME',
        mode='lines',
        line=dict(color=cfg.NAVY, width=1.5),
        hovertemplate='<b>LME</b><br>%{x|%Y-%m-%d}: %{y:.3f} USD/lb<extra></extra>',
    ), secondary_y=True)
    fig.update_layout(
        **cfg.LAYOUT_BASE, xaxis=cfg.data_select,
        yaxis=dict(**cfg.GRID, title='Inventarios (t)'),
        yaxis2=dict(**cfg.GRID, title='Precio (USD/lb)', overlaying='y', side='right'),
        bargap=0,
    )
    return fig


# ── Tasas implícitas (Chile/Brazil/Peru/Mexico/Colombia) ──────────────
def build_implicitas(df: pd.DataFrame) -> go.Figure:
    imp_cols = [c for c in df.columns if isinstance(c, str) and "Implicita" in c]
    dataimpl = df[['Fecha'] + imp_cols].reset_index().sort_values('Fecha', ascending=False)
    fechapivot = df['Fecha'].max() - pd.Timedelta(days=365)
    dataimpl = dataimpl[dataimpl['Fecha'] >= fechapivot]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataimpl['Fecha'], y=dataimpl['Implicita Chile'], name='Implícita Chile', line=dict(color=cfg.NAVY)))
    fig.add_trace(go.Scatter(x=dataimpl['Fecha'], y=dataimpl['Implicita Brazil'], name='Implícita Brazil', line=dict(color=cfg.GOLD)))
    fig.add_trace(go.Scatter(x=dataimpl['Fecha'], y=dataimpl['Implicita Peru'], name='Implicita Peru', line=dict(color=cfg.GRAY)))
    fig.add_trace(go.Scatter(x=dataimpl['Fecha'], y=dataimpl['Implicita Mexico'], name='Implicita Mexico', line=dict(color=cfg.TEAL)))
    fig.add_trace(go.Scatter(x=dataimpl['Fecha'], y=dataimpl['Implicita Colombia'], name='Implicita Colombia', line=dict(color=cfg.RED)))
    fig.update_layout(**cfg.LAYOUT_BASE, xaxis=dict(**cfg.GRID), yaxis=dict(**cfg.GRID))
    return fig


# ── Fixing de la banca (SQL: fixing) ──────────────────────────────────
_FIXING_COLOR_MAP = {
    'AFP': '#2E86DE',
    'Emp_real': '#1B2631',
    'CB': '#8B0000',
    'Bancos': '#7d3c98',
    'CS': '#e84393',
    'Emp_financiera': '#6c5ce7',
    'FFMM': '#00CEc9',
    'NR': '#C8A951',
    'Otros': '#7A7A7A',
    'BCCh': '#14284B',
}


def build_fixing(fixing: pd.DataFrame) -> go.Figure:
    fixing = fixing.copy()
    tot_fixing = fixing.groupby('Fixing', as_index=False)['Pos_neta'].sum()
    tot_fixing['NombreInformanteNorm'] = 'Total'
    tot_fixing['SectorContNorm'] = None
    tot_fixing = tot_fixing[['NombreInformanteNorm', 'SectorContNorm', 'Fixing', 'Pos_neta']]

    fixing = pd.concat([fixing, tot_fixing], ignore_index=True)
    fixing['Fixing'] = pd.to_datetime(fixing['Fixing'])
    fecha_fixing = pd.Timestamp.today().normalize()
    fixing_hoy = fixing[fixing['Fixing'] == fecha_fixing]

    tot_agente = fixing_hoy.groupby('NombreInformanteNorm', as_index=False)['Pos_neta'].sum()
    orden = ['Total'] + [b for b in fixing_hoy['NombreInformanteNorm'].unique() if b != 'Total']

    fig = go.Figure()
    for sector, color in _FIXING_COLOR_MAP.items():
        df_sector = fixing_hoy[fixing_hoy['SectorContNorm'].fillna('Otros') == sector]
        if len(df_sector) > 0:
            fig.add_trace(go.Bar(
                x=df_sector['NombreInformanteNorm'],
                y=df_sector['Pos_neta'].round(0),
                name=sector,
                marker_color=color,
            ))

    fig.add_trace(go.Scatter(
        x=tot_agente['NombreInformanteNorm'],
        y=tot_agente['Pos_neta'].round(0),
        mode='markers',
        marker=dict(color='black', size=6),
        name='Posición Neta',
    ))
    fig.update_layout(
        barmode='relative',
        **cfg.LAYOUT_BASE, xaxis=dict(categoryorder='array', categoryarray=orden),
        height=600,
    )
    return fig


# ── Heatmap Gamma Proxy (SQL: heatmap) ────────────────────────────────
def build_heatmap_gp(heatmap: pd.DataFrame) -> go.Figure:
    y_order = sorted(heatmap['Strike_bucket'].unique())
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=heatmap['venc'], y=heatmap['Strike_bucket'], z=heatmap['gamma_proxy'],
        xgap=2, ygap=2, colorscale='RdYlGn',
    ))
    fig.update_yaxes(type='category', categoryorder='array', categoryarray=y_order)
    fig.update_xaxes(type='category')
    fig.update_layout(**cfg.LAYOUT_BASE, xaxis=dict(**cfg.GRID), yaxis=dict(**cfg.GRID))
    return fig


def _add_update_footer(fig: go.Figure, footer_text: str) -> go.Figure:
    """Fecha de datos + fecha/hora de generación, como anotación DENTRO de la
    figura (no en el HTML alrededor) — así viaja con el gráfico cuando se
    copia individualmente ('Copiar' -> PNG) o se manda por correo, no sólo
    cuando se ve el dashboard completo.

    Va en el margen SUPERIOR (yshift fijo en px sobre y=1), no el inferior:
    ahí es donde vive la leyenda (y=-0.15 "paper", relativo al alto del área
    de ploteo) y en los gráficos con muchas series (monedas base100, 6-10
    items) envuelve a 2-3 filas y se comía el espacio reservado. El margen
    superior no lo usa nada más en ningún gráfico de este archivo.
    """
    fig.add_annotation(
        text=footer_text,
        xref='paper', yref='paper', x=0, y=1,
        xanchor='left', yanchor='bottom', yshift=8,
        showarrow=False,
        font=dict(size=8.5, color='#9aa2b1', family='Barlow, Arial, sans-serif'),
    )
    return fig


def _polish_hover(fig: go.Figure) -> go.Figure:
    """Uniforma el hover de los traces que no definieron hovertemplate propio."""
    for trace in fig.data:
        if trace.type in ('candlestick', 'heatmap') or trace.hoverinfo == 'skip' or trace.hovertemplate:
            continue
        name = trace.name or ''
        trace.update(hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}: %{{y:,.2f}}<extra></extra>")
    return fig


# ── Registro de figuras (sin dependencias SQL) ─────────────────────────
FIGURE_BUILDERS: dict[str, Callable[[pd.DataFrame], go.Figure]] = {
    "clp_candlestick":    build_clp_candlestick,
    "clp_bollinger":      build_clp_bollinger,
    "clp_medias_moviles": build_clp_medias_moviles,
    "clp_cobre":          build_clp_cobre,
    "clp_dxy":            build_clp_dxy,
    "clp_sr":             build_clp_sr,
    "clp_rsi":            build_clp_rsi,
    "cobre_sr":           build_cobre_sr,
    "dxy_sr":             build_dxy_sr,
    "vol_1w":             build_vol_1w,
    "fig_g10":            build_fig_g10,
    "fig_comparables":    build_fig_comparables,
    "fig_latam":          build_fig_latam,
    "fig_emerg":          build_fig_emerg,
    "fig_rsi_dxy":        build_rsi_dxy,
    "rsi_cobre":          build_rsi_cobre,
    "cobre_ny_spread":    build_cobre_ny_spread,
    "tot_gs":             build_tot_gs,
    "tot_ct":             build_tot_ct,
    "fwd_puntas":         build_fwd_puntas,
    "fwd_puntas12m":      build_fwd_puntas12m,
    "tpm_spread":         build_tpm_spread,
    "tcr":                build_tcr,
    "pos_nr_sr":          build_pos_nr_sr,
    "clp_nr":             build_clp_nr,
    "contrato_hga":       build_contrato_hga,
    "contrato_lma":       build_contrato_lma,
    "contrato_cl1":       build_contrato_cl1,
    "inv_comex":          build_inv_comex,
    "inv_lme":            build_inv_lme,
    "implicitas":         build_implicitas,
}

# Figuras que necesitan un DataFrame extra (Excel local: Intradía/CarryTrade).
EXCEL_EXTRA_BUILDERS: dict[str, tuple[str, Callable]] = {
    "clp_intradia": ("intradia", build_clp_intradia),
    "carry_trade":  ("carrytrade", build_carry_trade),
}

# Figuras que dependen de datos SQL en vivo (Get_Data) — ver docstring del módulo.
SQL_BUILDERS: dict[str, tuple[str, Callable]] = {
    "variacion_dia":  ("monedas", build_variacion_dia),
    "clp_dist":       ("distclp", build_clp_dist),
    "variacion_1mes": ("monedas1m", build_variacion_1mes),
    "gamma_plot":     ("gammaproxy", build_gamma_plot),
    "fixing":         ("fixing", build_fixing),
    "heatmap_gp":     ("heatmap", build_heatmap_gp),
    # vol_clp necesita df + clp_intra, se arma aparte en build_all_figures.
}

REBASE_FIG_IDS = {"fig_g10", "fig_latam", "fig_emerg", "fig_comparables"}


def build_all_figures(
    df: pd.DataFrame,
    *,
    intradia: pd.DataFrame | None = None,
    carrytrade: pd.DataFrame | None = None,
    monedas: pd.DataFrame | None = None,
    clp_intra: pd.DataFrame | None = None,
    distclp: pd.DataFrame | None = None,
    monedas1m: pd.DataFrame | None = None,
    gammaproxy: pd.DataFrame | None = None,
    fixing: pd.DataFrame | None = None,
    heatmap: pd.DataFrame | None = None,
    footer_text: str | None = None,
) -> dict[str, go.Figure]:
    figures = {fig_id: builder(df) for fig_id, builder in FIGURE_BUILDERS.items()}

    extra_frames = {"intradia": intradia, "carrytrade": carrytrade}
    for fig_id, (frame_name, builder) in EXCEL_EXTRA_BUILDERS.items():
        frame = extra_frames[frame_name]
        if frame is not None:
            figures[fig_id] = builder(frame)

    sql_frames = {
        "monedas": monedas, "distclp": distclp, "monedas1m": monedas1m,
        "gammaproxy": gammaproxy, "fixing": fixing, "heatmap": heatmap,
    }
    for fig_id, (frame_name, builder) in SQL_BUILDERS.items():
        frame = sql_frames[frame_name]
        if frame is not None:
            figures[fig_id] = builder(frame)

    if clp_intra is not None:
        figures["vol_clp"] = build_vol_clp(df, clp_intra)

    for fig in figures.values():
        _polish_hover(fig)
        if footer_text:
            _add_update_footer(fig, footer_text)
    return figures
