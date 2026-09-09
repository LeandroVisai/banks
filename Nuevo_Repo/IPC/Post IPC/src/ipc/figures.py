"""Los gráficos que se construyen ACÁ, fuera del notebook.

Los del notebook llegan ya hechos por figuras.json (ver data.py); acá
viven los que necesitan datos que el notebook no tiene: lo que llena el
operador (esperado vs efectivo) y lo que se deriva de los analíticos.

── Agregar un gráfico ────────────────────────────────────────────────
Tres pasos, cada uno en un archivo distinto:

  1. el dato: una hoja más en data.py, o
     nada si ya está en `datasets`
  2. el builder acá + su línea en BUILDERS, diciendo qué datasets consume
  3. el slot en layout.py, para decir dónde va

Si un dataset no está, la figura se saltea sola y la tarjeta sale como
"Pendiente" con el fig_id visible.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as cfg


# ═════════════════════════════════════════════════════════════════════
# ESPERADO VS EFECTIVO
# ═════════════════════════════════════════════════════════════════════

def build_esperado_vs_efectivo(expectativas: pd.DataFrame, analiticos: pd.DataFrame) -> go.Figure:
    """Gráfico N°3 del informe del banco: el IPC mensual que esperaba cada
    fuente (barras) contra el efectivo (línea con marcadores).

    Se muestran los últimos 18 meses con expectativa cargada. El efectivo se
    cruza por mes desde los analíticos del BCCh, así que el mes en curso —que
    tiene expectativa pero todavía no tiene dato— sale sin punto en la línea.
    """
    e = expectativas.tail(18).copy()
    efectivo = analiticos[["Fecha", "IPC General"]].copy()
    efectivo["Fecha"] = efectivo["Fecha"].dt.to_period("M").dt.to_timestamp()
    d = e.merge(efectivo, on="Fecha", how="left")

    fig = go.Figure()
    for fuente, color in cfg.FUENTES_EXPECTATIVA.items():
        if fuente not in d.columns:
            continue
        fig.add_trace(go.Bar(
            x=d["Fecha"], y=d[fuente], name=fuente, marker=dict(color=color),
            hovertemplate=f"<b>{fuente}</b> %{{x|%b %Y}}: %{{y:.2f}}%<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=d["Fecha"], y=d["IPC General"], name="IPC efectivo", mode="lines+markers",
        line=dict(color=cfg.EFECTIVO, width=2.2), marker=dict(size=8, symbol="diamond"),
        hovertemplate="<b>Efectivo</b> %{x|%b %Y}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **cfg.LAYOUT_BASE, barmode="group", bargap=0.25,
        xaxis=dict(**cfg.data_select, tickformat="%b %y", dtick="M1"),
        yaxis=dict(**cfg.GRID, title="Variación mensual · %", ticksuffix="%"),
    )
    return fig


def build_incidencias_feedback(divisiones: pd.DataFrame, ine_divisiones: pd.DataFrame) -> go.Figure:
    """Gráfico N°1 del banco: por división, el rango de incidencia que
    esperaba el mercado (mínimo–máximo entre instituciones) y el dato
    efectivo del INE.

    El rango se dibuja como barra flotante (base = mínimo, largo = máximo −
    mínimo); el promedio y el efectivo van como marcadores encima. Cuando el
    efectivo cae fuera del rango, esa división sorprendió.
    """
    ine = ine_divisiones.copy()
    ine["Nombre"] = ine["Nombre"].str.strip().str.upper()
    d = divisiones.merge(ine[["Nombre", "Incidencia"]], on="Nombre", how="left")
    d["Etiqueta"] = d["Corto"].fillna(d["Nombre"])
    d = d.sort_values("Incidencia", key=lambda s: s.abs(), ascending=True)

    n_inst = len(divisiones.attrs.get("instituciones", []))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=d["Etiqueta"], x=d["Maximo"] - d["Minimo"], base=d["Minimo"], orientation="h",
        name=f"Rango mercado ({n_inst} inst.)",
        marker=dict(color="rgba(148,104,38,0.28)", line=dict(color=cfg.GOLD, width=1)),
        customdata=np.stack([d["Minimo"], d["Maximo"]], axis=1),
        hovertemplate="<b>%{y}</b><br>rango: %{customdata[0]:.3f} a %{customdata[1]:.3f} pp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=d["Etiqueta"], x=d["Promedio"], mode="markers", name="Promedio mercado",
        marker=dict(color=cfg.GOLD, size=9, symbol="line-ns", line=dict(width=2.5, color=cfg.GOLD)),
        hovertemplate="<b>%{y}</b> promedio: %{x:.3f} pp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=d["Etiqueta"], x=d["Incidencia"], mode="markers", name="Efectivo INE",
        marker=dict(color=cfg.EFECTIVO, size=11, symbol="diamond"),
        hovertemplate="<b>%{y}</b> efectivo: %{x:.3f} pp<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#9aa2b1", width=1))
    fig.update_layout(
        **cfg.LAYOUT_BASE, height=max(440, 32 * len(d) + 140),
        xaxis=dict(**cfg.GRID, title="Incidencia mensual · pp", ticksuffix=" pp"),
        yaxis=dict(**cfg.GRID, automargin=True),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
# VARIACIONES DEL MES (derivado de los analíticos)
# ═════════════════════════════════════════════════════════════════════

_ANALITICOS = [
    ("IPC General",       "General"),
    ("IPC SAE",           "SAE"),
    ("IPC Bienes",        "Bienes"),
    ("IPC Servicios",     "Servicios"),
    ("IPC Transables",    "Transables"),
    ("IPC No transables", "No transables"),
    ("IPC Alimentos",     "Alimentos"),
    ("IPC Energía",       "Energía"),
    ("IPC Frutas y verduras", "Frutas y verd."),
    ("IPC Vivienda",      "Vivienda"),
]


def build_variaciones_mes(analiticos: pd.DataFrame) -> go.Figure:
    """Gráfico N°2 del banco: la variación del último mes de cada agregado
    analítico, con el mes anterior al lado para ver el cambio."""
    a = analiticos.sort_values("Fecha")
    ult, ant = a.iloc[-1], a.iloc[-2]
    cols = [(c, e) for c, e in _ANALITICOS if c in a.columns]
    etiquetas = [e for _, e in cols]
    v_ult = [float(ult[c]) for c, _ in cols]
    v_ant = [float(ant[c]) for c, _ in cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=etiquetas, y=v_ant, name=f"{ant['Fecha']:%b %Y}",
        marker=dict(color="rgba(0,23,48,0.22)"),
        hovertemplate="<b>%{x}</b> mes anterior: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=etiquetas, y=v_ult, name=f"{ult['Fecha']:%b %Y}",
        marker=dict(color=[cfg.NAVY if v >= 0 else cfg.RED for v in v_ult]),
        text=[f"{v:+.1f}" for v in v_ult], textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{x}</b>: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **cfg.LAYOUT_BASE, barmode="group", bargap=0.3,
        xaxis=dict(**cfg.GRID, tickangle=-30),
        yaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#9aa2b1",
                   title="Variación mensual · %", ticksuffix="%"),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
# REGISTRO
# ═════════════════════════════════════════════════════════════════════

# fig_id -> (datasets que necesita, builder)
#
# Las claves de datasets son las de data.cargar_datasets(): las del notebook
# (analiticos, difusion, canasta, grupos, ine_divisiones), las del operador
# (expectativas, divisiones).
BUILDERS: dict[str, tuple[tuple[str, ...], Callable[..., go.Figure]]] = {
    # Esperado vs efectivo (Excel del operador)
    "esperado_vs_efectivo": (("expectativas", "analiticos"),   build_esperado_vs_efectivo),
    "incidencias_feedback": (("divisiones", "ine_divisiones"), build_incidencias_feedback),
    # Derivados del notebook
    "variaciones_mes":      (("analiticos",),                  build_variaciones_mes),
}

# Ninguna figura del IPC es base 100 recalculable.
REBASE_FIG_IDS: set[str] = set()


def _add_update_footer(fig, footer_text: str):
    """Fecha de datos + de generación, DENTRO de la figura. Funciona tanto
    sobre go.Figure como sobre el dict ya serializado del notebook."""
    anotacion = dict(
        text=footer_text, xref="paper", yref="paper", x=0, y=1,
        xanchor="left", yanchor="bottom", yshift=8, showarrow=False,
        font=dict(size=8.5, color="#9aa2b1", family="Barlow, Arial, sans-serif"),
    )
    if isinstance(fig, dict):
        fig.setdefault("layout", {}).setdefault("annotations", []).append(anotacion)
    else:
        fig.add_annotation(**anotacion)
    return fig


def _polish_hover(fig: go.Figure) -> go.Figure:
    for trace in fig.data:
        if trace.type in ("heatmap", "bar", "histogram", "treemap") or trace.hoverinfo == "skip" \
                or trace.hovertemplate:
            continue
        trace.update(hovertemplate=f"<b>{trace.name or ''}</b><br>%{{x}}: %{{y:,.2f}}<extra></extra>")
    return fig


def build_all_figures(datasets: dict[str, pd.DataFrame], *,
                      footer_text: str | None = None) -> tuple[dict[str, go.Figure], list[str]]:
    """Construye las figuras de BUILDERS que tengan sus datasets. Devuelve
    `(figuras, avisos)`; un builder que falle no tumba la corrida."""
    figuras: dict[str, go.Figure] = {}
    avisos: list[str] = []
    for fig_id, (necesita, builder) in BUILDERS.items():
        faltan = [d for d in necesita if d not in datasets or datasets[d].empty]
        if faltan:
            avisos.append(f"{fig_id}: falta {', '.join(faltan)}")
            continue
        try:
            figuras[fig_id] = builder(*(datasets[d] for d in necesita))
        except Exception as exc:
            avisos.append(f"{fig_id}: {type(exc).__name__} — {exc}")
    for fig in figuras.values():
        _polish_hover(fig)
        if footer_text:
            _add_update_footer(fig, footer_text)
    return figuras, avisos
