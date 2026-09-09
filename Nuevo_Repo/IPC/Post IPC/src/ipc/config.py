
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
# El informe IPC es HÍBRIDO: el análisis sigue viviendo en el notebook, que
# deja outputs/figuras.json y outputs/Output IPC BI.xlsx como siempre. Lo
# que se agrega es la capa de publicación, con la misma infraestructura que
# el Informe Cambiario (templates, correo, versión final).
#
#     notebooks/01.Analisis IPC.ipynb   ──►  outputs/figuras.json + Output IPC BI.xlsx
#     data/entrada_operador/            ──►  lo que llena el operador (expectativas)
#                                              │
#                                              ▼
#     InformeIPC.py                     ──►  dashboards/InformeIPC.html
#     "Guardar versión final"           ──►  versiones_finales/
#     generar_email.py                  ──►  correos/InformeIPC.eml
ROOT          = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "templates"
DATA_DIR      = ROOT / "data"

# Lo que deja el notebook (no se toca desde acá)
FIGURAS_NOTEBOOK = ROOT / "outputs" / "figuras.json"
EXCEL_NOTEBOOK   = ROOT / "outputs" / "Output IPC BI.xlsx"
INE_2023         = DATA_DIR / "ipc2023.xlsx"

# Lo que llena el operador
EXPECTATIVAS = DATA_DIR / "entrada_operador" / "Expectativas IPC.xlsx"

# Lo que se publica
OUTPUTS_DIR           = ROOT / "dashboards"
VERSIONES_FINALES_DIR = ROOT / "versiones_finales"
CORREOS_DIR           = ROOT / "correos"
CACHE_DIR             = ROOT / "_cache"

OUTPUT_HTML   = OUTPUTS_DIR / "InformeIPC.html"
TEMPLATE      = TEMPLATES_DIR / "informe.html"
OUTPUT_EML    = CORREOS_DIR / "InformeIPC.eml"

OUTPUTS_FIGURES_DIR = CACHE_DIR / "figures"
# Estado de la última corrida: el mes publicado y la fecha del asunto del correo.
# Lo escribe InformeIPC.py; lo leen generar_email.py y 00_validar_entrada.py
# (para saber si el archivo del INE que dejaste en data/entrada/ es un mes nuevo).
ESTADO_CORRIDA      = ROOT / "outputs" / "estado_corrida.json"

# ── Paleta ────────────────────────────────────────────────────────────
from .theme import (  # noqa: F401  (re-export)
    NAVY, GOLD, MAROON, TEAL, CYAN, BLUE, GRAY, RED,
    PCT_P10, PCT_P25, PCT_P50, PCT_P75, PCT_P90, PCT_P100,
)

# ── Layouts base de Plotly ───────────────────────────────────────────
# Idénticos a los de cambiario: los dos informes salen del mismo
# departamento y se ven iguales. Ver la nota larga en cambiariobanco/config.py
# sobre los márgenes y la leyenda.
GRID = dict(showgrid=False, zeroline=False)

LAYOUT_BASE = dict(
    template='plotly_white',
    font=dict(family='Arial, sans-serif', size=12, color='#333333'),
    hovermode='closest',
    hoverlabel=dict(
        bgcolor='#ffffff',
        bordercolor=GOLD,
        font=dict(family='Barlow, Arial, sans-serif', size=12, color=NAVY),
        namelength=-1,
    ),
    margin=dict(l=50, r=25, t=42, b=88),
    legend=dict(
        orientation='h', yanchor='top', y=-0.34, xanchor='center', x=0.5,
        font=dict(size=10), tracegroupgap=2,
    ),
)

data_select = dict(**GRID, rangeslider=dict(visible=False))

# Colores de las fuentes de expectativas. Se definen una vez para que el
# gráfico de barras, el histograma y cualquier gráfico nuevo pinten igual a
# la misma fuente.
FUENTES_EXPECTATIVA = {
    "Seguros":   "#946826",   # GOLD
    "EOF":       "#0074D9",   # BLUE
    "EEE":       "#39CCCC",   # TEAL
    "Bloomberg": "#85144b",   # MAROON
}
EFECTIVO = "#001730"          # NAVY
