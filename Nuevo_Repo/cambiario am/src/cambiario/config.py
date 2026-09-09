
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parents[2]
TEMPLATES_DIR    = ROOT / "templates"
OUTPUTS_DIR      = ROOT / "outputs"
VERSIONES_FINALES_DIR = ROOT / "versiones_finales"

OUTPUT_HTML          = OUTPUTS_DIR / "CambiarioAM.html"
TEMPLATE             = TEMPLATES_DIR / "CambiarioAM_template.html"
OUTPUTS_FIGURES_DIR  = OUTPUTS_DIR / "figures"

OUTPUT_EML     = OUTPUTS_DIR / "CambiarioAM.eml"
EMAIL_TEMPLATE = TEMPLATES_DIR / "CambiarioAM_email_template.html"

MARKET_TABLE_FILE = OUTPUTS_DIR / "market_table.json"
CLP_STAT_FILE     = OUTPUTS_DIR / "clp_stat.json"

DATA_PATH = Path(r"\\fileserver\VOLM\GMN\DOMA\Reportes HTML\Informe Cambiario\data\Datos BI Informe Cambiario.xlsm")


GET_DATA_MODULE_PATH = r"d:\DOMA-DACE\isepulveda\Production\Modulos\Datos"

# ── Paleta ────────────────────────────────────────────────────────────
NAVY   = "#001730"
GOLD   = "#946826"
MAROON = "#85144b"
TEAL   = "#39CCCC"
CYAN   = "#7FDBFF"
BLUE   = "#0074D9"
GRAY   = "rgba(128,128,128,0.20)"
RED    = "#b42e20"

# Paleta de percentiles para gráficos S/R — saturada y sin colores claros,
# para que cada banda se distinga claramente sobre fondo blanco.
PCT_P10  = "#0c8599"
PCT_P25  = "#1f6feb"
PCT_P50  = "#c0392b"
PCT_P75  = "#8e44ad"
PCT_P90  = "#1e8449"
PCT_P100 = "#2c2c2c"

# ── Layouts base de Plotly ───────────────────────────────────────────
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
    # t=42, no 30: deja lugar arriba del área de ploteo para el footer de
    # fecha de datos/generación (figures._add_update_footer()). Se probó
    # primero abajo, pero la leyenda de los gráficos con muchas series
    # (monedas base100: 6-10 items) envuelve a 2-3 filas y tapaba el texto
    # — el margen superior, en cambio, no lo usa nada más en ningún gráfico.
    #
    # b=88 (no 40) + legend.y=-0.34 (no -0.15): con leyenda horizontal y
    # varias series (ej. Medias Móviles: 7 items -> 4 filas envueltas),
    # Plotly reserva el margen inferior según lo que la leyenda necesita, NO
    # según este número — bajarlo no sirve de nada si ya alcanza, y con los
    # valores viejos (b=40, y=-0.15) la leyenda quedaba pegada al eje X (8px
    # de aire, medido). Lo que sí mueve la aguja es legend.y: cuanto más
    # negativo, más margen total termina reservando Plotly, y ahí sí aparece
    # el respiro. Con estos valores: ~17-54px de aire según cuántas filas
    # tenga la leyenda (medido en varios gráficos), y de paso el ÁREA DE
    # PLOTEO queda con el mismo alto (250px) en todos los gráficos, tengan
    # leyenda de 1 fila o de 4.
    margin=dict(l=50, r=25, t=42, b=88),
    legend=dict(
        orientation='h', yanchor='top', y=-0.34, xanchor='center', x=0.5,
        font=dict(size=10), tracegroupgap=2,
    ),
)

data_select = dict(
    **GRID,
    rangeslider=dict(visible=False),
)

EMBED = dict(
    **LAYOUT_BASE,
    xaxis=data_select,
)

# Layout de los gráficos S/R: mismo margen base pero con más espacio a la
# derecha para los tags P10/P25/… que se agregan al final de cada línea.
SR_LAYOUT = {**LAYOUT_BASE, 'margin': dict(l=50, r=46, t=42, b=88)}
