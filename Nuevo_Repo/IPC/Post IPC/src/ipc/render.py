"""Render del informe: junta el template Jinja con el CSS, los JS y el
payload compacto, y devuelve UN archivo HTML autocontenido.

En el código fuente el informe está partido en cinco archivos
(`informe.html`, `informe.css`, `chart.js`, `ui.js`, `save.js`) porque un
monolito de 2.300 líneas era imposible de mantener. En la salida son uno
solo: el analista se descarga la "versión final" y la manda adjunta, así que
no puede depender de archivos vecinos.
"""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from plotly.offline import get_plotlyjs_version

from . import theme

# El orden en que aparecen los botones, de la ventana más larga a la más
# corta. Es la misma lista para el selector global y para el de cada
# gráfico — se define una vez acá y el template la usa en los dos lugares.
PERIODS: list[tuple[str, str]] = [
    ("todo", "Todo"),
    ("1a",   "1A"),
    ("ytd",  "YTD"),
    ("6m",   "6M"),
    ("3m",   "3M"),
    ("1m",   "1M"),
    ("mtd",  "MTD"),
]

_ASSETS = ("informe.css", "chart.js", "ui.js", "save.js")


def _leer_assets(template_dir: Path) -> dict[str, str]:
    return {nombre: (template_dir / nombre).read_text(encoding="utf-8") for nombre in _ASSETS}


def render_html(
    nav_groups: list,
    payload: dict,
    portada: dict,
    *,
    template_dir: Path,
    template_name: str,
    title: str,
    title_short: str,
    subtitle: str,
    footer_text: str,
) -> str:
    """`portada` es lo único que cambia de un informe a otro en el template.

    Es un dict con bloques opcionales — el template dibuja los que vengan:

        comentarios   ["Escenario Internacional", ...]  títulos de los
                      bloques editables (cambiario manda 3; el IPC los suyos)
        hero          {label, unit, value, fecha, stats:[{label, value, cls}]}
                      el número grande sobre navy (cambiario: USD/CLP)
        kpis          [{label, value, sub, cls}]  fila de tarjetas (IPC)
        mercado       [filas del snapshot]  + mercado_titulo opcional

    save.js lee estos mismos bloques del DOM para componer el correo, así que
    un informe nuevo no tiene que tocar ni el template ni el JS: alcanza con
    armar este dict.
    """
    assets = _leer_assets(template_dir)

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template(template_name)
    return tpl.render(
        title=title,
        title_short=title_short,
        subtitle=subtitle,
        footer_text=footer_text,
        nav_groups=nav_groups,
        periods=PERIODS,
        portada=portada,
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        # La versión del CDN se toma de la plotly.py instalada en vez de estar
        # escrita a mano: es la que generó las figuras, así que no pueden
        # desalinearse al actualizar el requirements.
        plotly_js_version=get_plotlyjs_version(),
        css=theme.css_root() + assets["informe.css"],
        js_chart=assets["chart.js"],
        js_ui=assets["ui.js"],
        js_save=assets["save.js"],
    )
