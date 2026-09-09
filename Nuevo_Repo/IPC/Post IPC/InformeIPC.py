"""Publica el Informe IPC con la infraestructura del Informe Cambiario.

    python InformeIPC.py [--open]

No recalcula nada del análisis: lee lo que dejó el notebook (figuras.json y
Output IPC BI.xlsx) y lo que llenó el operador (data/entrada_operador/), y
arma el HTML con los comentarios, el anexo para pegar y "Guardar versión
final".

Flujo mensual completo:

    1) scripts/00_validar_entrada.py + 01_ejecutar_notebook.py
    2) llenar data/entrada_operador/Expectativas IPC.xlsx
    3) python InformeIPC.py --open      este script
    4) escribir el análisis y "Guardar versión final"
    5) python generar_email.py
"""
import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ipc import config as cfg
from ipc import data, figures, layout, persist, render, tables

def main() -> None:
    if not cfg.FIGURAS_NOTEBOOK.exists():
        sys.exit(f"[ERROR] Falta {cfg.FIGURAS_NOTEBOOK} — corré primero el notebook "
                 "(scripts/01_ejecutar_notebook.py).")

    datasets = data.cargar_datasets()
    fecha = tables.fecha_informe(datasets)
    footer_text = (f"IPC de {data.mes_largo_es(fecha)}")

    # Las 8 del notebook llegan hechas; las demás se construyen acá.
    figuras_nb, ultimo_mes = data.cargar_figuras_notebook()
    if ultimo_mes != fecha.strftime("%Y-%m"):
        sys.exit(f"[ERROR] Desfase: figuras.json es de {ultimo_mes} y el Excel de {fecha:%Y-%m}. "
                    "Vuelve a correr el notebook de punta a punta.")
    for fig in figuras_nb.values():
        figures._add_update_footer(fig, footer_text)
    figs, avisos = figures.build_all_figures(datasets, footer_text=footer_text)
    figures_serialized = {**figuras_nb, **persist.serialize_figures(figs)}

    nav_groups = layout.build_nav_groups()
    layout.annotate_layout(nav_groups, figures_serialized)

    persist.write_section_files(nav_groups, figures_serialized, cfg.OUTPUTS_FIGURES_DIR)
    persist.write_json({
        "ultimo_periodo": int(ultimo_mes.replace("-", "")),
        "ultimo_mes": ultimo_mes,
        "fecha": data.mes_largo_es(fecha),
        "fecha_corrida": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_figuras": len(figures_serialized),
    }, cfg.ESTADO_CORRIDA)

    portada = tables.build_portada(datasets)

    payload = persist.compact_payload(
        figures_serialized, rebase_ids=figures.REBASE_FIG_IDS, nav_groups=nav_groups
    )
    # Las tablas de detalle viajan junto a las figuras; ui.js las arma.
    payload["tablas"] = tables.build_tablas(datasets)
    html = render.render_html(
        nav_groups, payload, portada,
        template_dir=cfg.TEMPLATE.parent, template_name=cfg.TEMPLATE.name,
        # El mes del dato va en el título: la fecha del masthead es la de hoy
        # (la pone ui.js) y en el IPC lo que se lee es el mes.
        title=f"Informe Post IPC - {data.mes_largo_es(fecha)}",
        title_short=f"Informe Post IPC {data.mes_largo_es(fecha)}",
        subtitle="Departamento de Operaciones Mercado Abierto",
        footer_text=footer_text,
    )
    cfg.OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_HTML.write_text(html, encoding="utf-8")

    # Se cuentan las que se ven, no las que existen: una figura del notebook
    # sin slot en layout.py no se publica ni pesa.
    total_slots = sum(1 for g in nav_groups for it in g["items"] for s in it["slots"] if s.get("fig_id"))
    print(f"[OK] {len(payload['figs'])}/{total_slots} figura(s) · IPC de {data.mes_largo_es(fecha)} "
          f"+ {len(payload['tablas'])} tabla(s) -> {cfg.OUTPUT_HTML} "
          f"({cfg.OUTPUT_HTML.stat().st_size / 1_048_576:.1f} MB)")
    for a in avisos:
        print(f"     [PENDIENTE] {a}")
    print(f"     1) Abrí {cfg.OUTPUT_HTML.name}, escribí el análisis, click 'Guardar versión final',")
    print(f"        soltá el .html descargado en {cfg.VERSIONES_FINALES_DIR}")
    print(f"     2) python generar_email.py  -> arma {cfg.OUTPUT_EML}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Publica el Informe IPC.")
    p.add_argument("--open", action="store_true", help="Abrir HTML en el navegador")
    args = p.parse_args()
    main()
    if args.open:
        webbrowser.open(cfg.OUTPUT_HTML.resolve().as_uri())
