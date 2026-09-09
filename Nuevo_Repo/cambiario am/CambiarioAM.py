import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.cambiario import config as cfg
from src.cambiario import data, figures, layout, persist, render, tables


def main() -> None:
    data_path = data.resolve_data_path()
    df = data.load_data(data_path)
    intradia = data.load_intradia(data_path)
    carrytrade = data.load_carrytrade(data_path)

    monedas    = data.load_monedas()
    clp_intra  = data.load_clp_intra()
    fixing     = data.load_fixing()
    distclp    = data.load_distclp()
    monedas1m  = data.load_monedas1m()
    gammaproxy = data.load_gammaproxy()
    heatmap    = data.load_heatmap()

    data_fecha = data.fecha_corta_es(df["Fecha"].max())
    generado_ts = data.fecha_corta_es(datetime.now(), with_time=True)
    footer_text = f"Datos al {data_fecha}  ·  Generado {generado_ts}"

    figs = figures.build_all_figures(
        df,
        intradia=intradia, carrytrade=carrytrade,
        monedas=monedas, clp_intra=clp_intra, distclp=distclp,
        monedas1m=monedas1m, gammaproxy=gammaproxy,
        fixing=fixing, heatmap=heatmap,
        footer_text=footer_text,
    )
    figures_serialized = persist.serialize_figures(figs)

    market_table = tables.build_market_table(df)
    clp_stat = tables.build_clp_stat(df)
    candle_table = tables.build_candle_table(df)
    sr_tables = tables.build_sr_tables(df)

    nav_groups = layout.build_nav_groups(
        candle_table,
        sr_tables["clp"], sr_tables["cobre"], sr_tables["dxy"], sr_tables["nr"],
    )
    layout.annotate_layout(nav_groups, figures_serialized)

    persist.write_section_files(nav_groups, figures_serialized, cfg.OUTPUTS_FIGURES_DIR)
    persist.write_json(market_table, cfg.MARKET_TABLE_FILE)
    persist.write_json(clp_stat, cfg.CLP_STAT_FILE)

    title = "Informe Cambiario AM - DOMA"
    title_short = "Informe Cambiario AM"
    subtitle = "Departamento de Operaciones Mercado Abierto"

    html = render.render_html(
        nav_groups, figures_serialized, market_table, clp_stat,
        template_dir=cfg.TEMPLATE.parent,
        template_name=cfg.TEMPLATE.name,
        title=title, title_short=title_short, subtitle=subtitle,
    )
    cfg.OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"[OK] {len(figures_serialized)} figura(s) - {len(market_table)} indicadores -> {cfg.OUTPUT_HTML}")
    print(f"     1) Abrir {cfg.OUTPUT_HTML.name}, escribir el análisis, luego click 'Guardar versión final',")
    print(f"        soltar luego el .html descargado en {cfg.VERSIONES_FINALES_DIR}")
    print("     2) python generar_email.py  -> arma outputs/CambiarioAM.eml con esa versión adjunta")


if __name__ == "__main__":
    ruta_bandera = r"\\fileserver\VOLM\GMN\DOMA\Reportes HTML\Informe Cambiario\data\bandera.txt"
    if os.path.exists(ruta_bandera):
        p = argparse.ArgumentParser(description="Compila el dashboard Informe Cambiario AM.")
        p.add_argument("--open", action="store_true", help="Abrir HTML en el navegador")
        args = p.parse_args()
        main()
        if args.open:
            webbrowser.open(cfg.OUTPUT_HTML.resolve().as_uri())
        os.remove(ruta_bandera)
    else:
        sys.exit()