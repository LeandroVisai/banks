from __future__ import annotations


def build_nav_groups(
    candle_table: dict,
    clp_sr_table: dict,
    cobre_sr_table: dict,
    dxy_sr_table: dict,
    nr_sr_table: dict,
) -> list[dict]:
    """Navegación e insert de ids (para agregar más graficos).

    - cols: columnas base de la grilla
    - hero: el primer gráfico con figura queda a ancho completo (destacado)
    - rebase: True marca los gráficos base 100 que deben re-rebasearse al
      cambiar de período (ver figures.fig_base100_dynamic y el JS
      rebaseChart() en el template).
    """
    return [
        {"label": "Resumen", "items": [
            {"id": "portada", "label": "Drivers del día", "badge": "Overview", "cols": 2, "slots": [
                {"label": "Variación Monedas en el día", "fig_id": "variacion_dia"},
                {"label": "Monedas Intradía",           "fig_id": "clp_intradia"},
                {"label": "Evolución tipo de cambio",  "fig_id": "clp_candlestick", "table": candle_table},
            ]},
        ]},
        {"label": "Tipo de Cambio", "items": [
            {"id": "clp_tecnico", "label": "CLP - Análisis", "badge": "LIVE", "cols": 2, "slots": [
                {"label": "Medias Móviles",   "fig_id": "clp_medias_moviles"},
                {"label": "Bollinger CLP",    "fig_id": "clp_bollinger"},
                {"label": "RSI CLP",          "fig_id": "clp_rsi"},
                {"label": "S/R CLP",          "fig_id": "clp_sr", "table": clp_sr_table},
                {"label": "Volatilidad CLP",  "fig_id": "vol_clp"},
                {"label": "Fixing de la banca", "fig_id": "fixing"},
                {"label": "Histograma CLP",   "fig_id": "clp_dist"},
            ]},
        ]},
        {"label": "No Residentes", "items": [
            {"id": "no_residentes", "label": "No Residentes", "badge": "LIVE", "cols": 2, "hero": True, "slots": [
                {"label": "Derivados",     "fig_id": "clp_nr"},
                {"label": "Vol 1W",        "fig_id": "vol_1w"},
                {"label": "Punta forward", "fig_id": "fwd_puntas"},
                {"label": "S/R NR",        "fig_id": "pos_nr_sr", "table": nr_sr_table},
            ]},
        ]},
        {"label": "Monedas", "items": [
            {"id": "monedas", "label": "Monedas & Carry", "badge": "LIVE", "cols": 2, "hero": True, "slots": [
                {"label": "Monedas LATAM",        "fig_id": "fig_latam",       "rebase": True},
                {"label": "Monedas Emergentes",   "fig_id": "fig_emerg",       "rebase": True},
                {"label": "Monedas Commodities",  "fig_id": "fig_comparables", "rebase": True},
                {"label": "Monedas G10",          "fig_id": "fig_g10",         "rebase": True},
                {"label": "Monedas vs USD 1 mes", "fig_id": "variacion_1mes"},
                {"label": "Carry trade",          "fig_id": "carry_trade"},
            ]},
        ]},
        {"label": "Drivers", "items": [
            {"id": "cobre", "label": "Cobre", "badge": "LIVE", "cols": 2, "hero": True, "slots": [
                {"label": "Cobre vs CLP", "fig_id": "clp_cobre"},
                {"label": "S/R Cobre",    "fig_id": "cobre_sr", "table": cobre_sr_table},
                {"label": "RSI Cobre",    "fig_id": "rsi_cobre"},
                {"label": "Inventarios COMEX",   "fig_id": "inv_comex"},
                {"label": "Inventarios Londres", "fig_id": "inv_lme"},
                {"label": "Spread COMEX-LME",    "fig_id": "cobre_ny_spread"},
            ]},
            {"id": "dxy", "label": "Dólar - DXY", "badge": "LIVE", "cols": 2, "hero": True, "slots": [
                {"label": "DXY vs CLP", "fig_id": "clp_dxy"},
                {"label": "S/R DXY",    "fig_id": "dxy_sr", "table": dxy_sr_table},
                {"label": "RSI DXY",    "fig_id": "fig_rsi_dxy"},
            ]},
        ]},
        {"label": "Expectativas", "items": [
            {"id": "expectativas", "label": "Expectativas", "badge": "LIVE", "cols": 2, "slots": [
                {"label": "Expectativas de TPM US - CL",       "fig_id": "tpm_spread"},
                {"label": "Puntas FWD Chile",                  "fig_id": "fwd_puntas12m"},
                {"label": "Curva de contratos: Cobre COMEX",   "fig_id": "contrato_hga"},
                {"label": "Curva de contratos: Cobre Londres", "fig_id": "contrato_lma"},
                {"label": "Curva de contratos: Petróleo",      "fig_id": "contrato_cl1"},
            ]},
        ]},
        {"label": "Otros", "items": [
            {"id": "otros", "label": "Otros", "badge": "LIVE", "cols": 2, "slots": [
                {"label": "Gamma Proxy",         "fig_id": "gamma_plot"},
                {"label": "Heatmap Gamma Proxy", "fig_id": "heatmap_gp"},
                {"label": "Tasas implícitas",    "fig_id": "implicitas"},
                {"label": "Tipo de Cambio Real", "fig_id": "tcr"},
                {"label": "Terms Of Trade GS",   "fig_id": "tot_gs"},
                {"label": "Terms Of Trade CT",   "fig_id": "tot_ct"},
            ]},
        ]},
    ]


_DEFAULT_CARD_HEIGHT = 440


def annotate_layout(groups: list, figs: dict) -> None:
    for group in groups:
        for item in group["items"]:
            live, pending = [], []
            for slot in item["slots"]:
                slot["pending"] = slot.get("fig_id") not in figs
                (pending if slot["pending"] else live).append(slot)
                if not slot["pending"]:
                    fig_height = figs[slot["fig_id"]].get("layout", {}).get("height")
                    if fig_height and fig_height > _DEFAULT_CARD_HEIGHT:
                        slot["card_height"] = fig_height

            rest = live
            if item.get("hero") and live:
                live[0]["wide"] = True
                rest = live[1:]
            if len(rest) % 2 == 1:
                rest[-1]["wide"] = True

            item["slots"] = live + pending
