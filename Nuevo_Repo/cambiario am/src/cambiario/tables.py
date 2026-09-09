"""Tablas y stats resumen del Informe (portada, S/R, candlestick)."""
from __future__ import annotations

import pandas as pd

from . import data

_DRIVER_CLP = {
    "Cobre":        +1,
    "DXY":          -1,
    "Dif. SPC–OIS": +1,
    "Petróleo":     -1,
    "Pto. Fwd 30D":  0,
}


def build_market_table(df: pd.DataFrame) -> list[dict]:
    d = df.dropna(subset=["CLP Cierre"]).reset_index(drop=True)
    spread_spc_ois = (d["Misceláneos SPC 1Y"] - d["Misceláneos OIS 1Y"]) * 100

    # modo: "pct" = variación porcentual | "bp" = diff en bp | "pts" = diff en puntos
    series_config = [
        ("Cobre",        "Comex · USD/lb",   d["Drivers Precio Cobre HGA"]/100, 2, "pct"),
        ("DXY",          "Índice dólar",      d["Drivers DXY "],             2, "pct"),
        ("Dif. SPC–OIS 1Y", "Spread tasa · pb", spread_spc_ois,                1, "bp"),
        ("Petróleo",     "WTI · USD/bbl",  d["Drivers Petróleo"],         2, "pct"),
        ("Pto. Fwd 30D", "CLP · puntos",     d["Puntos Forward 1M"],        2, "pts"),
    ]

    market_table = []
    for nombre, unidad, serie, dec, modo in series_config:
        idx = serie.last_valid_index()
        fmt = f"{{:.{dec}f}}"
        ultimo = float(serie.iloc[idx]) if idx is not None else None

        prev1_ok = idx is not None and idx >= 1 and not pd.isna(serie.iloc[idx - 1])
        prev5_ok = idx is not None and idx >= 5 and not pd.isna(serie.iloc[idx - 5])

        if modo == "pct":
            vd_raw = (float(serie.iloc[idx]) / float(serie.iloc[idx - 1]) - 1) * 100 if prev1_ok else None
            vs_raw = (float(serie.iloc[idx]) / float(serie.iloc[idx - 5]) - 1) * 100 if prev5_ok else None
            fmt_v  = lambda v, d=dec: f"{v:+.{d}f}%"
        else:
            vd_raw = float(serie.iloc[idx]) - float(serie.iloc[idx - 1]) if prev1_ok else None
            vs_raw = float(serie.iloc[idx]) - float(serie.iloc[idx - 5]) if prev5_ok else None
            sfx    = " bp" if modo == "bp" else " pts"
            fmt_v  = lambda v, d=dec, s=sfx: f"{v:+.{d}f}{s}"

        clp_dir = _DRIVER_CLP.get(nombre, 0)

        def _clp_cls(raw, _dir=clp_dir):
            if raw is None: return ""
            if _dir == 0: return "mar"
            return "pos" if raw * _dir > 0 else "neg"

        if vd_raw is not None and clp_dir != 0:
            estado, estado_cls = ("↑ Aprecia", "pill-pos") if vd_raw * clp_dir > 0 else ("↓ Deprecia", "pill-neg")
        elif vd_raw is not None:
            estado, estado_cls = "Neutral", "pill-mar"
        else:
            estado, estado_cls = "—", ""

        market_table.append({
            "nombre":     nombre,
            "unidad":     unidad,
            "ultimo":     fmt.format(ultimo) if ultimo is not None else "—",
            "var_dia":    fmt_v(vd_raw) if vd_raw is not None else "—",
            "var_sem":    fmt_v(vs_raw) if vs_raw is not None else "—",
            "estado":     estado,
            "estado_cls": estado_cls,
            "vd_cls":     _clp_cls(vd_raw),
            "vs_cls":     _clp_cls(vs_raw),
        })
    return market_table


def build_clp_stat(df: pd.DataFrame) -> dict:
    df = df.rename(columns={"CLP Cierre": "CLP"})

    clp_s = df.dropna(subset=["CLP"]).reset_index(drop=True)
    ci = clp_s["CLP"].last_valid_index()
    clp_last = float(clp_s["CLP"].iloc[ci])

    clp_vd = (clp_last / float(clp_s["CLP"].iloc[ci - 1]) - 1) * 100 \
        if ci >= 1 and not pd.isna(clp_s["CLP"].iloc[ci - 1]) else None

    clp_vs = (clp_last / float(clp_s["CLP"].iloc[ci - 5]) - 1) * 100 \
        if ci >= 5 and not pd.isna(clp_s["CLP"].iloc[ci - 5]) else None

    return {
        "value":  f"{clp_last:.2f}",
        "fecha":  clp_s["Fecha"].iloc[ci].strftime("%d %b %Y"),
        "vd_fmt": f"{clp_vd:+.2f}%" if clp_vd is not None else "—",
        "vs_fmt": f"{clp_vs:+.2f}%" if clp_vs is not None else "—",
        "vd_cls": ("neg" if clp_vd > 0 else "pos") if clp_vd is not None else "",
        "vs_cls": ("neg" if clp_vs > 0 else "pos") if clp_vs is not None else "",
    }


def sr_table(levels: list[tuple[str, float]], fmt: str) -> dict:
    """Tira compacta de percentiles P10..P100 para no depender solo de la
    leyenda oculta del gráfico (ver figures._pct_lines/_add_pct_labels).

    "kind": "pct" -> el template la renderiza como tira horizontal de chips
    (una fila, ~40px) en vez de una <table> vertical de 7 filas (~200px);
    ver .pct-strip en CambiarioAM_template.html. `fmt` viaja en la tabla
    para que el JS del slider (S/R dinámico) sepa con qué formato re-mostrar
    los niveles recalculados sin tener que adivinarlo del texto ya formateado.
    """
    return {
        "kind": "pct",
        "fmt": fmt,
        "headers": ["Percentil", "Nivel"],
        "rows": [[p, format(v, fmt)] for p, v in levels],
    }


def build_candle_table(df: pd.DataFrame) -> dict:
    candle = data.candlestick_window(df, months=2)
    last_day = candle.dropna(subset=['CLP Max', 'CLP minimo']).iloc[-1]
    soporte, resistencia = data.support_resistance(candle['CLP Cierre'])

    return {
        "headers": ["Indicador", "Valor"],
        "rows": [
            ["Máximo del día", f"{last_day['CLP Max']:,.2f}"],
            ["Mínimo del día", f"{last_day['CLP minimo']:,.2f}"],
            ["Soporte",        f"{soporte:,.2f}"],
            ["Resistencia",    f"{resistencia:,.2f}"],
        ],
    }


def build_sr_tables(df: pd.DataFrame) -> dict[str, dict]:
    """Un `sr_table()` por indicador, calculado con el mismo percentile_summary()
    que usan los gráficos S/R correspondientes en figures.py."""
    clp   = data.percentile_summary(df, 'Fecha', 'CLP Cierre')
    cobre = data.percentile_summary(df, 'Fecha', 'Drivers Precio Cobre HGA')
    dxy   = data.percentile_summary(df, 'Fecha', 'Drivers DXY ')
    nr    = data.percentile_summary(
        df, 'Fecha', 'No Residentes Posición OffShore',
        transform=lambda s: s * -1, output_col='CLPBODM Index',
    )

    def _levels(summary):
        return [("P10", summary['p10']), ("P25", summary['p25']), ("P50", summary['p50']),
                ("P75", summary['p75']), ("P90", summary['p90']), ("P100", summary['p100'])]

    return {
        "clp":   sr_table(_levels(clp), ",.0f"),
        "cobre": sr_table(_levels(cobre), ".2f"),
        "dxy":   sr_table(_levels(dxy), ".2f"),
        "nr":    sr_table(_levels(nr), ",.0f"),
    }
