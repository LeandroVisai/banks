"""
Series de tiempo macro como contexto directo (sin vectorización).

Toma el `QueryAnalysis` (variables + rango de fechas) y devuelve un bloque
de texto tabular legible por el LLM, listo para inyectar bajo
`<datos_historicos>...</datos_historicos>` en el prompt.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import db
from .query_analysis import QueryAnalysis

log = logging.getLogger(__name__)


# Variable de taxonomía → series_id que la representan
VARIABLE_TO_SERIES: dict[str, list[str]] = {
    "TASA_INTERES":              ["tpm", "exp_tpm_12m", "fed_funds_rate"],
    "INFLACION":                 ["ipc_anual", "ipc_mensual", "ipcx_anual",
                                  "exp_inflacion_12m", "exp_inflacion_24m"],
    "PIB":                       ["pib_trimestral", "imacec_mensual"],
    "TIPO_CAMBIO":               ["usdclp_spot"],
    "EXPECTATIVAS_INFLACIONARIAS": ["exp_inflacion_12m", "exp_inflacion_24m"],
    "TASAS_LARGO_PLAZO":         ["bcu_5y", "btp_5y"],
    "COMMODITIES":               ["precio_cobre", "precio_petroleo_wti"],
    "MERCADO_LABORAL":           ["desempleo"],
    "RIESGO_CREDITO":            ["cds_chile_5y"],
}

# Cuántas observaciones traer según frecuencia (cuando no hay rango)
_DEFAULT_LIMIT: dict[str, int] = {
    "diario":     30,
    "mensual":    18,
    "trimestral":  8,
    "anual":       5,
}


def _series_for(variables: list[str]) -> list[str]:
    out: list[str] = []
    for var in variables:
        for sid in VARIABLE_TO_SERIES.get(var, []):
            if sid not in out:
                out.append(sid)
    return out


def _fmt_value(value, unit: str) -> str:
    if value is None:
        return "N/D"
    val = float(value)
    if any(u in unit for u in ("CLP", "USD/lb", "USD/barril", "pb")):
        return f"{val:,.0f}"
    return f"{val:.2f}"


def _format_block(meta: dict, rows: list[dict]) -> str:
    if not rows:
        return ""
    name = meta["series_name"]
    unit = meta["unit"]
    src = meta["source"]
    freq = meta["frequency"]

    header = f"{name}  [fuente: {src} | {freq} | unidad: {unit}]"
    lines = [header, "-" * len(header)]
    for r in rows:
        d = r["date"]
        date_str = d.strftime("%Y-%m") if freq in ("mensual", "trimestral") else d.isoformat()
        val_str = _fmt_value(r["value"], unit)
        note = f"  ({r['notes']})" if r.get("notes") else ""
        lines.append(f"  {date_str}  {val_str}{note}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

async def build_context(analysis: QueryAnalysis) -> tuple[str, list[dict]]:
    """
    Devuelve (bloque_texto, lista_series_usadas).

    `lista_series_usadas` es metadata para persistir en `chat_messages.historical_series`.
    """
    if not analysis.needs_historical_data:
        return "", []

    series_ids = _series_for(analysis.variables)
    if not series_ids:
        return "", []

    metas = await db.fetch_series_meta(series_ids)
    blocks: list[str] = []
    used: list[dict] = []

    for sid in series_ids:
        meta = metas.get(sid)
        if not meta:
            continue
        limit = _DEFAULT_LIMIT.get(meta["frequency"], 18)
        rows = await db.fetch_series_rows(
            sid, limit=limit,
            date_from=analysis.date_from,
            date_to=analysis.date_to,
        )
        if not rows:
            continue

        block = _format_block(meta, rows)
        if not block:
            continue
        blocks.append(block)
        used.append({
            "series_id": sid,
            "series_name": meta["series_name"],
            "unit": meta["unit"],
            "frequency": meta["frequency"],
            "n_observations": len(rows),
            "first_date": rows[0]["date"].isoformat(),
            "last_date": rows[-1]["date"].isoformat(),
        })

    if not blocks:
        return "", []
    return "\n\n".join(blocks), used
