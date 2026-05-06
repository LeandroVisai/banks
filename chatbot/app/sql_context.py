"""
Series de tiempo macro como contexto directo para el prompt del LLM.

Consulta el Data Warehouse (SQL Server) en tiempo real vía dw_store,
usando el catálogo data_pipeline/series_catalog.yaml como fuente de
metadata (id, name, unit, sql_table, sql_column, etc.).

El resultado se inyecta bajo <datos_historicos>...</datos_historicos>
en el prompt, sin necesidad de vectorización ni parquets intermedios.
"""
from __future__ import annotations

import logging
from typing import Optional

from data_pipeline import dw_store

from .query_analysis import QueryAnalysis
from .settings import settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Variable de taxonomía → series_ids representativas
# ─────────────────────────────────────────────────────────────────────────────

VARIABLE_TO_SERIES: dict[str, list[str]] = {
    "TASA_INTERES": [
        "spc_3m_clp", "spc_1y_clp", "spc_2y_clp",
        "spread_mipr_3m", "spread_mipr_12m",
        "sofr_3m", "sofr_12m",
    ],
    "TASAS_LARGO_PLAZO": [
        "btp_5y", "btp_10y",
        "btu_5y", "btu_10y",
        "ust_5y", "ust_10y",
        "spc_5y_clp", "spc_10y_clp",
    ],
    "TASA_INTERES_MERCADO": [
        "spread_dap_swap_1m_clp", "spread_dap_swap_3m_clp",
        "spread_prime_swap_1m_clp", "spread_prime_swap_3m_clp",
        "tib_tasa",
    ],
    "TIPO_CAMBIO": [
        "usdclp", "dxy", "monedas_latam", "monedas_comparables",
    ],
    "COMMODITIES": [
        "cobre",
    ],
}

_DEFAULT_LIMIT: dict[str, int] = {
    "diario":     45,
    "mensual":    18,
    "trimestral":  8,
    "anual":       5,
}

_MAX_SERIES_IN_CONTEXT = 6


# ─────────────────────────────────────────────────────────────────────────────
# Resolución de series por variable
# ─────────────────────────────────────────────────────────────────────────────

def _series_for(variables: list[str]) -> list[str]:
    out: list[str] = []
    for var in variables:
        for sid in VARIABLE_TO_SERIES.get(var, []):
            if sid not in out:
                out.append(sid)
            if len(out) >= _MAX_SERIES_IN_CONTEXT:
                return out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

async def build_context(analysis: QueryAnalysis) -> tuple[str, list[dict]]:
    """
    Devuelve (bloque_texto, lista_series_usadas).

    Consulta el DW en tiempo real. Si DW_SERVER no está configurado,
    retorna vacío con una advertencia en lugar de romper el chatbot.
    """
    if not analysis.needs_historical_data:
        return "", []

    series_ids = _series_for(analysis.variables)
    if not series_ids:
        return "", []

    if not settings.get_data_path:
        log.warning("GET_DATA_PATH no configurado — datos históricos no disponibles")
        return "", []
    get_data_path = settings.get_data_path

    blocks: list[str] = []
    used: list[dict] = []

    for sid in series_ids:
        meta = dw_store.get_series_meta(sid, catalog_path=settings.catalog_path)
        if meta is None:
            log.debug("Serie no en catálogo: %s", sid)
            continue

        limit = _DEFAULT_LIMIT.get(meta.get("frequency", "diario"), 45)
        use_limit = limit if not (analysis.date_from and analysis.date_to) else None

        try:
            df = await dw_store.fetch_series(
                sid,
                get_data_path=get_data_path,
                catalog_path=settings.catalog_path,
                date_from=analysis.date_from,
                date_to=analysis.date_to,
                limit=use_limit,
            )
        except Exception as e:
            log.warning("fetch_series(%s) falló: %s", sid, e)
            continue

        if df.empty:
            continue

        block = dw_store.format_table(meta, df)
        if not block:
            continue
        blocks.append(block)
        used.append({
            "series_id": sid,
            "series_name": meta["name"],
            "unit": meta.get("unit", ""),
            "frequency": meta.get("frequency", ""),
            "n_observations": len(df),
            "first_date": df["date"].iloc[0].date().isoformat(),
            "last_date": df["date"].iloc[-1].date().isoformat(),
        })

    if not blocks:
        return "", []
    return "\n\n".join(blocks), used
