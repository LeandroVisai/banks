"""Endpoint /v1/kpis — indicadores económicos clave desde parquets reales."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from banks_rag.config.paths import ROOT

router = APIRouter(prefix="/v1", tags=["kpis"])

_PARQUET_BASE = ROOT / "data_pipeline" / "parquet"


def _query(path: Path, sql_template: str) -> list[dict]:
    import duckdb
    sql = sql_template.replace("{p}", str(path))
    with duckdb.connect(":memory:") as con:
        rel = con.sql(sql)
        cols = [d[0] for d in rel.description]
        return [{c: v for c, v in zip(cols, row)} for row in rel.fetchall()]


def _entry(label: str, value: str, delta: str, direction: str,
           fecha: str | None, available: bool) -> dict[str, Any]:
    return {
        "label": label, "value": value, "delta": delta,
        "dir": direction, "fecha": fecha, "available": available,
    }


def _pct_dir(pct: float | None) -> str:
    if pct is None:
        return "neu"
    return "up" if pct >= 0 else "down"


@router.get("/kpis", summary="Indicadores KPI desde parquets")
async def get_kpis() -> dict[str, Any]:
    """Retorna los indicadores del KPI strip. Valores no disponibles retornan value='—'."""
    kpis: list[dict[str, Any]] = []

    # TPM — no hay parquet con el nivel actual; viene de Comunicados del BCCh
    kpis.append(_entry("TPM", "—", "—", "neu", None, False))

    # USD/CLP
    try:
        p = _PARQUET_BASE / "clp_monto.parquet"
        rows = await asyncio.to_thread(
            _query, p,
            "SELECT Fecha, CLP FROM read_parquet('{p}') ORDER BY Fecha DESC LIMIT 2",
        )
        if rows:
            last = rows[0]["CLP"]
            prev = rows[1]["CLP"] if len(rows) > 1 else None
            pct = (last - prev) / prev * 100 if prev else None
            kpis.append(_entry(
                "USD/CLP", f"{last:,.1f}",
                f"{pct:+.1f}%" if pct is not None else "—",
                _pct_dir(pct), str(rows[0]["Fecha"])[:10], True,
            ))
        else:
            kpis.append(_entry("USD/CLP", "—", "—", "neu", None, False))
    except Exception:
        kpis.append(_entry("USD/CLP", "—", "—", "neu", None, False))

    # IPC 12m — expectativas de inflación
    try:
        p = _PARQUET_BASE / "expectativas_inflacion.parquet"
        rows = await asyncio.to_thread(
            _query, p,
            "SELECT Fecha, Valor FROM read_parquet('{p}') "
            "WHERE Serie = '12M' ORDER BY Fecha DESC LIMIT 2",
        )
        if rows:
            last = rows[0]["Valor"]
            prev = rows[1]["Valor"] if len(rows) > 1 else None
            pp = last - prev if prev is not None else None
            kpis.append(_entry(
                "IPC 12m", f"{last:.1f}%",
                f"{pp:+.1f}pp" if pp is not None else "—",
                _pct_dir(pp), str(rows[0]["Fecha"])[:10], True,
            ))
        else:
            kpis.append(_entry("IPC 12m", "—", "—", "neu", None, False))
    except Exception:
        kpis.append(_entry("IPC 12m", "—", "—", "neu", None, False))

    # Cobre (USD/lb — parquet almacena USc/lb, dividir por 100)
    try:
        p = _PARQUET_BASE / "cobre_dxy.parquet"
        rows = await asyncio.to_thread(
            _query, p,
            "SELECT Fecha, Cobre / 100.0 AS Cobre FROM read_parquet('{p}') ORDER BY Fecha DESC LIMIT 2",
        )
        if rows:
            last = rows[0]["Cobre"]
            prev = rows[1]["Cobre"] if len(rows) > 1 else None
            pct = (last - prev) / prev * 100 if prev else None
            kpis.append(_entry(
                "Cobre", f"{last:.3f}",
                f"{pct:+.1f}%" if pct is not None else "—",
                _pct_dir(pct), str(rows[0]["Fecha"])[:10], True,
            ))
        else:
            kpis.append(_entry("Cobre", "—", "—", "neu", None, False))
    except Exception:
        kpis.append(_entry("Cobre", "—", "—", "neu", None, False))

    # IPSA — sin parquet disponible
    kpis.append(_entry("IPSA", "—", "—", "neu", None, False))

    return {"kpis": kpis}
