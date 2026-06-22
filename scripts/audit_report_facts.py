#!/usr/bin/env python3
"""Diagnóstico facts↔texto del informe de parquets (Parte 4 del plan de verificación).

Responde dos preguntas sobre datos REALES:

  1. ¿Los datos LLEGAN BIEN CALCULADOS?  Por cada dataset imprime los ``facts`` de
     ``compute_facts`` (lo que ve el LLM) y RECOMPUTA de forma INDEPENDIENTE con
     DuckDB el número clave de cada ventana (suma de flujos / variación de stock).
     Marca ✗ si difieren → caza errores de cálculo SIN involucrar al modelo.

  2. ¿Dónde se EQUIVOCA el LLM?  Con ``--with-llm`` corre el pipeline real
     (modelo chico local, p.ej. Qwen2.5-7B en la RTX 3080) y muestra, por dataset,
     facts → párrafo → veredicto del verificador (direcciones corregidas, cifras
     marcadas) lado a lado.

Uso:
    # Solo cálculo (sin modelo): rápido, valida los facts contra el parquet
    python scripts/audit_report_facts.py --segment ffmm
    python scripts/audit_report_facts.py --segment afp
    python scripts/audit_report_facts.py --segment no_residentes

    # Con el LLM local (lee BANKS_LLM_* del .env; baja map-max-tokens para ctx chico)
    python scripts/audit_report_facts.py --segment ffmm --with-llm --map-max-tokens 1024
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import pathlib
import sys

import duckdb

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.reporting.parquet_facts import (  # noqa: E402
    _quote_ident,
    _quote_str,
    compute_facts,
    detect_roles,
    facts_to_text,
)
from banks_rag.application.reporting.parquet_report import (  # noqa: E402
    _window_specs,
    select_datasets,
)
from banks_rag.infrastructure.sql.parquet_catalog_loader import (  # noqa: E402
    ParquetDataset,
    get_parquet_dir,
    load_parquet_catalog,
)

_OK = "✓"
_BAD = "✗"


def _anchor_windows(last_date: str, window_specs: list[tuple[str, int]]) -> list[dict]:
    out = []
    anchor = datetime.date.fromisoformat(last_date)
    for label, days in window_specs:
        out.append({
            "label": label,
            "start": (anchor - datetime.timedelta(days=days)).isoformat(),
            "end": last_date,
        })
    return out


def _independent_window_value(
    con: duckdb.DuckDBPyConnection, dataset: ParquetDataset, parquet_path: pathlib.Path,
    win: dict, *, is_flow: bool,
) -> float | None:
    """Recomputa, SIN pasar por el código de series, el número clave de la ventana:
    suma de flujos (flujo) o variación primer→último del total por fecha (stock)."""
    roles = detect_roles(parquet_path, con)
    if roles.date_col is None or not roles.value_cols:
        return None
    src = f"read_parquet({_quote_str(parquet_path.as_posix())})"
    dcol = _quote_ident(roles.date_col)
    vcol = _quote_ident(roles.value_cols[0])
    scale = dataset.value_scale
    where = (
        f"TRY_CAST({dcol} AS DATE) BETWEEN DATE '{win['start']}' AND DATE '{win['end']}'"
    )
    if is_flow:
        row = con.execute(f"SELECT SUM({vcol}) FROM {src} WHERE {where}").fetchone()
        return None if row[0] is None else float(row[0]) * scale
    # Stock: total por fecha (suma de categorías) → variación primer vs último.
    rows = con.execute(
        f"SELECT TRY_CAST({dcol} AS DATE) AS d, SUM({vcol}) FROM {src} "
        f"WHERE {where} GROUP BY d ORDER BY d"
    ).fetchall()
    rows = [r for r in rows if r[1] is not None]
    if len(rows) < 2:
        return None
    return (float(rows[-1][1]) - float(rows[0][1])) * scale


def _facts_total_value(facts: dict, label: str, *, is_flow: bool) -> float | None:
    for v in facts.get("total_ventanas", []) or facts.get("windows_variation", []) or []:
        if v.get("label") == label and v.get("variacion"):
            var = v["variacion"]
            return var.get("flujo_periodo") if is_flow else var.get("cambio_absoluto")
    return None


def _check_dataset(dataset: ParquetDataset, parquet_dir: pathlib.Path,
                   window_specs: list[tuple[str, int]]) -> tuple[dict | None, list[str]]:
    facts = compute_facts(dataset, parquet_dir, window_specs)
    if not facts or facts.get("shape") in (None, "empty", "unknown", "snapshot"):
        return facts, []
    is_flow = bool(facts.get("is_flow"))
    last_date = facts.get("last_date")
    if not last_date:
        return facts, []
    parquet_path = dataset.parquet_path(parquet_dir)
    con = duckdb.connect()
    out: list[str] = []
    try:
        for win in _anchor_windows(last_date, window_specs):
            fact_val = _facts_total_value(facts, win["label"], is_flow=is_flow)
            indep = _independent_window_value(con, dataset, parquet_path, win, is_flow=is_flow)
            if fact_val is None or indep is None:
                out.append(f"    {win['label']}: facts={fact_val} indep={indep} (sin comparación)")
                continue
            ok = abs(fact_val - indep) <= max(abs(indep) * 0.01, 0.5)
            mark = _OK if ok else _BAD
            kind = "suma flujo" if is_flow else "variación"
            out.append(f"    {mark} {win['label']} [{kind}]: facts={fact_val:,.2f} indep={indep:,.2f}")
    finally:
        con.close()
    return facts, out


def _run_calc_only(datasets, parquet_dir, window_specs, *, show_facts: bool) -> int:
    bad = 0
    for d in datasets:
        kind = d.value_kind or "(heurística)"
        print(f"\n=== [{d.id}] {d.name}  ·  value_kind={kind} ===")
        facts, checks = _check_dataset(d, parquet_dir, window_specs)
        if facts is None:
            print("    sin parquet / sin datos")
            continue
        for line in checks:
            print(line)
            if line.lstrip().startswith(_BAD):
                bad += 1
        if show_facts and facts:
            print("    ── facts_to_text ──")
            for ln in facts_to_text(facts).splitlines():
                print(f"      {ln}")
    return bad


async def _run_with_llm(args, datasets) -> None:
    from banks_rag.application.reporting import generate_parquet_report
    from banks_rag.config import get_settings

    settings = get_settings()
    if settings.llm_family in ("mock", ""):
        print("WARN BANKS_LLM_FAMILY es 'mock'/vacío: configura qwen + backend en .env.")
        return
    if settings.llm_backend == "openai_compat":
        from banks_rag.infrastructure.llm.openai_compat_engine import OpenAICompatEngine
        llm = OpenAICompatEngine.from_settings(settings)
    else:
        from banks_rag.infrastructure.llm.llama_cpp_engine import LlamaCppEngine
        llm = LlamaCppEngine.from_settings(settings)
    print(f"Cargando LLM (backend={settings.llm_backend})…", flush=True)
    await llm.load()

    from banks_rag.application.reporting.parquet_facts import facts_to_text

    ids = [d.id for d in datasets]
    report = await generate_parquet_report(
        llm, dataset_ids=ids, windows=tuple(args.windows),
        map_max_tokens=args.map_max_tokens, think=False, verify=True,
    )
    vr = report.verification

    print("\n" + "#" * 70)
    print("# SÍNTESIS")
    print("#" * 70)
    print("\n── (1) PRIMERA GENERACIÓN (antes del verificador) ──")
    print(report.raw_overview_md or "(sin texto)")
    print("\n── (2) VERIFICADA (arreglada) ──")
    print(report.overview_md)
    if report.raw_overview_md == report.overview_md:
        print("  (sin cambios del verificador)")
    if vr is not None:
        print(f"\n[VERIFICACIÓN] {vr.summary()}")
        for i in vr.residual_for("__sintesis__"):
            print(f"  {_BAD} residual: {i.detail}")

    for s in report.sections:
        print("\n" + "=" * 70)
        print(f"=== [{s.dataset_id}] {s.name} ({s.status}) ===")
        print("\n── (1) PRIMERA GENERACIÓN (antes del verificador) ──")
        print(s.raw_paragraph or "(sin texto)")
        print("\n── (2) VERIFICADA (arreglada) ──")
        print(s.paragraph)
        if s.raw_paragraph == s.paragraph:
            print("  (sin cambios del verificador)")
        if vr is not None:
            for i in vr.residual_for(s.dataset_id):
                print(f"  {_BAD} residual: {i.detail}")
        print("\n── (3) DATOS DIRECTO DEL PARQUET (facts calculados en Python) ──")
        if s.facts:
            print(facts_to_text(s.facts))
        else:
            print("(sin facts)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnóstico facts↔texto del informe de parquets.")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--segment", default=None, help="Segmento o alias (ffmm, afp, no_residentes).")
    sel.add_argument("--datasets", default=None, help="CSV de dataset ids.")
    ap.add_argument("--windows", default="7d,30d", help="Ventanas CSV (default 7d,30d).")
    ap.add_argument("--show-facts", action="store_true", help="Imprime el facts_to_text completo.")
    ap.add_argument("--with-llm", action="store_true", help="Corre el pipeline real + verificador.")
    ap.add_argument("--map-max-tokens", type=int, default=1024, help="Tokens MAP (ctx chico local).")
    args = ap.parse_args()
    args.windows = [w.strip() for w in args.windows.split(",") if w.strip()]

    entries = load_parquet_catalog()
    parquet_dir = get_parquet_dir()
    dataset_ids = [s.strip() for s in args.datasets.split(",")] if args.datasets else None
    selection = select_datasets(entries, segment=args.segment, dataset_ids=dataset_ids)
    window_specs = _window_specs(tuple(args.windows))

    print(f"Diagnóstico — {selection.selector_desc}  ({len(selection.datasets)} datasets, "
          f"ventanas {', '.join(args.windows)})")

    bad = _run_calc_only(selection.datasets, parquet_dir, window_specs, show_facts=args.show_facts)
    print(f"\n>>> Chequeo de cálculo: {bad} discrepancia(s) facts↔parquet "
          f"({'TODO OK' if bad == 0 else 'REVISAR'})")

    if args.with_llm:
        asyncio.run(_run_with_llm(args, selection.datasets))


if __name__ == "__main__":
    main()
