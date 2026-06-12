#!/usr/bin/env python3
"""Diagnóstico del informe parquet: aísla las DOS etapas y muestra el error REAL.

El informe (`parquet_report.py`) hace por cada dataset:
    1) Python lee el parquet y calcula los hechos  (compute_facts)
    2) UNA llamada al LLM redacta el párrafo        (llm.generate)
Cuando "todos fallan", el informe pone el mismo texto genérico sin decir CUÁL
de las dos etapas reventó. Este script corre ambas por separado contra UN
dataset y vuelca el traceback completo, para saber si el problema es de datos
(DuckDB/parquet) o del servidor LLM (llama-server).

Uso (en la máquina donde falló — Mac o el H100):
    python scripts/diag_parquet_report.py
    python scripts/diag_parquet_report.py --dataset flujos_ffmm
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import traceback

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _line(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flujos_ffmm", help="dataset_id a probar")
    args = ap.parse_args()

    from banks_rag.config import get_settings
    from banks_rag.infrastructure.sql.parquet_catalog_loader import (
        get_dataset,
        get_parquet_dir,
        load_parquet_catalog,
    )

    settings = get_settings()
    _line("0) CONFIGURACIÓN (.env efectivo)")
    print(f"  llm_backend     = {settings.llm_backend}")
    print(f"  llm_family      = {settings.llm_family}")
    print(f"  llm_base_url    = {getattr(settings, 'llm_base_url', '—')}")
    print(f"  llm_model_path  = {getattr(settings, 'llm_model_path', '—')}")
    print(f"  parquet_dir     = {get_parquet_dir()}")
    if settings.llm_family in ("mock", ""):
        print("  ⚠  BANKS_LLM_FAMILY es mock/vacío → el informe no genera nada real.")

    entries = load_parquet_catalog()
    ds = get_dataset(entries, args.dataset)
    if ds is None:
        print(f"\n✗ dataset {args.dataset!r} no está en el catálogo. Abortando.")
        return

    # ── Etapa 1: cálculo de hechos en Python (DuckDB sobre el parquet real) ──
    _line(f"1) DATOS — compute_facts({ds.id})")
    facts = None
    try:
        from banks_rag.application.reporting.parquet_facts import (
            compute_facts,
            facts_to_text,
        )

        parquet_path = ds.parquet_path(get_parquet_dir())
        print(f"  parquet: {parquet_path}  (existe={parquet_path.exists()})")
        facts = compute_facts(ds, get_parquet_dir(), [("última semana", 7), ("último mes", 30)])
        if not facts:
            print("  ✗ compute_facts devolvió None/vacío (parquet sin datos útiles).")
        else:
            print(f"  ✓ OK — shape={facts.get('shape')} last_date={facts.get('last_date')}")
            print("  --- hechos (primeras líneas) ---")
            print("  " + "\n  ".join(facts_to_text(facts).splitlines()[:8]))
    except Exception:
        print("  ✗ FALLÓ EL CÁLCULO DE DATOS (problema de DuckDB/parquet, NO del LLM):")
        traceback.print_exc()
        print("\n  → Si esto falla en TODOS los datasets, el problema es la copia de "
              "parquets o la versión de duckdb en esta máquina, no el servidor LLM.")
        return

    # ── Etapa 2: una llamada real al LLM (contra el llama-server) ────────────
    _line("2) LLM — load() + una generación contra el servidor")
    if settings.llm_family in ("mock", ""):
        print("  (saltado: family=mock)")
        return
    try:
        if settings.llm_backend == "openai_compat":
            from banks_rag.infrastructure.llm.openai_compat_engine import OpenAICompatEngine
            llm = OpenAICompatEngine.from_settings(settings)
        else:
            from banks_rag.infrastructure.llm.llama_cpp_engine import LlamaCppEngine
            llm = LlamaCppEngine.from_settings(settings)
        print("  cargando engine…", flush=True)
        await llm.load()
        print("  ✓ engine cargado / servidor sano. Generando 1 párrafo de prueba…", flush=True)
        result = await llm.generate(
            [
                {"role": "system", "content": "Eres un analista. Responde en una frase."},
                {"role": "user", "content": "Di 'el diagnóstico funciona' y nada más."},
            ],
            tools=None,
            temperature=0.3,
            top_p=0.8,
            max_tokens=64,
        )
        print(f"  ✓ OK — el LLM respondió: {result.text!r}")
        print("\n  → Datos y LLM funcionan AISLADOS. Si el informe igual falla, el "
              "problema está en el código de generate_parquet_report desplegado "
              "(¿es la versión nueva?) o en el timeout bajo carga.")
    except Exception:
        print("  ✗ FALLÓ LA LLAMADA AL LLM (problema del servidor llama-server, NO de los datos):")
        traceback.print_exc()
        print("\n  → Los datos SÍ se calcularon (etapa 1 OK). El error está en el "
              "servidor: ¿está corriendo start_llama_server.ps1? ¿responde /health "
              "en llm_base_url? ¿timeout? Revisa también BANKS_LLM_FAMILY/BACKEND.")


if __name__ == "__main__":
    asyncio.run(main())
