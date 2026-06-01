"""Verificación end-to-end de los 8 subagentes especialistas contra el LLM real.

Corre cada subagente (``run_subagent``) con una tarea de su dominio y reporta:
tools usadas, evidencia numérica, ``finish_reason`` y un extracto del análisis.

Usa el LLM configurado vía ``.env`` (``BANKS_LLM_*``), de modo que el MISMO
script sirve local (modelo chico en CPU/GPU) y en la H100 (Qwen3.6-27B-Q4):
solo cambia el ``.env``.

Requisitos:
  - ``.env`` con BANKS_LLM_FAMILY=qwen y BANKS_LLM_MODEL_PATH apuntando al .gguf.
  - Variables de proceso PGUSER/PGPASSWORD/... (los especialistas document/policy
    consultan Postgres) y RAG_EMBEDDING_MODEL (embedder; debe ser el MISMO con
    que se ingestó el corpus — ver memoria del proyecto sobre la dimensión).

Uso:
    python scripts/verify_subagents.py                 # los 8
    python scripts/verify_subagents.py --subagent fx   # solo uno
    python scripts/verify_subagents.py --max-iters 6   # techo de iteraciones
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

# Permite ejecutar sin instalar el paquete (resuelve banks_rag desde src/).
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.agent.conversation_loop import run_subagent  # noqa: E402
from banks_rag.application.agent.subagents import SUBAGENTS  # noqa: E402
from banks_rag.config import get_settings  # noqa: E402
from banks_rag.domain.agent import AgentState  # noqa: E402
from banks_rag.infrastructure.llm.llama_cpp_engine import LlamaCppEngine  # noqa: E402

# Tarea autocontenida por especialista (el subagente NO ve el historial).
TASKS: dict[str, str] = {
    "fx": "Dame el último nivel del dólar observado USD/CLP y su variación reciente.",
    "no_residentes": "¿Cuál es la posición reciente de los inversionistas no residentes en renta fija local (RFL)?",
    "afp": "¿Cómo se reparte la cartera de las AFP entre activos nacionales e internacionales?",
    "fondos_mutuos": "¿Cuáles han sido los flujos recientes de los fondos mutuos (FFMM)?",
    "renta_fija": "¿Cuál es el último nivel del spread BTP vs SPC a 10 años?",
    "liquidez": "¿Cuál es el LCR más reciente del sistema bancario?",
    "document": "¿Qué señalan los comunicados o minutas recientes del Banco Central sobre la inflación?",
    "policy": "¿Cuál fue la última decisión de TPM del Consejo y su justificación principal?",
}


async def _run(only: str | None, max_iters: int) -> None:
    settings = get_settings()
    print(
        f"LLM: {settings.llm_family} {settings.llm_model_path} "
        f"(n_ctx={settings.llm_n_ctx}, gpu_layers={settings.llm_n_gpu_layers})",
        flush=True,
    )
    if settings.llm_family in ("mock", ""):
        print("⚠️  BANKS_LLM_FAMILY es 'mock'/vacío: configura qwen + BANKS_LLM_MODEL_PATH en .env.")
        return

    t0 = time.time()
    eng = LlamaCppEngine.from_settings(settings)
    print("cargando modelo...", flush=True)
    await eng.load()
    print(f"modelo listo en {time.time() - t0:.0f}s", flush=True)

    # Pre-carga del embedder (document/policy lo usan; evita el timeout de tool
    # por la carga en frío en la primera búsqueda).
    try:
        from banks_rag.infrastructure.embeddings import build_default_embedder
        await asyncio.to_thread(build_default_embedder().encode_text, ["warmup"])
        print("embedder pre-cargado\n", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warmup embedder falló (document/policy puede tardar): {exc}\n", flush=True)

    keys = [only] if only else list(SUBAGENTS)
    summary: list[tuple] = []
    for key in keys:
        spec = SUBAGENTS[key]
        task = TASKS[key]
        state = AgentState()
        print("=" * 80)
        print(f"SUBAGENTE [{key}] — {spec.display_name}")
        print(f"tarea: {task}", flush=True)
        t1 = time.time()
        try:
            sub = await run_subagent(
                spec, task, llm=eng, state=state,
                max_iterations=max_iters, max_tool_result_tokens=3000,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  EXCEPCIÓN: {type(exc).__name__}: {exc}", flush=True)
            summary.append((key, "EXC", 0, [], False))
            continue
        ev = state.evidence_tool_calls.get(key, 0)
        tools = [t["tool"] for t in state.tool_trace]
        print(
            f"\nANÁLISIS ({time.time() - t1:.0f}s, {sub.iterations} iters, "
            f"finish={sub.finish_reason}):\n{sub.analysis[:400]}",
            flush=True,
        )
        for t in state.tool_trace:
            args = json.dumps(t["arguments"], ensure_ascii=False)[:60]
            print(f"    {t['tool']}({args}) -> {t['result_summary']}", flush=True)
        ok = ev > 0 and sub.finish_reason != "ungrounded"
        print(f"  => {'OK' if ok else 'WARN'} evidence_tools={ev}", flush=True)
        summary.append((key, sub.finish_reason, sub.iterations, tools, ok))

    print("\n" + "#" * 80)
    print("RESUMEN")
    for key, fr, it, tools, ok in summary:
        dedup: list[str] = []
        for t in tools:
            if not dedup or dedup[-1] != t:
                dedup.append(t)
        print(f"  {'OK  ' if ok else 'WARN'} {key:14s} finish={fr:16s} iters={it} tools={dedup}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verifica los 8 subagentes contra el LLM configurado.")
    ap.add_argument("--subagent", default=None, choices=sorted(SUBAGENTS), help="Correr solo este.")
    ap.add_argument("--max-iters", type=int, default=5, help="Techo de iteraciones por subagente.")
    args = ap.parse_args()
    asyncio.run(_run(args.subagent, args.max_iters))


if __name__ == "__main__":
    main()
