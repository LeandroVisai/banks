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

# Windows: el stdout por defecto es cp1252 y revienta con acentos/emojis del
# informe (UnicodeEncodeError). Forzar utf-8 para que el informe se imprima bien.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Permite ejecutar sin instalar el paquete (resuelve banks_rag desde src/).
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.agent.conversation_loop import run_subagent  # noqa: E402
from banks_rag.application.agent.report import run_report  # noqa: E402
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


async def _load_engine():
    """Construye, carga y precalienta el engine (o None si la config es mock)."""
    settings = get_settings()
    print(
        f"LLM: {settings.llm_family} {settings.llm_model_path} "
        f"(n_ctx={settings.llm_n_ctx}, gpu_layers={settings.llm_n_gpu_layers}, "
        f"vision={'sí' if settings.llm_mmproj_path else 'no'})",
        flush=True,
    )
    if settings.llm_family in ("mock", ""):
        print("⚠️  BANKS_LLM_FAMILY es 'mock'/vacío: configura qwen + BANKS_LLM_MODEL_PATH en .env.")
        return None
    t0 = time.time()
    eng = LlamaCppEngine.from_settings(settings)
    print("cargando modelo...", flush=True)
    await eng.load()
    print(f"modelo listo en {time.time() - t0:.0f}s", flush=True)
    try:
        from banks_rag.infrastructure.embeddings import build_default_embedder
        await asyncio.to_thread(build_default_embedder().encode_text, ["warmup"])
        print("embedder pre-cargado\n", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warmup embedder falló (document/policy puede tardar): {exc}\n", flush=True)
    return eng


async def _run_report(topic: str, scope: str) -> None:
    """Genera un informe completo (run_report) y reporta su forma."""
    eng = await _load_engine()
    if eng is None:
        return
    t1 = time.time()
    res = await run_report(topic, llm=eng, scope=scope)
    print("=" * 80)
    print(f"INFORME (scope={scope}, {time.time() - t1:.0f}s, {res.iterations} iters):\n")
    print(res.response)
    print("\n" + "-" * 80)
    print(f"gráficos generados: {len(res.charts)}", flush=True)
    for c in res.charts:
        print(f"  - gráfico {c['id']}: {c.get('title')} ({c.get('dataset_id')}, {c.get('chart_type')})")
    print(f"series usadas: {len(res.series_used)} | chunks: {len(res.chunks_seen)} "
          f"| citas: {res.cited_refs} | tool calls: {len(res.tool_trace)}")
    if res.ungrounded_numbers:
        print(f"⚠️ cifras sin fundar: {res.ungrounded_numbers}")


async def _run(only: str | None, max_iters: int) -> None:
    eng = await _load_engine()
    if eng is None:
        return

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
                max_iterations=max_iters, max_tool_result_tokens=900,
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
    ap = argparse.ArgumentParser(
        description="Verifica los subagentes (y opcionalmente un informe) contra el LLM del .env.",
    )
    ap.add_argument("--subagent", default=None, choices=sorted(SUBAGENTS), help="Correr solo este.")
    ap.add_argument("--max-iters", type=int, default=5, help="Techo de iteraciones por subagente.")
    ap.add_argument(
        "--report", action="store_true",
        help="En vez de los subagentes, genera un INFORME completo (run_report).",
    )
    ap.add_argument(
        "--topic", default="condiciones financieras recientes del mercado chileno",
        help="Tema del informe (con --report).",
    )
    ap.add_argument(
        "--scope", default="auto", choices=["auto", "full"],
        help="Alcance del informe: 'auto' rutea por el tema, 'full' corre los 8.",
    )
    args = ap.parse_args()
    if args.report:
        asyncio.run(_run_report(args.topic, args.scope))
    else:
        asyncio.run(_run(args.subagent, args.max_iters))


if __name__ == "__main__":
    main()
