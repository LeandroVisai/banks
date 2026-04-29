#!/usr/bin/env python3
"""
test_search.py — Evalúa calidad de retrieval contra benchmark de 25 queries.

Métricas calculadas:
  Precision@1   — ¿el top-1 pertenece a los docs relevantes?
  Recall@5      — ¿algún doc relevante aparece en top-5?
  MRR           — Mean Reciprocal Rank del primer hit relevante
  NDCG@5        — Normalized Discounted Cumulative Gain

Uso:
  python3 test_search.py                       # evalúa y muestra resultados
  python3 test_search.py --baseline            # evalúa + guarda en test_baseline.json
  python3 test_search.py --compare <file>      # compara contra métricas guardadas
  python3 test_search.py --query q01,q05,q11   # evalúa solo esas queries
  python3 test_search.py --top-k 5             # tamaño de la lista recuperada (default 5)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

BASE            = Path(__file__).parent
BENCHMARK_PATH  = BASE / "queries_benchmark.json"
BASELINE_PATH   = BASE / "test_baseline.json"
SEARCH_SCRIPT   = BASE / "04_search.py"

TOP_K_DEFAULT = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_search(query: str, k: int) -> list[dict] | None:
    """Llama a 04_search.py --json y devuelve los resultados parseados."""
    cmd = [sys.executable, str(SEARCH_SCRIPT), query, str(k), "--json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"    [error] search falló: {result.stderr[:200]}")
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print("    [error] timeout en búsqueda")
        return None
    except json.JSONDecodeError as e:
        print(f"    [error] JSON inválido: {e}")
        return None
    except Exception as e:
        print(f"    [error] {e}")
        return None


def extract_doc_ids(results: list[dict]) -> list[str]:
    """Extrae document_ids desde los resultados de 04_search.py."""
    doc_ids = []
    for r in results:
        doc_id = r.get("document_id") or r.get("doc_id") or ""
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def extract_sections(results: list[dict]) -> list[str]:
    return [r.get("section_type", "") for r in results]


# ── Métricas ─────────────────────────────────────────────────────────────────

def precision_at_k(retrieved_docs: list[str], relevant_docs: list[str], k: int = 1) -> float:
    top = retrieved_docs[:k]
    return 1.0 if any(d in relevant_docs for d in top) else 0.0


def recall_at_k(retrieved_docs: list[str], relevant_docs: list[str], k: int = 5) -> float:
    top = retrieved_docs[:k]
    return 1.0 if any(d in relevant_docs for d in top) else 0.0


def reciprocal_rank(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    for i, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], relevant_docs: list[str], k: int = 5) -> float:
    def dcg(docs: list[str], rel_set: list[str], k_: int) -> float:
        return sum(
            (1.0 / math.log2(i + 2))
            for i, d in enumerate(docs[:k_])
            if d in rel_set
        )
    ideal_hits = min(len(relevant_docs), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg(retrieved_docs, relevant_docs, k) / idcg


# ── Evaluación ────────────────────────────────────────────────────────────────

def evaluate_query(bq: dict, k: int) -> dict:
    query_id    = bq["id"]
    query_text  = bq["query"]
    relevant    = bq["relevant_docs"]
    exp_section = bq.get("expected_section", "")
    min_imp     = bq.get("expected_importance_min", 0.0)

    results = run_search(query_text, k)
    if results is None:
        return {
            "id": query_id, "query": query_text, "status": "error",
            "precision_at_1": 0, "recall_at_5": 0, "mrr": 0, "ndcg_at_5": 0,
        }

    retrieved_docs    = extract_doc_ids(results)
    retrieved_sections = extract_sections(results)

    p1   = precision_at_k(retrieved_docs, relevant, 1)
    r5   = recall_at_k(retrieved_docs, relevant, 5)
    mrr  = reciprocal_rank(retrieved_docs, relevant)
    ndcg = ndcg_at_k(retrieved_docs, relevant, k)

    # Sección correcta en top-1
    top1_section_ok = (
        retrieved_sections[0] == exp_section if retrieved_sections and exp_section else None
    )

    # Importance del top-1
    top1_importance = results[0].get("importance_score", 0.0) if results else 0.0
    importance_ok   = top1_importance >= min_imp if min_imp > 0 else None

    return {
        "id": query_id,
        "query": query_text,
        "status": "ok",
        "relevant_docs": relevant,
        "retrieved_docs_top5": retrieved_docs[:5],
        "precision_at_1": p1,
        "recall_at_5": r5,
        "mrr": round(mrr, 4),
        "ndcg_at_5": round(ndcg, 4),
        "top1_section_correct": top1_section_ok,
        "top1_importance": round(top1_importance, 4),
        "importance_meets_min": importance_ok,
    }


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return {"n_queries": 0, "n_errors": len(results)}
    return {
        "n_queries":      len(results),
        "n_errors":       sum(1 for r in results if r["status"] == "error"),
        "precision_at_1": round(sum(r["precision_at_1"] for r in ok) / len(ok), 4),
        "recall_at_5":    round(sum(r["recall_at_5"]    for r in ok) / len(ok), 4),
        "mrr":            round(sum(r["mrr"]            for r in ok) / len(ok), 4),
        "ndcg_at_5":      round(sum(r["ndcg_at_5"]      for r in ok) / len(ok), 4),
        "section_accuracy": round(
            sum(1 for r in ok if r.get("top1_section_correct")) /
            sum(1 for r in ok if r.get("top1_section_correct") is not None)
            if any(r.get("top1_section_correct") is not None for r in ok) else 0.0,
            4
        ),
    }


# ── Comparación contra baseline ───────────────────────────────────────────────

def compare_reports(current: dict, baseline_path: Path) -> None:
    if not baseline_path.exists():
        print(f"[error] baseline no encontrado: {baseline_path}")
        return

    baseline = json.loads(baseline_path.read_text())
    cm = current["aggregate"]
    bm = baseline["aggregate"]

    W = 60
    print("\n" + "═" * W)
    print("  COMPARACIÓN vs BASELINE")
    print("═" * W)

    metrics = ["precision_at_1", "recall_at_5", "mrr", "ndcg_at_5", "section_accuracy"]
    for m in metrics:
        cv = cm.get(m, 0)
        bv = bm.get(m, 0)
        delta = cv - bv
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
        color_open = "" if delta >= 0 else ""
        print(f"  {m:<22} baseline={bv:.4f}  actual={cv:.4f}  {arrow} {delta:+.4f}")

    print("═" * W)


# ── Tabla de resultados ───────────────────────────────────────────────────────

def print_table(results: list[dict], agg: dict) -> None:
    W = 78
    print("\n" + "═" * W)
    print("  RESULTADOS POR QUERY")
    print("═" * W)
    header = f"  {'ID':<5} {'P@1':>5} {'R@5':>5} {'MRR':>6} {'NDCG@5':>7}  {'Sec OK':>6}  Query"
    print(header)
    print("-" * W)
    for r in results:
        if r["status"] == "error":
            print(f"  {r['id']:<5} {'ERROR':>5} {'':>5} {'':>6} {'':>7}  {'':>6}  {r['query'][:40]}")
            continue
        sec = ("✓" if r["top1_section_correct"] else "✗") if r["top1_section_correct"] is not None else "-"
        print(
            f"  {r['id']:<5} {r['precision_at_1']:>5.1f} {r['recall_at_5']:>5.1f}"
            f" {r['mrr']:>6.4f} {r['ndcg_at_5']:>7.4f}  {sec:>6}  {r['query'][:38]}"
        )

    print("─" * W)
    print(
        f"  {'AVG':<5} {agg.get('precision_at_1', 0):>5.4f} {agg.get('recall_at_5', 0):>5.4f}"
        f" {agg.get('mrr', 0):>6.4f} {agg.get('ndcg_at_5', 0):>7.4f}"
        f"  {agg.get('section_accuracy', 0):>5.4f}  (n={agg.get('n_queries', 0)}, errors={agg.get('n_errors', 0)})"
    )
    print("═" * W)

    print("\n  Guía de interpretación:")
    print("  Precision@1  ≥ 0.70 = bueno    ≥ 0.85 = excelente")
    print("  MRR          ≥ 0.65 = bueno    ≥ 0.80 = excelente")
    print("  NDCG@5       ≥ 0.60 = bueno    ≥ 0.75 = excelente")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa calidad de retrieval RAG")
    parser.add_argument("--baseline", action="store_true", help=f"Guarda resultados en {BASELINE_PATH.name}")
    parser.add_argument("--compare", metavar="FILE", help="Compara contra métricas guardadas")
    parser.add_argument("--query", metavar="IDS", help="Subset de queries (ej. q01,q05,q11)")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, metavar="K", help=f"Top-K resultados (default {TOP_K_DEFAULT})")
    args = parser.parse_args()

    if not BENCHMARK_PATH.exists():
        print(f"ERROR: {BENCHMARK_PATH.name} no encontrado")
        sys.exit(1)
    if not SEARCH_SCRIPT.exists():
        print(f"ERROR: {SEARCH_SCRIPT.name} no encontrado")
        sys.exit(1)

    benchmark = json.loads(BENCHMARK_PATH.read_text())

    # Filtrar si --query
    if args.query:
        ids = {q.strip() for q in args.query.split(",")}
        benchmark = [q for q in benchmark if q["id"] in ids]
        if not benchmark:
            print(f"ERROR: ninguna query encontrada con ids {ids}")
            sys.exit(1)

    k = args.top_k
    print(f"Evaluando {len(benchmark)} queries  (top-k={k})")
    print(f"Script de búsqueda: {SEARCH_SCRIPT.name}\n")

    query_results: list[dict] = []
    for bq in benchmark:
        print(f"  [{bq['id']}] {bq['query'][:55]}...")
        res = evaluate_query(bq, k)
        query_results.append(res)
        p1 = res["precision_at_1"]
        mrr = res.get("mrr", 0)
        marker = "✓" if p1 == 1.0 else ("~" if mrr > 0 else "✗")
        print(f"       {marker} P@1={p1:.1f}  MRR={mrr:.4f}  docs={res.get('retrieved_docs_top5', [])[:2]}")

    agg = aggregate(query_results)
    print_table(query_results, agg)

    report = {
        "top_k": k,
        "aggregate": agg,
        "per_query": query_results,
    }

    if args.baseline:
        BASELINE_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n  Baseline guardado en: {BASELINE_PATH.name}")

    if args.compare:
        compare_reports(report, Path(args.compare))

    output_path = BASE / "test_results.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  Resultados guardados en: {output_path.name}")


if __name__ == "__main__":
    main()
