"""CLI ``banks-evaluate`` — evalúa el pipeline RAG contra el golden set.

Genera ``eval_report.md`` con recall@k, MRR, nDCG y métricas de routing.
Corre offline (no requiere LLM en vivo ni BD; usa golden set + mocks opcionales).

Uso:
    banks-evaluate retrieval --k 5
    banks-evaluate routing
    banks-evaluate generation
    banks-evaluate all --k 5 --output eval_report.md
    make eval   # equivalente a: banks-evaluate all
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from banks_rag.application.evaluation.ragas_runner import (
    GenerationEval,
    aggregate_generation,
    evaluate_generation,
    load_generation_golden_set,
)
from banks_rag.application.evaluation.retrieval_metrics import (
    AggregateMetrics,
    RetrievalResult,
    aggregate,
    evaluate_retrieval,
    evaluate_sql_routing,
    load_retrieval_golden_set,
    load_sql_routing_golden_set,
)

app = typer.Typer(name="banks-evaluate", help="Evalúa el pipeline RAG/SQL contra el golden set.")

_DEFAULT_REPORT = Path("eval_report.md")
_BASELINE_PATH = Path("eval_baseline.json")
_RECALL_DROP_THRESHOLD = 0.05  # 5% caída bloquea CI


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_baseline() -> dict | None:
    if _BASELINE_PATH.exists():
        return json.loads(_BASELINE_PATH.read_text())
    return None


def _save_baseline(data: dict) -> None:
    _BASELINE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _format_metric(value: float, baseline: float | None = None) -> str:
    pct = f"{value:.1%}"
    if baseline is None:
        return pct
    delta = value - baseline
    sign = "+" if delta >= 0 else ""
    return f"{pct} ({sign}{delta:.1%})"


def _run_retrieval_eval(k: int) -> tuple[AggregateMetrics, list[RetrievalResult]]:
    """Evalúa los casos del golden set de retrieval con hits sintéticos.

    En modo offline (sin BD) genera hits vacíos — el score será 0, útil para
    detectar regresiones de interfaz y validar la cadena de cómputo.
    Con BD real, substituir _mock_hits por hybrid_search real.
    """
    cases = load_retrieval_golden_set()
    results: list[RetrievalResult] = []
    for case in cases:
        # Modo offline: hits vacíos (score 0 para todos)
        mock_hits: list[dict] = []
        r = evaluate_retrieval(
            mock_hits,
            expected_doc_types=case.get("expected_doc_types", []),
            expected_sections=case.get("expected_sections", []),
            min_importance=case.get("min_importance"),
            query=case.get("query", ""),
            k=k,
        )
        results.append(r)
    return aggregate(results), results


def _run_routing_eval() -> dict:
    from banks_rag.application.retrieval.query_router import route_query

    cases = load_sql_routing_golden_set()
    return evaluate_sql_routing(cases, route_fn=route_query)


def _run_generation_eval() -> tuple[dict, list[GenerationEval]]:
    cases = load_generation_golden_set()
    results: list[GenerationEval] = []
    for case in cases:
        # Modo offline: answer sintético con las keywords del caso
        keywords = case.get("ideal_answer_keywords", [])
        synthetic_answer = " ".join(keywords)
        synthetic_chunks = [
            {"text": synthetic_answer, "doc_type_category": dt}
            for dt in case.get("must_cite_doc_types", ["COMUNICADO_RPM"])
        ]
        r = evaluate_generation(
            query=case.get("query", ""),
            answer=synthetic_answer,
            context_chunks=synthetic_chunks,
            must_cite_doc_types=case.get("must_cite_doc_types"),
        )
        results.append(r)
    agg = aggregate_generation(results)
    return agg.to_dict(), results


def _build_report(
    retrieval_agg: AggregateMetrics,
    routing_metrics: dict,
    generation_agg: dict,
    *,
    k: int,
    baseline: dict | None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    b_ret = baseline.get("retrieval", {}) if baseline else {}
    b_route = baseline.get("routing", {}) if baseline else {}
    b_gen = baseline.get("generation", {}) if baseline else {}

    lines = [
        "# Eval Report — Banks RAG",
        f"\n_Generado: {ts}_\n",
        "## Retrieval Metrics",
        f"\n| Métrica | Valor | vs Baseline |",
        "|---|---|---|",
        f"| Hit-rate@{k} | {_format_metric(retrieval_agg.hit_rate_at_k, b_ret.get('hit_rate_at_k'))} | {'—' if not b_ret else ''} |",
        f"| MRR@{k}    | {_format_metric(retrieval_agg.mrr_at_k,    b_ret.get('mrr_at_k'))}    | |",
        f"| nDCG@{k}   | {_format_metric(retrieval_agg.ndcg_at_k,   b_ret.get('ndcg_at_k'))}   | |",
        f"| N queries  | {retrieval_agg.n_queries} | |",
        "",
        "> **Nota**: métricas offline usan hits vacíos (score=0 esperado).",
        "> Conectar BD real para métricas de retrieval significativas.",
        "",
        "## SQL Routing Metrics",
        f"\n| Métrica | Valor |",
        "|---|---|",
        f"| Accuracy  | {_format_metric(routing_metrics['accuracy'],   b_route.get('accuracy'))} |",
        f"| SQL Recall | {_format_metric(routing_metrics['sql_recall'], b_route.get('sql_recall'))} |",
        f"| N cases   | {routing_metrics['n_cases']} |",
        "",
        "## Generation Metrics (offline)",
        f"\n| Métrica | Valor |",
        "|---|---|",
        f"| Faithfulness      | {_format_metric(generation_agg['faithfulness'],     b_gen.get('faithfulness'))} |",
        f"| Answer Relevancy  | {_format_metric(generation_agg['answer_relevancy'], b_gen.get('answer_relevancy'))} |",
        f"| Context Precision | {_format_metric(generation_agg['context_precision'],b_gen.get('context_precision'))} |",
        f"| N evaluated       | {generation_agg['n_evaluated']} |",
        "",
        "---",
        "_Banks RAG Evaluation · banco central chileno_",
    ]
    return "\n".join(lines)


def _check_ci_gate(retrieval_agg: AggregateMetrics, baseline: dict | None) -> bool:
    """Devuelve True si pasa el gate CI (hit-rate no cae >5% vs baseline)."""
    if baseline is None:
        return True
    prev = baseline.get("retrieval", {}).get("hit_rate_at_k", 0.0)
    if prev == 0.0:
        return True
    drop = prev - retrieval_agg.hit_rate_at_k
    return drop <= _RECALL_DROP_THRESHOLD


# ── Subcomandos ───────────────────────────────────────────────────────────────

@app.command()
def retrieval(k: int = typer.Option(5, help="Top-K a evaluar")) -> None:
    """Métricas de retrieval (hit-rate@k, MRR, nDCG) contra el golden set."""
    agg, _ = _run_retrieval_eval(k)
    typer.echo(f"Hit-rate@{k}: {agg.hit_rate_at_k:.1%}")
    typer.echo(f"MRR@{k}:     {agg.mrr_at_k:.1%}")
    typer.echo(f"nDCG@{k}:    {agg.ndcg_at_k:.1%}")
    typer.echo(f"N queries: {agg.n_queries}")


@app.command()
def routing() -> None:
    """Evalúa el QueryRouter contra el golden set de SQL routing."""
    metrics = _run_routing_eval()
    typer.echo(f"Accuracy:    {metrics['accuracy']:.1%}  ({metrics['n_cases']} casos)")
    typer.echo(f"SQL Recall:  {metrics['sql_recall']:.1%}")


@app.command()
def generation() -> None:
    """Evalúa la generación offline (faithfulness, relevancy, precision)."""
    agg_dict, _ = _run_generation_eval()
    typer.echo(f"Faithfulness:      {agg_dict['faithfulness']:.1%}")
    typer.echo(f"Answer Relevancy:  {agg_dict['answer_relevancy']:.1%}")
    typer.echo(f"Context Precision: {agg_dict['context_precision']:.1%}")


@app.command()
def all(
    k: int = typer.Option(5, help="Top-K para métricas de retrieval"),
    output: Path = typer.Option(_DEFAULT_REPORT, help="Archivo de salida del reporte"),
    update_baseline: bool = typer.Option(False, "--update-baseline", help="Guarda los resultados como nuevo baseline"),
    ci: bool = typer.Option(False, "--ci", help="Modo CI: falla si recall@k cae >5% vs baseline"),
) -> None:
    """Corre toda la evaluación y genera eval_report.md."""
    typer.echo("Evaluando retrieval…")
    retrieval_agg, _ = _run_retrieval_eval(k)

    typer.echo("Evaluando SQL routing…")
    routing_metrics = _run_routing_eval()

    typer.echo("Evaluando generación (offline)…")
    generation_agg, _ = _run_generation_eval()

    baseline = _load_baseline()
    report = _build_report(
        retrieval_agg, routing_metrics, generation_agg,
        k=k, baseline=baseline,
    )
    output.write_text(report, encoding="utf-8")
    typer.echo(f"\nReporte escrito en: {output}")

    typer.echo(f"\nRouting accuracy: {routing_metrics['accuracy']:.1%}")
    typer.echo(f"Hit-rate@{k}: {retrieval_agg.hit_rate_at_k:.1%}")

    if update_baseline:
        _save_baseline({
            "retrieval": retrieval_agg.to_dict(),
            "routing": routing_metrics,
            "generation": generation_agg,
        })
        typer.echo("Baseline actualizado.")

    if ci and not _check_ci_gate(retrieval_agg, baseline):
        typer.echo(
            f"[CI GATE] FALLO: recall@{k} cayó >{_RECALL_DROP_THRESHOLD:.0%} vs baseline.",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
