"""Tests unitarios de Fase 6 — Evaluación reproducible.

Cubre:
- retrieval_metrics: recall@k, MRR, nDCG, aggregate, evaluate_sql_routing.
- ragas_runner: evaluate_generation offline, aggregate_generation.
- evaluate CLI: _check_ci_gate, _build_report (smoke).
- golden set: carga sin BD ni LLM real.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from banks_rag.application.evaluation.retrieval_metrics import (
    AggregateMetrics,
    RetrievalResult,
    _is_hit,
    _mrr,
    _ndcg,
    aggregate,
    evaluate_retrieval,
    evaluate_sql_routing,
    load_retrieval_golden_set,
    load_sql_routing_golden_set,
)
from banks_rag.application.evaluation.ragas_runner import (
    GenerationEval,
    _answer_relevancy_offline,
    _context_precision_offline,
    _faithfulness_offline,
    _tokenize,
    aggregate_generation,
    evaluate_generation,
    load_generation_golden_set,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(doc_type: str, section: str = "DECISION", importance: float = 0.8) -> dict:
    return {
        "doc_type_category": doc_type,
        "section_type": section,
        "importance_score": importance,
        "text": f"texto del chunk {doc_type} {section}",
    }


def _write_jsonl(tmp_path: Path, name: str, records: list[dict]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


# ── Tests _is_hit ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestIsHit:
    def test_matches_doc_type(self) -> None:
        c = _chunk("COMUNICADO")
        assert _is_hit(c, ["COMUNICADO"], []) is True

    def test_rejects_wrong_doc_type(self) -> None:
        c = _chunk("MINUTA")
        assert _is_hit(c, ["COMUNICADO"], []) is False

    def test_matches_section(self) -> None:
        c = _chunk("COMUNICADO", section="DECISION")
        assert _is_hit(c, [], ["DECISION"]) is True

    def test_both_conditions_required(self) -> None:
        c = _chunk("COMUNICADO", section="ANALISIS")
        assert _is_hit(c, ["COMUNICADO"], ["DECISION"]) is False

    def test_empty_criteria_always_matches(self) -> None:
        c = _chunk("CUALQUIER_TIPO")
        assert _is_hit(c, [], []) is True


# ── Tests MRR y nDCG ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMrrNdcg:
    def test_mrr_first_position(self) -> None:
        assert _mrr([1, 0, 0]) == pytest.approx(1.0)

    def test_mrr_second_position(self) -> None:
        assert _mrr([0, 1, 0]) == pytest.approx(0.5)

    def test_mrr_no_hit(self) -> None:
        assert _mrr([0, 0, 0]) == pytest.approx(0.0)

    def test_ndcg_perfect(self) -> None:
        assert _ndcg([1, 0, 0]) == pytest.approx(1.0)

    def test_ndcg_no_hit(self) -> None:
        assert _ndcg([0, 0, 0]) == pytest.approx(0.0)

    def test_ndcg_hit_at_2(self) -> None:
        # DCG = 1/log2(3) ≈ 0.631, IDCG = 1/log2(2) = 1.0
        result = _ndcg([0, 1, 0])
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert result == pytest.approx(expected)


# ── Tests evaluate_retrieval ──────────────────────────────────────────────────

@pytest.mark.unit
class TestEvaluateRetrieval:
    def test_perfect_retrieval(self) -> None:
        hits = [_chunk("COMUNICADO", "DECISION")]
        r = evaluate_retrieval(hits, expected_doc_types=["COMUNICADO"],
                               expected_sections=["DECISION"], k=5)
        assert r.hit_rate_at_k == pytest.approx(1.0)
        assert r.mrr_at_k == pytest.approx(1.0)
        assert r.ndcg_at_k == pytest.approx(1.0)

    def test_zero_hits(self) -> None:
        r = evaluate_retrieval([], expected_doc_types=["COMUNICADO"], k=5)
        assert r.hit_rate_at_k == pytest.approx(0.0)
        assert r.mrr_at_k == pytest.approx(0.0)

    def test_hit_at_rank_3(self) -> None:
        hits = [
            _chunk("MINUTA"), _chunk("RESEARCH"),
            _chunk("COMUNICADO", "DECISION"),
        ]
        r = evaluate_retrieval(hits, expected_doc_types=["COMUNICADO"],
                               expected_sections=["DECISION"], k=5)
        assert r.hit_rate_at_k == pytest.approx(1.0)
        assert r.mrr_at_k == pytest.approx(1 / 3)
        assert 3 in r.matched_at

    def test_min_importance_filter(self) -> None:
        hits = [_chunk("COMUNICADO", "DECISION", importance=0.3)]
        r = evaluate_retrieval(hits, expected_doc_types=["COMUNICADO"],
                               expected_sections=["DECISION"],
                               min_importance=0.7, k=5)
        assert r.hit_rate_at_k == pytest.approx(0.0)

    def test_top_k_respected(self) -> None:
        hits = [_chunk("MINUTA")] * 10 + [_chunk("COMUNICADO", "DECISION")]
        r = evaluate_retrieval(hits, expected_doc_types=["COMUNICADO"],
                               expected_sections=["DECISION"], k=5)
        assert 11 not in r.matched_at  # rank 11 fuera del k=5


# ── Tests aggregate ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestAggregate:
    def test_average_over_results(self) -> None:
        r1 = RetrievalResult("q1", 5, 1.0, 1.0, 1.0, 1, [1])
        r2 = RetrievalResult("q2", 5, 0.0, 0.0, 0.0, 0, [])
        agg = aggregate([r1, r2])
        assert agg.hit_rate_at_k == pytest.approx(0.5)
        assert agg.mrr_at_k == pytest.approx(0.5)
        assert agg.n_queries == 2

    def test_empty_list(self) -> None:
        agg = aggregate([])
        assert agg.n_queries == 0
        assert agg.hit_rate_at_k == pytest.approx(0.0)


# ── Tests evaluate_sql_routing ────────────────────────────────────────────────

@pytest.mark.unit
class TestEvaluateSqlRouting:
    def _make_route_fn(self, sql: bool, rag: bool = True, visual: bool = False):
        decision = MagicMock()
        decision.sql = sql
        decision.rag = rag
        decision.visual = visual
        return lambda q: decision

    def test_all_correct_sql(self) -> None:
        cases = [
            {"query": "¿cuánto vale el dólar?", "expected_route": "sql"},
            {"query": "dame el USD/CLP", "expected_route": "sql"},
        ]
        metrics = evaluate_sql_routing(cases, route_fn=self._make_route_fn(sql=True))
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["sql_recall"] == pytest.approx(1.0)

    def test_no_correct(self) -> None:
        cases = [{"query": "query", "expected_route": "sql"}]
        metrics = evaluate_sql_routing(cases, route_fn=self._make_route_fn(sql=False))
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_empty_cases(self) -> None:
        metrics = evaluate_sql_routing([], route_fn=self._make_route_fn(sql=True))
        assert metrics["n_cases"] == 0


# ── Tests load golden sets ────────────────────────────────────────────────────

@pytest.mark.unit
class TestLoadGoldenSets:
    def test_load_retrieval_from_file(self, tmp_path) -> None:
        records = [
            {"query": "q1", "expected_doc_types": ["COMUNICADO"], "expected_sections": ["DECISION"]},
            {"query": "q2", "expected_doc_types": ["MINUTA"], "expected_sections": []},
        ]
        p = _write_jsonl(tmp_path, "retrieval.jsonl", records)
        loaded = load_retrieval_golden_set(p)
        assert len(loaded) == 2
        assert loaded[0]["query"] == "q1"

    def test_load_sql_routing_from_file(self, tmp_path) -> None:
        records = [{"query": "precio cobre", "expected_query_id": "precio_cobre", "expected_route": "sql"}]
        p = _write_jsonl(tmp_path, "sql.jsonl", records)
        loaded = load_sql_routing_golden_set(p)
        assert loaded[0]["expected_query_id"] == "precio_cobre"

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        loaded = load_retrieval_golden_set(tmp_path / "no_existe.jsonl")
        assert loaded == []

    def test_real_retrieval_golden_set_loads(self) -> None:
        cases = load_retrieval_golden_set()
        assert len(cases) >= 20, "El golden set de retrieval debe tener al menos 20 casos"

    def test_real_sql_routing_golden_set_loads(self) -> None:
        cases = load_sql_routing_golden_set()
        assert len(cases) >= 20

    def test_real_generation_golden_set_loads(self) -> None:
        cases = load_generation_golden_set()
        assert len(cases) >= 10


# ── Tests tokenize y helpers offline ─────────────────────────────────────────

@pytest.mark.unit
class TestOfflineHelpers:
    def test_tokenize_removes_stopwords(self) -> None:
        tokens = _tokenize("el precio del dólar hoy")
        assert "el" not in tokens
        assert "del" not in tokens
        assert "precio" in tokens
        assert "dólar" in tokens

    def test_tokenize_empty(self) -> None:
        assert _tokenize("") == set()

    def test_faithfulness_high_overlap(self) -> None:
        answer = "TPM tasa inflación banco central"
        chunks = [{"text": "TPM tasa inflación banco central chile"}]
        score = _faithfulness_offline(answer, chunks)
        assert score > 0.5

    def test_faithfulness_no_overlap(self) -> None:
        answer = "cobre precio commodity"
        chunks = [{"text": "tasa inflación banco"}]
        score = _faithfulness_offline(answer, chunks)
        assert score < 0.3

    def test_answer_relevancy_related(self) -> None:
        score = _answer_relevancy_offline(
            "¿cuál es la TPM actual?",
            "La TPM actual es 5.5% según el Banco Central",
        )
        assert score > 0.2

    def test_context_precision_with_matching_doc_type(self) -> None:
        answer = "TPM decidió mantener tasa"
        chunks = [{"text": "TPM decidió mantener tasa política", "doc_type_category": "COMUNICADO"}]
        score = _context_precision_offline(answer, chunks, ["COMUNICADO"])
        assert score > 0.5

    def test_context_precision_wrong_doc_type(self) -> None:
        answer = "TPM decidió mantener tasa"
        chunks = [{"text": "TPM decidió mantener tasa política", "doc_type_category": "MINUTA"}]
        score = _context_precision_offline(answer, chunks, ["COMUNICADO"])
        assert score == pytest.approx(0.0)


# ── Tests evaluate_generation ─────────────────────────────────────────────────

@pytest.mark.unit
class TestEvaluateGeneration:
    def test_returns_generation_eval(self) -> None:
        r = evaluate_generation(
            "¿qué decidió el Consejo?",
            "El Consejo decidió bajar la TPM.",
            [{"text": "El Consejo decidió bajar la TPM.", "doc_type_category": "COMUNICADO"}],
        )
        assert isinstance(r, GenerationEval)
        assert 0.0 <= r.faithfulness <= 1.0
        assert 0.0 <= r.answer_relevancy <= 1.0
        assert 0.0 <= r.context_precision <= 1.0
        assert r.mode == "offline"

    def test_empty_answer_scores_zero(self) -> None:
        r = evaluate_generation(
            "query",
            "",
            [{"text": "TPM banco central", "doc_type_category": "COMUNICADO"}],
        )
        assert r.faithfulness == pytest.approx(0.0)
        assert r.answer_relevancy == pytest.approx(0.0)

    def test_ragas_fallback_when_not_installed(self) -> None:
        # use_ragas=True pero el paquete no está: debe caer a offline
        r = evaluate_generation(
            "query", "respuesta banco central TPM",
            [{"text": "banco central TPM", "doc_type_category": "COMUNICADO"}],
            use_ragas=True,
        )
        assert r.mode == "offline"


# ── Tests aggregate_generation ────────────────────────────────────────────────

@pytest.mark.unit
class TestAggregateGeneration:
    def test_average(self) -> None:
        r1 = GenerationEval("q1", faithfulness=1.0, answer_relevancy=1.0, context_precision=1.0)
        r2 = GenerationEval("q2", faithfulness=0.0, answer_relevancy=0.0, context_precision=0.0)
        agg = aggregate_generation([r1, r2])
        assert agg.faithfulness == pytest.approx(0.5)
        assert agg.answer_relevancy == pytest.approx(0.5)
        assert agg.n_evaluated == 2

    def test_empty(self) -> None:
        agg = aggregate_generation([])
        assert agg.n_evaluated == 0


# ── Tests CI gate y reporte ───────────────────────────────────────────────────

@pytest.mark.unit
class TestCiGate:
    def test_no_baseline_always_passes(self) -> None:
        from banks_rag.interface.cli.evaluate import _check_ci_gate
        agg = AggregateMetrics(n_queries=5, hit_rate_at_k=0.0, mrr_at_k=0.0, ndcg_at_k=0.0, k=5)
        assert _check_ci_gate(agg, None) is True

    def test_no_drop_passes(self) -> None:
        from banks_rag.interface.cli.evaluate import _check_ci_gate
        baseline = {"retrieval": {"hit_rate_at_k": 0.8}}
        agg = AggregateMetrics(n_queries=5, hit_rate_at_k=0.8, mrr_at_k=0.0, ndcg_at_k=0.0, k=5)
        assert _check_ci_gate(agg, baseline) is True

    def test_small_drop_passes(self) -> None:
        from banks_rag.interface.cli.evaluate import _check_ci_gate
        baseline = {"retrieval": {"hit_rate_at_k": 0.8}}
        agg = AggregateMetrics(n_queries=5, hit_rate_at_k=0.76, mrr_at_k=0.0, ndcg_at_k=0.0, k=5)
        assert _check_ci_gate(agg, baseline) is True  # 4% < 5%

    def test_large_drop_fails(self) -> None:
        from banks_rag.interface.cli.evaluate import _check_ci_gate
        baseline = {"retrieval": {"hit_rate_at_k": 0.8}}
        agg = AggregateMetrics(n_queries=5, hit_rate_at_k=0.7, mrr_at_k=0.0, ndcg_at_k=0.0, k=5)
        assert _check_ci_gate(agg, baseline) is False  # 10% > 5%


@pytest.mark.unit
class TestBuildReport:
    def test_report_contains_sections(self) -> None:
        from banks_rag.interface.cli.evaluate import _build_report
        agg = AggregateMetrics(n_queries=10, hit_rate_at_k=0.8, mrr_at_k=0.7, ndcg_at_k=0.75, k=5)
        routing = {"accuracy": 0.9, "sql_recall": 0.88, "n_cases": 30}
        gen = {"faithfulness": 0.85, "answer_relevancy": 0.8, "context_precision": 0.9, "n_evaluated": 15}
        report = _build_report(agg, routing, gen, k=5, baseline=None)
        assert "Hit-rate@5" in report
        assert "SQL Routing" in report
        assert "Generation" in report
        assert "80.0%" in report  # hit-rate

    def test_report_shows_delta_vs_baseline(self) -> None:
        from banks_rag.interface.cli.evaluate import _build_report
        agg = AggregateMetrics(n_queries=5, hit_rate_at_k=0.9, mrr_at_k=0.8, ndcg_at_k=0.85, k=5)
        routing = {"accuracy": 0.9, "sql_recall": 0.9, "n_cases": 10}
        gen = {"faithfulness": 0.9, "answer_relevancy": 0.9, "context_precision": 0.9, "n_evaluated": 5}
        baseline = {"retrieval": {"hit_rate_at_k": 0.8, "mrr_at_k": 0.7, "ndcg_at_k": 0.75}}
        report = _build_report(agg, routing, gen, k=5, baseline=baseline)
        assert "+10.0%" in report  # delta recall 0.9 - 0.8 = +10%
