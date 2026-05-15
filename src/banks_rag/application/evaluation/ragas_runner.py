"""Evaluación de generación RAG: faithfulness, relevancy, context_precision.

Dos modos:
  1. **Offline simple** (sin RAGAS ni LLM): heurísticas basadas en solapamiento
     de palabras clave entre la respuesta y los chunks de contexto.
     Siempre disponible, no requiere dependencias adicionales.

     ADVERTENCIA: el ``faithfulness`` offline es solapamiento léxico (Jaccard).
     Una respuesta puede repetir el vocabulario del contexto y aun así alucinar
     cifras o relaciones — el solapamiento no detecta contradicciones. Úsese
     SOLO como smoke-test, NUNCA como gate de calidad de producción.

  2. **RAGAS** (opcional): usa el paquete ``ragas`` si está instalado y se
     provee un LLM compatible. Se activa automáticamente cuando ``use_ragas=True``
     y el paquete está disponible. Este es el modo válido para medir calidad.

Uso típico (offline):
    result = evaluate_generation(
        query="¿Qué decidió el Consejo?",
        answer="El Consejo decidió mantener la TPM en 5.5%.",
        context_chunks=[{"text": "El Consejo acordó mantener la TPM..."}],
    )
    print(result)  # GenerationEval(faithfulness=0.8, relevancy=0.9, ...)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_GOLDEN_PATH = Path(__file__).parents[4] / "data" / "golden_set" / "generation.jsonl"

_STOPWORDS_ES = frozenset(
    "el la los las un una de del en y a que es con se por para lo su al"
    " no si pero también más así como".split()
)


@dataclass
class GenerationEval:
    """Scores para una respuesta generada."""

    query: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    mode: str = "offline"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregateGenerationMetrics:
    """Promedios sobre todas las evaluaciones de generación."""

    n_evaluated: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    mode: str

    def to_dict(self) -> dict:
        return asdict(self)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS_ES and len(t) > 2}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _faithfulness_offline(answer: str, chunks: list[dict]) -> float:
    """Solapamiento léxico entre la respuesta y la unión de los chunks."""
    ans_tokens = _tokenize(answer)
    ctx_tokens: set[str] = set()
    for c in chunks:
        ctx_tokens |= _tokenize(c.get("text", ""))
    return _overlap(ans_tokens, ctx_tokens)


def _answer_relevancy_offline(query: str, answer: str) -> float:
    """Solapamiento léxico entre la query y la respuesta."""
    return _overlap(_tokenize(query), _tokenize(answer))


def _context_precision_offline(
    answer: str,
    chunks: list[dict],
    must_cite_doc_types: list[str],
) -> float:
    """Fracción de chunks relevantes (por doc_type) que contribuyen a la respuesta."""
    if not chunks:
        return 0.0
    ans_tokens = _tokenize(answer)
    relevant = 0
    for c in chunks:
        if must_cite_doc_types and c.get("doc_type_category", "") not in must_cite_doc_types:
            continue
        if _overlap(ans_tokens, _tokenize(c.get("text", ""))) > 0.05:
            relevant += 1
    denom = sum(
        1 for c in chunks
        if not must_cite_doc_types or c.get("doc_type_category", "") in must_cite_doc_types
    ) or len(chunks)
    return relevant / denom


def evaluate_generation(
    query: str,
    answer: str,
    context_chunks: list[dict],
    *,
    must_cite_doc_types: list[str] | None = None,
    use_ragas: bool = False,
    ragas_llm: Any | None = None,
    ragas_embeddings: Any | None = None,
) -> GenerationEval:
    """Evalúa la calidad de una respuesta generada.

    Args:
        query: pregunta original.
        answer: texto de la respuesta generada.
        context_chunks: chunks recuperados pasados como contexto al LLM.
        must_cite_doc_types: tipos de doc que deben aparecer en el contexto.
        use_ragas: intentar usar el paquete ``ragas`` si está disponible.
        ragas_llm: LLM compatible con RAGAS (solo si use_ragas=True).
        ragas_embeddings: embeddings compatibles con RAGAS (solo si use_ragas=True).

    Returns:
        ``GenerationEval`` con scores 0–1.
    """
    doc_types = must_cite_doc_types or []

    if use_ragas:
        try:
            return _evaluate_with_ragas(
                query, answer, context_chunks,
                must_cite_doc_types=doc_types,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
        except Exception:
            pass  # fallback offline

    faith = _faithfulness_offline(answer, context_chunks)
    relevancy = _answer_relevancy_offline(query, answer)
    precision = _context_precision_offline(answer, context_chunks, doc_types)

    return GenerationEval(
        query=query,
        faithfulness=round(faith, 4),
        answer_relevancy=round(relevancy, 4),
        context_precision=round(precision, 4),
        mode="offline",
    )


def _evaluate_with_ragas(
    query: str,
    answer: str,
    context_chunks: list[dict],
    *,
    must_cite_doc_types: list[str],
    llm: Any,
    embeddings: Any,
) -> GenerationEval:
    """Delegación a RAGAS cuando está instalado."""
    from ragas import evaluate as ragas_evaluate  # type: ignore[import]
    from ragas.metrics import (  # type: ignore[import]
        answer_relevancy,
        context_precision,
        faithfulness,
    )
    from datasets import Dataset  # type: ignore[import]

    contexts = [c.get("text", "") for c in context_chunks]
    ds = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
    })
    kwargs: dict[str, Any] = {"metrics": [faithfulness, answer_relevancy, context_precision]}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    scores = ragas_evaluate(ds, **kwargs)
    row = scores.to_pandas().iloc[0]
    return GenerationEval(
        query=query,
        faithfulness=float(row.get("faithfulness", 0.0)),
        answer_relevancy=float(row.get("answer_relevancy", 0.0)),
        context_precision=float(row.get("context_precision", 0.0)),
        mode="ragas",
    )


def aggregate_generation(results: list[GenerationEval]) -> AggregateGenerationMetrics:
    if not results:
        return AggregateGenerationMetrics(
            n_evaluated=0, faithfulness=0.0, answer_relevancy=0.0,
            context_precision=0.0, mode="offline",
        )
    n = len(results)
    mode = results[0].mode
    return AggregateGenerationMetrics(
        n_evaluated=n,
        faithfulness=round(sum(r.faithfulness for r in results) / n, 4),
        answer_relevancy=round(sum(r.answer_relevancy for r in results) / n, 4),
        context_precision=round(sum(r.context_precision for r in results) / n, 4),
        mode=mode,
    )


def load_generation_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    """Lee el golden set de generación desde el JSONL."""
    p = path or _GOLDEN_PATH
    if not p.exists():
        return []
    cases = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases
