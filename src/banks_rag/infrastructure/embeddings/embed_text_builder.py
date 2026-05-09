"""Construcción del texto que va al modelo de embeddings.

Inyecta señal semántica del enriquecimiento como prefijo compacto:

    [TEMPORAL_PREFIX] [DOC_TYPE | SECTION | top_vars]→INDICATOR  texto...

- ``TEMPORAL_PREFIX`` viene de ``taxonomy.TEMPORAL_SOURCE_PREFIXES`` (e.g.
  ``[DAILY 2024-03-15]`` para Monitor PM, ``[PERIOD_MONTHLY 2024-03]`` para
  Comunicados/Research, ``[PERIOD_QUARTERLY 2024-03]`` para IPOM/IEF).
- Las top-3 variables se ordenan por importance (CRITICAL → HIGH → MEDIUM).
- ``→LEADING`` / ``→LAGGING`` indica el tipo de la variable más importante.

Modelos E5 reciben además el prefijo ``"passage: "`` al inicio. Modelos Qwen3
no llevan prefijo en passages (la instrucción es solo para queries en retrieval).

El builder se desactiva con env var ``RAG_PURE_TEXT=1`` (texto plano sin contexto).
"""

from __future__ import annotations

import os

from banks_rag.domain_knowledge.taxonomy import (
    DEFAULT_TEMPORAL_PREFIX,
    TEMPORAL_SOURCE_PREFIXES,
)

from .base import is_e5_model

E5_PASSAGE_PREFIX = "passage: "

_VAR_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}


def metadata_context_enabled() -> bool:
    """``True`` salvo que ``RAG_PURE_TEXT=1`` desactive la inyección de contexto."""
    return os.environ.get("RAG_PURE_TEXT", "0") != "1"


def _temporal_prefix(doc_type: str, date_str: str) -> str:
    """Prefijo temporal según tipo de fuente y fecha disponible."""
    base = TEMPORAL_SOURCE_PREFIXES.get(doc_type, DEFAULT_TEMPORAL_PREFIX)
    if base == "[DAILY]":
        return f"[DAILY {date_str}]" if date_str else "[DAILY]"
    if date_str and len(date_str) >= 7:
        return base.replace("]", f" {date_str[:7]}]")
    return base


def _top_variables(economic_variables: dict, k: int = 3) -> list[tuple[str, dict]]:
    """Top-K variables ordenadas por importance level (CRITICAL primero)."""
    return sorted(
        economic_variables.items(),
        key=lambda kv: _VAR_ORDER.get(kv[1].get("importance", "MEDIUM"), 3),
    )[:k]


def _indicator_suffix(top_var_meta: dict | None) -> str:
    if not top_var_meta:
        return ""
    indicator = top_var_meta.get("indicator_type")
    if indicator == "LEADING":
        return "→LEADING"
    if indicator == "LAGGING":
        return "→LAGGING"
    return ""


def build_embed_text(chunk: dict, model_name: str, *, with_metadata: bool | None = None) -> str:
    """Construye el texto que va al modelo de embeddings.

    Args:
        chunk: dict del enriched chunk (con ``doc_type_category``, ``section_type``,
            ``economic_variables``, ``document_date`` o ``chunk_date``, ``text``).
        model_name: nombre del modelo (usado para detectar familia E5/Qwen).
        with_metadata: si ``None``, lee el flag de env (``RAG_PURE_TEXT``).
            Pásalo explícito en tests para forzar comportamiento.

    Returns:
        Texto listo para ``model.encode()``.
    """
    base = chunk["text"]
    use_meta = metadata_context_enabled() if with_metadata is None else with_metadata

    if use_meta:
        parts: list[str] = [
            chunk.get("doc_type_category", ""),
            chunk.get("section_type", ""),
        ]
        vars_sorted = _top_variables(chunk.get("economic_variables", {}))
        if vars_sorted:
            parts.append(", ".join(v[0] for v in vars_sorted))

        doc_type = chunk.get("doc_type_category", "")
        date_str = chunk.get("document_date") or chunk.get("chunk_date") or ""
        temporal_prefix = _temporal_prefix(doc_type, date_str)

        top_meta = vars_sorted[0][1] if vars_sorted else None
        suffix = _indicator_suffix(top_meta)

        ctx = temporal_prefix + " [" + " | ".join(p for p in parts if p) + "]" + suffix
        base = f"{ctx} {base}"

    if is_e5_model(model_name):
        base = E5_PASSAGE_PREFIX + base
    # Qwen embedding: passages sin prefijo (la instrucción se aplica solo a queries).
    return base
