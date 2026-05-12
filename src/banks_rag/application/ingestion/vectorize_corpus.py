"""Orquestador del paso de vectorización (legacy paso 02).

Toma chunks enriquecidos + un ``Embedder`` y produce los mismos chunks con:

  - ``embedding``: lista de floats (vector L2-normalizado).
  - ``embedding_dim``: dimensión del modelo.
  - ``embedding_model``: nombre del modelo usado.

Genera además:

  - ``excel_daily_records.json``: chunks de Monitor PM.
  - ``pdf_period_records.json``: chunks de PDFs.
  - ``cross_references.json``: enlaces bidireccionales (vía
    ``domain_knowledge.cross_references.build_cross_references``).

Política multimodal:
  - Si el embedder es VL y la imagen carga OK: dual-embedding
    ``IMAGE_WEIGHT * img + (1 - IMAGE_WEIGHT) * txt``, luego L2-normalizado.
  - Si el embedder no es VL, o la imagen falla: embedding del texto descriptivo
    (``"[Imagen p.N]"`` ya tiene contexto mínimo del chunker visual).

``IMAGE_WEIGHT`` (default 0.7) configurable con env var ``RAG_VISUAL_IMG_WEIGHT``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from banks_rag.domain_knowledge.cross_references import build_cross_references
from banks_rag.infrastructure.embeddings.embed_text_builder import build_embed_text


class _EmbedderLike(Protocol):
    """Subset del Protocol ``Embedder`` que necesita este orquestador."""

    name: str
    dim: int
    max_seq_length: int
    is_multimodal: bool

    def encode_text(self, texts: list[str], batch_size: int = 4) -> np.ndarray: ...
    def encode_image_safe(self, image_path: str) -> np.ndarray | None: ...
    def count_tokens(self, text: str) -> int: ...


@dataclass
class VectorizationReport:
    model: str
    dimension: int
    max_seq_length: int
    total_chunks: int
    text_chunks: int
    image_chunks: int
    vl_model: bool
    truncated_count: int
    metadata_context_enabled: bool
    excel_daily_count: int
    pdf_period_count: int
    image_weight: float = 0.7
    dual_embed_count: int = 0
    token_stats: dict | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "total_chunks": self.total_chunks,
            "text_chunks": self.text_chunks,
            "image_chunks": self.image_chunks,
            "vl_model": self.vl_model,
            "truncated_count": self.truncated_count,
            "metadata_context_enabled": self.metadata_context_enabled,
            "excel_daily_count": self.excel_daily_count,
            "pdf_period_count": self.pdf_period_count,
            "image_weight": self.image_weight,
            "dual_embed_count": self.dual_embed_count,
            "token_stats": self.token_stats,
        }


@dataclass
class VectorizationResult:
    chunks: list[dict] = field(default_factory=list)
    excel_daily_records: list[dict] = field(default_factory=list)
    pdf_period_records: list[dict] = field(default_factory=list)
    cross_references: dict = field(default_factory=dict)
    report: VectorizationReport | None = None


def _split_text_and_image(chunks: list[dict]) -> tuple[list[int], list[int]]:
    text_idxs = [i for i, c in enumerate(chunks) if not c.get("image_path")]
    image_idxs = [i for i, c in enumerate(chunks) if c.get("image_path")]
    return text_idxs, image_idxs


def _count_truncated(
    embedder: _EmbedderLike,
    texts: list[str],
) -> tuple[int, list[int]]:
    """Cuenta cuántos textos exceden ``max_seq_length`` y devuelve ``token_counts``."""
    token_counts: list[int] = []
    truncated = 0
    if embedder.max_seq_length <= 0:
        return 0, []
    for text in texts:
        n = embedder.count_tokens(text)
        if n == 0:
            continue
        token_counts.append(n)
        if n > embedder.max_seq_length:
            truncated += 1
    return truncated, token_counts


def _embed_image_chunks(
    embedder: _EmbedderLike,
    chunks: list[dict],
    image_idxs: list[int],
    image_weight: float = 0.7,
) -> tuple[dict[int, np.ndarray], int]:
    """Embebe chunks visuales con dual-embedding cuando el modelo es VL.

    Si el embedder es VL y la imagen carga OK:
        emb = L2_norm(image_weight * img_emb + (1 - image_weight) * txt_emb)
    Si no (modelo no VL, imagen faltante o falla de carga):
        emb = txt_emb   (fallback transparente)

    Returns:
        Tupla (dict idx→embedding, dual_embed_count).
    """
    out: dict[int, np.ndarray] = {}
    dual_count = 0
    for i in image_idxs:
        chunk = chunks[i]
        img_emb: np.ndarray | None = None
        if embedder.is_multimodal:
            img_emb = embedder.encode_image_safe(chunk.get("image_path", ""))

        text = build_embed_text(chunk, embedder.name)
        txt_emb: np.ndarray = embedder.encode_text([text], batch_size=1)[0]

        if img_emb is not None:
            combined = image_weight * img_emb + (1.0 - image_weight) * txt_emb
            norm = np.linalg.norm(combined)
            out[i] = combined / norm if norm > 0.0 else txt_emb
            dual_count += 1
        else:
            out[i] = txt_emb
    return out, dual_count


def vectorize_corpus(
    chunks: list[dict],
    embedder: _EmbedderLike,
    *,
    batch_size: int = 4,
    image_weight: float | None = None,
) -> VectorizationResult:
    """Vectoriza un corpus completo de chunks enriquecidos.

    Args:
        chunks: lista de dicts (los del JSON producido por ``enrich_corpus``).
            Se mutan agregándoles ``embedding``, ``embedding_dim``, ``embedding_model``.
        embedder: objeto que cumple el subset de ``Embedder`` requerido.
        batch_size: tamaño de batch para encode_text. Para Qwen3-Embedding-8B
            usar 4–8; para E5-multilingual-small puede ser 32+.
        image_weight: peso de la imagen en el dual-embedding (0–1). Si es None
            se lee ``RAG_VISUAL_IMG_WEIGHT`` (default 0.7).

    Returns:
        ``VectorizationResult`` con chunks enriquecidos in-place + records
        separados (excel/pdf) + cross-references + reporte.
    """
    if image_weight is None:
        image_weight = float(os.environ.get("RAG_VISUAL_IMG_WEIGHT", "0.7"))

    text_idxs, image_idxs = _split_text_and_image(chunks)

    # ── Texto: build prompts y embed batch ────────────────────────────────────
    text_prompts = [build_embed_text(chunks[i], embedder.name) for i in text_idxs]
    truncated, token_counts = _count_truncated(embedder, text_prompts)

    text_embeddings = (
        embedder.encode_text(text_prompts, batch_size=batch_size)
        if text_prompts
        else np.empty((0, embedder.dim), dtype=np.float32)
    )

    # ── Imágenes: dual-embedding (VL) o fallback texto ───────────────────────
    image_embeddings, dual_embed_count = _embed_image_chunks(
        embedder, chunks, image_idxs, image_weight=image_weight
    )

    # ── Combinar y poblar campo embedding en cada chunk ───────────────────────
    for idx, i in enumerate(text_idxs):
        chunks[i]["embedding"] = text_embeddings[idx].tolist()
        chunks[i]["embedding_dim"] = embedder.dim
        chunks[i]["embedding_model"] = embedder.name
    for i, emb in image_embeddings.items():
        chunks[i]["embedding"] = emb.tolist()
        chunks[i]["embedding_dim"] = embedder.dim
        chunks[i]["embedding_model"] = embedder.name

    # ── Records separados por tipo de fuente ──────────────────────────────────
    excel_daily = [c for c in chunks if c.get("doc_type_category") == "MONITOR_PM"]
    pdf_period = [c for c in chunks if c.get("doc_type_category") != "MONITOR_PM"]

    cross_refs = build_cross_references(excel_daily, pdf_period)

    from banks_rag.infrastructure.embeddings.embed_text_builder import (
        metadata_context_enabled,
    )

    token_stats = (
        {
            "min": min(token_counts),
            "mean": sum(token_counts) // len(token_counts),
            "max": max(token_counts),
        }
        if token_counts
        else None
    )

    report = VectorizationReport(
        model=embedder.name,
        dimension=embedder.dim,
        max_seq_length=embedder.max_seq_length,
        total_chunks=len(chunks),
        text_chunks=len(text_idxs),
        image_chunks=len(image_idxs),
        vl_model=embedder.is_multimodal,
        truncated_count=truncated,
        metadata_context_enabled=metadata_context_enabled(),
        excel_daily_count=len(excel_daily),
        pdf_period_count=len(pdf_period),
        image_weight=image_weight,
        dual_embed_count=dual_embed_count,
        token_stats=token_stats,
    )

    return VectorizationResult(
        chunks=chunks,
        excel_daily_records=excel_daily,
        pdf_period_records=pdf_period,
        cross_references=cross_refs,
        report=report,
    )
