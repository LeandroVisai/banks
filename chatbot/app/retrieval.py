"""
Retrieval híbrido: vector recall + BM25 recall → RRF → MMR → importance boost.

Mantiene paridad de comportamiento con `04_search.py` del pipeline padre,
pero todo es async (las recalls van en paralelo) y desacoplado de variables
de módulo globales.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable

import numpy as np

from . import db, embeddings
from .settings import settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_embedding(emb) -> np.ndarray:
    if isinstance(emb, str):
        return np.array([float(x) for x in emb.strip("[]").split(",")], dtype=np.float32)
    if isinstance(emb, (list, tuple)):
        return np.asarray(emb, dtype=np.float32)
    return np.asarray(emb, dtype=np.float32)


def _rrf_fuse(
    vec_hits: list[dict],
    lex_hits: list[dict],
    k: int,
) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for rank, row in enumerate(vec_hits, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        docs[cid] = dict(row)
    for rank, row in enumerate(lex_hits, start=1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in docs:
            docs[cid] = dict(row)
    ordered = sorted(docs.values(), key=lambda d: scores[d["chunk_id"]], reverse=True)
    for d in ordered:
        d["rrf_score"] = scores[d["chunk_id"]]
    return ordered


def _mmr_select(
    candidates: list[dict],
    query_vec: np.ndarray,
    k: int,
    lambda_: float,
) -> list[dict]:
    if not candidates:
        return []
    vecs = [_parse_embedding(c["embedding"]) for c in candidates]
    sims = np.array([float(np.dot(query_vec, v)) for v in vecs])
    selected: list[int] = []
    remaining = set(range(len(candidates)))
    while len(selected) < k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: sims[i])
        else:
            best = max(
                remaining,
                key=lambda i: (
                    lambda_ * sims[i]
                    - (1 - lambda_) * max(float(np.dot(vecs[i], vecs[s])) for s in selected)
                ),
            )
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


def _importance_boost(hits: list[dict], boost: float) -> list[dict]:
    for h in hits:
        h["final_score"] = h.get("rrf_score", 0.0) + boost * float(h.get("importance_score") or 0.0)
    hits.sort(key=lambda h: h["final_score"], reverse=True)
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

async def retrieve(query: str, k: int = None) -> list[dict]:
    """Top-k chunks fusionados y diversificados para `query`."""
    k = k or settings.rag_top_k

    # 1. Embedding de la query (en threadpool) en paralelo con… nada por ahora.
    query_vec = await embeddings.embed_query(query)

    # 2. Recalls en paralelo (vector + lexical)
    vec_task = asyncio.create_task(db.vector_recall(query_vec.tolist(), settings.rag_recall_n))
    lex_task = asyncio.create_task(db.lexical_recall(query, settings.rag_recall_n))
    vec_hits, lex_hits = await asyncio.gather(vec_task, lex_task)

    # 3. Fusión RRF
    fused = _rrf_fuse(vec_hits, lex_hits, k=settings.rrf_k)
    if not fused:
        return []

    # 4. MMR sobre el doble del top-k para tener margen
    mmr_pool = _mmr_select(fused, query_vec, k=k * 2, lambda_=settings.mmr_lambda)

    # 5. Re-rank por importance
    final = _importance_boost(mmr_pool, boost=settings.importance_boost)

    log.debug(
        "retrieve(query='%.50s…') vec=%d lex=%d fused=%d → %d",
        query, len(vec_hits), len(lex_hits), len(fused), min(len(final), k),
    )
    return final[:k]


def format_sources(chunks: Iterable[dict]) -> list[dict]:
    """Convierte chunks en referencias limpias para la respuesta API."""
    out: list[dict] = []
    for i, c in enumerate(chunks, start=1):
        date = str(c.get("chunk_date") or c.get("document_date") or "")
        out.append({
            "ref": i,
            "filename": c.get("filename", ""),
            "doc_type": c.get("doc_type_category", ""),
            "section": c.get("section_type", ""),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "date": date,
            "importance": round(float(c.get("importance_score") or 0.0), 3),
            "text_snippet": (c.get("text") or "")[:200],
        })
    return out
