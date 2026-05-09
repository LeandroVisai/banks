"""Orquestador del paso de persistencia (legacy paso 03).

Toma documentos + chunks vectorizados y los carga en PostgreSQL via ``PostgresRepo``.

Operaciones expuestas:
  - ``setup_corpus``: crea BD (si falta) + extension vector + schema + índices.
  - ``persist_corpus``: upsert de documentos + delete + insert de chunks.
  - ``corpus_stats``: queries de inspección formateadas.
  - ``detect_embedding_dim``: heurística para inferir la dimensión sin tocar BD.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from banks_rag.infrastructure.persistence.postgres_repo import PostgresRepo


@dataclass
class PersistenceResult:
    documents_upserted: int
    chunks_inserted: int
    chunks_skipped_invalid_dim: int


def detect_embedding_dim(chunks: list[dict] | None = None) -> int:
    """Detecta la dimensión del embedding en orden de prioridad:

    1. Variable de entorno ``RAG_EMBEDDING_DIM``.
    2. Campo ``embedding_dim`` del primer chunk.
    3. Longitud real del primer ``embedding``.
    4. Fallback ``4096`` (Qwen3-Embedding-8B / Qwen3-VL-Embedding-8B).
    """
    env_dim = os.getenv("RAG_EMBEDDING_DIM")
    if env_dim:
        return int(env_dim)
    if chunks:
        first = chunks[0]
        dim = first.get("embedding_dim") or len(first.get("embedding") or [])
        if dim:
            return int(dim)
    return 4096


def setup_corpus(
    repo: PostgresRepo,
    *,
    dim: int,
    drop_first: bool = False,
) -> None:
    """Crea BD (si falta), valida pgvector, aplica schema + índices.

    Levanta ``RuntimeError`` si la extensión ``vector`` no está disponible.
    """
    repo.ensure_database()
    if not repo.pgvector_available():
        raise RuntimeError(
            "La extensión 'vector' no está disponible en el servidor PostgreSQL. "
            "Compila pgvector desde source o instala el paquete: "
            "https://github.com/pgvector/pgvector"
        )
    repo.setup_schema(dim=dim, drop_first=drop_first)


def persist_corpus(
    repo: PostgresRepo,
    documents: list[dict],
    chunks: list[dict],
    *,
    expected_dim: int | None = None,
) -> PersistenceResult:
    """Carga corpus completo: upsert documents + delete chunks + insert chunks.

    Args:
        repo: ``PostgresRepo`` ya configurado.
        documents: lista de dicts con campos del schema.
        chunks: lista de dicts con campos del schema + ``embedding``.
        expected_dim: dimensión esperada. Si ``None``, se infiere de ``chunks``.

    Verifica que la dimensión esperada coincida con la dimensión real de la
    columna ``embedding`` en la tabla. Si difieren, levanta ``ValueError`` con
    instrucciones para resolver (cambiar prefijo, hacer reset, re-vectorizar).
    """
    dim = expected_dim or detect_embedding_dim(chunks)

    existing_dim = repo.existing_embedding_dim()
    if existing_dim is not None and existing_dim != dim:
        raise ValueError(
            f"Dimensión de embedding incompatible: tabla {repo.chunks_table} tiene "
            f"vector({existing_dim}) pero los chunks producen vector({dim}).\n"
            f"Opciones:\n"
            f"  1. Tabla nueva por modelo: setear RAG_TABLE_PREFIX y correr setup.\n"
            f"  2. Reset (destructivo): correr setup con drop_first=True.\n"
            f"  3. Re-vectorizar con el mismo modelo de la tabla existente."
        )

    repo.upsert_documents(documents)
    doc_ids = [d["document_id"] for d in documents]
    repo.delete_chunks_for_docs(doc_ids)
    inserted = repo.insert_chunks(chunks, expected_dim=dim)
    skipped = len(chunks) - inserted

    return PersistenceResult(
        documents_upserted=len(documents),
        chunks_inserted=inserted,
        chunks_skipped_invalid_dim=skipped,
    )


def corpus_stats(repo: PostgresRepo) -> dict:
    """Métricas de la BD actual (count + distribución por tipo + importance avg)."""
    return repo.stats()
