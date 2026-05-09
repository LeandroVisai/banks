"""Dependencias compartidas del API (DI vía closures + lazy state).

Mantiene una única instancia de:
  - LLM engine (cargada en lifespan).
  - Embedder (lazy load on first use).
  - PostgresRepo (creado por request, conexión barata).

Los routes leen vía ``request.app.state.<thing>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from banks_rag.infrastructure.embeddings import SentenceTransformersEmbedder
    from banks_rag.infrastructure.llm import LLMEngine
    from banks_rag.infrastructure.persistence import PostgresRepo


class AppState:
    """Container de singletons del API. Se monta en ``app.state.deps``."""

    def __init__(self) -> None:
        self.llm: "LLMEngine | None" = None
        self.embedder: "SentenceTransformersEmbedder | None" = None
        self.repo: "PostgresRepo | None" = None
