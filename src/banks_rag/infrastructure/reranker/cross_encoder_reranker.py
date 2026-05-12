"""CrossEncoderReranker — reranker de segunda pasada para hybrid_search.

Usa ``sentence-transformers`` ``CrossEncoder`` (bge-reranker-v2-m3 por defecto)
para re-puntuar los chunks después de RRF + MMR + importance_boost.

El reranker toma los top-N candidatos del pipeline existente, les asigna un
score cross-encoder (query, text) y los re-ordena. El beneficio es mayor
precisión a expensas de latencia (~20-100ms en GPU para N≤50).

Convención de paths:
    ``models/<owner>--<name>/`` para uso offline en H100.
    Si el directorio no existe se descarga desde HuggingFace.

Protocol ``Reranker``:
    Cualquier clase que implemente ``rerank(query, chunks, top_k)`` y tenga
    ``name``, ``loaded`` puede usarse como reranker en hybrid_search.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from banks_rag.config.paths import MODELS_DIR

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# Texto del chunk que se pasa al cross-encoder. El campo "text" es el principal;
# agregamos section_type como contexto breve si está disponible.
_TEXT_FIELD = "text"


@runtime_checkable
class Reranker(Protocol):
    """Protocol mínimo para un reranker de segunda pasada."""

    name: str
    loaded: bool

    def load(self) -> None:
        """Carga el modelo. Idempotente."""
        ...

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Re-ordena ``chunks`` por relevancia a ``query``.

        Args:
            query: texto de la consulta (clean_query del ParsedQuery).
            chunks: lista de dicts con al menos el campo ``text``.
            top_k: número de chunks a devolver. ``None`` = todos.

        Returns:
            Lista re-ordenada de mayor a menor score. Cada chunk recibe un
            campo adicional ``reranker_score`` (float).
        """
        ...

    def info(self) -> dict:
        """Metadata para health checks."""
        ...


class CrossEncoderReranker:
    """Reranker basado en ``CrossEncoder`` de sentence-transformers.

    Parámetros:
        model_name: nombre HuggingFace o path local. Default: bge-reranker-v2-m3.
        batch_size: pares (query, text) por batch en ``predict()``. Default 32.
        max_length: tokens máximos para la concatenación (query, text). Default 512.
        models_dir: directorio raíz para buscar modelos pre-descargados offline.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        max_length: int = 512,
        models_dir: Path | None = None,
    ) -> None:
        self._requested_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._models_dir = models_dir or MODELS_DIR
        self._model: Any = None
        self.name: str = model_name
        self.loaded: bool = False

    def _resolve_path(self, name: str) -> str:
        local = self._models_dir / name.replace("/", "--")
        return str(local) if local.exists() else name

    def load(self) -> None:
        """Carga el CrossEncoder. Idempotente."""
        if self.loaded:
            return
        from sentence_transformers import CrossEncoder

        resolved = self._resolve_path(self._requested_name)
        log.info("Cargando reranker: %s", resolved)
        t0 = time.monotonic()
        self._model = CrossEncoder(
            resolved,
            max_length=self._max_length,
        )
        self.loaded = True
        log.info("Reranker listo en %.1fs", time.monotonic() - t0)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Re-ordena chunks por score cross-encoder. Añade ``reranker_score``."""
        if not chunks:
            return chunks

        if not self.loaded:
            self.load()

        pairs = [(query, _extract_text(c)) for c in chunks]
        scores: list[float] = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        ).tolist()

        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        result = []
        limit = top_k if top_k is not None else len(scored)
        for score, chunk in scored[:limit]:
            c = dict(chunk)
            c["reranker_score"] = round(float(score), 6)
            result.append(c)
        return result

    def info(self) -> dict:
        return {
            "name": self.name,
            "loaded": self.loaded,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
        }


def _extract_text(chunk: dict) -> str:
    """Texto del chunk para el par (query, text). Incluye section_type si existe."""
    text = chunk.get(_TEXT_FIELD, "")
    section = chunk.get("section_type", "")
    if section:
        return f"[{section}] {text}"
    return text
