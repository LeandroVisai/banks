"""CrossEncoderReranker — reranker de segunda pasada para hybrid_search.

Usa ``sentence-transformers`` ``CrossEncoder`` (jina-reranker-v3 por defecto)
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
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from banks_rag.config.paths import MODELS_DIR

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "jinaai/jina-reranker-v3"

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
        model_name: nombre HuggingFace o path local. Default: jina-reranker-v3.
        batch_size: pares (query, text) por batch en ``predict()``. Default 32.
        max_length: tokens máximos para la concatenación (query, text). Default 2048.
        models_dir: directorio raíz para buscar modelos pre-descargados offline.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        max_length: int = 2048,
        models_dir: Path | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._requested_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._models_dir = models_dir or MODELS_DIR
        # Los rerankers Jina (y otros con código custom) requieren
        # trust_remote_code=True para cargar. bge no lo necesita pero lo tolera.
        self._trust_remote_code = trust_remote_code
        self._model: Any = None
        self.name: str = model_name
        self.loaded: bool = False
        # Si load() ya falló una vez, no reintentar en cada rerank (evita gastar
        # tiempo y log-spam): el reranker queda permanentemente degradado.
        self._load_failed: bool = False

    def _resolve_path(self, name: str) -> str:
        local = self._models_dir / name.replace("/", "--")
        return str(local) if local.exists() else name

    def load(self) -> None:
        """Carga el CrossEncoder. Idempotente.

        Pasa ``trust_remote_code`` cuando la versión de sentence-transformers lo
        soporta; si no, reintenta sin ese argumento (compatibilidad con versiones
        antiguas)."""
        if self.loaded:
            return
        from sentence_transformers import CrossEncoder

        resolved = self._resolve_path(self._requested_name)
        log.info("Cargando reranker: %s (trust_remote_code=%s)",
                 resolved, self._trust_remote_code)
        t0 = time.monotonic()
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = CrossEncoder(
                resolved,
                max_length=self._max_length,
                trust_remote_code=self._trust_remote_code,
                device=device,
            )
        except TypeError:
            # sentence-transformers antiguo: CrossEncoder no acepta esos kwargs.
            self._model = CrossEncoder(resolved, max_length=self._max_length)
        self.loaded = True
        log.info("Reranker listo en %.1fs (device=%s)", time.monotonic() - t0, device)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Re-ordena chunks por score cross-encoder. Añade ``reranker_score``.

        Best-effort total: si el modelo no carga (ausente, sin internet, falta
        trust_remote_code, arquitectura no soportada) o falla al puntuar (OOM,
        par malformado), se degrada con gracia devolviendo el orden previo
        recortado a ``top_k`` en vez de tumbar la búsqueda documental.
        """
        if not chunks:
            return chunks
        limit = top_k if top_k is not None else len(chunks)

        # Carga perezosa robusta: un fallo de carga NO debe propagarse a
        # search_documents (rompería el retrieval). Se marca como degradado.
        if not self.loaded:
            if self._load_failed:
                return chunks[:limit]
            try:
                self.load()
            except Exception:  # noqa: BLE001
                self._load_failed = True
                log.exception(
                    "Reranker %r no pudo cargar; se mantiene el orden de "
                    "retrieval (sin reranking)", self._requested_name,
                )
                return chunks[:limit]

        pairs = [(query, _extract_text(c)) for c in chunks]
        try:
            scores: list[float] = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            ).tolist()
        except Exception:  # noqa: BLE001
            log.exception("Reranker falló al puntuar; se mantiene el orden previo")
            return chunks[:limit]

        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        result = []
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


# ── Singleton compartido del proceso (análogo a build_default_embedder) ──────
_DEFAULT_RERANKER: CrossEncoderReranker | None = None
_DEFAULT_RERANKER_LOCK = threading.Lock()


def reranking_enabled() -> bool:
    """``True`` salvo que ``BANKS_RERANK_ENABLED`` esté en false (en el ``.env``).

    Lee de ``Settings`` (no de ``os.getenv``) para que la variable funcione desde
    el ``.env`` como el resto de las ``BANKS_*``. Permite apagar el cross-encoder
    sin tocar código (poca VRAM, aislar latencia, o si falta el modelo)."""
    from banks_rag.config import get_settings

    return get_settings().rerank_enabled


def build_default_reranker() -> CrossEncoderReranker | None:
    """Factory memoizada del reranker compartido. ``None`` si está deshabilitado.

    Respeta (desde el ``.env`` vía ``Settings``):
      - ``BANKS_RERANK_ENABLED`` (default true) — apaga el reranker si es false.
      - ``BANKS_RERANK_MODEL`` (default ``jinaai/jina-reranker-v3``) — resuelve a
        ``models/<owner>--<name>/`` para deploy offline en H100.

    No carga el modelo aquí (la carga es lazy en el primer ``rerank``). Es
    independiente de la dimensión de los embeddings del corpus: el cross-encoder
    puntúa pares ``(query, texto)`` crudos, así que sirve igual con el corpus
    4096-dim de Qwen3-VL que con cualquier otro embedder."""
    from banks_rag.config import get_settings

    global _DEFAULT_RERANKER
    if not reranking_enabled():
        return None
    if _DEFAULT_RERANKER is not None:
        return _DEFAULT_RERANKER
    with _DEFAULT_RERANKER_LOCK:
        if _DEFAULT_RERANKER is not None:
            return _DEFAULT_RERANKER
        model_name = get_settings().rerank_model or _DEFAULT_MODEL
        _DEFAULT_RERANKER = CrossEncoderReranker(model_name)
        return _DEFAULT_RERANKER


def reset_default_reranker() -> None:
    """Limpia el singleton del reranker. Pensado para tests (re-leer env vars)."""
    global _DEFAULT_RERANKER
    with _DEFAULT_RERANKER_LOCK:
        _DEFAULT_RERANKER = None
