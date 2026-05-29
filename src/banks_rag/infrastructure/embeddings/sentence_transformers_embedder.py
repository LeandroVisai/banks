"""Adapter de ``sentence-transformers`` que cumple el Protocol ``Embedder``.

Carga modelos texto (Qwen3-Embedding, E5) o multimodales (Qwen3-VL-Embedding).
Soporta:

- Cache local en ``models/<owner--name>/`` (descubierto al cargar).
- Cadena de fallback: si el modelo principal no carga, intenta el siguiente.
- Embedding L2-normalizado (``normalize_embeddings=True``).
- Embedding de imágenes para modelos VL (PIL → encode).

La carga es lazy: ``__init__`` solo guarda el nombre, ``_ensure_loaded`` hace
la inicialización pesada al primer ``encode``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from banks_rag.config import paths

from .base import is_vl_model

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbedder:
    """Adapter para modelos cargados con ``sentence-transformers``.

    Implementa el Protocol ``Embedder`` con carga lazy y fallback chain.
    """

    def __init__(
        self,
        model_name: str,
        *,
        models_dir: Path | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        self._requested_name = model_name
        # Convención del proyecto: modelos pre-descargados en `<repo_root>/models/<owner>--<name>/`
        # para deploy offline en H100. NO usar HF cache.
        self._models_dir = models_dir or (paths.ROOT / "models")
        self._fallback_models = fallback_models or []

        # Estado tras cargar:
        self.name: str = ""
        self.dim: int = 0
        self.max_seq_length: int = 0
        self.is_multimodal: bool = False
        self._model: SentenceTransformer | None = None
        self._tokenizer: Any = None
        # _load_lock serializa la carga del modelo (evita doble carga si dos
        # búsquedas concurrentes la disparan vía asyncio.to_thread).
        # _infer_lock serializa los forward passes sobre el mismo módulo torch:
        # la GPU es un recurso único y torch no garantiza reentrancia.
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _resolve_path(self, name: str) -> str:
        """Si existe el modelo en ``models/<owner--name>/`` usa la ruta local."""
        local = self._models_dir / name.replace("/", "--")
        return str(local) if local.exists() else name

    def _try_load(self, name: str) -> bool:
        """Intenta cargar un modelo. ``True`` si tuvo éxito."""
        from sentence_transformers import SentenceTransformer

        try:
            resolved = self._resolve_path(name)
            self._model = SentenceTransformer(resolved, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            self._model = None
            return False

        self.name = name
        self.dim = self._model.get_sentence_embedding_dimension()
        self.max_seq_length = getattr(self._model, "max_seq_length", 0) or 0
        self._tokenizer = getattr(self._model, "tokenizer", None)
        self.is_multimodal = is_vl_model(name)
        return True

    def _ensure_loaded(self) -> None:
        # Double-checked locking: el fast-path sin lock evita contención una vez
        # cargado; el lock solo protege la primera carga concurrente.
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            for candidate in [self._requested_name, *self._fallback_models]:
                if self._try_load(candidate):
                    return
            raise RuntimeError(
                f"No se pudo cargar ningún modelo de embeddings. Probados: "
                f"{[self._requested_name, *self._fallback_models]}"
            )

    # ── Protocol Embedder ─────────────────────────────────────────────────────

    def encode_text(
        self,
        texts: list[str],
        batch_size: int = 4,
    ) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        with self._infer_lock:
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

    def encode_image(self, image_paths: list[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None

        if not self.is_multimodal:
            raise RuntimeError(
                f"Modelo {self.name!r} no es multimodal — no soporta encode_image. "
                f"Usa Qwen3-VL-Embedding-8B o similar."
            )

        from PIL import Image as PILImage

        images = []
        for p in image_paths:
            if not p or not Path(p).exists():
                raise FileNotFoundError(f"Imagen no encontrada: {p}")
            images.append(PILImage.open(p).convert("RGB"))

        with self._infer_lock:
            return self._model.encode(
                images,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

    def encode_image_safe(self, image_path: str) -> np.ndarray | None:
        """Versión tolerante: retorna ``None`` si la imagen falla en lugar de raise.

        Útil para caer al fallback de texto descriptivo en el orquestador.
        """
        try:
            return self.encode_image([image_path])[0]
        except Exception:  # noqa: BLE001
            return None

    def count_tokens(self, text: str) -> int:
        self._ensure_loaded()
        if self._tokenizer is None:
            return 0
        ids = self._tokenizer.encode(text, add_special_tokens=True, truncation=False)
        return len(ids)


# Singleton de proceso: el embedder es caro (modelo de varios GB) y la carga es
# lazy, así que UNA instancia compartida carga el modelo una sola vez y lo
# reutiliza en cada búsqueda. Sin esto, cada `search_documents`/`search_visuals`
# construía una instancia nueva y recargaba el modelo desde disco — el origen
# del timeout de ~97s del Analista de Documentos.
_DEFAULT_EMBEDDER: SentenceTransformersEmbedder | None = None
_DEFAULT_EMBEDDER_LOCK = threading.Lock()


def build_default_embedder() -> SentenceTransformersEmbedder:
    """Factory memoizada: devuelve el embedder compartido del proceso.

    Respeta env vars del legacy:

    - ``RAG_EMBEDDING_MODEL``: modelo principal (default ``Qwen3-VL-Embedding-8B``,
      el embedder multimodal definitivo aprobado en la H100; se resuelve a
      ``models/Qwen3-VL-Embedding-8B`` para deploy offline). 4096-dim, L2.
    - ``SENTENCE_TRANSFORMERS_HOME``: cache de HF (auto-detectado en deploy offline).

    Fallback automático: ``intfloat/multilingual-e5-small`` (modelo chico ~80MB).

    La primera llamada fija el modelo (la env var se lee una sola vez); para
    cambiarla en tests usa :func:`reset_default_embedder`.
    """
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is not None:
        return _DEFAULT_EMBEDDER
    with _DEFAULT_EMBEDDER_LOCK:
        if _DEFAULT_EMBEDDER is not None:
            return _DEFAULT_EMBEDDER
        primary = os.environ.get("RAG_EMBEDDING_MODEL", "Qwen3-VL-Embedding-8B")
        _DEFAULT_EMBEDDER = SentenceTransformersEmbedder(
            primary,
            fallback_models=["intfloat/multilingual-e5-small"],
        )
        return _DEFAULT_EMBEDDER


def reset_default_embedder() -> None:
    """Limpia el singleton del embedder. Pensado para tests (re-leer env vars
    o evitar fuga de estado entre casos)."""
    global _DEFAULT_EMBEDDER
    with _DEFAULT_EMBEDDER_LOCK:
        _DEFAULT_EMBEDDER = None
