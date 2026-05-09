"""Protocol ``Embedder`` y helpers para detectar familia del modelo.

El Protocol define la interfaz mínima que debe cumplir cualquier adapter de
embeddings (SentenceTransformersEmbedder, futuros llama-cpp embedders, etc.).
Permite que ``application/ingestion/vectorize_corpus.py`` no dependa de
sentence-transformers concretamente.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Interfaz mínima para un proveedor de embeddings (texto y opcional imagen).

    Atributos:
        name: nombre/identificador del modelo cargado.
        dim: dimensión del vector de salida.
        max_seq_length: tokens máximos antes de truncar (0 si no aplica).
        is_multimodal: ``True`` si soporta ``encode_image``.
    """

    name: str
    dim: int
    max_seq_length: int
    is_multimodal: bool

    def encode_text(self, texts: list[str], batch_size: int = 4) -> np.ndarray:
        """Codifica una lista de textos. Embeddings normalizados L2."""
        ...

    def encode_image(self, image_paths: list[str]) -> np.ndarray:
        """Codifica imágenes (solo modelos multimodales). Normalizados L2."""
        ...

    def count_tokens(self, text: str) -> int:
        """Tokens del texto (sin límite). 0 si el modelo no expone tokenizer."""
        ...


# ── Detectores de familia del modelo (pure, sin I/O) ─────────────────────────


def is_e5_model(name: str) -> bool:
    """E5 family (intfloat/multilingual-e5-*, intfloat/e5-*) requiere prefix
    ``"query: "`` / ``"passage: "`` en el texto antes de embeber."""
    return "e5" in name.lower()


def is_qwen_embedding(name: str) -> bool:
    """Qwen embedding family (Qwen3-Embedding-8B, Qwen3-VL-Embedding-8B):
    documentos sin prefijo, queries con instrucción ``"Instruct: ..."``."""
    n = name.lower()
    return "qwen" in n and "embedding" in n


def is_vl_model(name: str) -> bool:
    """Modelos vision-language: aceptan imágenes y texto en el mismo espacio.

    Ejemplos: ``Qwen3-VL-Embedding-8B``, ``Qwen3-VL-Embedding``.
    """
    n = name.lower()
    return ("vl" in n or "vision" in n) and ("embedding" in n or "embed" in n)
