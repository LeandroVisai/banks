"""Embedding de queries — idéntico al de chatbot/."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import numpy as np

from .settings import EMBEDDING_CACHE_DIR, MODELS_DIR, settings

log = logging.getLogger(__name__)

_model = None
_model_name: Optional[str] = None


def _resolve_model_path() -> str:
    model_id = settings.embedding_model_id
    if EMBEDDING_CACHE_DIR.exists():
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(EMBEDDING_CACHE_DIR))
    local_dir = MODELS_DIR / model_id.replace("/", "--")
    if local_dir.exists():
        return str(local_dir)
    return model_id


def _load_model_sync():
    global _model, _model_name
    if _model is not None:
        return _model

    from sentence_transformers import SentenceTransformer

    name = _resolve_model_path()
    log.info("Cargando modelo de embedding: %s", name)
    try:
        _model = SentenceTransformer(name)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el embedding '{settings.embedding_model_id}': {e}"
        ) from e
    _model_name = name
    return _model


async def warm_up() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_model_sync)


def _encode_sync(text: str) -> np.ndarray:
    model = _load_model_sync()
    prefix = "query: " if "e5" in (_model_name or "").lower() else ""
    vec = model.encode(
        [prefix + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vec.astype(np.float32)


async def embed_query(text: str) -> np.ndarray:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_sync, text)
