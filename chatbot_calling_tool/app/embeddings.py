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
        _model = SentenceTransformer(name, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el embedding '{settings.embedding_model_id}': {e}"
        ) from e
    _model_name = name
    return _model


async def warm_up() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_model_sync)


_QWEN_QUERY_INSTRUCTION = (
    "Instruct: Dado el texto de una consulta financiera en español, "
    "recupera los fragmentos de documentos del Banco Central de Chile más relevantes.\nQuery: "
)


def _encode_sync(text: str) -> np.ndarray:
    model = _load_model_sync()
    name_lower = (_model_name or "").lower()
    if "e5" in name_lower:
        text = "query: " + text
    elif "qwen" in name_lower and "embedding" in name_lower:
        text = _QWEN_QUERY_INSTRUCTION + text
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vec.astype(np.float32)


async def embed_query(text: str) -> np.ndarray:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_sync, text)
