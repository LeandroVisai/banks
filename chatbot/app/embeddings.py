"""
Embedding de queries para el RAG.

`sentence-transformers` es síncrono y CPU-bound (modelo pequeño, ~80MB).
Lo aislamos en un threadpool para no bloquear el event loop de FastAPI.
"""
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
    """Decide qué ruta/ID pasarle a SentenceTransformer."""
    model_id = settings.embedding_model_id

    # 1. ¿Está en la cache compartida del pipeline padre?
    if EMBEDDING_CACHE_DIR.exists():
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(EMBEDDING_CACHE_DIR))

    # 2. ¿Está en chatbot/models/<id>/ ?
    local_dir = MODELS_DIR / model_id.replace("/", "--")
    if local_dir.exists():
        return str(local_dir)

    # 3. Caer al ID de HuggingFace (intentará descarga si hay internet)
    return model_id


def _load_model_sync():
    """Importa y carga sentence-transformers — bloquea, llamar desde executor."""
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
            f"No se pudo cargar el embedding '{settings.embedding_model_id}'.\n"
            f"Verifica que esté en {EMBEDDING_CACHE_DIR} o en {MODELS_DIR}.\n"
            f"Error: {e}"
        ) from e

    _model_name = name
    return _model


async def warm_up() -> None:
    """Carga el modelo en startup para que la primera query no pague el costo."""
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
