"""
LLM en proceso con vLLM AsyncLLMEngine.

Sin HTTP, sin OpenAI client, sin servidores intermedios. El motor se carga
una vez en `lifespan` y se usa desde los handlers.

Provee:
  - load_engine() / unload_engine()  → ciclo de vida
  - count_tokens(messages)           → presupuesto de tokens del prompt
  - generate(messages, ...)          → respuesta completa
  - generate_stream(messages, ...)   → deltas (no acumulativo)

vLLM emite outputs *acumulativos*; aquí los convertimos a *deltas* para
poder hacer SSE limpio en la API.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncIterator, Optional

from .settings import settings

log = logging.getLogger(__name__)

_engine = None         # vllm.AsyncLLMEngine
_tokenizer = None      # transformers.AutoTokenizer
_eos_token_ids: list[int] = []


# ─────────────────────────────────────────────────────────────────────────────
# Ciclo de vida
# ─────────────────────────────────────────────────────────────────────────────

async def load_engine() -> None:
    """Carga vLLM + tokenizer. Bloquea hasta que el modelo está listo."""
    global _engine, _tokenizer, _eos_token_ids

    if _engine is not None:
        return

    if not settings.model_path.exists():
        raise RuntimeError(
            f"No existe la carpeta del modelo: {settings.model_path}\n"
            f"Descarga los archivos del modelo y colócalos ahí."
        )

    log.info("Cargando modelo %s desde %s", settings.chatbot_model_name, settings.model_path)
    log.info(
        "  tensor_parallel=%d quant=%s gpu_mem=%.2f max_len=%d",
        settings.chatbot_tensor_parallel,
        settings.chatbot_quantization,
        settings.chatbot_gpu_mem_util,
        settings.chatbot_max_model_len,
    )

    # Imports diferidos: vLLM tarda ~5s en importar y carga CUDA al hacerlo
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from transformers import AutoTokenizer

    engine_args = AsyncEngineArgs(
        model=str(settings.model_path),
        tensor_parallel_size=settings.chatbot_tensor_parallel,
        gpu_memory_utilization=settings.chatbot_gpu_mem_util,
        max_model_len=settings.chatbot_max_model_len,
        dtype="bfloat16",
        quantization=settings.chatbot_quantization,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=256,
        trust_remote_code=True,
        disable_log_requests=True,
    )

    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    _tokenizer = AutoTokenizer.from_pretrained(
        str(settings.model_path),
        trust_remote_code=True,
    )

    # Tokens de fin: el chat template define estos como cadenas, los traducimos a IDs
    eos_candidates = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
    ids: list[int] = []
    for tok in eos_candidates:
        try:
            tid = _tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != _tokenizer.unk_token_id:
                ids.append(tid)
        except Exception:
            pass
    if _tokenizer.eos_token_id is not None:
        ids.append(_tokenizer.eos_token_id)
    _eos_token_ids = list(dict.fromkeys(ids))  # dedup

    log.info("Modelo cargado. EOS tokens: %s", _eos_token_ids)


async def unload_engine() -> None:
    global _engine, _tokenizer
    if _engine is not None:
        # vLLM no expone un shutdown explícito; descartamos referencias y
        # dejamos que CUDA libere al terminar el proceso.
        _engine = None
        _tokenizer = None
        log.info("LLM engine descargado")


def is_loaded() -> bool:
    return _engine is not None


def engine_info() -> dict:
    return {
        "loaded": is_loaded(),
        "model": settings.chatbot_model_name,
        "model_path": str(settings.model_path),
        "quantization": settings.chatbot_quantization,
        "tensor_parallel": settings.chatbot_tensor_parallel,
        "max_model_len": settings.chatbot_max_model_len,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tokenización
# ─────────────────────────────────────────────────────────────────────────────

def _render_prompt(messages: list[dict]) -> str:
    if _tokenizer is None:
        raise RuntimeError("Tokenizer no inicializado")
    return _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def count_tokens(messages: list[dict]) -> int:
    """Cuenta tokens del prompt completo (template aplicado)."""
    if _tokenizer is None:
        return 0
    prompt = _render_prompt(messages)
    return len(_tokenizer(prompt, add_special_tokens=False)["input_ids"])


def count_text_tokens(text: str) -> int:
    if _tokenizer is None:
        return len(text) // 4  # heurística previa al startup
    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])


# ─────────────────────────────────────────────────────────────────────────────
# Generación
# ─────────────────────────────────────────────────────────────────────────────

def _sampling_params(temperature: float, top_p: float, max_tokens: int):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=max(temperature, 0.0),
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=_eos_token_ids or None,
    )


async def generate(
    messages: list[dict],
    *,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
) -> tuple[str, int]:
    """Genera la respuesta completa. Retorna (texto, n_tokens_generados)."""
    if _engine is None:
        raise RuntimeError("LLM engine no cargado")

    params = _sampling_params(
        temperature if temperature is not None else settings.chatbot_temperature,
        top_p if top_p is not None else settings.chatbot_top_p,
        max_tokens if max_tokens is not None else settings.chatbot_max_tokens,
    )

    prompt = _render_prompt(messages)
    request_id = str(uuid.uuid4())

    final_text = ""
    n_tokens = 0
    async for output in _engine.generate(prompt, params, request_id):
        final_text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)

    return final_text.strip(), n_tokens


async def generate_stream(
    messages: list[dict],
    *,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
) -> AsyncIterator[str]:
    """
    Yieldea *deltas* (no acumulativo). Convierte el output cumulativo de vLLM
    en chunks listos para SSE.
    """
    if _engine is None:
        raise RuntimeError("LLM engine no cargado")

    params = _sampling_params(
        temperature if temperature is not None else settings.chatbot_temperature,
        top_p if top_p is not None else settings.chatbot_top_p,
        max_tokens if max_tokens is not None else settings.chatbot_max_tokens,
    )

    prompt = _render_prompt(messages)
    request_id = str(uuid.uuid4())

    sent_len = 0
    async for output in _engine.generate(prompt, params, request_id):
        text = output.outputs[0].text
        if len(text) > sent_len:
            delta = text[sent_len:]
            sent_len = len(text)
            yield delta


async def cancel_request(request_id: str) -> None:
    """Aborta una request en curso (usado si el cliente cierra la conexión)."""
    if _engine is not None:
        try:
            await _engine.abort(request_id)
        except Exception as e:
            log.warning("No se pudo abortar request %s: %s", request_id, e)
