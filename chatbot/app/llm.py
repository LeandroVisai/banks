"""
LLM en proceso con vLLM AsyncLLMEngine (o transformers en CPU para testing).

Sin HTTP, sin OpenAI client, sin servidores intermedios. El motor se carga
una vez en `lifespan` y se usa desde los handlers.

Provee:
  - load_engine() / unload_engine()  → ciclo de vida
  - count_tokens(messages)           → presupuesto de tokens del prompt
  - generate(messages, ...)          → respuesta completa
  - generate_stream(messages, ...)   → deltas (no acumulativo)

vLLM emite outputs *acumulativos*; aquí los convertimos a *deltas* para
poder hacer SSE limpio en la API.

Modos:
  - vLLM (default, GPU): producción en H100+
  - CPU (testing): usa transformers en CPU. Activar: USE_CPU_LLM=1
    Si CUDA no está disponible, cae a CPU automáticamente.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import AsyncIterator, Optional

from .settings import settings

log = logging.getLogger(__name__)

_USE_CPU_LLM = os.getenv("USE_CPU_LLM", "0") == "1"

_engine = None
_tokenizer = None
_eos_token_ids: list[int] = []
_is_cpu_mode = False


# ─────────────────────────────────────────────────────────────────────────────
# Ciclo de vida
# ─────────────────────────────────────────────────────────────────────────────

async def load_engine() -> None:
    global _engine, _tokenizer, _eos_token_ids, _is_cpu_mode

    if _engine is not None:
        return

    use_cpu_env = os.getenv("USE_CPU_LLM", "0") == "1"
    if use_cpu_env or _USE_CPU_LLM:
        _load_engine_cpu()
        return

    # Auto-detectar CUDA; si no hay, caer a CPU
    try:
        import importlib
        has_cuda = False
        if importlib.util.find_spec("torch") is not None:
            import torch
            has_cuda = torch.cuda.is_available()
        if not has_cuda:
            log.info("CUDA no disponible — usando backend CPU (transformers/mock)")
            _load_engine_cpu()
            return
    except Exception:
        log.info("No se pudo comprobar CUDA — cayendo a CPU")
        _load_engine_cpu()
        return

    try:
        _load_engine_vllm()
    except Exception as e:
        log.warning("No se pudo cargar vLLM: %s — cayendo a CPU", e)
        _load_engine_cpu()


def _load_engine_cpu() -> None:
    global _engine, _tokenizer, _eos_token_ids, _is_cpu_mode
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        if not settings.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {settings.model_path}")

        log.info("Cargando modelo en CPU desde %s (puede ser lento)...", settings.model_path)

        _tokenizer = AutoTokenizer.from_pretrained(
            str(settings.model_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(settings.model_path),
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        _engine = pipeline(
            "text-generation", model=model, tokenizer=_tokenizer,
            device=-1, return_full_text=False,
        )

        ids: list[int] = []
        for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            try:
                tid = _tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid != _tokenizer.unk_token_id:
                    ids.append(tid)
            except Exception:
                pass
        if _tokenizer.eos_token_id is not None:
            ids.append(_tokenizer.eos_token_id)
        _eos_token_ids = list(dict.fromkeys(ids))
        _is_cpu_mode = True
        log.info("Modelo CPU cargado. EOS tokens: %s", _eos_token_ids)
        return
    except Exception as e:
        log.warning("No se pudo cargar modelo CPU con transformers: %s", e)

    # Mock ligero para pruebas de flujo sin modelo descargado
    class _MockEngine:
        pass

    _engine = _MockEngine()
    _tokenizer = None
    _eos_token_ids = []
    _is_cpu_mode = True
    log.info("Backend CPU mock activado (sin modelo, solo flujo)")


def _load_engine_vllm() -> None:
    global _engine, _tokenizer, _eos_token_ids, _is_cpu_mode

    if not settings.model_path.exists():
        raise RuntimeError(
            f"No existe la carpeta del modelo: {settings.model_path}\n"
            "Descarga los archivos del modelo y colócalos ahí."
        )

    log.info("Cargando modelo %s desde %s", settings.chatbot_model_name, settings.model_path)
    log.info(
        "  tensor_parallel=%d quant=%s gpu_mem=%.2f max_len=%d",
        settings.chatbot_tensor_parallel,
        settings.chatbot_quantization,
        settings.chatbot_gpu_mem_util,
        settings.chatbot_max_model_len,
    )

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
        str(settings.model_path), trust_remote_code=True,
    )

    ids: list[int] = []
    for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
        try:
            tid = _tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != _tokenizer.unk_token_id:
                ids.append(tid)
        except Exception:
            pass
    if _tokenizer.eos_token_id is not None:
        ids.append(_tokenizer.eos_token_id)
    _eos_token_ids = list(dict.fromkeys(ids))
    _is_cpu_mode = False
    log.info("Modelo vLLM cargado. EOS tokens: %s", _eos_token_ids)


async def unload_engine() -> None:
    global _engine, _tokenizer
    if _engine is not None:
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
        "mode": "CPU (testing)" if _is_cpu_mode else "vLLM (production)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tokenización
# ─────────────────────────────────────────────────────────────────────────────

def _render_prompt(messages: list[dict]) -> str:
    if _tokenizer is None:
        parts = []
        for msg in messages:
            parts.append(f"[{msg.get('role','user').upper()}] {msg.get('content','')}")
        parts.append("[ASSISTANT]")
        return "\n".join(parts)
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def count_tokens(messages: list[dict]) -> int:
    if _tokenizer is None:
        return max(1, len(_render_prompt(messages)) // 4)
    prompt = _render_prompt(messages)
    return len(_tokenizer(prompt, add_special_tokens=False)["input_ids"])


def count_text_tokens(text: str) -> int:
    if _tokenizer is None:
        return max(1, len(text) // 4)
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
    if _engine is None:
        raise RuntimeError("LLM engine no cargado")
    if _is_cpu_mode:
        return _generate_cpu(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    return await _generate_vllm(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def _generate_cpu(
    messages: list[dict],
    *,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
) -> tuple[str, int]:
    user_msg = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    prompt = _render_prompt(messages)
    max_new_tokens = max_tokens or settings.chatbot_max_tokens

    if hasattr(_engine, "__call__") and _tokenizer is not None:
        try:
            outputs = _engine(
                prompt,
                max_new_tokens=int(max_new_tokens),
                do_sample=(temperature or settings.chatbot_temperature) > 0,
                temperature=float(temperature or settings.chatbot_temperature),
                top_p=float(top_p or settings.chatbot_top_p),
            )
            text = outputs[0].get("generated_text", "") if isinstance(outputs, list) else str(outputs)
            return text.strip(), max(1, len(text) // 4)
        except Exception as e:
            log.warning("transformers pipeline falló, usando mock: %s", e)

    answer = (
        "[modo CPU mock] Infraestructura OK: recibí la consulta y armé el contexto RAG/SQL. "
        f"Consulta: {user_msg}\n"
        "En H100 con vLLM este bloque se reemplaza por la respuesta real del modelo."
    )
    return answer, max(1, len(answer) // 4)


async def _generate_vllm(
    messages: list[dict],
    *,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
) -> tuple[str, int]:
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
    if _engine is None:
        raise RuntimeError("LLM engine no cargado")
    if _is_cpu_mode:
        async for delta in _generate_stream_cpu(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens):
            yield delta
    else:
        async for delta in _generate_stream_vllm(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens):
            yield delta


async def _generate_stream_cpu(
    messages: list[dict], *, temperature=None, top_p=None, max_tokens=None,
) -> AsyncIterator[str]:
    text, _ = _generate_cpu(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    for i in range(0, len(text), 50):
        yield text[i:i + 50]
        await asyncio.sleep(0.05)


async def _generate_stream_vllm(
    messages: list[dict], *, temperature=None, top_p=None, max_tokens=None,
) -> AsyncIterator[str]:
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
            yield text[sent_len:]
            sent_len = len(text)


async def cancel_request(request_id: str) -> None:
    if _engine is not None and not _is_cpu_mode:
        try:
            await _engine.abort(request_id)
        except Exception as e:
            log.warning("No se pudo abortar request %s: %s", request_id, e)
