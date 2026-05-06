"""
LLM en proceso con vLLM AsyncLLMEngine + soporte de TOOL CALLING.

Diferencias vs chatbot/llm.py:
  - `apply_chat_template(..., tools=tools)` inyecta las definiciones al prompt.
  - `generate_with_tools()` retorna texto + tool_calls parseados.
  - El parser maneja el formato Qwen3 / Hermes-style: <tool_call>{"name":...,
    "arguments":{...}}</tool_call>.

Modelos soportados (con tool calling nativo):
  - Qwen3.6-35B-A3B   → formato <tool_call> JSON
  - Kimi-K2-Instruct  → formato compatible con OpenAI tool calls
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .settings import settings

log = logging.getLogger(__name__)

_engine = None
_tokenizer = None
_eos_token_ids: list[int] = []


# ─────────────────────────────────────────────────────────────────────────────
# Estructuras
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """Una llamada a herramienta parseada del output del LLM."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_message_block(self) -> dict:
        """Bloque que se inserta como assistant message en la siguiente iteración."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class GenerationResult:
    """Resultado de una llamada al LLM en modo agentic."""
    text: str                              # texto de la respuesta (sin los tool_call markers)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_text: str = ""                     # texto completo tal como salió del modelo
    n_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Ciclo de vida (igual al chatbot/ clásico)
# ─────────────────────────────────────────────────────────────────────────────

async def load_engine() -> None:
    global _engine, _tokenizer, _eos_token_ids
    if _engine is not None:
        return
    if not settings.model_path.exists():
        raise RuntimeError(f"No existe la carpeta del modelo: {settings.model_path}")

    log.info("Cargando modelo (agentic) %s desde %s",
             settings.chatbot_model_name, settings.model_path)

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
        enable_prefix_caching=True,           # crucial: cada iteración del agente reusa
        enable_chunked_prefill=True,          # prefijo + tools + history sin recomputar.
        max_num_seqs=256,
        trust_remote_code=True,
        disable_log_requests=True,
    )
    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    _tokenizer = AutoTokenizer.from_pretrained(
        str(settings.model_path), trust_remote_code=True,
    )

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
    _eos_token_ids = list(dict.fromkeys(ids))
    log.info("Modelo cargado. EOS tokens: %s", _eos_token_ids)


async def unload_engine() -> None:
    global _engine, _tokenizer
    _engine = None
    _tokenizer = None


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
        "supports_tool_calling": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tokenización con tools
# ─────────────────────────────────────────────────────────────────────────────

def _render_prompt(messages: list[dict], tools: Optional[list[dict]] = None) -> str:
    """
    Renderiza el prompt con el chat template del modelo, inyectando tools
    si vienen. Qwen3 y Kimi K2 soportan el argumento `tools=`.
    """
    if _tokenizer is None:
        raise RuntimeError("Tokenizer no inicializado")
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    return _tokenizer.apply_chat_template(messages, **kwargs)


def count_tokens(messages: list[dict], tools: Optional[list[dict]] = None) -> int:
    if _tokenizer is None:
        return 0
    prompt = _render_prompt(messages, tools=tools)
    return len(_tokenizer(prompt, add_special_tokens=False)["input_ids"])


def count_text_tokens(text: str) -> int:
    if _tokenizer is None:
        return len(text) // 4
    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])


# ─────────────────────────────────────────────────────────────────────────────
# Parser de tool calls
# ─────────────────────────────────────────────────────────────────────────────

# Qwen3 / Hermes-style: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Algunos modelos emiten ```json ... ``` o sólo el JSON crudo al final
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```")


def _parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """
    Extrae tool calls del texto generado. Retorna (tool_calls, texto_limpio).
    El texto limpio es lo que va al usuario (sin los <tool_call> markers).
    """
    calls: list[ToolCall] = []

    # 1. Formato preferido: <tool_call>...</tool_call>
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
            name = payload.get("name")
            args = payload.get("arguments", {})
            if isinstance(args, str):
                # algunos modelos emiten arguments como string-JSON anidado
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                calls.append(ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                ))
        except json.JSONDecodeError as e:
            log.warning("tool_call con JSON inválido: %s", e)

    cleaned = _TOOL_CALL_RE.sub("", text).strip()

    # 2. Fallback: si no encontró nada Y el output es un JSON crudo, intentar
    if not calls:
        m = _FENCED_JSON_RE.search(text)
        candidate = m.group(1) if m else (text.strip() if text.strip().startswith("{") else None)
        if candidate:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict) and "name" in payload and "arguments" in payload:
                    calls.append(ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=payload["name"],
                        arguments=payload.get("arguments", {}) or {},
                    ))
                    cleaned = ""
            except json.JSONDecodeError:
                pass

    return calls, cleaned


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
    tools: Optional[list[dict]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> GenerationResult:
    """Genera una respuesta. Si vienen tools, parsea tool_calls del output."""
    if _engine is None:
        raise RuntimeError("LLM engine no cargado")

    params = _sampling_params(
        temperature if temperature is not None else settings.chatbot_temperature,
        top_p if top_p is not None else settings.chatbot_top_p,
        max_tokens if max_tokens is not None else settings.chatbot_max_tokens,
    )

    prompt = _render_prompt(messages, tools=tools)
    request_id = str(uuid.uuid4())

    raw_text = ""
    n_tokens = 0
    finish_reason = "stop"
    async for output in _engine.generate(prompt, params, request_id):
        raw_text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason or "stop"

    if tools:
        tool_calls, cleaned = _parse_tool_calls(raw_text)
    else:
        tool_calls, cleaned = [], raw_text.strip()

    return GenerationResult(
        text=cleaned,
        tool_calls=tool_calls,
        raw_text=raw_text,
        n_tokens=n_tokens,
        finish_reason=finish_reason,
    )


async def cancel_request(request_id: str) -> None:
    if _engine is not None:
        try:
            await _engine.abort(request_id)
        except Exception:
            pass
