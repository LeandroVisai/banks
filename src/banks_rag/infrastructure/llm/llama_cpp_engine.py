"""LlamaCppEngine — adapter llama-cpp-python que implementa el Protocol LLMEngine.

Soporta Qwen3.6-27B-UD-Q4_K_XL.gguf y Gemma4-26B-A4B-it.gguf sobre H100 offline.

Async-first: llama.cpp es blocking; todas las llamadas pesadas corren en un
ThreadPoolExecutor dedicado para no bloquear el event loop de FastAPI.

Tool calling:
  1. Si llama-cpp-python devuelve ``tool_calls`` estructurado (native) → se usa directo.
  2. Si el modelo emite ``<tool_call>{...}</tool_call>`` en texto → ``parse_tool_calls``.
  Qwen3 soporta ambos; Gemma 4 usa principalmente el texto.

Qwen3 thinking mode:
  Los bloques ``<think>...</think>`` se eliminan del texto final antes de
  entregarlo al usuario y al parser de tool calls.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from banks_rag.domain.agent import GenerationResult, ToolCall
from banks_rag.infrastructure.llm.tool_call_parser import parse_tool_calls

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_CHAT_FORMAT_BY_FAMILY = {
    "qwen": "chatml",
    "gemma": "gemma",
}


def _detect_chat_format(model_path: str) -> str:
    """Infiere el chat format desde el nombre del archivo GGUF."""
    p = model_path.lower()
    if "qwen" in p:
        return "chatml"
    if "gemma" in p:
        return "gemma"
    return "chatml"


def _parse_args(raw: str | dict) -> dict:
    """Parsea ``arguments`` de un tool call, tolerando string-JSON anidado."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


class LlamaCppEngine:
    """Motor de inferencia llama.cpp que implementa el Protocol ``LLMEngine``.

    Parámetros:
        model_path: ruta absoluta al archivo ``.gguf``.
        n_ctx: contexto máximo en tokens (default 16 384).
        n_gpu_layers: capas a offloadear en GPU; ``-1`` = todas (H100).
        temperature, top_p, max_tokens: defaults de generación (sobreescribibles
            por llamada en ``generate()``).
        chat_format: ``"chatml"`` (Qwen3) o ``"gemma"``; ``None`` = auto-detect
            desde el nombre del archivo.
        n_threads: workers del ThreadPoolExecutor (default 1; llama.cpp usa
            internamente sus propios threads para inferencia).
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 16_384,
        n_gpu_layers: int = -1,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 2_048,
        chat_format: str | None = None,
        n_threads: int = 1,
    ) -> None:
        self.model_path = model_path
        self.name = Path(model_path).stem
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._chat_format = chat_format or _detect_chat_format(model_path)
        self._model: Any = None
        # Una sola instancia del modelo no es concurrente: con n_threads=1
        # las requests a /v1/chat se serializan en este executor. Es el
        # comportamiento correcto (evita corromper el estado del modelo),
        # pero define el techo de throughput — escalar requiere réplicas
        # del proceso o una cola con back-pressure, no subir n_threads.
        self._executor = ThreadPoolExecutor(
            max_workers=n_threads,
            thread_name_prefix="llama_cpp",
        )
        self.loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Carga el modelo GGUF en GPU. Idempotente."""
        if self.loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._sync_load)

    def _sync_load(self) -> None:
        from llama_cpp import Llama  # importación lazy para tests sin el binario

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo GGUF no encontrado: {self.model_path}")

        log.info(
            "Cargando %s (n_ctx=%d, n_gpu_layers=%d, chat_format=%s)",
            self.name,
            self._n_ctx,
            self._n_gpu_layers,
            self._chat_format,
        )
        t0 = time.monotonic()
        self._model = Llama(
            model_path=str(path),
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            chat_format=self._chat_format,
            verbose=False,
        )
        self.loaded = True
        log.info("Modelo listo en %.1fs", time.monotonic() - t0)

    async def unload(self) -> None:
        """Libera el modelo y la VRAM. Idempotente."""
        if not self.loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._sync_unload)

    def _sync_unload(self) -> None:
        self._model = None
        self.loaded = False
        log.info("Modelo %s descargado", self.name)

    # ── Generación ────────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        if not self.loaded:
            raise RuntimeError(f"Modelo {self.name!r} no cargado. Llama load() primero.")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._sync_generate,
                messages,
                tools=tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
        )

    def _sync_generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "top_p": top_p if top_p is not None else self._top_p,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._model.create_chat_completion(**kwargs)
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason: str = choice.get("finish_reason") or "stop"
        n_tokens: int = response.get("usage", {}).get("completion_tokens", 0)
        raw_text: str = message.get("content") or ""

        # Qwen3 thinking mode: eliminar bloque <think> del texto entregado
        cleaned = _THINK_RE.sub("", raw_text).strip()

        # 1. Native tool calls (llama-cpp structured output)
        native_calls: list[dict] = message.get("tool_calls") or []
        if native_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=tc["function"]["name"],
                    arguments=_parse_args(tc["function"].get("arguments", "{}")),
                )
                for i, tc in enumerate(native_calls)
            ]
            return GenerationResult(
                text=cleaned,
                tool_calls=tool_calls,
                raw_text=raw_text,
                n_tokens=n_tokens,
                finish_reason=finish_reason,
            )

        # 2. Text-embedded tool calls (<tool_call>...</tool_call> o fenced JSON)
        tool_calls, final_text = parse_tool_calls(cleaned)
        return GenerationResult(
            text=final_text,
            tool_calls=tool_calls,
            raw_text=raw_text,
            n_tokens=n_tokens,
            finish_reason="tool_calls" if tool_calls else finish_reason,
        )

    # ── Conteo de tokens ──────────────────────────────────────────────────────

    # El chat template real (tokens de rol, separadores, formato de tools)
    # añade overhead que la concatenación en texto plano no refleja. Se
    # aplica un margen para no subestimar y arriesgar exceder n_ctx.
    _CHAT_TEMPLATE_OVERHEAD = 1.15

    def count_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> int:
        """Estima tokens del prompt completo (concatenación + margen de template)."""
        if not self.loaded or self._model is None:
            return 0
        parts: list[str] = []
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))
        for m in messages:
            parts.append(f"{m.get('role', '')}: {m.get('content', '')}")
        raw = self.count_text_tokens("\n".join(parts))
        return int(raw * self._CHAT_TEMPLATE_OVERHEAD)

    def count_text_tokens(self, text: str) -> int:
        """Cuenta tokens de texto plano usando el tokenizer del modelo."""
        if not self.loaded or self._model is None:
            return 0
        try:
            return len(self._model.tokenize(text.encode("utf-8", errors="replace")))
        except Exception:
            return 0

    # ── Info / health ─────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "loaded": self.loaded,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "chat_format": self._chat_format,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
        }

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Any) -> "LlamaCppEngine":
        """Construye el engine desde el objeto Settings del servicio.

        Espera que ``settings`` tenga:
            ``llm_model_path``, ``llm_n_ctx``, ``llm_n_gpu_layers``,
            ``llm_temperature``, ``llm_top_p``, ``llm_max_tokens``.
        """
        model_path = getattr(settings, "model_path_resolved", None) or settings.llm_model_path
        return cls(
            str(model_path),
            n_ctx=settings.llm_n_ctx,
            n_gpu_layers=settings.llm_n_gpu_layers,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            max_tokens=settings.llm_max_tokens,
        )
