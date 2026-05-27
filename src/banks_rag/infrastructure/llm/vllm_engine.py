"""VLLMEngine — backend de inferencia via servidor vLLM (OpenAI-compatible API).

Arquitectura: vLLM corre como proceso separado (systemd banks-vllm.service) y
expone una API OpenAI-compatible en localhost:8000. Este engine usa
``openai.AsyncOpenAI`` para hablar con él.

Ventajas vs llama.cpp:
- Batching continuo (PagedAttention) → 10-20× más throughput.
- Tool calling nativo estructurado (no text-parsing).
- Soporte de streaming nativo (disponible para implementar).
- El proceso FastAPI no carga el modelo (sin costo de VRAM en el API).

Invariantes:
- ``openai`` e ``httpx`` se importan de forma lazy dentro de ``load()``
  para que los tests unitarios corran sin esos paquetes instalados.
- El servidor vLLM es externo: ``unload()`` cierra el cliente HTTP pero
  no apaga el servidor.
- La interfaz ``LLMEngine`` se respeta en su totalidad; el agente loop
  y el API no necesitan cambios.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from banks_rag.domain.agent import GenerationResult, ToolCall
from banks_rag.infrastructure.llm.tool_call_parser import parse_tool_calls

if TYPE_CHECKING:
    import openai as _openai_t

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOKENS_PER_WORD = 1.35
_CHAT_TEMPLATE_OVERHEAD = 1.15


def _parse_args(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parsea argumentos de un tool call tolerando string-JSON anidado."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


class VLLMEngine:
    """Motor de inferencia respaldado por un servidor vLLM externo.

    El servidor debe estar corriendo antes de llamar a ``load()``.
    Se lanza ``RuntimeError`` si el servidor no responde.

    Uso típico::

        engine = VLLMEngine.from_settings(settings)
        await engine.load()          # verifica conectividad
        result = await engine.generate(messages, tools=schemas)
        await engine.unload()        # cierra el cliente HTTP

    Variables de entorno (todas con prefijo ``BANKS_``):
        BANKS_VLLM_BASE_URL, BANKS_VLLM_MODEL, BANKS_VLLM_API_KEY,
        BANKS_VLLM_ENABLE_THINKING, BANKS_VLLM_TOOL_CHOICE,
        BANKS_VLLM_TIMEOUT, BANKS_VLLM_TEMPERATURE,
        BANKS_VLLM_TOP_P, BANKS_VLLM_MAX_TOKENS
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "none",
        enable_thinking: bool = True,
        tool_choice: str = "auto",
        timeout: float = 120.0,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> None:
        self.model_path: str = model
        self.name: str = model.split("/")[-1] if "/" in model else model
        self.loaded: bool = False

        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._enable_thinking = enable_thinking
        self._tool_choice = tool_choice
        self._timeout = timeout
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens

        self._client: _openai_t.AsyncOpenAI | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Conecta con el servidor vLLM y verifica que el modelo esté disponible.

        Idempotente: segunda llamada es no-op si ya está cargado.
        Lanza ``RuntimeError`` si el servidor no responde.
        """
        if self.loaded:
            return

        import httpx  # lazy — no disponible en entorno de tests sin instalación
        import openai  # lazy

        self._client = openai.AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=httpx.Timeout(self._timeout),
        )

        try:
            models_page = await self._client.models.list()
            available = [m.id for m in models_page.data]
            if self._model not in available:
                log.warning(
                    "vLLM: modelo %r no encontrado. Disponibles: %s",
                    self._model,
                    available,
                )
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo conectar al servidor vLLM en {self._base_url}: {exc}"
            ) from exc

        self.loaded = True
        log.info("VLLMEngine listo — model=%s base_url=%s", self._model, self._base_url)

    async def unload(self) -> None:
        """Cierra el cliente HTTP. No apaga el servidor vLLM."""
        if not self.loaded:
            return
        if self._client is not None:
            await self._client.close()
            self._client = None
        self.loaded = False
        log.info("VLLMEngine desconectado — servidor vLLM sigue corriendo")

    # ── Generación ───────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Genera una respuesta. Parsea ``tool_calls`` si el modelo los emite.

        Maneja el thinking mode de Qwen3:
        - Si vLLM separa ``reasoning_content`` + ``content`` → usa ``content`` directamente.
        - Si el modelo emite ``<think>...</think>`` embebido → elimina por regex.

        Fallback de tool parsing: si el modelo no emite ``tool_calls`` estructurados,
        intenta parsear ``<tool_call>`` tags del texto (``tool_call_parser.py``).
        """
        if not self.loaded:
            raise RuntimeError(
                f"VLLMEngine {self.name!r} no está cargado. Llamar a load() primero."
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "top_p": top_p if top_p is not None else self._top_p,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = self._tool_choice

        assert self._client is not None
        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message
        finish_reason: str = choice.finish_reason or "stop"
        n_tokens: int = response.usage.completion_tokens if response.usage else 0
        raw_content: str = message.content or ""

        # Thinking mode — preferir reasoning_content si vLLM lo separa
        if self._enable_thinking:
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning is not None:
                cleaned = raw_content.strip()
            else:
                cleaned = _THINK_RE.sub("", raw_content).strip()
        else:
            cleaned = raw_content.strip()

        # Tool calls nativos (vLLM con --tool-call-parser)
        native_calls = message.tool_calls or []
        if native_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id or f"call_{uuid.uuid4().hex[:12]}",
                    name=tc.function.name,
                    arguments=_parse_args(tc.function.arguments),
                )
                for tc in native_calls
            ]
            return GenerationResult(
                text=cleaned,
                tool_calls=tool_calls,
                raw_text=raw_content,
                n_tokens=n_tokens,
                finish_reason=finish_reason,
            )

        # Fallback: parsear <tool_call> tags del texto (poco frecuente con vLLM)
        tool_calls, final_text = parse_tool_calls(cleaned)
        return GenerationResult(
            text=final_text,
            tool_calls=tool_calls,
            raw_text=raw_content,
            n_tokens=n_tokens,
            finish_reason="tool_calls" if tool_calls else finish_reason,
        )

    # ── Token counting ───────────────────────────────────────────────────────

    def count_text_tokens(self, text: str) -> int:
        """Estimado de tokens por conteo de palabras (sin tokenizer)."""
        if not text:
            return 0
        return max(1, int(len(text.split()) * _TOKENS_PER_WORD))

    def count_tokens(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        """Estimado de tokens del prompt completo incluyendo overhead de chat template."""
        parts: list[str] = []
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))
        for m in messages:
            role = m.get("role", "")
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            parts.append(f"{role}: {content}")
        raw = self.count_text_tokens("\n".join(parts))
        return int(raw * _CHAT_TEMPLATE_OVERHEAD)

    # ── Metadata ─────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Metadata del engine para health checks y el endpoint ``/readyz``."""
        return {
            "name": self.name,
            "model_path": self.model_path,
            "loaded": self.loaded,
            "backend": "vllm",
            "base_url": self._base_url,
            "enable_thinking": self._enable_thinking,
            "tool_choice": self._tool_choice,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
        }

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Any) -> "VLLMEngine":
        """Construye el engine desde ``Settings`` (``banks_rag.config``)."""
        return cls(
            base_url=settings.vllm_base_url,
            model=settings.vllm_model,
            api_key=settings.vllm_api_key,
            enable_thinking=settings.vllm_enable_thinking,
            tool_choice=settings.vllm_tool_choice,
            timeout=settings.vllm_timeout,
            temperature=settings.vllm_temperature,
            top_p=settings.vllm_top_p,
            max_tokens=settings.vllm_max_tokens,
        )
