"""Protocol ``LLMEngine`` — interfaz mínima para el motor de inferencia.

Cualquier backend (vLLM, llama.cpp, mock para tests) implementa este Protocol.
La implementación concreta para Qwen3.6 y Gemma 4 sobre llama.cpp viene en
Fase 3 del plan de transformación.

Async-first porque el agente loop ejecuta tool calls concurrentes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from banks_rag.domain.agent import GenerationResult


@runtime_checkable
class LLMEngine(Protocol):
    """Motor de inferencia LLM con soporte de tool calling.

    Atributos:
        name: nombre/familia del modelo (e.g. ``"Qwen3.6-27B-Q4_K_XL"``).
        model_path: path absoluto al modelo (GGUF o directorio HF).
        loaded: ``True`` si está listo para generar.
    """

    name: str
    model_path: str
    loaded: bool

    async def load(self) -> None:
        """Carga el modelo en GPU. Idempotente. Lanza si falla."""
        ...

    async def unload(self) -> None:
        """Libera el modelo y la memoria GPU."""
        ...

    async def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Genera una respuesta. Si vienen ``tools``, parsea ``tool_calls`` del output."""
        ...

    def count_tokens(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        """Cuenta tokens del prompt completo (incluyendo el chat template)."""
        ...

    def count_text_tokens(self, text: str) -> int:
        """Cuenta tokens de un texto plano (sin chat template)."""
        ...

    def info(self) -> dict:
        """Metadata del engine para health checks."""
        ...
