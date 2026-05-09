"""Unit tests del conversation_loop con LLM mock."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from banks_rag.application.agent import run_agent
from banks_rag.application.agent.tools.registry import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    register,
    reset_registry,
)
from banks_rag.domain.agent import GenerationResult, ToolCall


@dataclass
class _MockLLM:
    """LLM mock que devuelve respuestas pre-programadas en orden."""

    responses: list[GenerationResult] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    async def generate(self, messages, **kwargs):
        self.calls.append({"n_messages": len(messages), "kwargs": kwargs})
        if not self.responses:
            return GenerationResult(text="default", n_tokens=5)
        return self.responses.pop(0)

    def count_text_tokens(self, text: str) -> int:
        return len(text) // 4


@pytest.fixture
def clean_registry():
    snapshot_registry = dict(TOOL_REGISTRY)
    snapshot_schemas = list(TOOL_SCHEMAS)
    reset_registry()
    yield
    reset_registry()
    TOOL_REGISTRY.update(snapshot_registry)
    TOOL_SCHEMAS.extend(snapshot_schemas)


@pytest.mark.unit
class TestRunAgent:
    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_immediately(self, clean_registry) -> None:
        llm = _MockLLM(responses=[
            GenerationResult(text="Respuesta directa", n_tokens=10),
        ])
        result = await run_agent("Hola", history=[], llm=llm)
        assert result.response == "Respuesta directa"
        assert result.iterations == 1
        assert result.tool_trace == []
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_single_tool_call_then_response(self, clean_registry) -> None:
        # Tool: devuelve un chunk y registra una ref [1]
        @register("my_search", {
            "type": "function",
            "function": {
                "name": "my_search",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        })
        async def my_search(state):
            ref = state.add_chunk({
                "chunk_id": "x1",
                "filename": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "section_type": "DECISION",
                "doc_type_category": "COMUNICADO",
                "importance_score": 0.9,
            })
            return {"ref": ref, "n_results": 1}

        llm = _MockLLM(responses=[
            GenerationResult(
                text="",
                tool_calls=[ToolCall(id="c1", name="my_search", arguments={})],
                n_tokens=20,
            ),
            GenerationResult(text="Encontré [1].", n_tokens=15),
        ])

        result = await run_agent("¿Qué dijo el comunicado?", history=[], llm=llm)
        assert result.response == "Encontré [1]."
        assert result.iterations == 2
        assert result.cited_refs == [1]
        assert len(result.tool_trace) == 1
        assert result.tool_trace[0]["tool"] == "my_search"
        assert len(result.chunks_seen) == 1
        assert result.total_tokens == 35

    @pytest.mark.asyncio
    async def test_max_iterations_fallback(self, clean_registry) -> None:
        @register("loop_tool", {
            "type": "function",
            "function": {
                "name": "loop_tool",
                "description": "x",
                "parameters": {"type": "object"},
            },
        })
        async def loop_tool(state):
            return {"ok": True}

        # LLM siempre llama tools — agotamos iteraciones.
        responses = [
            GenerationResult(
                text="",
                tool_calls=[ToolCall(id=f"c{i}", name="loop_tool", arguments={})],
                n_tokens=5,
            )
            for i in range(10)
        ]
        llm = _MockLLM(responses=responses)

        result = await run_agent("loop", history=[], llm=llm, max_iterations=3)
        assert result.iterations == 3
        assert result.finish_reason == "max_iterations"
        assert "iteraciones" in result.response.lower()

    @pytest.mark.asyncio
    async def test_invalid_citations_cleaned(self, clean_registry) -> None:
        # Agente cita [5] sin tener 5 chunks → debería removerse.
        llm = _MockLLM(responses=[
            GenerationResult(text="La TPM bajó [5] según fuentes.", n_tokens=10),
        ])
        result = await run_agent("?", history=[], llm=llm)
        assert "[5]" not in result.response
        assert result.cited_refs == []

    @pytest.mark.asyncio
    async def test_history_is_passed_to_llm(self, clean_registry) -> None:
        llm = _MockLLM(responses=[GenerationResult(text="ok", n_tokens=1)])
        history = [
            {"role": "user", "content": "Pregunta previa"},
            {"role": "assistant", "content": "Respuesta previa"},
        ]
        await run_agent("Nueva pregunta", history=history, llm=llm)
        # System + 2 history + user = 4 messages
        assert llm.calls[0]["n_messages"] == 4
