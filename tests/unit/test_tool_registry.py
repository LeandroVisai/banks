"""Unit tests para tool registry y dispatcher."""

from __future__ import annotations

import pytest

from banks_rag.application.agent.tools.registry import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    dispatch,
    register,
    reset_registry,
    signature_hint,
)
from banks_rag.domain.agent import AgentState


@pytest.fixture
def clean_registry():
    """Limpia y restaura el registry alrededor de cada test."""
    snapshot_registry = dict(TOOL_REGISTRY)
    snapshot_schemas = list(TOOL_SCHEMAS)
    reset_registry()
    yield
    reset_registry()
    TOOL_REGISTRY.update(snapshot_registry)
    TOOL_SCHEMAS.extend(snapshot_schemas)


@pytest.mark.unit
class TestRegister:
    def test_register_adds_to_registry(self, clean_registry) -> None:
        schema = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        @register("my_tool", schema)
        async def my_tool(state, **kwargs):
            return {"ok": True}

        assert "my_tool" in TOOL_REGISTRY
        assert any(s["function"]["name"] == "my_tool" for s in TOOL_SCHEMAS)

    def test_register_rejects_inconsistent_schema(self, clean_registry) -> None:
        schema = {"type": "function", "function": {"name": "wrong_name"}}
        with pytest.raises(ValueError, match="inconsistente"):
            register("my_tool", schema)


@pytest.mark.unit
class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_calls_registered_tool(self, clean_registry) -> None:
        schema = {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo args",
                "parameters": {"type": "object"},
            },
        }

        @register("echo", schema)
        async def echo(state, **kwargs):
            return {"echoed": kwargs}

        state = AgentState()
        result, duration = await dispatch(state, "echo", {"x": 1, "y": 2})
        assert result == {"echoed": {"x": 1, "y": 2}}
        assert duration >= 0

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool_returns_error(self, clean_registry) -> None:
        state = AgentState()
        result, _ = await dispatch(state, "no_such_tool", {})
        assert "error" in result
        assert "no_such_tool" in result["error"]
        assert "available_tools" in result

    @pytest.mark.asyncio
    async def test_dispatch_invalid_args_returns_error(self, clean_registry) -> None:
        schema = {
            "type": "function",
            "function": {
                "name": "needs_x",
                "description": "needs x",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            },
        }

        @register("needs_x", schema)
        async def needs_x(state, x: str):
            return {"x": x}

        state = AgentState()
        # Pasar argumentos sin "x"
        result, _ = await dispatch(state, "needs_x", {})
        assert "error" in result
        assert "expected_signature" in result

    @pytest.mark.asyncio
    async def test_dispatch_runtime_error_captured(self, clean_registry) -> None:
        schema = {
            "type": "function",
            "function": {
                "name": "broken",
                "description": "raises",
                "parameters": {"type": "object"},
            },
        }

        @register("broken", schema)
        async def broken(state):
            raise RuntimeError("boom")

        state = AgentState()
        result, _ = await dispatch(state, "broken", {})
        assert "error" in result
        assert "boom" in result["error"]


@pytest.mark.unit
def test_signature_hint(clean_registry) -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "x",
            "description": "x",
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
    }

    @register("x", schema)
    async def x(state, a: str):
        return {"a": a}

    hint = signature_hint("x")
    assert hint is not None
    assert hint["properties"]["a"]["type"] == "string"

    assert signature_hint("nonexistent") is None
