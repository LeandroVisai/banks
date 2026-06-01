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
class TestArgumentNormalization:
    """dispatch tolera alias de args (modelos que inventan nombres) y descarta
    kwargs no aceptados en vez de fallar con TypeError."""

    @staticmethod
    def _register_query_tool() -> None:
        schema = {
            "type": "function",
            "function": {
                "name": "qtool",
                "description": "needs query",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"],
                },
            },
        }

        @register("qtool", schema)
        async def qtool(state, query: str, top_k: int = 5):
            return {"query": query, "top_k": top_k}

    @pytest.mark.asyncio
    async def test_alias_maps_to_canonical(self, clean_registry) -> None:
        self._register_query_tool()
        state = AgentState()
        result, _ = await dispatch(state, "qtool", {"keywords": "tasa de interés"})
        assert result == {"query": "tasa de interés", "top_k": 5}

    @pytest.mark.asyncio
    async def test_query_list_coerced_to_string(self, clean_registry) -> None:
        self._register_query_tool()
        state = AgentState()
        result, _ = await dispatch(state, "qtool", {"search_term": ["spread", "btp", "spc"]})
        assert result["query"] == "spread btp spc"

    @pytest.mark.asyncio
    async def test_unknown_kwarg_dropped_not_error(self, clean_registry) -> None:
        self._register_query_tool()
        state = AgentState()
        # `bogus` no es alias ni parámetro: se descarta y la tool corre igual.
        result, _ = await dispatch(state, "qtool", {"query": "x", "bogus": 123})
        assert result == {"query": "x", "top_k": 5}

    @pytest.mark.asyncio
    async def test_explicit_canonical_not_overwritten_by_alias(self, clean_registry) -> None:
        self._register_query_tool()
        state = AgentState()
        result, _ = await dispatch(state, "qtool", {"query": "real", "q": "alias"})
        assert result["query"] == "real"

    @pytest.mark.asyncio
    async def test_series_split_into_dataset_and_column(self, clean_registry) -> None:
        schema = {
            "type": "function",
            "function": {
                "name": "ctool",
                "description": "needs dataset_id+column",
                "parameters": {
                    "type": "object",
                    "properties": {"dataset_id": {"type": "string"}, "column": {"type": "string"}},
                    "required": ["dataset_id", "column"],
                },
            },
        }

        @register("ctool", schema)
        async def ctool(state, dataset_id: str, column: str):
            return {"dataset_id": dataset_id, "column": column}

        state = AgentState()
        # El modelo empaqueta "dataset.columna" en `series`.
        result, _ = await dispatch(state, "ctool", {"series": "clp_monto.usdclp"})
        assert result == {"dataset_id": "clp_monto", "column": "usdclp"}

    @pytest.mark.asyncio
    async def test_varkw_tool_passthrough_untouched(self, clean_registry) -> None:
        schema = {
            "type": "function",
            "function": {
                "name": "anytool",
                "description": "accepts anything",
                "parameters": {"type": "object"},
            },
        }

        @register("anytool", schema)
        async def anytool(state, **kwargs):
            return {"got": kwargs}

        state = AgentState()
        # Con **kwargs no se remapea ni descarta nada.
        result, _ = await dispatch(state, "anytool", {"keywords": ["a"], "bogus": 1})
        assert result == {"got": {"keywords": ["a"], "bogus": 1}}

    @pytest.mark.asyncio
    async def test_query_optional_omitted_is_ok(self, clean_registry) -> None:
        # query opcional: si la tool lo declara con default, omitirlo no es error.
        schema = {
            "type": "function",
            "function": {
                "name": "browsetool",
                "description": "query opcional",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }

        @register("browsetool", schema)
        async def browsetool(state, query: str | None = None):
            return {"query": query}

        state = AgentState()
        result, _ = await dispatch(state, "browsetool", {"segment_bogus": "x"})
        assert result == {"query": None}


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
