"""Tests del wiring de sub-agentes (Fase B): specs, delegación y filtrado de tools."""

from __future__ import annotations

import pytest

from banks_rag.application.agent.subagents import (
    DELEGATE_SCHEMAS,
    SUBAGENTS,
    SubAgentSpec,
    subagent_for_delegate,
    tool_schemas_for,
)


@pytest.mark.unit
class TestSubagentSpecs:
    def test_four_subagents_registered(self) -> None:
        assert set(SUBAGENTS) == {"document", "quant", "policy", "market"}

    def test_each_spec_is_self_consistent(self) -> None:
        for key, spec in SUBAGENTS.items():
            assert spec.key == key
            assert spec.delegate_tool.startswith("delegate_to_")
            assert spec.system_prompt
            assert spec.tool_names  # ningún especialista sin tools

    def test_subagents_have_distinct_prompts(self) -> None:
        prompts = {spec.system_prompt for spec in SUBAGENTS.values()}
        assert len(prompts) == 4

    def test_subagent_keys_appear_in_prompt(self) -> None:
        """Cada prompt debe describir al especialista correcto (sanity check)."""
        assert "Documentos" in SUBAGENTS["document"].system_prompt
        assert "Cuantitativo" in SUBAGENTS["quant"].system_prompt
        assert "Política Monetaria" in SUBAGENTS["policy"].system_prompt
        assert "Mercados" in SUBAGENTS["market"].system_prompt


@pytest.mark.unit
class TestDelegateSchemas:
    def test_one_schema_per_subagent(self) -> None:
        names = [s["function"]["name"] for s in DELEGATE_SCHEMAS]
        assert names == [
            "delegate_to_document_analyst",
            "delegate_to_quant_analyst",
            "delegate_to_policy_analyst",
            "delegate_to_market_analyst",
        ]

    def test_every_delegate_requires_task(self) -> None:
        for schema in DELEGATE_SCHEMAS:
            params = schema["function"]["parameters"]
            assert params["required"] == ["task"]
            assert "task" in params["properties"]


@pytest.mark.unit
class TestSubagentForDelegate:
    def test_resolves_known_delegate(self) -> None:
        spec = subagent_for_delegate("delegate_to_market_analyst")
        assert spec is not None
        assert spec.key == "market"

    def test_unknown_delegate_returns_none(self) -> None:
        assert subagent_for_delegate("delegate_to_nobody") is None


@pytest.mark.unit
class TestToolSchemasFor:
    def test_returns_declared_subset_in_order(self) -> None:
        spec = SUBAGENTS["quant"]
        schemas = tool_schemas_for(spec)
        names = [s["function"]["name"] for s in schemas]
        assert names == list(spec.tool_names)

    def test_document_analyst_has_no_sql_tools(self) -> None:
        names = {s["function"]["name"] for s in tool_schemas_for(SUBAGENTS["document"])}
        assert "execute_query" not in names
        assert "search_documents" in names

    def test_raises_on_unregistered_tool(self) -> None:
        bad = SubAgentSpec(
            key="bad",
            display_name="Bad",
            delegate_tool="delegate_to_bad",
            delegate_description="x",
            system_prompt="x",
            tool_names=("herramienta_inexistente",),
        )
        with pytest.raises(ValueError, match="no registradas"):
            tool_schemas_for(bad)
