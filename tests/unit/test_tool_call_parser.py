"""Unit tests para parser de tool calls."""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.llm.tool_call_parser import parse_tool_calls


@pytest.mark.unit
class TestParseToolCalls:
    def test_no_tool_calls_in_plain_text(self) -> None:
        calls, cleaned = parse_tool_calls("Esta es una respuesta normal sin tool calls.")
        assert calls == []
        assert cleaned == "Esta es una respuesta normal sin tool calls."

    def test_qwen_format_single_call(self) -> None:
        text = (
            'Voy a buscar información.\n'
            '<tool_call>\n'
            '{"name": "search_documents", "arguments": {"query": "TPM 2024"}}\n'
            '</tool_call>'
        )
        calls, cleaned = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "search_documents"
        assert calls[0].arguments == {"query": "TPM 2024"}
        assert "<tool_call>" not in cleaned
        assert calls[0].id.startswith("call_")

    def test_multiple_tool_calls(self) -> None:
        text = (
            '<tool_call>{"name": "tool_a", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "tool_b", "arguments": {"y": 2}}</tool_call>'
        )
        calls, cleaned = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "tool_a"
        assert calls[1].name == "tool_b"

    def test_arguments_as_json_string(self) -> None:
        # Algunos modelos serializan arguments dos veces.
        text = '<tool_call>{"name": "foo", "arguments": "{\\"x\\": 1}"}</tool_call>'
        calls, _ = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments == {"x": 1}

    def test_invalid_json_skipped(self) -> None:
        text = '<tool_call>{not valid json}</tool_call>'
        calls, _ = parse_tool_calls(text)
        assert calls == []

    def test_fenced_json_fallback(self) -> None:
        text = (
            'Llamo a la tool:\n```json\n'
            '{"name": "search_documents", "arguments": {"query": "x"}}\n'
            '```'
        )
        calls, _ = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "search_documents"

    def test_raw_json_fallback(self) -> None:
        text = '{"name": "search_documents", "arguments": {"query": "y"}}'
        calls, cleaned = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "search_documents"
        assert cleaned == ""

    def test_each_call_gets_unique_id(self) -> None:
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        )
        calls, _ = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].id != calls[1].id
