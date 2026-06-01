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


@pytest.mark.unit
class TestParseXmlToolCalls:
    """Formato XML/Hermes que emite Qwen3 con la plantilla nativa del GGUF."""

    def test_xml_single_param(self) -> None:
        text = (
            "Voy a buscar el dataset.\n"
            "<tool_call>\n"
            "<function=discover_query>\n"
            "<parameter=query>\nspread BTP vs SPC a 10 años\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls, cleaned = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "discover_query"
        assert calls[0].arguments == {"query": "spread BTP vs SPC a 10 años"}
        assert cleaned == "Voy a buscar el dataset."

    def test_xml_multiple_params(self) -> None:
        text = (
            "<tool_call><function=discover_query>"
            "<parameter=query>LCR</parameter>"
            "<parameter=segment>liquidez_bancaria</parameter>"
            "</function></tool_call>"
        )
        calls, _ = parse_tool_calls(text)
        assert calls[0].arguments == {"query": "LCR", "segment": "liquidez_bancaria"}

    def test_xml_numeric_param_coerced(self) -> None:
        # Un valor JSON-parseable se convierte a su tipo nativo (int, no str).
        text = (
            "<tool_call><function=execute_query>"
            "<parameter=dataset_id>clp_monto</parameter>"
            "<parameter=limit>5</parameter>"
            "</function></tool_call>"
        )
        calls, _ = parse_tool_calls(text)
        assert calls[0].arguments == {"dataset_id": "clp_monto", "limit": 5}
        assert isinstance(calls[0].arguments["limit"], int)

    def test_xml_truncated_no_closing_tags(self) -> None:
        # Generación cortada por max_tokens: faltan </parameter></function></tool_call>.
        text = "<tool_call>\n<function=discover_query>\n<parameter=query>\nLCR sistema bancario"
        calls, _ = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "discover_query"
        assert calls[0].arguments == {"query": "LCR sistema bancario"}

    def test_xml_no_params(self) -> None:
        text = "<tool_call><function=list_documents></function></tool_call>"
        calls, _ = parse_tool_calls(text)
        assert calls[0].name == "list_documents"
        assert calls[0].arguments == {}

    def test_json_format_takes_precedence(self) -> None:
        # Si viene JSON válido, no se intenta el parser XML.
        text = '<tool_call>{"name": "search_documents", "arguments": {"query": "x"}}</tool_call>'
        calls, _ = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "search_documents"
