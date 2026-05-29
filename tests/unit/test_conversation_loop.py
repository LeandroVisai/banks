"""Unit tests del orquestador multi-agente (Fase B) con LLM mock.

El mock comparte su lista de respuestas entre orquestador y sub-agentes: las
respuestas se consumen en el orden en que el loop llama al LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from banks_rag.application.agent import run_agent, run_subagent
from banks_rag.application.agent.conversation_loop import _format_chunks_seen
from banks_rag.application.agent.prompts import AFP_ANALYST_PROMPT
from banks_rag.application.agent.subagents import SubAgentSpec
from banks_rag.application.agent.tools.registry import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    register,
    reset_registry,
)
from banks_rag.domain.agent import AgentState, GenerationResult, ToolCall


@dataclass
class _MockLLM:
    """LLM mock que devuelve respuestas pre-programadas en orden."""

    responses: list[GenerationResult] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    async def generate(self, messages, *, tools=None, **kwargs):
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "n_messages": len(messages),
            "tools": tools,
        })
        if not self.responses:
            return GenerationResult(text="default", n_tokens=5)
        return self.responses.pop(0)

    def count_text_tokens(self, text: str) -> int:
        return len(text) // 4


def _delegate(call_id: str, delegate_tool: str, task: str) -> GenerationResult:
    return GenerationResult(
        text="",
        tool_calls=[ToolCall(id=call_id, name=delegate_tool, arguments={"task": task})],
        n_tokens=10,
    )


@dataclass
class _RoutedLLM:
    """LLM mock que responde según qué agente lo llama y el turno de ese agente.

    Detecta el agente por su system prompt. A diferencia de ``_MockLLM`` (lista
    posicional única), es determinista bajo delegación concurrente: cada
    sub-agente sigue su propio guion sin importar el orden de interleaving.
    """

    scripts: dict[str, list[GenerationResult]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    # Cada prompt abre con "Eres el <rol>"; el prefijo exacto identifica al
    # agente sin colisiones (los sub-agentes mencionan "coordinador" en su
    # cuerpo, así que un `in` genérico no sirve).
    _OPENINGS = (
        ("Eres el coordinador", "orchestrator"),
        ("Eres el Analista de Mercado Cambiario (FX)", "fx"),
        ("Eres el Analista de No Residentes (NR)", "no_residentes"),
        ("Eres el Analista de Fondos de Pensiones (AFP)", "afp"),
        ("Eres el Analista de Fondos Mutuos (FFMM)", "fondos_mutuos"),
        ("Eres el Analista de Renta Fija", "renta_fija"),
        ("Eres el Analista de Liquidez y Balance", "liquidez"),
        ("Eres el Analista de Documentos", "document"),
        ("Eres el Analista de Política Monetaria", "policy"),
    )

    @classmethod
    def _agent_of(cls, system_prompt: str) -> str:
        # `in` (no startswith): _with_today() antepone la fecha al system
        # prompt, así que el rol ya no está al inicio. "Eres el coordinador" no
        # aparece en los cuerpos de los sub-agentes (dicen "El coordinador"),
        # por lo que no hay colisión.
        for opening, key in cls._OPENINGS:
            if opening in system_prompt:
                return key
        return "?"

    async def generate(self, messages, *, tools=None, **kwargs):
        key = self._agent_of(messages[0]["content"])
        turn = self.counts.get(key, 0)
        self.counts[key] = turn + 1
        self.calls.append(key)
        script = self.scripts.get(key, [])
        if turn < len(script):
            return script[turn]
        return GenerationResult(text="(sin guion)", n_tokens=1)

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


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_answers_trivial_question_directly(self) -> None:
        """Sin delegar: un saludo se responde directo en una iteración."""
        llm = _MockLLM(responses=[
            GenerationResult(text="Hola, soy el asistente del BCCh.", n_tokens=10),
        ])
        result = await run_agent("Hola", history=[], llm=llm)
        assert result.response == "Hola, soy el asistente del BCCh."
        assert result.iterations == 1
        assert result.tool_trace == []
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_delegates_to_subagent_and_synthesizes(self) -> None:
        llm = _MockLLM(responses=[
            # Orquestador iter 1: delega al analista AFP.
            _delegate("c1", "delegate_to_afp_analyst", "¿DV01 de los fondos de pensiones?"),
            # Sub-agente AFP: responde directo (sin tools, sin cifras → se conserva).
            GenerationResult(text="No tengo el detalle aún.", n_tokens=20),
            # Orquestador iter 2: sintetiza.
            GenerationResult(text="El especialista no entregó el detalle.", n_tokens=15),
        ])
        result = await run_agent("¿DV01 de las AFP?", history=[], llm=llm)

        assert result.response == "El especialista no entregó el detalle."
        assert result.iterations == 2
        assert result.total_tokens == 25  # solo cuentan las llamadas del orquestador

        # La traza registra la delegación, etiquetada como del orquestador.
        assert len(result.tool_trace) == 1
        entry = result.tool_trace[0]
        assert entry["tool"] == "delegate_to_afp_analyst"
        assert entry["agent"] == "orquestador"
        assert "Fondos de Pensiones" in entry["result_summary"]

        # El sub-agente recibió su system prompt y el task — no la conversación.
        # _with_today() antepone la fecha, así que el prompt está contenido.
        sub_call = llm.calls[1]
        assert sub_call["n_messages"] == 2
        assert AFP_ANALYST_PROMPT in sub_call["messages"][0]["content"]
        assert sub_call["messages"][1]["content"] == "¿DV01 de los fondos de pensiones?"
        assert len(llm.calls) == 3

    @pytest.mark.asyncio
    async def test_concurrent_delegation_to_two_subagents(self) -> None:
        """El orquestador delega a dos especialistas en un mismo turno.

        Determinista pese al interleaving de asyncio: cada agente sigue su
        propio guion en _RoutedLLM, ruteado por system prompt.
        """
        llm = _RoutedLLM(scripts={
            "orchestrator": [
                GenerationResult(text="", tool_calls=[
                    ToolCall(id="o1", name="delegate_to_fx_analyst",
                             arguments={"task": "panorama cambiario"}),
                    ToolCall(id="o2", name="delegate_to_renta_fija_analyst",
                             arguments={"task": "curva BTP"}),
                ], n_tokens=15),
                GenerationResult(text="Cambiario estable; curva al alza.", n_tokens=12),
            ],
            "fx": [GenerationResult(text="El mercado cambiario está estable.", n_tokens=8)],
            "renta_fija": [GenerationResult(text="La curva BTP subió.", n_tokens=8)],
        })
        result = await run_agent("cambiario y curva", history=[], llm=llm)

        assert result.response == "Cambiario estable; curva al alza."
        assert result.iterations == 2
        # Ambas delegaciones quedan en la traza, tagueadas al orquestador.
        delegations = [t for t in result.tool_trace if t["tool"].startswith("delegate_")]
        assert {t["tool"] for t in delegations} == {
            "delegate_to_fx_analyst", "delegate_to_renta_fija_analyst",
        }
        assert all(t["agent"] == "orquestador" for t in delegations)
        # Cada especialista corrió exactamente una vez.
        assert llm.calls.count("fx") == 1
        assert llm.calls.count("renta_fija") == 1

    @pytest.mark.asyncio
    async def test_unknown_delegate_returns_structured_error(self) -> None:
        """Si el LLM inventa un especialista, el dispatch responde con error."""
        llm = _MockLLM(responses=[
            GenerationResult(
                text="",
                tool_calls=[ToolCall(id="c1", name="delegate_to_ghost", arguments={"task": "x"})],
                n_tokens=5,
            ),
            GenerationResult(text="No pude completar la consulta.", n_tokens=10),
        ])
        result = await run_agent("pregunta", history=[], llm=llm)
        assert result.response == "No pude completar la consulta."
        assert result.tool_trace[0]["result_summary"].startswith("ERROR")

    @pytest.mark.asyncio
    async def test_max_iterations_fallback(self) -> None:
        llm = _MockLLM(responses=[
            _delegate("c1", "delegate_to_afp_analyst", "tarea 1"),
            GenerationResult(text="análisis del sub-agente", n_tokens=5),
            _delegate("c2", "delegate_to_afp_analyst", "tarea 2"),
        ])
        result = await run_agent("loop", history=[], llm=llm, max_iterations=2)
        assert result.iterations == 2
        assert result.finish_reason == "max_iterations"
        assert "iteraciones" in result.response.lower()

    @pytest.mark.asyncio
    async def test_invalid_citations_cleaned(self) -> None:
        llm = _MockLLM(responses=[
            GenerationResult(text="La TPM bajó [5] según fuentes.", n_tokens=10),
        ])
        result = await run_agent("?", history=[], llm=llm)
        assert "[5]" not in result.response
        assert result.cited_refs == []
        assert result.invalid_refs == [5]

    @pytest.mark.asyncio
    async def test_flags_ungrounded_figures_in_final_answer(self) -> None:
        """Anti-alucinación (orquestador): cifras sin respaldo de tool quedan
        reportadas en ungrounded_numbers — red de seguridad para gerentes."""
        llm = _MockLLM(responses=[
            GenerationResult(text="La TPM está en 5,25% y el dólar en 942.", n_tokens=10),
        ])
        result = await run_agent("nivel de la TPM", history=[], llm=llm)
        # Ninguna tool de evidencia corrió → ambas cifras son no fundadas.
        assert 5.25 in result.ungrounded_numbers
        assert 942.0 in result.ungrounded_numbers

    @pytest.mark.asyncio
    async def test_history_is_passed_to_orchestrator(self) -> None:
        llm = _MockLLM(responses=[GenerationResult(text="ok", n_tokens=1)])
        history = [
            {"role": "user", "content": "Pregunta previa"},
            {"role": "assistant", "content": "Respuesta previa"},
        ]
        await run_agent("Nueva pregunta", history=history, llm=llm)
        # System + 2 history + user = 4 mensajes.
        assert llm.calls[0]["n_messages"] == 4


# ─────────────────────────────────────────────────────────────────────────────
# Sub-agente especialista
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRunSubagent:
    @pytest.mark.asyncio
    async def test_subagent_dispatches_tool_and_shares_state(self, clean_registry) -> None:
        """Un sub-agente ejecuta sus tools y acumula chunks en el state compartido."""

        @register("fake_tool", {
            "type": "function",
            "function": {
                "name": "fake_tool",
                "description": "tool de prueba",
                "parameters": {"type": "object", "properties": {}},
            },
        })
        async def fake_tool(state):
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

        spec = SubAgentSpec(
            key="tester",
            display_name="Analista de Prueba",
            delegate_tool="delegate_to_tester",
            delegate_description="x",
            system_prompt="Eres un analista de prueba.",
            tool_names=("fake_tool",),
        )
        llm = _MockLLM(responses=[
            GenerationResult(
                text="",
                tool_calls=[ToolCall(id="c1", name="fake_tool", arguments={})],
                n_tokens=8,
            ),
            GenerationResult(text="Encontré evidencia [1].", n_tokens=12),
        ])
        state = AgentState()

        sub = await run_subagent(spec, "busca algo", llm=llm, state=state)

        assert sub.key == "tester"
        assert sub.display_name == "Analista de Prueba"
        assert sub.analysis == "Encontré evidencia [1]."
        assert sub.iterations == 2
        assert sub.total_tokens == 20

        # El chunk y la traza viven en el state compartido, etiquetados al sub-agente.
        assert len(state.chunks_seen) == 1
        assert len(state.tool_trace) == 1
        assert state.tool_trace[0]["tool"] == "fake_tool"
        assert state.tool_trace[0]["agent"] == "tester"

    @pytest.mark.asyncio
    async def test_subagent_answers_without_tools(self, clean_registry) -> None:
        spec = SubAgentSpec(
            key="tester",
            display_name="Analista de Prueba",
            delegate_tool="delegate_to_tester",
            delegate_description="x",
            system_prompt="Eres un analista de prueba.",
            tool_names=(),
        )
        llm = _MockLLM(responses=[GenerationResult(text="Respuesta directa.", n_tokens=5)])
        state = AgentState()

        sub = await run_subagent(spec, "tarea simple", llm=llm, state=state)
        assert sub.analysis == "Respuesta directa."
        assert sub.iterations == 1
        assert state.tool_trace == []


@pytest.mark.unit
class TestFormatChunksSeen:
    """Fase 3: los chunks VISUAL se exponen con image_url/caption para el frontend."""

    def test_visual_chunk_exposes_image_url(self) -> None:
        state = AgentState()
        state.add_chunk({
            "chunk_id": "doc_img_0001", "kind": "VISUAL",
            "visual_caption": "Curva swap CLP — último IPoM",
            "image_path": "images/doc/p3.png", "section_type": None,
            "doc_type_category": "IPOM", "document_date": "2026-03-31",
            "importance_score": 0.8, "filename": "ipom_mar2026.pdf",
            "page_start": 12, "page_end": 12,
        })
        out = _format_chunks_seen(state)[0]
        assert out["kind"] == "VISUAL"
        assert out["image_url"] == "/v1/images/doc_img_0001"
        assert out["caption"] == "Curva swap CLP — último IPoM"

    def test_text_chunk_has_no_image_url(self) -> None:
        state = AgentState()
        state.add_chunk({
            "chunk_id": "doc_0001", "kind": "TEXT", "section_type": "DECISION",
            "doc_type_category": "COMUNICADO_RPM", "document_date": "2026-03-31",
            "importance_score": 0.9, "filename": "com.pdf",
            "page_start": 1, "page_end": 1,
        })
        out = _format_chunks_seen(state)[0]
        assert out["kind"] == "TEXT"
        assert "image_url" not in out
        assert "caption" not in out

    def test_handles_chunkkind_enum(self) -> None:
        from banks_rag.domain.documents import ChunkKind
        state = AgentState()
        state.add_chunk({
            "chunk_id": "v9", "kind": ChunkKind.VISUAL,
            "visual_caption": "Gráfico", "image_path": "images/x.png",
            "doc_type_category": "IPOM",
        })
        out = _format_chunks_seen(state)[0]
        assert out["kind"] == "VISUAL"
        assert out["image_url"] == "/v1/images/v9"

    @pytest.mark.asyncio
    async def test_subagent_figures_without_evidence_are_discarded(self, clean_registry) -> None:
        """Anti-alucinación: cifras sin tool de evidencia → análisis descartado.

        Reproduce el caso DV01/AFP de los logs: el especialista 'responde' con
        cifras detalladas sin haber ejecutado ninguna consulta.
        """
        spec = SubAgentSpec(
            key="afp",
            display_name="Analista de Fondos de Pensiones (AFP)",
            delegate_tool="delegate_to_afp_analyst",
            delegate_description="x",
            system_prompt="Eres un analista de prueba.",
            tool_names=(),
        )
        llm = _MockLLM(responses=[
            GenerationResult(
                text="El DV01 promedió 0,85% con un máximo de 0,92%.", n_tokens=20,
            ),
        ])
        state = AgentState()
        sub = await run_subagent(spec, "DV01 de los fondos de pensiones", llm=llm, state=state)

        assert sub.finish_reason == "ungrounded"
        assert "0,85" not in sub.analysis and "0.85" not in sub.analysis
        assert "no dispongo" in sub.analysis.lower() or "no pude" in sub.analysis.lower()

    @pytest.mark.asyncio
    async def test_subagent_figures_with_evidence_are_kept(self, clean_registry) -> None:
        """Si una tool de evidencia entregó el número, la cifra se conserva."""

        @register("get_series_stats", {
            "type": "function",
            "function": {
                "name": "get_series_stats",
                "description": "stats de prueba",
                "parameters": {"type": "object", "properties": {}},
            },
        })
        async def _fake_stats(state):
            return {"mean": 0.85, "max": 0.92, "n": 120}

        spec = SubAgentSpec(
            key="afp",
            display_name="Analista de Fondos de Pensiones (AFP)",
            delegate_tool="delegate_to_afp_analyst",
            delegate_description="x",
            system_prompt="Eres un analista de prueba.",
            tool_names=("get_series_stats",),
        )
        llm = _MockLLM(responses=[
            GenerationResult(
                text="",
                tool_calls=[ToolCall(id="c1", name="get_series_stats", arguments={})],
                n_tokens=8,
            ),
            GenerationResult(text="El DV01 promedió 0,85% (máx 0,92%).", n_tokens=15),
        ])
        state = AgentState()
        sub = await run_subagent(spec, "DV01 fondos de pensiones", llm=llm, state=state)

        assert sub.finish_reason != "ungrounded"
        assert "0,85" in sub.analysis
