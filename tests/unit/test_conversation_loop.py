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
# run_agent — router determinista + especialistas en paralelo + síntesis única
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRunAgent:
    @pytest.mark.asyncio
    async def test_greeting_answered_directly_no_specialists(self) -> None:
        """Saludo: router → sin especialistas → síntesis directa en 1 llamada."""
        llm = _MockLLM(responses=[
            GenerationResult(text="Hola, soy el asistente del BCCh.", n_tokens=10),
        ])
        result = await run_agent("Hola", history=[], llm=llm)
        assert result.response == "Hola, soy el asistente del BCCh."
        assert result.tool_trace == []        # no corrió ningún especialista
        assert len(llm.calls) == 1            # solo la síntesis
        assert llm.calls[0]["tools"] is None  # síntesis SIN tools → no puede loopear

    @pytest.mark.asyncio
    async def test_single_domain_routes_runs_specialist_then_synthesizes(self) -> None:
        """'DV01 de las AFP' → router=[afp] → subagente AFP → síntesis."""
        llm = _MockLLM(responses=[
            GenerationResult(text="El DV01 no está disponible aún.", n_tokens=10),  # subagente afp
            GenerationResult(text="Respuesta final del coordinador.", n_tokens=8),  # síntesis
        ])
        result = await run_agent("¿DV01 de las AFP?", history=[], llm=llm)

        assert result.response == "Respuesta final del coordinador."
        assert len(llm.calls) == 2
        # Llamada 0 = subagente AFP: su system prompt + la pregunta como task.
        assert AFP_ANALYST_PROMPT in llm.calls[0]["messages"][0]["content"]
        assert llm.calls[0]["messages"][1]["content"] == "¿DV01 de las AFP?"
        # Llamada 1 = síntesis: SIN tools.
        assert llm.calls[1]["tools"] is None

    @pytest.mark.asyncio
    async def test_thinking_mode_adaptive_quant_thinks_synthesis_does_not(self) -> None:
        """adaptive: el especialista cuantitativo (afp) razona (sin /no_think);
        la síntesis no (con /no_think)."""
        llm = _MockLLM(responses=[
            GenerationResult(text="DV01 reciente.", n_tokens=8),    # subagente afp
            GenerationResult(text="Respuesta final.", n_tokens=6),  # síntesis
        ])
        await run_agent("¿DV01 de las AFP?", history=[], llm=llm, thinking_mode="adaptive")
        afp_system = llm.calls[0]["messages"][0]["content"]
        synth_system = llm.calls[1]["messages"][0]["content"]
        assert "/no_think" not in afp_system     # cuant razona
        assert synth_system.endswith("/no_think")  # síntesis no

    @pytest.mark.asyncio
    async def test_attachments_context_injected(self) -> None:
        """El contenido adjunto se inyecta en el task del especialista y en la síntesis."""
        llm = _MockLLM(responses=[
            GenerationResult(text="análisis del especialista", n_tokens=5),
            GenerationResult(text="respuesta final", n_tokens=5),
        ])
        await run_agent(
            "¿DV01 de las AFP?", history=[], llm=llm,
            attachments_context="--- Datos adjuntos: x.csv ---\nFecha,Cobre\n2026-05-20,624.9",
        )
        # Llamada 0 = especialista: ve el adjunto en su task.
        afp_task = llm.calls[0]["messages"][1]["content"]
        assert "CONTENIDO ADJUNTO" in afp_task and "624.9" in afp_task
        # Última llamada = síntesis: también ve el adjunto.
        synth_user = llm.calls[-1]["messages"][-1]["content"]
        assert "CONTENIDO ADJUNTO" in synth_user

    @pytest.mark.asyncio
    async def test_no_attachments_no_injection(self) -> None:
        """Sin adjuntos, el task del especialista es la pregunta limpia."""
        llm = _MockLLM(responses=[
            GenerationResult(text="análisis", n_tokens=5),
            GenerationResult(text="final", n_tokens=5),
        ])
        await run_agent("¿DV01 de las AFP?", history=[], llm=llm)
        assert "CONTENIDO ADJUNTO" not in llm.calls[0]["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_thinking_mode_off_everywhere(self) -> None:
        """off: ni el especialista ni la síntesis razonan."""
        llm = _MockLLM(responses=[
            GenerationResult(text="DV01 reciente.", n_tokens=8),
            GenerationResult(text="Respuesta final.", n_tokens=6),
        ])
        await run_agent("¿DV01 de las AFP?", history=[], llm=llm, thinking_mode="off")
        assert llm.calls[0]["messages"][0]["content"].endswith("/no_think")
        assert llm.calls[1]["messages"][0]["content"].endswith("/no_think")

    @pytest.mark.asyncio
    async def test_multi_domain_runs_specialists_in_parallel(self) -> None:
        """Cross-dominio → varios especialistas en paralelo + una síntesis."""
        llm = _RoutedLLM(scripts={
            "renta_fija": [GenerationResult(text="La curva BTP subió.", n_tokens=8)],
            "policy": [GenerationResult(text="El Consejo mantuvo la TPM.", n_tokens=8)],
            "orchestrator": [GenerationResult(text="Curva al alza; TPM sin cambios.", n_tokens=12)],
        })
        # "postura del Consejo" → policy; "curva BTP" → renta_fija.
        result = await run_agent(
            "compara la postura del Consejo con la curva BTP", history=[], llm=llm,
        )
        assert result.response == "Curva al alza; TPM sin cambios."
        assert llm.calls.count("renta_fija") == 1
        assert llm.calls.count("policy") == 1
        assert llm.calls.count("orchestrator") == 1   # una sola síntesis

    @pytest.mark.asyncio
    async def test_invalid_citations_cleaned(self) -> None:
        # Saludo → sin especialistas → chunks_seen vacío → [5] es inválida.
        llm = _MockLLM(responses=[
            GenerationResult(text="La TPM bajó [5] según fuentes.", n_tokens=10),
        ])
        result = await run_agent("hola", history=[], llm=llm)
        assert "[5]" not in result.response
        assert result.cited_refs == []
        assert result.invalid_refs == [5]

    @pytest.mark.asyncio
    async def test_flags_ungrounded_figures_in_synthesis(self) -> None:
        """Si la síntesis emite cifras sin respaldo de tool, se reportan."""
        llm = _MockLLM(responses=[
            GenerationResult(text="La TPM está en 5,25% y el dólar en 942.", n_tokens=10),
        ])
        result = await run_agent("hola", history=[], llm=llm)  # sin especialistas → 0 evidencia
        assert 5.25 in result.ungrounded_numbers
        assert 942.0 in result.ungrounded_numbers

    @pytest.mark.asyncio
    async def test_history_is_passed_to_synthesis(self) -> None:
        llm = _MockLLM(responses=[GenerationResult(text="ok", n_tokens=1)])
        history = [
            {"role": "user", "content": "Pregunta previa"},
            {"role": "assistant", "content": "Respuesta previa"},
        ]
        await run_agent("Hola", history=history, llm=llm)
        # Saludo → única llamada = síntesis: system + 2 history + user = 4 mensajes.
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
            system_prompt="Eres un analista de prueba.",
            tool_names=(),
        )
        llm = _MockLLM(responses=[GenerationResult(text="Respuesta directa.", n_tokens=5)])
        state = AgentState()

        sub = await run_subagent(spec, "tarea simple", llm=llm, state=state)
        assert sub.analysis == "Respuesta directa."
        assert sub.iterations == 1
        assert state.tool_trace == []

    @pytest.mark.asyncio
    async def test_think_false_injects_no_think(self, clean_registry) -> None:
        """think=False anexa /no_think al system prompt del especialista."""
        spec = SubAgentSpec(
            key="tester", display_name="X",
            system_prompt="Eres un analista de prueba.", tool_names=(),
        )
        llm = _MockLLM(responses=[GenerationResult(text="ok", n_tokens=3)])
        await run_subagent(spec, "t", llm=llm, state=AgentState(), think=False)
        system_msg = llm.calls[0]["messages"][0]["content"]
        assert system_msg.endswith("/no_think")

    @pytest.mark.asyncio
    async def test_think_true_no_directive(self, clean_registry) -> None:
        spec = SubAgentSpec(
            key="tester", display_name="X",
            system_prompt="Eres un analista de prueba.", tool_names=(),
        )
        llm = _MockLLM(responses=[GenerationResult(text="ok", n_tokens=3)])
        await run_subagent(spec, "t", llm=llm, state=AgentState(), think=True)
        assert "/no_think" not in llm.calls[0]["messages"][0]["content"]


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
    async def test_repeated_identical_tool_call_runs_once(self, clean_registry) -> None:
        """Anti-loop: si el modelo repite la MISMA tool con los MISMOS args en
        iteraciones distintas, se ejecuta una sola vez (la 2ª es un nudge)."""
        calls = {"n": 0}

        @register("fake_tool", {
            "type": "function",
            "function": {"name": "fake_tool", "description": "x",
                         "parameters": {"type": "object", "properties": {}}},
        })
        async def fake_tool(state):
            calls["n"] += 1
            return {"n_results": 1}

        spec = SubAgentSpec(
            key="afp", display_name="X",
            system_prompt="Eres un analista de prueba.",
            tool_names=("fake_tool",),
        )
        llm = _MockLLM(responses=[
            GenerationResult(text="", tool_calls=[
                ToolCall(id="c1", name="fake_tool", arguments={})], n_tokens=5),
            GenerationResult(text="", tool_calls=[
                ToolCall(id="c2", name="fake_tool", arguments={})], n_tokens=5),
            GenerationResult(text="Listo, sin más datos.", n_tokens=5),
        ])
        state = AgentState()
        sub = await run_subagent(spec, "tarea", llm=llm, state=state, max_iterations=4)

        assert calls["n"] == 1            # dispatch real solo la primera vez
        assert sub.analysis == "Listo, sin más datos."

    @pytest.mark.asyncio
    async def test_last_iteration_injects_synthesis_nudge(self, clean_registry) -> None:
        """Consolidación: en la última iteración se inyecta el nudge (sin tools) y
        el especialista redacta con la evidencia reunida en vez de agotar sin
        concluir (caso cobre: tenía stats pero entregaba un 'plan')."""
        from banks_rag.application.agent.prompts import FINAL_SYNTHESIS_NUDGE

        @register("get_series_stats", {
            "type": "function",
            "function": {"name": "get_series_stats", "description": "x",
                         "parameters": {"type": "object", "properties": {}}},
        })
        async def _stats(state, **kw):
            return {"mean": 1.73, "max": 2.08, "last": 1.76}

        spec = SubAgentSpec(
            key="liquidez", display_name="Analista de Liquidez y Balance",
            system_prompt="Eres un analista de prueba.",
            tool_names=("get_series_stats",),
        )
        llm = _MockLLM(responses=[
            GenerationResult(text="", tool_calls=[
                ToolCall(id="c1", name="get_series_stats", arguments={"q": "a"})], n_tokens=5),
            GenerationResult(text="", tool_calls=[
                ToolCall(id="c2", name="get_series_stats", arguments={"q": "b"})], n_tokens=5),
            GenerationResult(text="El LCR más reciente es 1,76 (al 12-may-2026).", n_tokens=10),
        ])
        state = AgentState()
        sub = await run_subagent(spec, "LCR del sistema", llm=llm, state=state, max_iterations=3)

        # Consolidó con la evidencia (no es el fallback de max_iterations).
        assert "1,76" in sub.analysis
        assert sub.finish_reason != "max_iterations"
        # El nudge se inyectó en la última llamada, que NO ofreció tools.
        last_msgs = llm.calls[-1]["messages"]
        assert any(FINAL_SYNTHESIS_NUDGE in (m.get("content") or "") for m in last_msgs)
        assert llm.calls[-1]["tools"] is None

    @pytest.mark.asyncio
    async def test_synthesis_nudge_not_injected_on_single_iteration(self, clean_registry) -> None:
        """Con max_iterations=1 (is_last en la iter 1) NO se inyecta el nudge:
        no hubo ronda de tools previa que consolidar."""
        from banks_rag.application.agent.prompts import FINAL_SYNTHESIS_NUDGE

        spec = SubAgentSpec(
            key="afp", display_name="X",
            system_prompt="Eres un analista de prueba.",
            tool_names=(),
        )
        llm = _MockLLM(responses=[GenerationResult(text="Respuesta directa.", n_tokens=5)])
        await run_subagent(spec, "tarea", llm=llm, state=AgentState(), max_iterations=1)

        first_msgs = llm.calls[0]["messages"]
        assert not any(FINAL_SYNTHESIS_NUDGE in (m.get("content") or "") for m in first_msgs)

    @pytest.mark.asyncio
    async def test_subagent_figures_without_evidence_are_discarded(self, clean_registry) -> None:
        """Anti-alucinación: cifras sin tool de evidencia → análisis descartado.

        Reproduce el caso DV01/AFP de los logs: el especialista 'responde' con
        cifras detalladas sin haber ejecutado ninguna consulta.
        """
        spec = SubAgentSpec(
            key="afp",
            display_name="Analista de Fondos de Pensiones (AFP)",
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
