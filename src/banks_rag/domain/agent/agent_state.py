"""Estado mutable del agente durante una sesión de chat.

Acumula chunks vistos (con refs globales [1], [2], ...), series consultadas
y la traza completa de tool calls. Las tools mutan este objeto para que el
agente pueda citar en su respuesta final.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Estado del agente durante un único turno (request → response)."""

    chunks_seen: list[dict] = field(default_factory=list)        # con refs [1], [2], ...
    chunk_id_to_ref: dict[str, int] = field(default_factory=dict)
    series_used: dict[str, dict] = field(default_factory=dict)
    tool_trace: list[dict] = field(default_factory=list)
    # Grounding numérico: todos los números que las herramientas entregaron en
    # este turno. Una cifra de la respuesta solo es válida si matchea aquí
    # (ver application/agent/numeric_grounding.py).
    grounded_numbers: list[float] = field(default_factory=list)
    # Cuántas tools que producen EVIDENCIA numérica corrió cada agente
    # (key = agent_label). Si un especialista emite cifras con 0 evidencia,
    # es alucinación.
    evidence_tool_calls: dict[str, int] = field(default_factory=dict)

    def add_grounded_numbers(self, numbers: list[float]) -> None:
        """Acumula números entregados por una herramienta (evidencia citable)."""
        self.grounded_numbers.extend(numbers)

    def note_evidence_tool(self, agent_label: str) -> None:
        """Registra que ``agent_label`` corrió una tool de evidencia numérica."""
        self.evidence_tool_calls[agent_label] = (
            self.evidence_tool_calls.get(agent_label, 0) + 1
        )

    def add_chunk(self, chunk: dict) -> int:
        """Asigna una ref global al chunk. Idempotente: si ya tiene ref, retorna esa.

        Returns:
            El número de ref (1-based) para usar en citas ``[N]``.
        """
        cid = str(chunk.get("chunk_id"))
        if cid in self.chunk_id_to_ref:
            return self.chunk_id_to_ref[cid]
        self.chunks_seen.append(chunk)
        ref = len(self.chunks_seen)
        self.chunk_id_to_ref[cid] = ref
        return ref

    def add_series(self, series_id: str, meta: dict, rows: list[dict]) -> None:
        """Registra una serie consultada. ``rows`` debe tener key ``date`` por fila.

        Si las filas traen además ``value`` (numérico), se guardan los ``points``
        ``[[iso_date, value], ...]`` para que el frontend grafique la serie en la
        respuesta. Se capan a ``MAX_SERIES_POINTS`` (los más recientes) para no
        inflar el payload del chat; sin ``value`` la serie no es graficable."""
        points = _extract_points(rows)
        self.series_used[series_id] = {
            "series_id": series_id,
            "series_name": meta.get("series_name") or meta.get("name", ""),
            "unit": meta.get("unit", ""),
            "frequency": meta.get("frequency", ""),
            "n_observations": len(rows),
            "first_date": _isoformat(rows[0]["date"]) if rows else None,
            "last_date": _isoformat(rows[-1]["date"]) if rows else None,
            "points": points,
        }

    def add_tool_call_trace(
        self,
        *,
        iteration: int,
        tool: str,
        arguments: dict,
        result_summary: str,
        result_size_chars: int,
        duration_ms: int,
        agent: str = "",
    ) -> None:
        """Registra una tool call. ``agent`` identifica qué agente la emitió
        (``orquestador`` o la ``key`` de un sub-agente especialista)."""
        self.tool_trace.append({
            "iteration": iteration,
            "tool": tool,
            "arguments": arguments,
            "result_summary": result_summary,
            "result_size_chars": result_size_chars,
            "duration_ms": duration_ms,
            "agent": agent,
        })


def _isoformat(value) -> str | None:
    """Convierte date/datetime a ISO; deja strings tal cual."""
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return str(value)


# Tope de puntos por serie en la respuesta: suficiente para un gráfico, evita
# payloads enormes. Si la serie excede, se conservan los más recientes.
MAX_SERIES_POINTS = 500


def _extract_points(rows: list[dict]) -> list[list]:
    """``[[iso_date, value], ...]`` de las filas que traen ``date`` y ``value``
    numérico. Vacío si las filas no tienen valores (serie no graficable)."""
    points: list[list] = []
    for r in rows:
        if "value" not in r or r["value"] is None:
            continue
        try:
            value = float(r["value"])
        except (TypeError, ValueError):
            continue
        date = _isoformat(r.get("date"))
        if date is not None:
            points.append([date, value])
    if len(points) > MAX_SERIES_POINTS:
        points = points[-MAX_SERIES_POINTS:]
    return points
