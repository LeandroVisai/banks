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
        """Registra una serie consultada. ``rows`` debe tener key ``date`` por fila."""
        self.series_used[series_id] = {
            "series_id": series_id,
            "series_name": meta.get("series_name") or meta.get("name", ""),
            "unit": meta.get("unit", ""),
            "frequency": meta.get("frequency", ""),
            "n_observations": len(rows),
            "first_date": _isoformat(rows[0]["date"]) if rows else None,
            "last_date": _isoformat(rows[-1]["date"]) if rows else None,
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
    ) -> None:
        self.tool_trace.append({
            "iteration": iteration,
            "tool": tool,
            "arguments": arguments,
            "result_summary": result_summary,
            "result_size_chars": result_size_chars,
            "duration_ms": duration_ms,
        })


def _isoformat(value) -> str | None:
    """Convierte date/datetime a ISO; deja strings tal cual."""
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return str(value)
