"""Estado mutable del agente durante una sesión de chat.

Acumula chunks vistos (con refs globales [1], [2], ...), series consultadas
y la traza completa de tool calls. Las tools mutan este objeto para que el
agente pueda citar en su respuesta final.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chart_types import chart_family


@dataclass
class AgentState:
    """Estado del agente durante un único turno (request → response)."""

    chunks_seen: list[dict] = field(default_factory=list)        # con refs [1], [2], ...
    chunk_id_to_ref: dict[str, int] = field(default_factory=dict)
    series_used: dict[str, dict] = field(default_factory=dict)
    # Gráficos generados por plot_series (specs Vega-Lite) para que el frontend
    # los renderice. Espacio de ids propio (gráfico 1, 2, ...), separado de las
    # citas [N] de chunks para no romper la verificación de citas.
    charts: list[dict] = field(default_factory=list)
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

    def add_chart(self, chart: dict) -> int:
        """Registra un gráfico (spec Vega-Lite) generado por ``plot_series``.

        Espacio de ids propio (1, 2, ...), separado de las citas ``[N]`` de
        chunks para no romper la verificación de citas. El ``chart_id`` se
        inyecta en el dict almacenado para que el frontend pueda referenciarlo.

        Returns:
            El ``chart_id`` (1-based) para referenciar el gráfico en la respuesta.
        """
        chart_id = len(self.charts) + 1
        self.charts.append({"chart_id": chart_id, **chart})
        return chart_id

    def add_series(
        self,
        series_id: str,
        meta: dict,
        rows: list[dict],
        *,
        chart_type: str | None = None,
        chart_hint: str | None = None,
    ) -> None:
        """Registra una serie consultada para que el frontend la grafique.

        Cada fila puede ser:
          - **temporal**: ``{"date": ..., "value": ...}`` → eje X de fechas;
          - **categórica**: ``{"category": "BTP", "value": ...}`` → eje X de
            etiquetas (composiciones, cortes transversales).

        Solo las filas con ``value`` numérico producen ``points``
        ``[[x, value], ...]`` (``x`` = ISO date o etiqueta). Se capan a
        ``MAX_SERIES_POINTS`` para no inflar el payload.

        ``chart_type`` (``"line"``/``"area"``/``"bar"``/``"grouped_bar"``/
        ``"stacked_bar"``) lo fija el llamador o, si es ``None``, lo infiere
        :func:`infer_chart_type` según la forma del dato (temporal → ``line``;
        categórico → ``bar``) y el ``chart_hint`` — el ``chart_type`` canónico
        que el catálogo de parquets declara para el dataset de origen
        (``stacked_area``, ``grouped_bar``, ...). El frontend respeta este
        tipo en vez de graficar todo como línea."""
        points, x_is_date = _extract_points(rows)
        if chart_type is None:
            chart_type = infer_chart_type(points, x_is_date=x_is_date, hint=chart_hint)
        first_date = _isoformat(rows[0].get("date")) if (x_is_date and rows) else None
        last_date = _isoformat(rows[-1].get("date")) if (x_is_date and rows) else None
        self.series_used[series_id] = {
            "series_id": series_id,
            "series_name": meta.get("series_name") or meta.get("name", ""),
            "unit": meta.get("unit", ""),
            "frequency": meta.get("frequency", ""),
            "n_observations": len(rows),
            "first_date": first_date,
            "last_date": last_date,
            "chart_type": chart_type,
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

# Mínimo de puntos temporales para que una línea tenga sentido. Por debajo de
# esto (1-2 cifras), una serie temporal se grafica mejor como barra.
_MIN_LINE_POINTS = 3

# Máximo de puntos temporales que rinden como barras. Una familia de barras
# (flujos semanales, variaciones) con más observaciones que esto se degrada a
# línea: cientos de barras son ilegibles en el panel compacto del chat.
_MAX_BAR_POINTS = 31


def infer_chart_type(
    points: list[list], *, x_is_date: bool, hint: str | None = None,
) -> str:
    """Clasificador determinista del tipo de gráfico según la forma del dato.

    No todo es una línea: el eje X categórico (composiciones, cortes
    transversales) se grafica como barra; una serie temporal con suficientes
    puntos como línea. Pocos puntos discretos sobre un eje temporal también
    rinden mejor como barra que como una "línea" de dos vértices.

    ``hint`` es el ``chart_type`` canónico que el catálogo de parquets declara
    para el dataset de origen (``stacked_area``, ``grouped_bar``, ...). Cuando
    la forma del dato lo permite, la familia del hint manda: así el gráfico es
    consistente con el que el tablero construye para ese dataset.

    Devuelve ``"line"`` | ``"area"`` | ``"bar"``. La distinción fina (apilado,
    agrupado) la resuelve el frontend con el chart_type explícito."""
    if not x_is_date:
        return "bar"
    if len(points) < _MIN_LINE_POINTS:
        return "bar"
    if hint:
        family = chart_family(hint)
        if family == "area":
            return "area"
        if family in ("bar", "grouped_bar", "stacked_bar"):
            # Barras solo si son pocas observaciones; si no, línea legible.
            return "bar" if len(points) <= _MAX_BAR_POINTS else "line"
    return "line"


def _extract_points(rows: list[dict]) -> tuple[list[list], bool]:
    """``([[x, value], ...], x_is_date)`` de las filas con ``value`` numérico.

    ``x`` es la fecha ISO (filas con ``date``) o la etiqueta de categoría (filas
    con ``category``, p. ej. una composición). Si alguna fila trae ``category``
    la serie se considera categórica (``x_is_date=False``). Vacío si ninguna fila
    tiene valor (serie no graficable)."""
    categorical = any(
        isinstance(r, dict) and r.get("category") is not None for r in rows
    )
    points: list[list] = []
    for r in rows:
        if "value" not in r or r["value"] is None:
            continue
        try:
            value = float(r["value"])
        except (TypeError, ValueError):
            continue
        if categorical:
            cat = r.get("category")
            x = str(cat) if cat is not None else None
        else:
            x = _isoformat(r.get("date"))
        if x is not None:
            points.append([x, value])
    if len(points) > MAX_SERIES_POINTS:
        points = points[-MAX_SERIES_POINTS:]
    return points, (not categorical)
