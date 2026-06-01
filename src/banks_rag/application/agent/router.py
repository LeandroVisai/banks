"""Router determinista de especialistas (reemplaza el loop LLM del orquestador).

En vez de que un LLM decida a qué `delegate_to_*` llamar (lo que con Qwen
provocaba nombres inventados, re-delegación y loops), el ruteo es **determinista**
y reusa lo que ya está validado contra los logs reales:

  - `domain_knowledge.financial_aliases.matched_concepts` → especialistas de
    mercado (fx, afp, no_residentes, fondos_mutuos, renta_fija, liquidez);
  - regex de señales de corpus → policy (decisiones/TPM) y/o document (IPoM,
    research, riesgos, gráficos).

Devuelve 1–3 `SubAgentSpec`. `run_agent` los corre en paralelo y luego sintetiza
una sola vez. Función pura (sin I/O ni LLM) → trivial de testear.
"""

from __future__ import annotations

import re
import unicodedata

from banks_rag.domain_knowledge.financial_aliases import matched_concepts

from .subagents import SUBAGENTS, SubAgentSpec

# Tope de especialistas por consulta (cobertura cross-dominio sin disparar la
# latencia: corren en paralelo, pero más de 3 raramente aporta).
MAX_SPECIALISTS = 3

# Señales de POLÍTICA MONETARIA → especialista `policy` (decisiones del Consejo,
# nivel/▼/▲ de TPM, votación, postura). Texto sin acentos (ver _norm).
_POLICY_RE = re.compile(
    r"\b(tpm|politica monetaria|consejo|comunicado|minuta|"
    r"decidi|decision de tasa|votacion|reunion de politica|postura|"
    r"sesgo|tasa de politica)\b"
)

# Señales de CORPUS documental → especialista `document` (informes, research,
# Fed, IPoM/IEF, riesgos/escenarios, contenido cualitativo). IPoM va aquí (su
# contenido —gráficos, proyecciones— vive en el corpus, no es una decisión TPM).
_DOC_RE = re.compile(
    r"\b(ipom|ief|informe|reporte|research|jpmorgan|jp morgan|fed|"
    r"riesgo|escenario|proyeccion|que dice|que dijo|antecedente|"
    r"balance de riesgos|monitor pm)\b"
)

# Señales de GRÁFICO/figura/tabla → document (search_visuals; solo IPoM tiene visuales).
_VISUAL_RE = re.compile(
    r"\b(grafico|figura|imagen|chart|tabla|muestrame|muestra|visualiza|"
    r"diagrama|curva del)\b"
)

# Saludos / preguntas sobre capacidades → SIN especialistas (respuesta directa
# en la síntesis). Evita correr subagentes inútiles ante "hola" o "qué puedes
# hacer". Solo aplica si el mensaje es corto y NO trae señales de dato/corpus.
_GREETING_RE = re.compile(
    r"^\s*[¿¡?!.,\s]*(hola|buenas|buenos dias|buenas tardes|buenas noches|hey|"
    r"saludos|gracias|ok|listo|que puedes hacer|que sabes hacer|que informacion|"
    r"que info|quien eres|ayuda|help|test|prueba)\b"
)


def _norm(text: str) -> str:
    """Minúsculas sin acentos, para que las regex matcheen 'política'/'politica'."""
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(ch) != "Mn"
    )
    return stripped.lower()


def _routing_text(message: str, history: list[dict] | None) -> str:
    """Texto para rutear: la pregunta + el último turno del usuario.

    Incluir el último mensaje del usuario resuelve anáforas baratas
    ("¿de dónde salió eso?", "y en marzo?") rutando por el tema previo.
    """
    last_user = ""
    for turn in reversed(history or []):
        if turn.get("role") == "user" and turn.get("content"):
            last_user = turn["content"]
            break
    return f"{message} {last_user}".strip()


def select_specialists(
    message: str, history: list[dict] | None = None,
) -> list[SubAgentSpec]:
    """Selecciona 1–3 especialistas para la consulta (determinista).

    Orden: especialistas de mercado (por concepto) → policy → document. Si nada
    matchea, default a [document, policy] (el corpus cubre lo general).
    """
    text = _routing_text(message, history)
    norm = _norm(text)

    keys: list[str] = []

    # 1. Mercado: keys de especialista desde los conceptos detectados.
    for concept in matched_concepts(text):
        k = concept.specialist
        if k and k in SUBAGENTS and k not in keys:
            keys.append(k)

    # 2. Corpus / política monetaria.
    if _POLICY_RE.search(norm) and "policy" not in keys:
        keys.append("policy")
    if (_DOC_RE.search(norm) or _VISUAL_RE.search(norm)) and "document" not in keys:
        keys.append("document")

    # 3a. Saludo/capacidades SIN ninguna señal de dato/corpus → sin especialistas
    # (la síntesis responde directo). Solo el mensaje actual, no el history.
    if not keys and _GREETING_RE.search(_norm(message)):
        return []

    # 3b. Default: substantivo pero sin señales → corpus general.
    if not keys:
        keys = ["document", "policy"]

    return [SUBAGENTS[k] for k in keys[:MAX_SPECIALISTS]]
