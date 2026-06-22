"""Verificador de afirmaciones del informe — POST-síntesis, híbrido.

Corre DESPUÉS de redactar los párrafos y la síntesis. Su trabajo es que el
informe NO entregue afirmaciones incorrectas ni contradictorias (el caso que
detectó la analista: el mismo "Tipo 1" descrito como entrada en un bloque y como
salida en otro). Dos capas:

1. **Determinista (la garantía dura).** La fuente de verdad son los ``facts`` ya
   calculados en Python por ``parquet_facts.compute_facts`` (no el LLM). Por cada
   párrafo se construye un *ledger* desde sus facts y se contrasta la prosa:
   - DIRECCIÓN: una palabra direccional (entrada/aporte vs salida/rescate) pegada
     a una categoría debe coincidir con el SIGNO del flujo del período de esa
     categoría en los facts. Si la contradice → se AUTOCORRIGE la palabra (es un
     swap binario y grounded, bajo riesgo).
   - CIFRAS: todo número de la prosa debería trazar a un hecho del dataset; los
     que no, se MARCAN (no se reescriben: elegir "el" número correcto es ambiguo).
   - CONTRADICCIÓN intra-texto: la misma categoría afirmada en ambas direcciones.

2. **Crítico LLM (red de seguridad semántica).** Si se pasa un ``llm``, una
   llamada SIN tools con ``REPORT_VERIFIER_PROMPT`` recibe los facts + el borrador
   y devuelve correcciones/contradicciones en JSON. Solo se aplican las que son
   sustituciones de texto seguras; el resto se marca.

Acción global (decisión del usuario): **corregir + reintentar, marcar lo
residual**. Las direcciones se autocorrigen; los párrafos con cifras sin sustento
se REGENERAN una vez (si se pasó un callback ``regenerate``); lo que persiste
queda marcado para que el analista lo vea. Best-effort: el verificador NUNCA
tumba la generación (si algo revienta, loguea y deja el borrador).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Marca reservada para la "categoría" del agregado/síntesis (no es un dataset).
SYNTHESIS_KEY = "__sintesis__"

# ─────────────────────────────────────────────────────────────────────────────
# Léxico de dirección (flujos)
# ─────────────────────────────────────────────────────────────────────────────
_DIR_POS = frozenset({
    "entrada", "entradas", "aporte", "aportes", "ingreso", "ingresos",
    "suscripcion", "suscripciones", "captacion", "captaciones",
    "inflow", "inflows", "aportaron", "ingresaron", "entro", "entraron",
})
_DIR_NEG = frozenset({
    "salida", "salidas", "rescate", "rescates", "retiro", "retiros",
    "desinversion", "desinversiones", "outflow", "outflows", "fuga", "fugas",
    "rescataron", "salieron", "retiraron",
})
_DIR_WORDS = _DIR_POS | _DIR_NEG


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _dir_sign(word: str) -> int:
    w = _strip_accents(word.lower())
    if w in _DIR_POS:
        return 1
    if w in _DIR_NEG:
        return -1
    return 0


def _canonical_dir(sign: int, *, plural: bool) -> str:
    base = "entrada" if sign > 0 else "salida"
    return base + ("s" if plural else "")


# ─────────────────────────────────────────────────────────────────────────────
# Ledger: la verdad numérica/direccional desde los facts
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Ledger:
    """Hechos autoritativos de UN dataset (o la unión, para la síntesis)."""

    is_flow: bool = False
    numbers: set[float] = field(default_factory=set)
    # categoría → conjunto de signos del flujo del período observados (por ventana).
    # Un signo de la prosa que NO esté en este conjunto contradice los datos.
    direction_by_cat: dict[str, set[int]] = field(default_factory=dict)
    categories: set[str] = field(default_factory=set)

    def merge(self, other: Ledger) -> None:
        self.is_flow = self.is_flow or other.is_flow
        self.numbers |= other.numbers
        self.categories |= other.categories
        for cat, signs in other.direction_by_cat.items():
            self.direction_by_cat.setdefault(cat, set()).update(signs)


def _add_num(acc: set[float], value: Any) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        acc.add(round(float(value), 2))


def _walk_variation(var: dict | None, acc: set[float]) -> int | None:
    """Agrega los números de un dict de variación/flujo y devuelve su signo
    direccional (signo del flujo del período si es flujo; ``None`` si no aplica)."""
    if not var:
        return None
    for k in ("valor_inicio", "valor_fin", "cambio_absoluto", "variacion_pct",
              "minimo", "maximo", "flujo_periodo"):
        _add_num(acc, var.get(k))
    if var.get("is_flow_sum"):
        flow = var.get("flujo_periodo")
        if isinstance(flow, (int, float)):
            return (flow > 0) - (flow < 0)
    return None


def build_ledger(facts: dict | None) -> Ledger:
    """Construye el ledger autoritativo desde el dict de ``compute_facts``."""
    led = Ledger()
    if not facts:
        return led
    led.is_flow = bool(facts.get("is_flow"))

    def reg_dir(cat: str, sign: int | None) -> None:
        if sign is None or not cat:
            return
        led.categories.add(cat)
        led.direction_by_cat.setdefault(cat, set()).add(sign)

    # Total agregado / ventana simple.
    for v in facts.get("total_ventanas", []) or []:
        reg_dir("total", _walk_variation(v.get("variacion"), led.numbers))
    for v in facts.get("windows_variation", []) or []:
        _walk_variation(v.get("variacion"), led.numbers)

    # Drivers (contribuciones) por ventana.
    for c in facts.get("contribuciones", []) or []:
        for d in c.get("drivers", []) or []:
            cat = str(d.get("categoria") or "")
            _add_num(led.numbers, d.get("cambio_absoluto"))
            _add_num(led.numbers, d.get("variacion_pct"))
            val = d.get("nivel_fin") if led.is_flow else None
            if isinstance(val, (int, float)):
                reg_dir(cat, (val > 0) - (val < 0))
            elif cat:
                led.categories.add(cat)

    # Por categoría / por columna.
    for item in (facts.get("por_categoria", []) or []) + (facts.get("por_columna", []) or []):
        cat = str(item.get("categoria") or item.get("columna") or "")
        if cat:
            led.categories.add(cat)
        _add_num(led.numbers, item.get("ultimo_valor"))
        for v in item.get("ventanas", []) or []:
            reg_dir(cat, _walk_variation(v.get("variacion"), led.numbers))

    # Composición de corte (solo stocks).
    comp = facts.get("composicion_corte") or facts.get("composition")
    if comp:
        _add_num(led.numbers, comp.get("total"))
        for it in comp.get("breakdown", []) or []:
            if it.get("categoria"):
                led.categories.add(str(it["categoria"]))
            _add_num(led.numbers, it.get("valor"))
            _add_num(led.numbers, it.get("share_pct"))

    st = facts.get("estadisticas")
    if st:
        for k in ("ultimo_valor", "minimo", "maximo", "media", "percentil_ultimo_valor"):
            _add_num(led.numbers, st.get(k))
    return led


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de la prosa
# ─────────────────────────────────────────────────────────────────────────────

# Token numérico: dígitos con separadores . , y decimales, opcional signo.
_NUMBER_RE = re.compile(r"[-+]?\d[\d.,]*\d|\d")


def _parse_number(tok: str) -> float | None:
    """Parsea un número tolerando formato ES (1.943,09) y US (1,943.09)."""
    s = tok.strip()
    if not re.search(r"\d", s):
        return None
    neg = s.startswith("-")
    s = s.lstrip("+-")
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        # El último separador es el decimal.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        frac = s.rsplit(",", 1)[1]
        s = s.replace(",", ".") if (s.count(",") == 1 and len(frac) <= 2) else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        val = float(s)
    except ValueError:
        return None
    return -val if neg else val


def extract_numbers(text: str) -> list[tuple[str, float, int]]:
    """``[(token, valor, pos)]`` de los números de la prosa (pos = offset)."""
    out: list[tuple[str, float, int]] = []
    for m in _NUMBER_RE.finditer(text):
        val = _parse_number(m.group(0))
        if val is not None:
            out.append((m.group(0), val, m.start()))
    return out


def _category_spans(text: str, categories: Iterable[str]) -> list[tuple[str, int, int]]:
    """``[(categoria, inicio, fin)]`` de las menciones de categorías conocidas +
    el patrón genérico ``Tipo N``. Match sin distinguir mayúsculas/acentos."""
    spans: list[tuple[str, int, int]] = []
    norm = _strip_accents(text.lower())
    for cat in sorted(set(categories), key=len, reverse=True):
        if not cat:
            continue
        ncat = _strip_accents(cat.lower())
        start = 0
        while True:
            i = norm.find(ncat, start)
            if i < 0:
                break
            spans.append((cat, i, i + len(ncat)))
            start = i + len(ncat)
    for m in re.finditer(r"tipo\s+\d+", norm):
        spans.append((text[m.start():m.end()], m.start(), m.end()))
    return spans


def _direction_spans(text: str) -> list[tuple[str, int, int, int]]:
    """``[(palabra, signo, inicio, fin)]`` de las palabras direccionales."""
    out: list[tuple[str, int, int, int]] = []
    for m in re.finditer(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text):
        sign = _dir_sign(m.group(0))
        if sign:
            out.append((m.group(0), sign, m.start(), m.end()))
    return out


# Distancia máxima (caracteres) para atribuir una palabra direccional a una categoría.
_ATTACH_WINDOW = 70


# ─────────────────────────────────────────────────────────────────────────────
# Issues + reporte
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Issue:
    kind: str             # "direction" | "ungrounded_number" | "contradiction" | "llm"
    where: str            # dataset_id o SYNTHESIS_KEY
    detail: str
    span: str = ""
    suggestion: str = ""
    auto_corrected: bool = False
    resolved: bool = False


@dataclass
class VerificationReport:
    issues: list[Issue] = field(default_factory=list)
    n_direction_fixed: int = 0
    n_regenerated: int = 0
    llm_used: bool = False

    @property
    def residual(self) -> list[Issue]:
        return [i for i in self.issues if not i.resolved]

    def residual_for(self, where: str) -> list[Issue]:
        return [i for i in self.issues if i.where == where and not i.resolved]

    def summary(self) -> str:
        return (
            f"direcciones corregidas={self.n_direction_fixed} "
            f"párrafos regenerados={self.n_regenerated} "
            f"marcas residuales={len(self.residual)} "
            f"(de {len(self.issues)} hallazgos; crítico LLM={'sí' if self.llm_used else 'no'})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Capa determinista
# ─────────────────────────────────────────────────────────────────────────────


def correct_directions(text: str, ledger: Ledger, *, where: str) -> tuple[str, list[Issue]]:
    """Autocorrige palabras direccionales que contradicen el signo del flujo del
    período en los facts. Devuelve ``(texto_corregido, issues)``.

    Solo actúa en series de FLUJOS y cuando la categoría tiene un signo ÚNICO en el
    ledger (si la categoría tiene entrada en una ventana y salida en otra, ambas
    direcciones son válidas y no se toca)."""
    if not ledger.is_flow or not ledger.direction_by_cat:
        return text, []
    cat_spans = _category_spans(text, ledger.categories)
    dir_spans = _direction_spans(text)
    issues: list[Issue] = []
    # Reemplazos (inicio, fin, nuevo) aplicados de atrás hacia adelante.
    edits: list[tuple[int, int, str]] = []
    for word, sign, dstart, dend in dir_spans:
        # Categoría más cercana dentro de la ventana de atribución.
        best, bestdist = None, _ATTACH_WINDOW + 1
        for cat, cstart, cend in cat_spans:
            dist = dstart - cend if dstart >= cend else cstart - dend
            if 0 <= dist < bestdist:
                best, bestdist = cat, dist
        if best is None:
            continue
        signs = ledger.direction_by_cat.get(best) or _signs_for(best, ledger)
        if not signs or len(signs) != 1:
            continue  # desconocido o ambiguo (week+/month−): no tocar
        truth = next(iter(signs))
        if sign != truth:
            fixed = _canonical_dir(truth, plural=word.lower().endswith("s"))
            edits.append((dstart, dend, fixed))
            issues.append(Issue(
                kind="direction", where=where, auto_corrected=True, resolved=True,
                detail=f"'{word}' sobre {best} contradice el flujo del período "
                       f"({'entrada' if truth > 0 else 'salida'}): corregido a '{fixed}'",
                span=word, suggestion=fixed,
            ))
    for dstart, dend, fixed in sorted(edits, reverse=True):
        text = text[:dstart] + fixed + text[dend:]
    return text, issues


def _signs_for(cat: str, ledger: Ledger) -> set[int]:
    """Resuelve el signo de una mención ``Tipo N`` contra las claves del ledger
    (que pueden venir como 'Tipo 1'); match laxo por contención normalizada."""
    ncat = _strip_accents(cat.lower())
    for key, signs in ledger.direction_by_cat.items():
        nk = _strip_accents(key.lower())
        if nk == ncat or nk in ncat or ncat in nk:
            return signs
    return set()


def _is_grounded(value: float, numbers: set[float], *, rel_tol: float = 0.01) -> bool:
    """``True`` si ``value`` coincide (tolerancia relativa) con algún número de los
    facts. Compara por MAGNITUD (la prosa suele escribir el valor absoluto y llevar
    el signo en la palabra: "reducción de 147,79" = -147,79); la corrección de signo
    la maneja el chequeo de dirección, no el de grounding."""
    if not numbers:
        return False
    av = abs(value)
    for n in numbers:
        an = abs(n)
        if abs(an - av) <= max(an * rel_tol, 0.05):
            return True
    # La prosa redondea: igualdad de magnitud entera también vale.
    return any(round(abs(n)) == round(av) for n in numbers)


# Números que no se marcan aunque no estén en los facts (años, días, ordinales).
def _ignorable_number(tok: str, value: float) -> bool:
    if re.fullmatch(r"20\d{2}", tok):        # años (2024, 2026…)
        return True
    # Enteros pequeños: 'Tipo 6', 'los 3 tipos', '30 de abril', días/meses/ordinales.
    # Una cifra financiera de esa magnitud llevaría decimales o unidad explícita.
    if value == int(value) and abs(value) <= 31:
        return True
    return False


def flag_ungrounded_numbers(text: str, ledger: Ledger, *, where: str) -> list[Issue]:
    """Marca (no reescribe) cifras de la prosa que no trazan a ningún hecho."""
    issues: list[Issue] = []
    for tok, val, pos in extract_numbers(text):
        if _ignorable_number(tok, val) or _is_grounded(val, ledger.numbers):
            continue
        ctx = text[max(0, pos - 25): pos + len(tok) + 25].strip()
        issues.append(Issue(
            kind="ungrounded_number", where=where,
            detail=f"cifra '{tok}' sin respaldo en los datos del dataset",
            span=ctx,
        ))
    return issues


def detect_intra_contradiction(text: str, ledger: Ledger, *, where: str) -> list[Issue]:
    """Misma categoría afirmada en AMBAS direcciones dentro del mismo texto, sin
    que el ledger lo justifique (no es week+ / month−)."""
    cat_spans = _category_spans(text, ledger.categories)
    dir_spans = _direction_spans(text)
    by_cat: dict[str, set[int]] = {}
    for _word, sign, dstart, dend in dir_spans:
        best, bestdist = None, _ATTACH_WINDOW + 1
        for cat, cstart, cend in cat_spans:
            dist = dstart - cend if dstart >= cend else cstart - dend
            if 0 <= dist < bestdist:
                best, bestdist = cat, dist
        if best is not None:
            by_cat.setdefault(best, set()).add(sign)
    issues: list[Issue] = []
    for cat, signs in by_cat.items():
        if len(signs) > 1:
            led_signs = ledger.direction_by_cat.get(cat) or _signs_for(cat, ledger)
            if len(led_signs) <= 1:  # los datos NO justifican ambas direcciones
                issues.append(Issue(
                    kind="contradiction", where=where,
                    detail=f"{cat} aparece como entrada Y salida en el mismo texto "
                           f"sin sustento en los datos",
                    span=cat,
                ))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Crítico LLM (opcional)
# ─────────────────────────────────────────────────────────────────────────────


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_critic_json(raw: str) -> dict | None:
    t = _strip_code_fences(raw or "")
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except (json.JSONDecodeError, ValueError):
        return None


async def llm_review(
    draft: str, facts_digest: str, *, llm, max_tokens: int = 1536,
) -> tuple[str, list[Issue]]:
    """Pasada crítica: el LLM recibe los facts + el borrador y devuelve, en JSON,
    sustituciones seguras (``replace``) y contradicciones (``flag``). Aplica solo
    las sustituciones literales presentes en el texto; el resto se marca.
    Best-effort: cualquier error → ``(draft, [])``."""
    from banks_rag.application.agent.prompts import NO_THINK_DIRECTIVE, REPORT_VERIFIER_PROMPT

    user = (
        f"{NO_THINK_DIRECTIVE}\n\nHECHOS CALCULADOS (la única verdad numérica):\n"
        f"{facts_digest}\n\nBORRADOR DEL INFORME:\n{draft}\n\n"
        "Devuelve SOLO el JSON pedido."
    )
    try:
        result = await llm.generate(
            [
                {"role": "system", "content": REPORT_VERIFIER_PROMPT},
                {"role": "user", "content": user},
            ],
            tools=None, temperature=0.2, top_p=0.8, max_tokens=max_tokens,
        )
    except Exception:
        log.exception("crítico LLM falló; se conserva el borrador")
        return draft, []

    data = _parse_critic_json(getattr(result, "text", "") or "")
    if not data:
        return draft, []

    issues: list[Issue] = []
    out = draft
    for corr in data.get("corrections", []) or []:
        if not isinstance(corr, dict):
            continue
        find, repl = str(corr.get("find", "")), str(corr.get("replace", ""))
        if not (find and repl and find in out):
            continue
        # SEGURIDAD: el crítico (sobre todo en modelos chicos) tiende a "corregir"
        # una cifra correcta por otra del contexto equivocado (cifra válida pero mal
        # ubicada). NO aplicamos cambios que alteren NÚMEROS: solo dejamos pasar
        # ediciones que preservan las cifras (swaps de dirección / redacción). Un
        # cambio numérico se reporta como marca, no se aplica.
        nums_find = {round(v, 2) for _t, v, _p in extract_numbers(find)}
        nums_repl = {round(v, 2) for _t, v, _p in extract_numbers(repl)}
        if nums_find != nums_repl:
            issues.append(Issue(
                kind="llm", where=SYNTHESIS_KEY,
                detail=f"crítico LLM sugirió un cambio de cifras (NO aplicado por seguridad): "
                       f"'{find}' → '{repl}' ({corr.get('reason', '')})".strip(),
                span=find, suggestion=repl,
            ))
            continue
        out = out.replace(find, repl, 1)
        issues.append(Issue(
            kind="llm", where=SYNTHESIS_KEY, auto_corrected=True, resolved=True,
            detail=f"crítico LLM: '{find}' → '{repl}' ({corr.get('reason', '')})".strip(),
            span=find, suggestion=repl,
        ))
    for flag in data.get("contradictions", []) or []:
        detail = flag if isinstance(flag, str) else str((flag or {}).get("detail", flag))
        if detail:
            issues.append(Issue(kind="llm", where=SYNTHESIS_KEY,
                                detail=f"crítico LLM: {detail}"))
    return out, issues


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador
# ─────────────────────────────────────────────────────────────────────────────

# Callback opcional para regenerar el párrafo de una sección con una nota de
# corrección. Lo provee parquet_report (reusa el prompt del redactor) para no
# acoplar este módulo a la mecánica del LLM/prompt.
RegenerateFn = Callable[[Any, list[Issue]], Awaitable[str]]


async def verify_report(
    report: Any,
    *,
    llm: Any | None = None,
    regenerate: RegenerateFn | None = None,
    facts_digest: str = "",
    redundant_ids: set[str] | None = None,
) -> VerificationReport:
    """Verifica y corrige ``report`` IN-PLACE; devuelve el ``VerificationReport``.

    ``report`` debe exponer ``sections`` (con ``.dataset_id``, ``.paragraph``,
    ``.status`` y ``.facts``) y ``overview_md``. Determinista siempre; con ``llm``
    además corre el crítico sobre la síntesis; con ``regenerate`` reintenta los
    párrafos que queden con cifras sin sustento. ``redundant_ids`` son datasets
    cuyo párrafo NO alimentó la síntesis (vistas no_text): se verifican por sí
    mismos pero se excluyen del ledger-unión de la síntesis, para que el ledger
    refleje exactamente lo que la síntesis contiene."""
    redundant_ids = redundant_ids or set()
    vr = VerificationReport()
    try:
        union = Ledger()
        ok_sections = [s for s in getattr(report, "sections", []) if getattr(s, "status", "") == "ok"]

        # ── Por párrafo: dirección (autocorrige) + cifras/contradicción (marca) ──
        for s in ok_sections:
            led = build_ledger(getattr(s, "facts", None))
            if s.dataset_id not in redundant_ids:
                union.merge(led)
            text = s.paragraph or ""
            text, dir_issues = correct_directions(text, led, where=s.dataset_id)
            s.paragraph = text
            vr.issues.extend(dir_issues)
            vr.n_direction_fixed += sum(1 for i in dir_issues if i.auto_corrected)
            vr.issues.extend(detect_intra_contradiction(text, led, where=s.dataset_id))
            vr.issues.extend(flag_ungrounded_numbers(text, led, where=s.dataset_id))

        # ── Regeneración acotada de párrafos aún marcados (si hay callback) ──
        if regenerate is not None:
            for s in ok_sections:
                residual = [i for i in vr.residual_for(s.dataset_id)
                            if i.kind in ("ungrounded_number", "contradiction")]
                if not residual:
                    continue
                try:
                    new_text = await regenerate(s, residual)
                except Exception:
                    log.exception("[%s] regeneración del párrafo falló", s.dataset_id)
                    continue
                if not new_text or not new_text.strip():
                    continue
                led = build_ledger(getattr(s, "facts", None))
                new_text, dir2 = correct_directions(new_text, led, where=s.dataset_id)
                re_residual = (detect_intra_contradiction(new_text, led, where=s.dataset_id)
                               + flag_ungrounded_numbers(new_text, led, where=s.dataset_id))
                if len(re_residual) < len(residual):
                    s.paragraph = new_text
                    vr.n_regenerated += 1
                    for i in residual:
                        i.resolved = True
                    vr.issues.extend(dir2)
                    vr.n_direction_fixed += sum(1 for i in dir2 if i.auto_corrected)
                    vr.issues.extend(re_residual)

        # ── Síntesis: dirección (autocorrige) + cifras (marca) contra la unión ──
        overview = getattr(report, "overview_md", "") or ""
        overview, syn_dir = correct_directions(overview, union, where=SYNTHESIS_KEY)
        vr.issues.extend(syn_dir)
        vr.n_direction_fixed += sum(1 for i in syn_dir if i.auto_corrected)
        vr.issues.extend(flag_ungrounded_numbers(overview, union, where=SYNTHESIS_KEY))

        # ── Crítico LLM sobre la síntesis (semántico, opcional) ──
        if llm is not None:
            overview, llm_issues = await llm_review(
                overview, facts_digest or _digest(union), llm=llm,
            )
            vr.issues.extend(llm_issues)
            vr.llm_used = True
        report.overview_md = overview
    except Exception:
        log.exception("el verificador falló; se conserva el informe sin verificar")
    log.info("Verificación del informe: %s", vr.summary())
    return vr


def _digest(ledger: Ledger) -> str:
    """Resumen compacto del ledger-unión para el crítico LLM cuando el caller no
    pasa un ``facts_digest`` más rico."""
    nums = ", ".join(f"{n:g}" for n in sorted(ledger.numbers)[:40])
    dirs = "; ".join(
        f"{cat}: {'entrada' if s == {1} else 'salida' if s == {-1} else 'mixto'}"
        for cat, s in list(ledger.direction_by_cat.items())[:20]
    )
    return f"Números válidos: {nums}\nDirección por categoría (flujos): {dirs}"
