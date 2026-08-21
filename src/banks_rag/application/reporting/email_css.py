"""CSS de la plantilla → estilos **inline**, y de ahí a algo que Word entienda.

Outlook de escritorio no renderiza con un motor web sino con el de **Word**: ignora
el ``<style>`` del ``<head>`` por completo y solo mira el atributo ``style=`` de cada
elemento. Traducir ese CSS es, entonces, obligatorio.

Acá se hace **genérico**: se parsea el ``<style>`` REAL del informe y se aplican sus
reglas por selector, con especificidad, como haría un navegador. Es la diferencia
con el mapa hardcodeado ``clase → estilo`` del ``reports_to_eml.py`` original, que
había que mantener a mano y que dejaba afuera todo lo que no estuviera en la lista
(``body``, las tablas, ``.block-table``…) — de ahí que el correo cayera a Times New
Roman y las tablas se desarmaran.

Alcance del selector: etiqueta, ``.clase``, ``#id``, compuestos (``.card.card-wide``),
descendencia (``.section-text p``), hijo (``>``) y hermano adyacente (``+``). Se
ignoran pseudo-clases/elementos y las at-rules (``@media``, ``@page``) — nada de eso
existe en un correo.

El HTML se reescribe con un **recorrido de tags por regex**, no con un parser que
serialice de nuevo: así el documento sale idéntico salvo el ``style=`` que tocamos
(un parser normalizaría mayúsculas y rompería, por ejemplo, ``viewBox`` de un SVG).
"""

from __future__ import annotations

import re

# ── Propiedades que el motor de Word no entiende ─────────────────────────────
# Dejarlas puestas es peor que sacarlas: Word ignora unas y malinterpreta otras
# (``max-width`` lo descarta y el elemento se va a ancho completo; ``display:grid``
# apila todo en una columna rarísima). Se descartan al volcar el estilo.
_DROP_PROPS = frozenset({
    "display", "grid", "grid-template-columns", "grid-template-rows", "grid-auto-rows",
    "grid-row", "grid-column", "gap", "row-gap", "column-gap", "align-content",
    "align-self", "justify-content", "align-items", "flex", "flex-direction", "flex-wrap",
    "max-width", "min-width", "max-height", "min-height",
    "overflow", "overflow-x", "overflow-y",
    "position", "top", "left", "right", "bottom", "z-index",
    "box-shadow", "transition", "cursor", "content", "box-sizing", "opacity",
    "pointer-events", "page-break-inside", "break-inside", "white-space",
    "fill-opacity", "stroke", "stroke-width",
})

# Funciones CSS que el motor de Word no sabe evaluar: si el valor las usa, Word
# descarta la declaración ENTERA y el elemento cae a su default (``.page`` con
# ``width:min(1200px, calc(100% - 40px))`` termina a ancho completo). Se descartan
# acá y el ancho lo fija la tabla contenedora del correo.
_UNPARSEABLE = ("min(", "max(", "clamp(", "calc(", "var(", "env(")

# ``display`` se descarta salvo estos valores, que Word sí respeta y que hacen falta
# (``.block-dates`` es un chip inline-block, las imágenes van en bloque).
_KEEP_DISPLAY = frozenset({"block", "inline-block", "inline", "none"})

_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input",
                   "link", "meta", "param", "source", "track", "wbr"})

_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_STYLE_BLOCK = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S | re.I)
_RE_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_RE_VAR = re.compile(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^)]*))?\)")
_RE_TAG = re.compile(r"<(/?)([a-zA-Z][^\s/>]*)((?:[^>\"']|\"[^\"]*\"|'[^']*')*?)(/?)>", re.S)
_RE_ATTR = re.compile(r"""([\w:-]+)\s*=\s*("[^"]*"|'[^']*'|[^\s"'>]+)""")


# ── Parseo del CSS ───────────────────────────────────────────────────────────

def _split_decls(block: str) -> list[tuple[str, str]]:
    out = []
    for decl in block.split(";"):
        prop, sep, value = decl.partition(":")
        if sep and prop.strip() and value.strip():
            out.append((prop.strip().lower(), value.strip()))
    return out


def _resolve_vars(value: str, variables: dict[str, str]) -> str:
    for _ in range(4):  # una var puede apuntar a otra
        if "var(" not in value:
            break
        value = _RE_VAR.sub(lambda m: variables.get(m.group(1), (m.group(2) or "").strip()), value)
    return value


def _parse_compound(part: str) -> tuple[str, frozenset[str], str] | None:
    """``div.card.wide#x`` → ``("div", {"card","wide"}, "x")``. ``None`` si no se
    puede representar (pseudo-clases, atributos, ``*``)."""
    if not part or any(ch in part for ch in ":[]*("):
        return None
    if not re.fullmatch(r"([a-zA-Z][\w-]*)?((?:[.#][\w-]+)*)", part):
        return None
    tag = re.match(r"[a-zA-Z][\w-]*", part)
    classes, ident = set(), ""
    for token in re.findall(r"[.#][\w-]+", part):
        if token[0] == ".":
            classes.add(token[1:])
        else:
            ident = token[1:]
    return (tag.group(0).lower() if tag else ""), frozenset(classes), ident


def _parse_selector(sel: str):
    """Selector → lista ``[(combinador, compuesto), …]`` o ``None`` si no aplica."""
    tokens = re.split(r"\s*([>+~])\s*|\s+", sel.strip())
    tokens = [t for t in tokens if t]
    if not tokens or tokens[0] in ">+~":
        return None
    parts, combinator = [], None
    for token in tokens:
        if token in ">+~":
            if token == "~":       # hermano general: no se usa en las plantillas
                return None
            combinator = token
            continue
        compound = _parse_compound(token)
        if compound is None:
            return None
        parts.append((combinator, compound))
        combinator = None
    return parts


def _specificity(parts) -> tuple[int, int, int]:
    ids = sum(1 for _c, (_t, _cls, i) in parts if i)
    classes = sum(len(cls) for _c, (_t, cls, _i) in parts)
    tags = sum(1 for _c, (t, _cls, _i) in parts if t)
    return ids, classes, tags


def parse_stylesheet(html: str):
    """``<style>`` del documento → ``[(especificidad, orden, partes, declaraciones)]``,
    con las custom properties de ``:root`` ya resueltas."""
    css = "\n".join(_RE_CSS_COMMENT.sub("", block) for block in _RE_STYLE_BLOCK.findall(html))

    variables: dict[str, str] = {}
    for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;}]+)", css):
        variables.setdefault(name, value.strip())

    rules, order = [], 0
    for prelude, body in _iter_blocks(css):
        selectors = prelude.strip()
        if not selectors or selectors.startswith("@"):
            continue          # at-rule (@media, @page, @font-face): en un correo no existe
        decls = [(p, _resolve_vars(v, variables)) for p, v in _split_decls(body)]
        if not decls:
            continue
        for sel in selectors.split(","):
            parts = _parse_selector(sel)
            if not parts:
                continue
            order += 1
            rules.append((_specificity(parts), order, parts, decls))
    return sorted(rules, key=lambda r: (r[0], r[1]))


def _iter_blocks(css: str):
    """Recorre el CSS de primer nivel y devuelve ``(prelude, cuerpo)`` por bloque.
    Las at-rules anidadas (``@media { … }``) se saltan enteras contando llaves."""
    i, n = 0, len(css)
    while i < n:
        open_brace = css.find("{", i)
        if open_brace < 0:
            return
        prelude = css[i:open_brace]
        depth, j = 1, open_brace + 1
        while j < n and depth:
            if css[j] == "{":
                depth += 1
            elif css[j] == "}":
                depth -= 1
            j += 1
        yield prelude, css[open_brace + 1:j - 1]
        i = j


# ── Matching contra la pila de elementos abiertos ────────────────────────────

class _Node:
    __slots__ = ("classes", "ident", "prev", "tag")

    def __init__(self, tag, classes, ident, prev):
        self.tag, self.classes, self.ident, self.prev = tag, classes, ident, prev


def _matches_compound(compound, node) -> bool:
    tag, classes, ident = compound
    if node is None:
        return False
    if tag and tag != node.tag:
        return False
    if ident and ident != node.ident:
        return False
    return classes <= node.classes


def _matches(parts, stack) -> bool:
    """Matching derecha→izquierda, como un navegador."""
    if not _matches_compound(parts[-1][1], stack[-1]):
        return False
    node = len(stack) - 1
    for i in range(len(parts) - 2, -1, -1):
        combinator, compound = parts[i + 1][0], parts[i][1]
        if combinator == ">":
            node -= 1
            if node < 0 or not _matches_compound(compound, stack[node]):
                return False
        elif combinator == "+":
            if not _matches_compound(compound, stack[node].prev):
                return False
        else:
            node -= 1
            while node >= 0 and not _matches_compound(compound, stack[node]):
                node -= 1
            if node < 0:
                return False
    return True


# ── Volcado a inline ─────────────────────────────────────────────────────────

def _wordsafe(decls: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out = []
    for prop, value in decls:
        if prop == "display":
            if value.split()[0] in _KEEP_DISPLAY:
                out.append((prop, value))
            continue
        if prop in _DROP_PROPS or prop.startswith(("grid-", "flex-", "--", "animation", "transform")):
            continue
        if any(fn in value for fn in _UNPARSEABLE):
            continue
        out.append((prop, value))
    return out


def _render_style(decls: list[tuple[str, str]]) -> str:
    last = {prop: i for i, (prop, _v) in enumerate(decls)}
    return ";".join(f"{p}:{v}" for i, (p, v) in enumerate(decls) if last[p] == i)


def _attrs(raw: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2).strip("\"'") for m in _RE_ATTR.finditer(raw)}


def _comment_spans(html: str) -> list[tuple[int, int]]:
    """Rangos de los comentarios HTML: adentro no se toca nada (pueden traer los
    bloques condicionales ``<!--[if mso]>`` que Outlook sí interpreta)."""
    return [(m.start(), m.end()) for m in _RE_COMMENT.finditer(html)]


def inline_css(html: str, *, extra: dict[str, str] | None = None, drop_style_tag: bool = True) -> str:
    """Vuelca el ``<style>`` del documento al atributo ``style=`` de cada elemento.

    ``extra`` agrega declaraciones por etiqueta (``{"td": "font-family:Arial"}``) con
    la prioridad más baja: sirve para reimponer la tipografía dentro de las tablas,
    donde Word no hereda la del ``body``.

    ``drop_style_tag`` saca el ``<style>`` del resultado (por defecto): si se queda,
    los clientes que SÍ leen CSS verían un layout distinto al de Outlook, y la copia
    "plana" dejaría de mostrar lo que realmente llega al correo.
    """
    rules = parse_stylesheet(html)
    extra_rules = {k.lower(): _split_decls(v) for k, v in (extra or {}).items()}
    comments = _comment_spans(html)
    skip = 0  # índice del próximo comentario que puede contener la posición actual

    stack: list[_Node] = []
    siblings: list[_Node | None] = [None]
    out, cursor = [], 0

    for m in _RE_TAG.finditer(html):
        while skip < len(comments) and comments[skip][1] <= m.start():
            skip += 1
        if skip < len(comments) and comments[skip][0] <= m.start():
            continue  # tag dentro de un comentario

        closing, tag, raw, self_closing = m.group(1), m.group(2).lower(), m.group(3), m.group(4)

        if closing:
            # Cierra hasta el tag homónimo (tolera markup con tags sin cerrar).
            for depth, node in enumerate(reversed(stack)):
                if node.tag == tag:
                    del stack[len(stack) - depth - 1:]
                    del siblings[len(siblings) - depth - 1:]
                    break
            continue

        attrs = _attrs(raw)
        if "data-email-final" in raw:   # atributo sin valor: no lo ve _attrs
            # Elemento ya resuelto para el correo (una imagen que reemplazó a un SVG
            # o a una tabla): su style se escribió sabiendo que iba a un correo, así
            # que la normalización de acá lo estropearía (le sacaría el max-width).
            if tag not in _VOID and not self_closing:
                stack.append(_Node(tag, frozenset(), "", siblings[-1]))
                siblings.append(None)
            continue
        node = _Node(tag, frozenset(attrs.get("class", "").split()),
                     attrs.get("id", ""), siblings[-1])

        decls: list[tuple[str, str]] = list(extra_rules.get(tag, ()))
        stack.append(node)
        for _spec, _order, parts, rule_decls in rules:
            if _matches(parts, stack):
                decls.extend(rule_decls)
        if attrs.get("style"):
            decls.extend(_split_decls(attrs["style"]))   # el inline propio manda

        style = _render_style(_wordsafe(decls))
        if style:
            without = _RE_ATTR.sub(lambda a: "" if a.group(1).lower() == "style" else a.group(0), raw)
            out.append(html[cursor:m.start()])
            out.append(f'<{m.group(2)}{without.rstrip()} style="{style}"{"/" if self_closing else ""}>')
            cursor = m.end()

        siblings[-1] = node
        if self_closing or tag in _VOID:
            stack.pop()
        else:
            siblings.append(None)

    out.append(html[cursor:])
    result = "".join(out)
    return _RE_STYLE_BLOCK.sub("", result) if drop_style_tag else result
