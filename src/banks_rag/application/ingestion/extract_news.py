"""Extracción del corpus de **contexto actual** (noticias scrapeadas).

Paralelo a ``extract_corpus`` (PDFs/Excel), pero para los JSON diarios de
noticias que alimentan la base aislada ``contexto_actual`` (ver
``docs`` y el CLI ``banks-ingest-news``). NO comparte storage con el corpus del
banco: vive en otra base de datos para que el agente no pueda cruzarlos por
error.

Entrada: un directorio con un JSON por día, nombrado con la fecha, p. ej.
``noticias_2026_01_20.json``. Cada JSON es una **lista de artículos** con campos
como ``title``, ``emailTitle``, ``source``, ``topic``, ``author``, ``text``,
``audience``, ``vpe``, ``url``/``pressLink``.

Dos cuidados con los datos reales del scraper:
  - **Fecha**: el campo ``date`` interno (``DD/MM/YYYY``) suele venir ``null``
    pero a veces trae el día real (un informe diario incluye notas de días
    previos). Se usa esa fecha si está; si no, la del nombre del archivo. Clave
    para el filtro por fecha / recency que aplica el agente en query-time.
  - **Mojibake**: el texto viene doble-codificado (UTF-8 leído como cp1252:
    ``inflaciÃ³n`` → ``inflación``). Se repara en la extracción (``fix_mojibake``)
    o los embeddings y lo que lee el agente quedarían corruptos.

Modelo de datos (reusa el dominio):
  - 1 artículo → 1 ``Document`` (``doc_type_category="NOTICIA"``).
  - el cuerpo del artículo → N ``Chunk`` (split por presupuesto de caracteres),
    todos con ``chunk_date`` = fecha del artículo.

Funciones puras (sin I/O salvo ``extract_news_file``/``extract_news``): fáciles
de testear con datos sintéticos.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from banks_rag.domain.documents import Chunk, ChunkKind, Document
from banks_rag.domain_knowledge.text_repair import fix_mojibake

log = logging.getLogger(__name__)

DOC_TYPE_NEWS = "NOTICIA"

# Presupuesto de caracteres por chunk del cuerpo (≈300-400 tokens). Un artículo
# corto cabe en un solo chunk; uno largo se parte en ventanas con solapamiento
# para no cortar ideas en seco entre chunks contiguos.
TARGET_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
# Cuerpos más cortos que esto no se trocean (no vale la pena).
MIN_SPLIT_CHARS = 1600

# Fecha en el nombre del archivo: noticias_2026_01_20 / noticias-2026-01-20 /
# 2026_01_20 ... Acepta separadores `_`, `-` o nada entre los grupos.
_DATE_IN_NAME_RE = re.compile(r"(20\d{2})[_\-.]?(\d{2})[_\-.]?(\d{2})")

# Fecha interna del artículo (campo ``date``), formato chileno DD/MM/YYYY. Suele
# venir null, pero cuando está presente es más precisa que la del archivo (un
# informe diario incluye notas de días previos).
_INTERNAL_DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(20\d{2})")

# Campos del JSON (case-insensitive) — espejo de jarvis_news/upload_store, pero
# este módulo es la fuente para la base de contexto del agente.
_FIELD_ALIASES = {
    "title": ("emailTitle", "title", "titulo", "título", "headline", "titular"),
    "short_title": ("title", "titulo", "título"),
    "source": ("source", "fuente", "medio", "publisher", "diario"),
    "topic": ("topic", "tema", "seccion", "sección", "categoria", "categoría"),
    "author": ("author", "autor", "byline"),
    "body": ("text", "content", "body", "texto", "contenido", "cuerpo", "articulo"),
    "url": ("url", "link", "enlace", "href", "permalink", "pressLink", "detailsUrl"),
}


@dataclass
class NewsExtractionReport:
    files_processed: int = 0
    files_skipped: list[str] = field(default_factory=list)
    documents_count: int = 0
    chunks_count: int = 0
    articles_empty: int = 0

    def to_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "documents_count": self.documents_count,
            "chunks_count": self.chunks_count,
            "articles_empty": self.articles_empty,
        }


@dataclass
class NewsExtractionResult:
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    report: NewsExtractionReport = field(default_factory=NewsExtractionReport)


# ── Helpers puros ──────────────────────────────────────────────────────────────


def _parse_internal_date(article: dict) -> str | None:
    """Fecha ISO del campo ``date`` del artículo (DD/MM/YYYY → YYYY-MM-DD), o None."""
    raw = article.get("date") or article.get("fecha")
    if not raw:
        return None
    m = _INTERNAL_DATE_RE.match(str(raw))
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year}-{month:02d}-{day:02d}"


def news_date_from_filename(name: str) -> str | None:
    """Extrae la fecha ISO ``YYYY-MM-DD`` del nombre del archivo, o ``None``.

    ``noticias_2026_01_20.json`` → ``"2026-01-20"``. Valida rangos mes/día para
    no aceptar números espurios.
    """
    m = _DATE_IN_NAME_RE.search(name or "")
    if not m:
        return None
    year, month, day = m.group(1), m.group(2), m.group(3)
    if not ("01" <= month <= "12" and "01" <= day <= "31"):
        return None
    return f"{year}-{month}-{day}"


def _pick(article: dict, kind: str) -> str:
    """Primer valor no vacío entre los alias del campo (case-insensitive).

    Repara el mojibake del scraper antes de devolver (ver ``fix_mojibake``)."""
    lower = {str(k).lower(): v for k, v in article.items()}
    for key in _FIELD_ALIASES[kind]:
        val = lower.get(key.lower())
        if val:
            return fix_mojibake(str(val).strip())
    return ""


# Palabras funcionales que NO son siglas aunque vengan en mayúscula ("LA"→"La").
_SOURCE_STOPWORDS = {"LA", "EL", "LOS", "LAS", "DE", "DEL", "LO", "UN", "UNA", "Y", "EN"}


def _titlecase_word(w: str) -> str:
    """Title-case respetando siglas reales y paréntesis.

    "DF"→"DF" (sigla), "LA"→"La" (artículo), "(PULSO)"→"(Pulso)", "tercera"→"Tercera"."""
    if 2 <= len(w) <= 4 and w.isupper() and w not in _SOURCE_STOPWORDS:
        return w  # sigla real: DF, FMI, BCI, CNN
    # Baja todo y sube la primera letra alfabética (respeta un '(' inicial).
    return re.sub(r"[a-záéíóúñü]", lambda m: m.group(0).upper(), w.lower(), count=1)


def parse_source_institution(source: str) -> str:
    """Deriva el medio desde el campo ``source``.

    ``"LA TERCERA (PULSO) - CHILE - ACTUALIDAD - 29/03/2026 0:00:00"`` →
    ``"La Tercera (Pulso)"``. Toma el segmento previo al primer ``-`` y lo
    normaliza a Title Case (preservando siglas reales en mayúscula)."""
    if not source:
        return "Prensa"
    # Separador " - " o " (en-dash) " usado por algunas fuentes.
    head = re.split("\\s[-\u2013]\\s", source.strip(), maxsplit=1)[0].strip()
    if not head:
        return "Prensa"
    return " ".join(_titlecase_word(w) for w in head.split())


def chunk_article_text(
    text: str,
    *,
    target_chars: int = TARGET_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
    min_split: int = MIN_SPLIT_CHARS,
) -> list[str]:
    """Parte el cuerpo en chunks de ~``target_chars`` con solapamiento.

    Greedy por párrafos: acumula párrafos hasta superar el presupuesto. Un
    párrafo que por sí solo lo excede se corta por ventana de caracteres con
    ``overlap``. Cuerpos cortos (< ``min_split``) se devuelven enteros."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= min_split:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > target_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            step = max(1, target_chars - overlap)
            for i in range(0, len(para), step):
                chunks.append(para[i:i + target_chars].strip())
            continue
        if current and len(current) + len(para) + 2 > target_chars:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _article_id(file_date: str, source: str, title: str, body: str) -> str:
    """ID estable y human-scannable: ``news_<fecha>_<hash8>``.

    El hash sobre (fuente+título+inicio del cuerpo) hace la ingesta idempotente:
    re-ingestar el mismo artículo produce el mismo id → upsert, no duplica."""
    raw = f"{source}|{title}|{body[:200]}".encode("utf-8", errors="replace")
    digest = hashlib.sha1(raw).hexdigest()[:8]
    return f"news_{file_date}_{digest}"


def extract_article(
    article: dict,
    *,
    file_date: str,
    filename: str,
    filepath: str,
    position: int,
) -> tuple[Document, list[Chunk]] | None:
    """Convierte un artículo (dict del JSON) en ``Document`` + ``Chunk``s.

    Devuelve ``None`` si el artículo no tiene cuerpo ni título (nada que indexar).
    El titular se antepone al cuerpo para que entre al embedding del primer chunk.
    """
    title = _pick(article, "title")
    body = _pick(article, "body")
    if not (title or body):
        return None

    source_raw = _pick(article, "source")
    institution = parse_source_institution(source_raw)
    topic = _pick(article, "topic") or "GENERAL"

    # Fecha del artículo: la interna (DD/MM/YYYY) si viene, si no la del archivo.
    # Un informe diario incluye notas de días previos; usar la propia es más fiel.
    article_date = _parse_internal_date(article) or file_date

    document_id = _article_id(article_date, source_raw, title, body)

    # El titular como lead del cuerpo (contexto para el embedding + lectura).
    lead = f"{title}.\n\n" if title else ""
    full_text = f"{lead}{body}".strip()

    pieces = chunk_article_text(full_text) or [full_text]
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        chunks.append(Chunk(
            chunk_id=f"{document_id}_c{i}",
            document_id=document_id,
            text=piece,
            char_count=len(piece),
            token_estimate=len(piece) // 4,
            page_start=0,
            page_end=0,
            position_in_doc=i,
            section_title_raw=topic,
            chunk_date=article_date,
            image_path=None,
            visual_caption=None,
            kind=ChunkKind.TEXT,
        ))

    doc = Document(
        document_id=document_id,
        filename=title[:200] or filename,
        filepath=f"{filepath}#art{position}",
        doc_type_category=DOC_TYPE_NEWS,
        institution=institution,
        document_date=article_date,
        total_pages=0,
        total_chunks=len(chunks),
        char_count=len(full_text),
        extraction_warnings=[],
    )
    return doc, chunks


def extract_news_file(path: Path) -> tuple[list[Document], list[Chunk], int]:
    """Extrae un JSON diario. Retorna ``(documents, chunks, n_articulos_vacios)``.

    La fecha sale del nombre del archivo (``news_date_from_filename``). Si el
    nombre no trae fecha válida, levanta ``ValueError`` (no podemos fechar la
    noticia y la ventana por fecha quedaría rota)."""
    file_date = news_date_from_filename(path.name)
    if not file_date:
        raise ValueError(
            f"No pude extraer la fecha del nombre {path.name!r}. "
            "Usa el patrón 'noticias_YYYY_MM_DD.json'."
        )

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        # Permite {"articles": [...]} / {"noticias": [...]}: primera lista de dicts.
        items = next(
            (v for v in raw.values() if isinstance(v, list) and v and isinstance(v[0], dict)),
            None,
        )
        articles = items or []
    elif isinstance(raw, list):
        articles = raw
    else:
        articles = []

    documents: list[Document] = []
    chunks: list[Chunk] = []
    empty = 0
    for pos, art in enumerate(articles):
        if not isinstance(art, dict):
            empty += 1
            continue
        out = extract_article(
            art, file_date=file_date, filename=path.name,
            filepath=path.name, position=pos,
        )
        if out is None:
            empty += 1
            continue
        doc, doc_chunks = out
        documents.append(doc)
        chunks.extend(doc_chunks)
    return documents, chunks, empty


def extract_news(source: Path) -> NewsExtractionResult:
    """Extrae todos los JSON diarios de ``source`` (archivo o directorio).

    Ingesta puramente aditiva: NO filtra por ventana ni purga lo viejo (el
    agente decide la frescura con el filtro por fecha en query-time). Si dos
    archivos traen el mismo artículo, comparten ``document_id`` → upsert."""
    files: list[Path]
    if source.is_dir():
        files = sorted(p for p in source.glob("*.json") if p.is_file())
    elif source.is_file():
        files = [source]
    else:
        raise FileNotFoundError(f"No existe el origen de noticias: {source}")

    result = NewsExtractionResult()
    for path in files:
        try:
            docs, chunks, empty = extract_news_file(path)
        except ValueError as exc:
            log.warning("Saltando %s: %s", path.name, exc)
            result.report.files_skipped.append(path.name)
            continue
        except json.JSONDecodeError as exc:
            log.warning("JSON inválido en %s: %s", path.name, exc)
            result.report.files_skipped.append(path.name)
            continue
        result.documents.extend(docs)
        result.chunks.extend(chunks)
        result.report.files_processed += 1
        result.report.articles_empty += empty

    result.report.documents_count = len(result.documents)
    result.report.chunks_count = len(result.chunks)
    return result
