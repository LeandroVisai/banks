"""Reparación de texto mal codificado (mojibake).

Vive en ``domain_knowledge`` (capa baja, sin dependencias hacia application/
infrastructure) para que la compartan tanto la ingesta de noticias
(``application/ingestion/extract_news``) como el contexto efímero de archivos
subidos en el chat (``infrastructure/uploads/upload_store``) sin violar la
arquitectura por capas.
"""

from __future__ import annotations

# Marcadores típicos del doble-encoding UTF-8↔cp1252. Si no aparece ninguno, el
# texto ya está bien y se devuelve intacto (la reparación es no-op).
_MOJIBAKE_MARKERS = ("Ã", "Â", "â")


def fix_mojibake(text: str) -> str:
    """Repara texto doble-codificado (UTF-8 leído como Windows-1252/Latin-1).

    Origen típico: un scraper guarda bytes UTF-8 interpretados como cp1252,
    dejando basura como ``inflaciÃ³n`` (→ ``inflación``), ``paÃ­s`` (→ ``país``),
    ``Â¿`` (→ ``¿``). Sin esto, los embeddings y lo que lee el agente quedan
    corruptos.

    Usa ``ftfy`` si está instalado (repara también comillas tipográficas y casos
    parciales); si no, hace el round-trip ``cp1252→utf-8`` (cubre los acentos,
    que es el grueso). Si el texto no tiene marcadores de mojibake, lo devuelve
    intacto.
    """
    if not text or not any(m in text for m in _MOJIBAKE_MARKERS):
        return text
    try:
        import ftfy  # lazy: dependencia opcional

        return ftfy.fix_text(text)
    except ImportError:
        pass
    for codec in ("cp1252", "latin-1"):
        try:
            return text.encode(codec).decode("utf-8")
        except (UnicodeError, ValueError):
            continue
    return text
