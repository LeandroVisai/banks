"""Subida de archivos del chat como **contexto efímero**.

Extrae el contenido de un archivo (PDF/TXT → texto; CSV/Excel → tabla + stats)
y lo guarda con un ``upload_id`` para que el siguiente turno del agente lo
inyecte como contexto. NO se indexa en pgvector ni en el catálogo: vive solo
mientras dure la conversación y caduca por TTL.
"""

from .upload_store import (
    MAX_UPLOAD_BYTES,
    UploadError,
    extract_upload,
    load_upload,
    save_upload,
    upload_images_dir,
)

__all__ = [
    "MAX_UPLOAD_BYTES",
    "UploadError",
    "extract_upload",
    "load_upload",
    "save_upload",
    "upload_images_dir",
]
