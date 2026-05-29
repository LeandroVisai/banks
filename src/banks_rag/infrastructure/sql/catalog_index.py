"""Índice semántico del parquet_catalog (descubrimiento por embeddings).

Completa la mitad "embeddings" del descubrimiento (la otra mitad —alias léxico—
vive en ``parquet_catalog_loader.search_datasets`` + ``domain_knowledge``).
Embebe la descripción de cada dataset con el embedder del proyecto
(Qwen3-VL-Embedding-8B) y, en query-time, puntúa cada dataset por similitud
coseno. Esos puntajes se inyectan en ``search_datasets`` vía ``extra_scores``.

Degradación elegante (clave para tests/offline):
  - Off por defecto: solo se activa con ``BANKS_CATALOG_SEMANTIC=true`` (en la
    H100) o pasando un ``embedder`` explícito (tests). Sin eso, devuelve ``{}``
    y el descubrimiento usa únicamente léxico+alias (que ya resuelve los casos
    de los logs).
  - Cualquier fallo (modelo ausente, error de carga) se captura → ``{}``. Nunca
    rompe ``discover_query``.

Persistencia: el índice se cachea en disco (``data/catalog_index.npz``) con un
hash del catálogo; se reconstruye solo si el YAML cambió.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass

import numpy as np

from banks_rag.config.paths import DATA_DIR
from banks_rag.infrastructure.sql.parquet_catalog_loader import ParquetDataset

log = logging.getLogger(__name__)

# Peso del aporte semántico al score de search_datasets. El coseno vive en
# [0, 1]; el léxico suma ~1/token y ~5/hint. Con weight=6 el semántico es un
# booster/tiebreaker útil sin pisar señales léxicas fuertes (hints exactos).
SEMANTIC_WEIGHT = 6.0
# Umbral mínimo de coseno para considerar un dataset (evita ruido de baja sim).
_MIN_SIM = 0.15

_CACHE_PATH = DATA_DIR / "catalog_index.npz"


def _enabled() -> bool:
    return os.getenv("BANKS_CATALOG_SEMANTIC", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def catalog_text(dataset: ParquetDataset) -> str:
    """Texto representativo de un dataset para embeber (id+name+desc+cols)."""
    cols = ", ".join(c.name for c in dataset.columns)
    return (
        f"{dataset.id}. {dataset.name}. {dataset.description.strip()} "
        f"Segmento: {dataset.segment}. Unidad: {dataset.unit}. Columnas: {cols}."
    )


def _catalog_hash(entries: list[ParquetDataset]) -> str:
    """Hash estable del contenido textual del catálogo (detecta staleness)."""
    h = hashlib.sha256()
    for e in entries:
        h.update(catalog_text(e).encode("utf-8"))
    return h.hexdigest()


@dataclass
class _Index:
    ids: list[str]
    matrix: np.ndarray  # (n_datasets, dim), filas L2-normalizadas
    catalog_hash: str


class CatalogSemanticIndex:
    """Singleton del índice semántico, con cache en memoria y disco."""

    def __init__(self) -> None:
        self._index: _Index | None = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._index = None

    def ensure(self, entries: list[ParquetDataset], embedder) -> _Index:
        """Devuelve el índice válido para ``entries``, construyéndolo si hace falta.

        Reusa el índice en memoria o en disco si su hash coincide con el del
        catálogo actual; si no, lo (re)construye y persiste (best-effort).
        """
        chash = _catalog_hash(entries)
        with self._lock:
            if self._index is not None and self._index.catalog_hash == chash:
                return self._index

            disk = _load_from_disk(chash)
            if disk is not None:
                self._index = disk
                return disk

            self._index = _build(entries, embedder, chash)
            _persist(self._index)
            return self._index


# Singleton del proceso.
_SINGLETON = CatalogSemanticIndex()


def _build(entries: list[ParquetDataset], embedder, catalog_hash: str) -> _Index:
    texts = [catalog_text(e) for e in entries]
    matrix = np.asarray(embedder.encode_text(texts), dtype=np.float32)
    # Re-normaliza por si el embedder no garantizara L2 (la similitud usa dot).
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return _Index(ids=[e.id for e in entries], matrix=matrix, catalog_hash=catalog_hash)


def _persist(index: _Index) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            _CACHE_PATH,
            ids=np.array(index.ids, dtype=object),
            matrix=index.matrix,
            catalog_hash=np.array(index.catalog_hash),
        )
    except Exception:
        log.debug("No se pudo persistir el índice semántico del catálogo", exc_info=True)


def _load_from_disk(expected_hash: str) -> _Index | None:
    if not _CACHE_PATH.exists():
        return None
    try:
        data = np.load(_CACHE_PATH, allow_pickle=True)
        disk_hash = str(data["catalog_hash"])
        if disk_hash != expected_hash:
            return None  # catálogo cambió → reconstruir
        return _Index(
            ids=list(data["ids"]),
            matrix=np.asarray(data["matrix"], dtype=np.float32),
            catalog_hash=disk_hash,
        )
    except Exception:
        log.debug("No se pudo cargar el índice semántico del catálogo", exc_info=True)
        return None


def catalog_semantic_scores(
    query: str,
    entries: list[ParquetDataset],
    *,
    embedder=None,
    weight: float = SEMANTIC_WEIGHT,
) -> dict[str, float]:
    """Puntajes semánticos por ``dataset_id`` para inyectar en ``search_datasets``.

    Args:
        query: pregunta del usuario.
        entries: datasets del catálogo.
        embedder: embedder a usar. Si es ``None`` y la feature está habilitada
            (``BANKS_CATALOG_SEMANTIC``), usa ``build_default_embedder``. Si es
            ``None`` y está deshabilitada, devuelve ``{}`` (sin cargar modelo).
        weight: escala del coseno → score aditivo.

    Returns:
        ``{dataset_id: score}`` con ``score = weight * cos`` para coseno ≥ umbral.
        ``{}`` ante cualquier fallo o si la feature está off.
    """
    if embedder is None:
        if not _enabled():
            return {}
        try:
            from banks_rag.infrastructure.embeddings import build_default_embedder
            embedder = build_default_embedder()
        except Exception:
            log.warning("Embedder no disponible; descubrimiento sin semántica", exc_info=True)
            return {}

    try:
        index = _SINGLETON.ensure(entries, embedder)
        qv = np.asarray(embedder.encode_text([query])[0], dtype=np.float32)
        n = float(np.linalg.norm(qv))
        if n == 0.0:
            return {}
        qv = qv / n
        sims = index.matrix @ qv  # coseno (filas normalizadas)
        return {
            ds_id: weight * float(sim)
            for ds_id, sim in zip(index.ids, sims, strict=False)
            if sim >= _MIN_SIM
        }
    except Exception:
        log.warning("Falló el scoring semántico del catálogo", exc_info=True)
        return {}


def reset_catalog_index() -> None:
    """Limpia el índice en memoria (tests)."""
    _SINGLETON.reset()
