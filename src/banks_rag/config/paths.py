"""Paths del repo y de runtime.

Centraliza las rutas en un único lugar para no esparcir constantes en cada
módulo. ``ROOT`` se infiere desde la ubicación de este archivo: ``src/banks_rag/config/paths.py``
está cuatro niveles por debajo de la raíz del repo.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Inputs
DATA_RAW_DIR = ROOT / "Datos_prueba"  # legacy; en Fase 8 se mueve a data/raw/

# Outputs intermedios (en Fase 8 se mueven a data/logs_intermedios/)
LOGS_DIR = ROOT / "logs"
IMAGES_DIR = ROOT / "images"

# Modelos pre-descargados (offline-friendly).
# Convención: ``models/<owner>--<name>/`` para sentence-transformers / HF;
# ``models/<archivo>.gguf`` para llama.cpp.
MODELS_DIR = ROOT / "models"

# Datos persistentes nuevos
DATA_DIR = ROOT / "data"
DATA_GOLDEN_SET_DIR = DATA_DIR / "golden_set"
DATA_SNAPSHOTS_DIR = DATA_DIR / "snapshots"

# Catálogo SQL (Fase 4)
SQL_CATALOG_DIR = ROOT / "sql_catalog"
SQL_CATALOG_YAML = SQL_CATALOG_DIR / "catalog.yaml"

# Outputs por etapa
DOCUMENTS_JSON = LOGS_DIR / "documents.json"
CHUNKS_JSON = LOGS_DIR / "chunks.json"
EXTRACTION_REPORT_JSON = LOGS_DIR / "extraction_report.json"
CHUNKS_ENRICHED_JSON = LOGS_DIR / "chunks_enriched.json"
CHUNKS_VECTORIZED_JSON = LOGS_DIR / "chunks_vectorized.json"
