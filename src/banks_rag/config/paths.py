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
# JSON diarios de noticias scrapeadas → base aislada "contexto_actual"
# (un JSON por día, nombrado 'noticias_YYYY_MM_DD.json'). Ver extract_news.py.
NEWS_RAW_DIR = ROOT / "data_pipeline" / "Noticias_scrapping"

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
# Logs de conversación del agente (JSONL diario) para evaluación de respuestas.
DATA_CHAT_LOGS_DIR = DATA_DIR / "chat_logs"
# Archivos subidos por el usuario en el chat (contexto efímero, con TTL).
DATA_UPLOADS_DIR = DATA_DIR / "uploads"

# Catálogo parquet (datasets crudos del DW; tools del agente arman SQL safe).
SQL_CATALOG_DIR = ROOT / "sql_catalog"
PARQUET_CATALOG_YAML = SQL_CATALOG_DIR / "parquet_catalog.yaml"

# Outputs por etapa
DOCUMENTS_JSON = LOGS_DIR / "documents.json"
CHUNKS_JSON = LOGS_DIR / "chunks.json"
EXTRACTION_REPORT_JSON = LOGS_DIR / "extraction_report.json"
CHUNKS_ENRICHED_JSON = LOGS_DIR / "chunks_enriched.json"
CHUNKS_VECTORIZED_JSON = LOGS_DIR / "chunks_vectorized.json"
