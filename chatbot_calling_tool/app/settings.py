"""Settings tipados con Pydantic — específicos para el chatbot agentic."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDING_CACHE_DIR = PROJECT_ROOT.parent / "models_cache"
SCHEMA_DIR = PROJECT_ROOT / "schema"

# data_pipeline vive un nivel arriba: banks/data_pipeline/
DATA_PIPELINE_DIR = PROJECT_ROOT.parent / "data_pipeline"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    chatbot_model_name: str = "Qwen3.6-35B-A3B"
    chatbot_quantization: str | None = "fp8"
    chatbot_tensor_parallel: int = Field(1, ge=1, le=16)
    chatbot_gpu_mem_util: float = Field(0.90, ge=0.1, le=1.0)
    chatbot_max_model_len: int = Field(16384, ge=512)
    chatbot_max_tokens: int = Field(2048, ge=1)
    chatbot_temperature: float = Field(0.2, ge=0.0, le=2.0)
    chatbot_top_p: float = Field(0.9, ge=0.0, le=1.0)

    # ── Agente ───────────────────────────────────────────────────────────────
    max_agent_iterations: int = Field(6, ge=1, le=20)
    max_tool_result_tokens: int = Field(1500, ge=100)
    tool_trace_enabled: bool = True

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model_id: str = "intfloat/multilingual-e5-small"

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    pghost: str = "localhost"
    pgport: int = 5432
    pguser: str = "postgres"
    pgpassword: str = "postgres"
    pgdatabase: str = "rag_banco"
    pg_pool_min: int = Field(2, ge=1)
    pg_pool_max: int = Field(8, ge=1)

    # ── RAG retrieval (usado por las tools) ─────────────────────────────────
    rag_table_prefix: str = ""
    rag_recall_n: int = Field(30, ge=5, le=200)
    rrf_k: int = Field(60, ge=1)
    mmr_lambda: float = Field(0.65, ge=0.0, le=1.0)

    # ── Conversación ─────────────────────────────────────────────────────────
    history_max_turns: int = Field(10, ge=0, le=50)

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(8081, ge=1, le=65535)  # 8081 ≠ 8080 del chatbot/ clásico
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # ── Series históricas — catálogo ─────────────────────────────────────────
    catalog_path: Path = Field(
        default_factory=lambda: DATA_PIPELINE_DIR / "series_catalog.yaml",
        description="YAML con la metadata de las series (id, name, unit, sql_table, etc.)",
    )

    # ── Data Warehouse — Get_Data module ─────────────────────────────────────
    # Ruta al directorio donde está Get_Data.py (el mismo módulo que usa Monitor.py)
    # En producción (Windows): D:\GOM\DACE\Nacho\Modulos
    get_data_path: str = Field(
        default="",
        description="Directorio que contiene Get_Data.py (igual que sys.path.append en Monitor.py)",
    )

    @field_validator("rag_table_prefix")
    @classmethod
    def _check_prefix(cls, v: str) -> str:
        if v and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*_?", v):
            raise ValueError(f"rag_table_prefix inválido: {v!r}")
        return v

    @field_validator("chatbot_quantization", mode="before")
    @classmethod
    def _empty_quant_to_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
            return None
        return v

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self.chatbot_model_name

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.pguser}:{self.pgpassword}"
            f"@{self.pghost}:{self.pgport}/{self.pgdatabase}"
        )

    @property
    def docs_table(self) -> str:
        return f"{self.rag_table_prefix}documents"

    @property
    def chunks_table(self) -> str:
        return f"{self.rag_table_prefix}chunks"


settings = Settings()
