"""Configuración tipada del servicio (Pydantic Settings).

Lee de:
  1. Variables de entorno (prefijo ``BANKS_``).
  2. Archivo ``.env`` en la raíz del repo (si existe).
  3. Defaults documentados.

Una sola fuente de verdad para CLI, API y workers. Centraliza compatibilidad
con env vars legacy (``PGHOST`` sin prefijo) y unifica las settings que antes
estaban duplicadas en ``chatbot/app/settings.py`` y
``chatbot_calling_tool/app/settings.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import DATA_CHAT_LOGS_DIR, MODELS_DIR, ROOT


class Settings(BaseSettings):
    """Settings unificadas del servicio."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="BANKS_",
        extra="ignore",
        case_sensitive=False,
    )

    # ── PostgreSQL (env legacy sin prefijo BANKS_) ───────────────────────────
    pg_host: str = Field(default_factory=lambda: os.getenv("PGHOST", "localhost"))
    pg_port: int = Field(default_factory=lambda: int(os.getenv("PGPORT", "5432")))
    pg_user: str = Field(default_factory=lambda: os.getenv(
        "PGUSER", os.getenv("USER", "postgres"),
    ))
    pg_password: str = Field(default_factory=lambda: os.getenv("PGPASSWORD", "postgres"))
    pg_database: str = Field(default_factory=lambda: os.getenv("PGDATABASE", "rag_banco"))

    # ── RAG (env legacy sin prefijo) ──────────────────────────────────────────
    rag_table_prefix: str = Field(default_factory=lambda: os.getenv("RAG_TABLE_PREFIX", ""))

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_family: Literal["qwen", "gemma", "mock"] = "mock"
    llm_model_path: str = ""
    llm_n_ctx: int = 16384
    llm_n_gpu_layers: int = -1  # todo a GPU
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_max_tokens: int = 2048

    # ── Agente ───────────────────────────────────────────────────────────────
    max_agent_iterations: int = 6
    max_tool_result_tokens: int = 1500
    history_max_turns: int = 10
    # Control del modo "thinking" de Qwen3 (soft switch /no_think):
    #   off      → sin thinking en ningún componente (mínima latencia).
    #   adaptive → thinking SOLO en especialistas multi-paso (cuantitativos),
    #              donde el razonamiento más rinde; sin thinking en documentales
    #              ni en la síntesis. Default (mejor balance latencia/calidad).
    #   on       → thinking en todo (máxima calidad, máxima latencia).
    thinking_mode: Literal["off", "adaptive", "on"] = "adaptive"

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    api_keys: str = ""  # CSV de API keys; vacío = auth desactivada

    @property
    def api_keys_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    # ── Chat logging ─────────────────────────────────────────────────────────
    chat_log_enabled: bool = True
    chat_log_dir: str = ""  # vacío = default DATA_CHAT_LOGS_DIR

    # ── Helpers ──────────────────────────────────────────────────────────────
    @property
    def models_dir(self) -> Path:
        return MODELS_DIR

    @property
    def model_path_resolved(self) -> Path | None:
        """Resuelve ``llm_model_path`` contra ``MODELS_DIR`` si es relativo."""
        if not self.llm_model_path:
            return None
        p = Path(self.llm_model_path)
        if p.is_absolute():
            return p
        return MODELS_DIR / self.llm_model_path

    @property
    def chat_log_dir_resolved(self) -> Path:
        """Directorio donde se persisten los logs de chat (JSONL diario)."""
        if not self.chat_log_dir:
            return DATA_CHAT_LOGS_DIR
        p = Path(self.chat_log_dir)
        return p if p.is_absolute() else ROOT / self.chat_log_dir


# Instancia global lazy. Tests pueden override via override_settings().
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(new: Settings) -> None:
    """Reemplaza la instancia global. Útil en tests."""
    global _settings
    _settings = new


def reset_settings() -> None:
    global _settings
    _settings = None
