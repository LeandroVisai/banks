.PHONY: help install install-dev install-eval lint format typecheck test test-unit test-integration ingest serve eval clean tree

# Colores
BLUE  := \033[36m
GREEN := \033[32m
RESET := \033[0m

help: ## Muestra esta ayuda
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(BLUE)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Instalación ─────────────────────────────────────────────────────────

install: ## Instala el paquete en modo editable
	pip install -e .

install-dev: ## Instala con dependencias de desarrollo (lint, test, typecheck)
	pip install -e ".[dev]"
	pre-commit install || true

install-eval: ## Instala con dependencias de evaluación (RAGAS)
	pip install -e ".[eval]"

install-offline: ## Instala desde wheels/ (servidor sin internet)
	pip install --no-index --find-links ./wheels/ -e .

# ── Calidad ─────────────────────────────────────────────────────────────

lint: ## Ejecuta ruff (lint)
	ruff check src tests scripts

format: ## Formatea código con ruff
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck: ## Ejecuta mypy
	mypy src/banks_rag

test: ## Ejecuta toda la suite de tests
	pytest

test-unit: ## Solo tests unitarios
	pytest -m unit

test-integration: ## Solo tests de integración
	pytest -m integration

test-cov: ## Tests con coverage report
	pytest --cov=banks_rag --cov-report=term-missing --cov-report=html

# ── Pipeline RAG ────────────────────────────────────────────────────────

ingest: ## Pipeline completo: extract → enrich → vectorize → persist
	banks-ingest full

ingest-extract: ## Solo extracción
	banks-ingest extract

ingest-enrich: ## Solo enriquecimiento
	banks-ingest enrich

ingest-vectorize: ## Solo vectorización (texto + multimodal)
	banks-ingest vectorize

ingest-persist: ## Solo carga a PostgreSQL
	banks-ingest persist

# ── Servicio ────────────────────────────────────────────────────────────

serve: ## Levanta el servicio FastAPI en :8080
	python -m banks_rag.interface.api.main

serve-dev: ## Servicio con reload
	uvicorn banks_rag.interface.api.main:app --reload --host 0.0.0.0 --port 8080

# ── Evaluación ──────────────────────────────────────────────────────────

eval: ## Corre golden set + RAGAS y emite reporte
	banks-eval all

eval-retrieval: ## Solo métricas de retrieval (recall@k, MRR)
	banks-eval retrieval

eval-routing: ## Solo métricas de routing SQL
	banks-eval routing

eval-ci: ## Gate CI: falla si recall@5 cae >5% vs baseline
	banks-eval all --ci

# ── Mantenimiento ───────────────────────────────────────────────────────

clean: ## Limpia archivos temporales
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	rm -rf build dist *.egg-info src/*.egg-info

tree: ## Muestra estructura de carpetas (ignora deps y datos)
	tree -L 3 -I '__pycache__|node_modules|*.egg-info|*.parquet|*.pdf|snapshots|images|models_cache' --dirsfirst

# ── SQL Catalog ─────────────────────────────────────────────────────────

extract-monitor-queries: ## Extrae queries de querys/Monitor.py al sql_catalog/
	python scripts/extract_monitor_queries.py
