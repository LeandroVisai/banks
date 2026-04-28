# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Sistema RAG para banco central chileno. Procesa PDFs financieros (Comunicados BCCh, Minutas del Consejo, reportes JPMorgan, Fed Statements) y un Excel de Monitor PM en un corpus semántico consultable con citación por página.

Stack: `pypdf` + `openpyxl` + `sentence-transformers` (multilingual-e5-small, 384-dim) + PostgreSQL + pgvector (HNSW) + búsqueda híbrida BM25/vector con RRF y MMR.

## Comandos esenciales

```bash
# Pipeline completo (extracción → enriquecimiento → embeddings → BD)
python3 run.py full

# Pasos individuales
python3 run.py step 0          # extracción PDF/Excel → chunks.json
python3 run.py step 1          # enriquecimiento → chunks_enriched.json
python3 run.py step 2          # embeddings → chunks_vectorized.json
python3 run.py step 3 reset    # drop + recrear schema PostgreSQL + cargar
python3 run.py step 3 stats    # métricas de la BD sin tocar nada

# Búsqueda
python3 04_search.py "tasa de interés 2022" 5
python3 04_search.py "commodities riesgos" 10 --json
python3 04_search.py "inflación" 5 --no-mmr

# Generar prompt RAG listo para LLM local (Ollama, llama.cpp)
python3 05_query.py "política monetaria 2022" 3

# Validación de que el sistema funciona correctamente
python3 04_search.py "política monetaria 2022" 3
# → top-1 debe ser comunicado1.pdf, sección DECISION, importance ≥ 0.90
```

## Arquitectura

### Flujo de datos

```
Datos_prueba/Comunicados/*.pdf
Datos_prueba/Minutas/*.pdf
Datos_prueba/Fed/*.pdf
Datos_prueba/Researchs/**/*.pdf
Datos_prueba/Monitor PM/textos_monitor_pm.xlsx
  → [00] chunks.json + documents.json        (extracción + chunking)
  → [01] chunks_enriched.json               (taxonomía semántica)
  → [02] chunks_vectorized.json             (embeddings normalizados)
  → [03] PostgreSQL rag_banco               (schema + HNSW + GIN)
  → [04] búsqueda híbrida                   (HNSW + BM25 → RRF → MMR)
  → [05] prompt para LLM                    (contexto delimitado + citas)
```

### Módulos y responsabilidades

| Archivo | Responsabilidad única |
|---|---|
| `models.py` | Dataclasses `Document`, `Chunk`, `EnrichedChunk` — única fuente de verdad de estructuras |
| `taxonomy.py` | Patterns de variables, secciones, entidades, boilerplate. Compartido por paso 1 y paso 4 |
| `00_generate_jsons.py` | PDF+Excel → chunks semánticos (por párrafos, target 600 chars, overlap 100) |
| `01_enrich_metadata.py` | Scoring de sección, variables económicas, importance_score |
| `02_vectorize.py` | Embeddings con E5-multilingual, prefijo contextual, normalización L2 |
| `03_database.py` | Schema PostgreSQL, HNSW index, bulk upsert, stats |
| `04_search.py` | Parse query NL, recall dual (vector+BM25), RRF, MMR, importance boost |
| `05_query.py` | Formatea contexto RAG + few-shot prompt para LLM local |
| `run.py` | Orquestador — delega a los scripts numerados vía subprocess |

### Procesamiento del Excel (Monitor PM)

El archivo `textos_monitor_pm.xlsx` tiene estructura tabular especial:
- **Filas** = fechas (cada fila es un día de mercado)
- **Columnas** = segmentos de mercado (ej. "Mercado Cambiario", "Renta Fija")
- **Celdas** = texto narrativo completo

Cada celda se convierte en un `Chunk` independiente con `chunk_date` = fecha de esa fila (ISO `YYYY-MM-DD`). El `section_type` se asigna directamente desde `MONITOR_PM_SECTION_MAP` en `taxonomy.py` con confianza 1.0 — nunca pasa por el scorer genérico de secciones.

### Invariantes críticos

- **`taxonomy.py` es la fuente de verdad**: si agregas una variable, sección, entidad o patrón, edita solo este archivo. Los pasos 1 y 4 lo importan en tiempo de ejecución.

- **doc_type se hereda del filepath, nunca del contenido del chunk**: `detect_doc_type()` en `00_generate_jsons.py` opera sobre la ruta relativa. Un chunk de una minuta que menciona "comunicado" no se re-clasifica.

- **Embeddings normalizados L2 + `vector_cosine_ops`**: los embeddings se normalizan en `02_vectorize.py`. El índice HNSW usa `vector_cosine_ops`. Cambiar uno sin el otro rompe la similitud.

- **models_cache/ se auto-detecta**: `02_vectorize.py` y `04_search.py` setean `SENTENCE_TRANSFORMERS_HOME=./models_cache` automáticamente si la carpeta existe junto al script. En el servidor sin internet, el modelo ya está ahí.

- **Filtro de variables incluye DECISION**: en `04_search.py`, `build_filters_sql()` siempre incluye chunks con `section_type='DECISION'` aunque no tengan la variable etiquetada.

- **chunk_date vs document_date**: los chunks del Monitor PM tienen `chunk_date` (fecha de la celda). Los PDFs tienen `chunk_date = NULL`. Los filtros de año en `04_search.py` usan `COALESCE(chunk_date year, document_year)` para manejar ambos casos correctamente.

## Taxonomía (`taxonomy.py`) — mapa de conceptos

| Constante | Propósito |
|---|---|
| `ECONOMIC_VARIABLES` | 14 variables con listas de keywords y nivel de importancia (CRITICAL/HIGH/MEDIUM) |
| `SECTION_KEYWORDS` | Patrones para clasificar secciones canónicas de PDFs |
| `MONITOR_PM_SECTION_MAP` | Mapeo directo columna Excel → section_type canónico (11 columnas) |
| `MONITOR_PM_SECTIONS` | Frozenset de los 11 section_type del Monitor PM (usado en importance scoring) |
| `ENTITY_KEYWORDS` | Keywords de entidades (PAIS_CHILE, BANCO_CENTRAL_CHILE, FEDERAL_RESERVE, etc.) |
| `BOILERPLATE_PATTERN` | Regex para detectar texto legal/disclaimer de reportes JPMorgan |
| `FORWARD_LOOKING_PATTERN` | Detecta lenguaje prospectivo |

### Importance score — señales y pesos (`01_enrich_metadata.py`)

```
CRITICAL variable presente   → +0.35  (+0.10 si hay 2+)
HIGH variable (sin CRITICAL) → +0.20
MEDIUM variable (sin HIGH)   → +0.08
Datos numéricos (satura 3)   → +0.20
Sección DECISION/VOTACION    → +0.25
Sección PROYECCION/RIESGOS   → +0.15
Sección ANALISIS             → +0.08
Sección Monitor PM           → +0.12  (las 11 secciones del Excel)
Forward-looking              → +0.08
Entidades (satura 2)         → +0.10
Sin variables ni datos       → -0.12  (penalización)
Boilerplate legal            → = 0.0  (forzado, no acumulable)
```

## Agregar nuevos tipos de documento

Dos cambios necesarios:

1. **`00_generate_jsons.py` → `detect_doc_type()`**: agregar rama `if` que reconozca la carpeta nueva (ej. `if "bce" in p: return "BCE_STATEMENT"`).

2. **`taxonomy.py` → `SECTION_KEYWORDS`**: agregar los patrones de texto característicos del nuevo tipo.

Luego correr `python3 run.py full` — el pipeline es idempotente.

## Ajustar parámetros de búsqueda

| Parámetro | Archivo | Qué controla |
|---|---|---|
| `IMPORTANCE_WEIGHTS` | `01_enrich_metadata.py` | Peso de cada señal en importance_score |
| `RECALL_N` | `04_search.py` | Candidatos por rama (vector + BM25) antes de RRF |
| `RRF_K` | `04_search.py` | Hiperparámetro de RRF (60 = estándar) |
| `MMR_LAMBDA` | `04_search.py` | 1.0 = solo relevancia, 0.0 = solo diversidad |
| `IMPORTANCE_BOOST` | `04_search.py` | Peso de importance en el re-rank final |

## Schema PostgreSQL (referencia rápida)

```sql
documents (document_id PK, doc_type_category, institution,
           document_date TEXT, document_year INT, extraction_warnings JSONB)

chunks    (chunk_id PK, document_id FK,
           text, text_tsv TSVECTOR,              -- BM25
           embedding VECTOR(384),                -- HNSW cosine
           section_type, section_confidence,
           economic_variables JSONB,             -- GIN jsonb_path_ops
           entities JSONB,                       -- GIN jsonb_path_ops
           importance_score, is_policy_decision, is_forward_looking,
           chunk_date DATE,                      -- solo Monitor PM, NULL para PDFs
           tags TEXT[])                          -- GIN
```

## Variables de entorno

| Variable | Default (main) | Default (EmbaddingGemma) | Cuándo cambiar |
|---|---|---|---|
| `PGDATABASE` | `rag_banco` | `rag_banco` | Usar otra BD |
| `PGUSER` / `PGPASSWORD` | SO / vacío | SO / vacío | Servidor con auth |
| `RAG_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | `EmbaddingGemma-300` | Cambiar modelo |
| `RAG_EMBEDDING_DIM` | auto-detectado (384) | auto-detectado desde chunks | Forzar dimensión sin leer JSON |
| `RAG_PURE_TEXT` | `0` | `0` | `1` = no inyectar metadata context en embedding |
| `SENTENCE_TRANSFORMERS_HOME` | auto-detectado | auto-detectado | Solo si `models_cache/` no está junto al script |

## Rama EmbaddingGemma

Esta rama usa el modelo `EmbaddingGemma-300` ubicado en `./models/EmbaddingGemma-300`.

### Diferencias respecto a `main`

| Aspecto | `main` | `EmbaddingGemma` |
|---|---|---|
| Modelo | `intfloat/multilingual-e5-small` (384-dim) | `EmbaddingGemma-300` (dim auto-detectada) |
| Prefijo embedding | `passage: ` / `query: ` (E5) | Sin prefijo |
| Fallback modelo | `all-MiniLM-L6-v2` | `intfloat/multilingual-e5-small` |
| Dimensión VECTOR | Hardcodeada 384 | Dinámica (lee de `chunks_vectorized.json` o `RAG_EMBEDDING_DIM`) |

### Setup para EmbaddingGemma

```bash
# 1. Asegurarse de que el modelo esté en ./models/EmbaddingGemma-300
ls models/EmbaddingGemma-300/

# 2. Correr el pipeline completo (genera embeddings con el nuevo modelo y recrea BD)
python3 run.py full

# 3. Validar
python3 04_search.py "política monetaria 2022" 3
```

### Invariante crítico de la rama

La dimensión del VECTOR en PostgreSQL debe coincidir con la dimensión real del modelo.
`03_database.py` la auto-detecta desde `logs/chunks_vectorized.json` (campo `embedding_dim`).
**Siempre correr `step 2` antes de `step 3 reset`** al cambiar de modelo.

## Gemelo de desarrollo

Este repo tiene un gemelo en `/Users/leandrovenegas/Desktop/Proyecto_rag/` (sandbox Mac). Mantener paridad entre ambos — cambios en uno se copian al otro. `taxonomy.py` es la fuente de verdad para patrones compartidos.
