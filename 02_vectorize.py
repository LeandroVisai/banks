#!/usr/bin/env python3
"""
PASO 2 — Vectorización de chunks enriquecidos.

Entrada:
  logs/chunks_enriched.json  (del paso 1)

Salida:
  logs/chunks_vectorized.json

Mejoras sobre el pipeline previo:
  1. Modelo por defecto multilingüe (paraphrase-multilingual-MiniLM-L12-v2,
     384-dim, 50+ idiomas). Mejor para documentos en español. Se puede
     override con env var RAG_EMBEDDING_MODEL. Fallback automático a
     all-MiniLM-L6-v2 si el multilingüe no se puede cargar.

  2. Embeddings normalizados L2 en almacenamiento → cosine similarity.
     El índice HNSW usa vector_cosine_ops (operador <=>).

  3. Texto embebido incluye un prefijo compacto de contexto:
        [DOC_TYPE | SECTION | vars]  texto...
     Esto inyecta señal semántica del enriquecimiento al embedding,
     mejorando retrieval. Se puede deshabilitar con RAG_PURE_TEXT=1.

  4. Detección real de truncamiento usando el tokenizer del modelo
     (no la heurística char/4). Logea cuántos chunks se truncan.

  5. Caché del modelo respetando SENTENCE_TRANSFORMERS_HOME si está
     seteado (para deploy offline).

Uso:
  python3 02_vectorize.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from taxonomy import DEFAULT_TEMPORAL_PREFIX, TEMPORAL_SOURCE_PREFIXES

INPUT_CHUNKS = Path("logs/chunks_enriched.json")
OUTPUT_CHUNKS = Path("logs/chunks_vectorized.json")
STATS_PATH = Path("logs/vectorization_report.json")
_LOGS_DIR = Path("logs")

_MODELS_DIR = Path(__file__).parent / "models"
_DEFAULT_MODEL_ID = os.environ.get("RAG_EMBEDDING_MODEL", "EmbeddingGemma")

# Si el modelo está en models/<nombre>, usamos esa ruta local directamente.
_LOCAL_MODEL_PATH = _MODELS_DIR / _DEFAULT_MODEL_ID
if _LOCAL_MODEL_PATH.exists():
    DEFAULT_MODEL = str(_LOCAL_MODEL_PATH)
else:
    DEFAULT_MODEL = _DEFAULT_MODEL_ID

FALLBACK_MODEL = "intfloat/multilingual-e5-small"
BATCH_SIZE = 32
USE_METADATA_CONTEXT = os.environ.get("RAG_PURE_TEXT", "0") != "1"

# Los modelos E5 requieren prefijo "passage: " para documentos y "query: " para queries.
# Gemma Embedding no requiere prefijos especiales.
E5_PASSAGE_PREFIX = "passage: "


def is_e5_model(name: str) -> bool:
    return "e5" in name.lower()


def build_embed_text(chunk: dict, model_name: str) -> str:
    """Construye el texto que va al modelo: prefijo temporal + contexto + texto."""
    base = chunk["text"]

    if USE_METADATA_CONTEXT:
        parts = [chunk.get("doc_type_category", ""), chunk.get("section_type", "")]
        var_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        vars_sorted = sorted(
            chunk.get("economic_variables", {}).items(),
            key=lambda kv: var_order.get(kv[1].get("importance", "MEDIUM"), 3),
        )[:3]
        if vars_sorted:
            parts.append(", ".join(v[0] for v in vars_sorted))

        # Prefijo temporal diferenciado por tipo de fuente
        doc_type = chunk.get("doc_type_category", "")
        temporal_prefix = TEMPORAL_SOURCE_PREFIXES.get(doc_type, DEFAULT_TEMPORAL_PREFIX)
        date_str = chunk.get("document_date") or chunk.get("chunk_date") or ""
        if temporal_prefix == "[DAILY]":
            temporal_prefix = f"[DAILY {date_str}]" if date_str else "[DAILY]"
        elif date_str and len(date_str) >= 7:
            temporal_prefix = temporal_prefix.replace("]", f" {date_str[:7]}]")

        # Indicar tipo de indicador de la variable más importante
        top_indicator = vars_sorted[0][1].get("indicator_type") if vars_sorted else None
        indicator_suffix = ""
        if top_indicator == "LEADING":
            indicator_suffix = "→LEADING"
        elif top_indicator == "LAGGING":
            indicator_suffix = "→LAGGING"

        ctx = temporal_prefix + " [" + " | ".join(p for p in parts if p) + "]" + indicator_suffix
        base = f"{ctx} {base}"

    if is_e5_model(model_name):
        base = E5_PASSAGE_PREFIX + base
    return base


def _period_contains_date(excel_date: str, pdf_date: str, pdf_doc_type: str) -> bool:
    """True si pdf_date's period (mes o trimestre) contiene excel_date."""
    if not excel_date or not pdf_date or len(excel_date) < 7 or len(pdf_date) < 7:
        return False
    try:
        ex_ym = excel_date[:7]
        pdf_ym = pdf_date[:7]
        if pdf_doc_type in ("IPOM", "IEF"):
            ex_q = (int(ex_ym[5:7]) - 1) // 3
            pdf_q = (int(pdf_ym[5:7]) - 1) // 3
            return ex_ym[:4] == pdf_ym[:4] and ex_q == pdf_q
        return ex_ym == pdf_ym
    except (ValueError, IndexError):
        return False


def _build_cross_references(
    excel_records: list[dict],
    pdf_records: list[dict],
) -> dict:
    """Genera índice bidireccional de referencias entre daily y period records."""
    excel_to_period: dict[str, list] = {}
    period_to_daily: dict[str, list] = {}

    for ex in excel_records:
        ex_id = ex.get("chunk_id", "")
        ex_date = ex.get("chunk_date") or ex.get("document_date") or ""
        ex_vars = set(ex.get("economic_variables", {}).keys())
        refs: list[dict] = []
        for pdf in pdf_records:
            pdf_id = pdf.get("chunk_id", "")
            pdf_date = pdf.get("document_date") or ""
            pdf_type = pdf.get("doc_type_category", "")
            pdf_vars = set(pdf.get("economic_variables", {}).keys())
            shared = sorted(ex_vars & pdf_vars)
            if shared and _period_contains_date(ex_date, pdf_date, pdf_type):
                refs.append({
                    "record_id": pdf_id,
                    "pointer_type": "period_context",
                    "shared_vars": shared,
                })
                period_to_daily.setdefault(pdf_id, []).append({
                    "record_id": ex_id,
                    "pointer_type": "daily_example",
                    "shared_vars": shared,
                })
        if refs:
            excel_to_period[ex_id] = refs

    return {"excel_to_period": excel_to_period, "period_to_daily": period_to_daily}


def load_model(name: str):
    from sentence_transformers import SentenceTransformer

    print(f"[02] Cargando modelo: {name}")
    return SentenceTransformer(name)


def main() -> int:
    if not INPUT_CHUNKS.exists():
        print(f"❌ Falta {INPUT_CHUNKS}. Corre: python3 01_enrich_metadata.py",
              file=sys.stderr)
        return 1

    try:
        import numpy as np
    except ImportError:
        print("❌ Falta numpy. pip install numpy", file=sys.stderr)
        return 1

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print("❌ Falta sentence-transformers. pip install sentence-transformers torch",
              file=sys.stderr)
        return 1

    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Cargar modelo con fallback
    model = None
    model_name_used = None
    for candidate in (DEFAULT_MODEL, FALLBACK_MODEL):
        try:
            model = load_model(candidate)
            model_name_used = candidate
            break
        except Exception as e:
            print(f"[02] ⚠ No se pudo cargar {candidate}: {e}")
    if model is None:
        print("❌ No se pudo cargar ningún modelo de embeddings.", file=sys.stderr)
        return 1

    max_seq_len = getattr(model, "max_seq_length", 256)
    tokenizer = getattr(model, "tokenizer", None)
    dim = model.get_sentence_embedding_dimension()
    print(f"[02] ✓ Modelo listo — dim={dim}, max_seq_length={max_seq_len}, "
          f"context_prefix={'ON' if USE_METADATA_CONTEXT else 'OFF'}")

    # Construir textos y medir truncamiento real
    embed_texts = [build_embed_text(c, model_name_used) for c in chunks]

    truncated = 0
    token_counts = []
    if tokenizer is not None:
        for text in embed_texts:
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            token_counts.append(len(ids))
            if len(ids) > max_seq_len:
                truncated += 1

    if truncated:
        pct = 100 * truncated // len(chunks)
        print(f"[02] ⚠ {truncated} chunks ({pct}%) exceden {max_seq_len} tokens y serán truncados")

    # Vectorizar (normalize_embeddings=True para que L2-norm=1 y dot=cosine)
    print(f"[02] Vectorizando {len(chunks)} chunks (batch={BATCH_SIZE})...")
    embeddings = model.encode(
        embed_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Guardar
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb.tolist()
        c["embedding_dim"] = dim
        c["embedding_model"] = model_name_used

    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    # Reporte
    report = {
        "model": model_name_used,
        "dimension": dim,
        "max_seq_length": max_seq_len,
        "total_chunks": len(chunks),
        "truncated_count": truncated,
        "metadata_context_enabled": USE_METADATA_CONTEXT,
        "token_stats": {
            "min": min(token_counts) if token_counts else 0,
            "mean": sum(token_counts) // len(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
        } if token_counts else None,
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    size_mb = OUTPUT_CHUNKS.stat().st_size / (1024 * 1024)
    print(f"[02] ✓ {OUTPUT_CHUNKS} ({size_mb:.1f} MB)")
    print(f"[02] ✓ {STATS_PATH}")
    if token_counts:
        print(f"[02] tokens por chunk: min={report['token_stats']['min']} "
              f"mean={report['token_stats']['mean']} max={report['token_stats']['max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
