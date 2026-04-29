#!/usr/bin/env python3
"""
compare_models.py — Compara calidad semántica de embeddings Qwen3 vs EmbeddingGemma.

La comparación se hace sobre PDFs únicamente (corpus común a ambos modelos),
descartando el Monitor PM del lado de Qwen para equiparar el corpus.

Métricas calculadas (todas sobre calidad semántica, no volumen):
  1. Cohesión intra-sección    — similitud coseno promedio entre chunks del mismo section_type
  2. Silhouette por sección    — separación inter-sección en el espacio vectorial
  3. Silhouette por doc_type   — el modelo distingue géneros de documentos
  4. Sharpness de retrieval    — ratio sim(top-1) / sim(top-10): discriminación de top result
  5. Distribución de similitudes — std alta = espacio mejor distribuido

Salida: model_comparison_report.json + tabla en consola
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

BASE        = Path(__file__).parent
QWEN_PATH   = BASE / "logs_qwen"  / "chunks_vectorized.json"
GEMMA_PATH  = BASE / "logs_gemma" / "chunks_vectorized.json"
REPORT_PATH = BASE / "model_comparison_report.json"

RANDOM_SEED      = 42
SHARPNESS_N      = 60    # chunks usados como queries para medir sharpness
MAX_PER_SECTION  = 150   # subsample por sección para controlar tiempo
MIN_SECTION_SIZE = 5     # mínimo para calcular cohesión de sección


# ── Utilidades ───────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    size_mb = path.stat().st_size / 1e6
    print(f"  Cargando {path.parent.name}/{path.name} ({size_mb:.1f} MB)...", end="", flush=True)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f" {len(data)} chunks OK")
    return data


def filter_pdfs(chunks: list[dict]) -> list[dict]:
    """Filtra Monitor PM (chunk_date != null) para comparación justa."""
    return [c for c in chunks if c.get("chunk_date") is None]


def to_embeddings(chunks: list[dict]) -> np.ndarray:
    """Extrae y renormaliza embeddings L2 (por si tienen drift numérico)."""
    emb = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return emb / norms


# ── Métricas ─────────────────────────────────────────────────────────────────

def intra_section_cohesion(chunks: list[dict], emb: np.ndarray) -> tuple[dict[str, float], float]:
    """Similitud coseno promedio entre pares de la misma sección."""
    by_section: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_section[c["section_type"]].append(i)

    results: dict[str, float] = {}
    for sec, idxs in by_section.items():
        if len(idxs) < MIN_SECTION_SIZE:
            continue
        if len(idxs) > MAX_PER_SECTION:
            random.seed(RANDOM_SEED)
            idxs = random.sample(idxs, MAX_PER_SECTION)
        mat = emb[idxs]
        sim_mat = mat @ mat.T
        n = len(idxs)
        avg = (sim_mat.sum() - n) / (n * (n - 1))
        results[sec] = float(avg)

    overall = float(np.mean(list(results.values()))) if results else 0.0
    return results, overall


def silhouette(emb: np.ndarray, labels: list[str], name: str) -> float | None:
    if not HAS_SKLEARN:
        return None
    unique = list(set(labels))
    if len(unique) < 2:
        return None
    label_ids = [unique.index(l) for l in labels]
    sample_size = min(2000, len(emb))
    try:
        return float(silhouette_score(emb, label_ids, sample_size=sample_size, random_state=RANDOM_SEED))
    except Exception as e:
        print(f"    [warn] silhouette({name}): {e}")
        return None


def retrieval_sharpness(emb: np.ndarray, n_queries: int = SHARPNESS_N, top_k: int = 10) -> float:
    """
    Para cada query-chunk, calcula sim(top-1) / sim(top-k).
    Ratio alto = el modelo discrimina claramente el resultado más relevante.
    """
    random.seed(RANDOM_SEED)
    n = len(emb)
    query_idxs = random.sample(range(n), min(n_queries, n))
    ratios: list[float] = []
    for qi in query_idxs:
        sims = emb[qi] @ emb.T
        sims[qi] = -2.0          # excluir self-match
        top = np.sort(sims)[::-1][:top_k]
        if len(top) == top_k and top[-1] > 1e-6:
            ratios.append(float(top[0] / top[-1]))
    return float(np.mean(ratios)) if ratios else 0.0


def sim_distribution(emb: np.ndarray, sample_size: int = 300) -> dict[str, float]:
    """Estadísticas de similitudes off-diagonal en una muestra."""
    random.seed(RANDOM_SEED)
    n = len(emb)
    idxs = random.sample(range(n), min(sample_size, n))
    mat = emb[idxs] @ emb[idxs].T
    flat = mat.flatten()
    off_diag = flat[flat < 0.9999]
    return {
        "mean": float(np.mean(off_diag)),
        "std":  float(np.std(off_diag)),
        "p90":  float(np.percentile(off_diag, 90)),
    }


# ── Análisis por modelo ───────────────────────────────────────────────────────

def analyze(label: str, chunks: list[dict]) -> dict:
    print(f"\n{'─'*60}")
    print(f"  [{label}]  total chunks: {len(chunks)}")

    pdf_chunks = filter_pdfs(chunks)
    monitor_pm = len(chunks) - len(pdf_chunks)
    print(f"  PDFs (sin Monitor PM): {len(pdf_chunks)}   Monitor PM filtrado: {monitor_pm}")

    emb = to_embeddings(pdf_chunks)
    dim = emb.shape[1]
    print(f"  Dimensión: {dim}")

    print("  → cohesión intra-sección...", end="", flush=True)
    coh_by_sec, avg_cohesion = intra_section_cohesion(pdf_chunks, emb)
    print(f"  avg={avg_cohesion:.4f}")

    section_labels = [c["section_type"] for c in pdf_chunks]
    doctype_labels = [c["doc_type_category"] for c in pdf_chunks]

    print("  → silhouette por sección...", end="", flush=True)
    sil_sec = silhouette(emb, section_labels, "section")
    print(f"  {sil_sec:.4f}" if sil_sec is not None else "  (sklearn no disponible)")

    print("  → silhouette por doc_type...", end="", flush=True)
    sil_doc = silhouette(emb, doctype_labels, "doctype")
    print(f"  {sil_doc:.4f}" if sil_doc is not None else "  (sklearn no disponible)")

    print("  → sharpness de retrieval...", end="", flush=True)
    sharp = retrieval_sharpness(emb)
    print(f"  {sharp:.4f}")

    print("  → distribución de similitudes...", end="", flush=True)
    sim_dist = sim_distribution(emb)
    print(f"  mean={sim_dist['mean']:.4f}  std={sim_dist['std']:.4f}")

    section_counts = defaultdict(int)
    for c in pdf_chunks:
        section_counts[c["section_type"]] += 1

    return {
        "model": label,
        "embedding_dim": dim,
        "total_chunks_file": len(chunks),
        "pdf_chunks_compared": len(pdf_chunks),
        "monitor_pm_excluded": monitor_pm,
        "sections_detected": dict(sorted(section_counts.items())),
        "avg_intra_section_cohesion": round(avg_cohesion, 4),
        "cohesion_by_section": {k: round(v, 4) for k, v in sorted(coh_by_sec.items())},
        "silhouette_by_section": round(sil_sec, 4) if sil_sec is not None else None,
        "silhouette_by_doctype": round(sil_doc, 4) if sil_doc is not None else None,
        "retrieval_sharpness": round(sharp, 4),
        "sim_distribution": {k: round(v, 4) for k, v in sim_dist.items()},
    }


# ── Tabla comparativa ─────────────────────────────────────────────────────────

def print_table(q: dict, g: dict) -> None:
    W = 64
    print("\n" + "═" * W)
    print("  COMPARACIÓN DE MODELOS DE EMBEDDING")
    print("═" * W)

    def winner_marker(qv, gv, higher_is_better: bool = True) -> tuple[str, str]:
        q_better = (qv > gv) if higher_is_better else (qv < gv)
        return ("✓", " ") if q_better else (" ", "✓")

    rows: list[tuple[str, str, str, bool]] = [
        ("Dimensión", str(q["embedding_dim"]), str(g["embedding_dim"]), True),
        ("Chunks PDF comparados", str(q["pdf_chunks_compared"]), str(g["pdf_chunks_compared"]), True),
        ("Cohesión intra-sección ↑", f"{q['avg_intra_section_cohesion']:.4f}", f"{g['avg_intra_section_cohesion']:.4f}", True),
        ("Silhouette sección ↑",
            f"{q['silhouette_by_section']:.4f}" if q['silhouette_by_section'] is not None else "N/A",
            f"{g['silhouette_by_section']:.4f}" if g['silhouette_by_section'] is not None else "N/A",
            True),
        ("Silhouette doc_type ↑",
            f"{q['silhouette_by_doctype']:.4f}" if q['silhouette_by_doctype'] is not None else "N/A",
            f"{g['silhouette_by_doctype']:.4f}" if g['silhouette_by_doctype'] is not None else "N/A",
            True),
        ("Sharpness retrieval ↑", f"{q['retrieval_sharpness']:.4f}", f"{g['retrieval_sharpness']:.4f}", True),
        ("Similitud media (↓ mejor)", f"{q['sim_distribution']['mean']:.4f}", f"{g['sim_distribution']['mean']:.4f}", False),
        ("Std similitudes ↑", f"{q['sim_distribution']['std']:.4f}", f"{g['sim_distribution']['std']:.4f}", True),
    ]

    header = f"  {'Métrica':<36} {'Qwen3':>10} {'Gemma':>10}"
    print(header)
    print("-" * W)
    for label, qv, gv, hib in rows:
        try:
            qfloat, gfloat = float(qv), float(gv)
            qm, gm = winner_marker(qfloat, gfloat, hib)
        except ValueError:
            qm, gm = " ", " "
        print(f"  {label:<36} {qm}{qv:>9} {gm}{gv:>9}")


# ── Decisión ─────────────────────────────────────────────────────────────────

def decide(q: dict, g: dict) -> tuple[str, str]:
    """Elige ganador por puntos ponderados. Prioriza calidad semántica."""
    scores = {"qwen": 0, "gemma": 0}
    details: list[str] = []

    def compare(metric_name: str, qv, gv, weight: int, higher_is_better: bool = True):
        if qv is None or gv is None:
            return
        q_wins = (qv > gv) if higher_is_better else (qv < gv)
        winner_label = "qwen" if q_wins else "gemma"
        scores[winner_label] += weight
        direction = "↑" if higher_is_better else "↓"
        details.append(
            f"{metric_name}{direction}: {'qwen' if q_wins else 'gemma'} "
            f"({qv:.4f} vs {gv:.4f}, peso={weight})"
        )

    compare("cohesión-sección",   q["avg_intra_section_cohesion"],  g["avg_intra_section_cohesion"],  weight=3)
    compare("silhouette-sección", q["silhouette_by_section"],        g["silhouette_by_section"],        weight=3)
    compare("silhouette-doctype", q["silhouette_by_doctype"],        g["silhouette_by_doctype"],        weight=2)
    compare("sharpness",          q["retrieval_sharpness"],          g["retrieval_sharpness"],          weight=3)
    compare("std-similitudes",    q["sim_distribution"]["std"],      g["sim_distribution"]["std"],      weight=1)
    # Similitud media baja = mejor discriminación (higher_is_better=False)
    compare("sim-media",          q["sim_distribution"]["mean"],     g["sim_distribution"]["mean"],     weight=1, higher_is_better=False)

    winner = max(scores, key=lambda k: scores[k])
    reason = f"Puntos: qwen={scores['qwen']} gemma={scores['gemma']}. " + " | ".join(details)
    return winner, reason


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    W = 64
    print("=" * W)
    print("  compare_models.py")
    print("  Qwen3-Embedding-0.6B  vs  EmbeddingGemma")
    print("  Base: PDFs únicamente (corpus común, excluye Monitor PM)")
    print("=" * W)

    for path in (QWEN_PATH, GEMMA_PATH):
        if not path.exists():
            print(f"\nERROR: no encontrado → {path}")
            print("  Ejecuta el pipeline completo primero:")
            print("  python3 run.py full")
            sys.exit(1)

    if not HAS_SKLEARN:
        print("\n[warn] sklearn no disponible — métricas silhouette omitidas")
        print("       pip install scikit-learn  para activarlas\n")

    print("\n[1/3] Cargando archivos")
    qwen_raw  = load_chunks(QWEN_PATH)
    gemma_raw = load_chunks(GEMMA_PATH)

    print("\n[2/3] Calculando métricas")
    qwen_m  = analyze("Qwen3-Embedding-0.6B",  qwen_raw)
    gemma_m = analyze("EmbeddingGemma", gemma_raw)

    print("\n[3/3] Comparación")
    print_table(qwen_m, gemma_m)

    winner, reason = decide(qwen_m, gemma_m)

    report = {
        "evaluation_basis": "PDFs only (18 documentos comunes, Monitor PM excluido para comparación justa)",
        "qwen": qwen_m,
        "gemma": gemma_m,
        "winner": winner,
        "reason": reason,
        "next_step": (
            f"Cargar logs_{winner}/chunks_vectorized.json en PostgreSQL: "
            "python3 run.py step 3 reset"
            + (
                " — NOTA: arreglar bug openpyxl y re-generar Gemma para incluir Monitor PM antes de cargar."
                if winner == "gemma" else ""
            )
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n" + "═" * W)
    print(f"  GANADOR: {winner.upper()}")
    print(f"  {reason[:W - 4]}")
    print(f"\n  Reporte completo: {REPORT_PATH.name}")
    print("═" * W)


if __name__ == "__main__":
    main()
