"""CLI ``banks-search`` — búsqueda híbrida sobre el corpus indexado."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from banks_rag.application.retrieval import (
    MONTH_NUM_TO_NAME,
    hybrid_search,
)
from banks_rag.config import paths
from banks_rag.domain.retrieval import SearchFilters
from banks_rag.infrastructure.embeddings import build_default_embedder
from banks_rag.infrastructure.persistence import PostgresRepo

app = typer.Typer(
    name="banks-search",
    help="Búsqueda híbrida (vector + BM25 → RRF → MMR → importance boost).",
    no_args_is_help=True,
)


def _format_text_output(result, k: int) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"QUERY: {result.query}")
    if result.clean_query != result.query:
        lines.append(f"  clean: {result.clean_query}")

    pf = result.parsed_filters or {}
    filters_repr = []
    if pf.get("exact_date"):
        filters_repr.append(f"fecha={pf['exact_date']}")
    else:
        if pf.get("year_from") is not None:
            yr = (
                f"{pf['year_from']}"
                if pf.get("year_from") == pf.get("year_to")
                else f"{pf['year_from']}–{pf['year_to']}"
            )
            filters_repr.append(f"año={yr}")
        if pf.get("month") is not None:
            filters_repr.append(
                f"mes={MONTH_NUM_TO_NAME.get(pf['month'], pf['month'])}"
            )
    for fld in ("doc_types", "variables", "sections", "institutions",
                "entities", "tags", "exclude_doc_types", "exclude_institutions"):
        if pf.get(fld):
            filters_repr.append(f"{fld}={','.join(pf[fld])}")
    if pf.get("min_importance", 0) > 0 or pf.get("max_importance", 1) < 1:
        filters_repr.append(
            f"imp={pf.get('min_importance', 0):.2f}-{pf.get('max_importance', 1):.2f}"
        )
    if pf.get("exclude_boilerplate"):
        filters_repr.append("no-boilerplate")
    if filters_repr:
        lines.append(f"  filtros: {' | '.join(filters_repr)}")
    lines.append("=" * 80)

    if not result.hits:
        lines.append("\nSin resultados.")
        return "\n".join(lines)

    if all(h.get("low_confidence") for h in result.hits):
        lines.append(
            "⚠  Resultados con baja confianza — la query puede estar fuera del dominio."
        )

    for i, r in enumerate(result.hits, start=1):
        vars_str = ",".join(r.get("economic_variables", {}).keys()) or "-"
        nums = r.get("numeric_values") or []
        nums_str = ", ".join(n.get("raw", "") for n in nums[:3]) or "-"
        tags_str = ",".join(r.get("tags") or []) or "-"

        lines.append(
            f"\n[{i}] {r['filename']} p.{r['page_start']}"
            + (f"-{r['page_end']}" if r['page_end'] != r['page_start'] else "")
            + f" · {r['doc_type_category']} · {r['section_type']}"
        )
        effective_date = r.get("chunk_date") or r.get("document_date") or "-"
        lines.append(
            f"    fecha: {effective_date} | imp: {r['importance_score']:.2f} | "
            f"rrf: {r.get('rrf_score', 0):.3f} | final: {r.get('final_score', 0):.3f}"
        )
        lines.append(f"    vars: {vars_str}")
        lines.append(f"    datos: {nums_str}")
        lines.append(f"    tags: {tags_str}")
        if r.get("image_path"):
            lines.append(f"    [imagen: {r['image_path']}]")
        text = r["text"].replace("\n", " ")
        lines.append(f"    > {text[:400]}" + ("…" if len(text) > 400 else ""))

    return "\n".join(lines)


def _format_json_output(result, k: int) -> str:
    def clean(r: dict) -> dict:
        return {
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "filename": r["filename"],
            "doc_type_category": r["doc_type_category"],
            "document_date": r.get("document_date"),
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "section_type": r["section_type"],
            "importance_score": float(r["importance_score"]),
            "rrf_score": float(r.get("rrf_score", 0)),
            "final_score": float(r.get("final_score", 0)),
            "economic_variables": r.get("economic_variables", {}),
            "numeric_values": r.get("numeric_values", []),
            "tags": r.get("tags") or [],
            "image_path": r.get("image_path"),
            "text": r["text"],
        }

    payload = {
        "query": result.query,
        "clean_query": result.clean_query,
        "filters": result.parsed_filters,
        "results": [clean(h) for h in result.hits],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


@app.command()
def main(
    query: str = typer.Argument(..., help="Consulta en lenguaje natural."),
    k: int = typer.Argument(5, help="Número de resultados (default 5)."),
    no_mmr: bool = typer.Option(
        False, "--no-mmr",
        help="Desactiva MMR (más relevancia, menos diversidad).",
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="Formato de salida JSON.",
    ),
    table_prefix: str = typer.Option(
        "",
        "--table-prefix", envvar="RAG_TABLE_PREFIX",
        help="Prefijo de tablas (e.g. 'qwen_').",
    ),
    config_file: Path | None = typer.Option(
        None, "--config",
        help="Archivo JSON con filtros avanzados (years, institutions, tags...).",
    ),
    institutions: str = typer.Option(
        "", "--institution",
        help="Instituciones (coma-separadas): BANCO_CENTRAL_CHILE, FEDERAL_RESERVE...",
    ),
    tags: str = typer.Option(
        "", "--tags",
        help="Tags (coma-separados): DECISION_POLITICA, FORWARD_LOOKING...",
    ),
    min_importance: float = typer.Option(0.0, "--min-importance"),
    exclude_boilerplate: bool = typer.Option(False, "--exclude-boilerplate"),
) -> None:
    """Ejecuta hybrid_search y formatea el resultado."""
    extra = SearchFilters()
    if config_file:
        config_data = json.loads(config_file.read_text(encoding="utf-8"))
        # Map a SearchFilters
        for f, val in config_data.items():
            if hasattr(extra, f):
                setattr(extra, f, val)

    if institutions:
        extra.institutions = [s.strip() for s in institutions.split(",") if s.strip()]
    if tags:
        extra.tags = [s.strip() for s in tags.split(",") if s.strip()]
    if min_importance > 0:
        extra.min_importance = min_importance
    if exclude_boilerplate:
        extra.exclude_boilerplate = True

    repo = PostgresRepo(prefix=table_prefix)
    embedder = build_default_embedder()

    result = hybrid_search(
        query, query_embedder=embedder, repo=repo, extra_filters=extra,
        k=k, use_mmr=not no_mmr,
    )

    if json_output:
        typer.echo(_format_json_output(result, k))
    else:
        typer.echo(_format_text_output(result, k))


if __name__ == "__main__":
    app()
