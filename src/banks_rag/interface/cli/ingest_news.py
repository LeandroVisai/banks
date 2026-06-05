"""CLI ``banks-ingest-news`` — ingesta del corpus de **contexto actual**.

Pipeline paralelo al de ``banks-ingest`` pero para las noticias scrapeadas, y
contra una **base de datos AISLADA** (``contexto_actual`` por defecto, env
``BANKS_CONTEXT_DB``): así el agente nunca cruza el corpus del banco con la
prensa por error.

    extract_news → enrich_news (taxonomía noticiera) → vectorize (Qwen3-VL) →
    persist (DB contexto_actual)

La ingesta es **aditiva e idempotente**: re-correr sobre los mismos JSON no
duplica (``document_id`` estable por artículo) y NO purga lo viejo. La frescura
la decide el agente en query-time con el filtro por fecha + recency.

Subcomandos:
  ``banks-ingest-news full``   end-to-end (default).
  ``banks-ingest-news setup``  crea/valida la base aislada + schema + índices.
  ``banks-ingest-news stats``  métricas de la base de contexto.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer

from banks_rag.application.ingestion import (
    corpus_stats,
    detect_embedding_dim,
    enrich_news,
    extract_news,
    persist_corpus,
    setup_corpus,
    vectorize_corpus,
)
from banks_rag.config import get_settings, paths
from banks_rag.domain.documents import ChunkKind

app = typer.Typer(
    name="banks-ingest-news",
    help="Ingesta de noticias (contexto actual) en la base Postgres aislada.",
    no_args_is_help=True,
)


def _context_repo():
    """``PostgresRepo`` apuntando a la base AISLADA de contexto (sin prefijo)."""
    from banks_rag.infrastructure.persistence import PostgresRepo

    return PostgresRepo(prefix="", database=get_settings().context_db)


def _enriched_to_dict(e) -> dict:
    """EnrichedChunk → dict (ChunkKind enum → string), como en banks-ingest."""
    d = asdict(e)
    if isinstance(d.get("kind"), ChunkKind):
        d["kind"] = d["kind"].value
    return d


@app.command("full")
def cmd_full(
    source: Path = typer.Option(
        paths.NEWS_RAW_DIR,
        "--source", "-s",
        help="Directorio (o archivo) con los JSON diarios 'noticias_YYYY_MM_DD.json'.",
    ),
    batch_size: int = typer.Option(
        4, "--batch-size", help="Batch size para vectorize (Qwen3: 4–8).",
    ),
) -> None:
    """extract → enrich → vectorize → persist en la base de contexto aislada."""
    if not source.exists():
        typer.secho(f"❌ origen no existe: {source}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    settings = get_settings()
    typer.echo(f"[news] origen: {source}")
    typer.echo(f"[news] base aislada: '{settings.context_db}' (tablas documents/chunks)")

    # ── extract ───────────────────────────────────────────────────────────────
    ext = extract_news(source)
    if not ext.documents:
        typer.secho(
            "❌ no se extrajo ninguna noticia (¿nombres sin fecha o JSON vacíos?).",
            fg=typer.colors.RED, err=True,
        )
        if ext.report.files_skipped:
            typer.secho(f"   archivos saltados: {ext.report.files_skipped}",
                        fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    typer.secho(
        f"[news] extract ✓ {ext.report.files_processed} archivos → "
        f"{ext.report.documents_count} noticias, {ext.report.chunks_count} chunks",
        fg=typer.colors.GREEN,
    )
    if ext.report.files_skipped:
        typer.secho(f"[news] ⚠ saltados (sin fecha/ilegibles): {ext.report.files_skipped}",
                    fg=typer.colors.YELLOW)

    # ── enrich (taxonomía noticiera) ──────────────────────────────────────────
    enr = enrich_news(ext.documents, ext.chunks)
    if enr.report:
        sent = enr.report.sentiment_histogram
        typer.secho(
            f"[news] enrich ✓ importance media={enr.report.avg_importance} · "
            f"sentimiento={sent}",
            fg=typer.colors.GREEN,
        )

    doc_dicts = [asdict(d) for d in ext.documents]
    chunk_dicts = [_enriched_to_dict(e) for e in enr.enriched_chunks]

    # ── vectorize (mismo Qwen3-VL del corpus del banco) ───────────────────────
    typer.echo("[news] cargando embedder...")
    from banks_rag.infrastructure.embeddings import build_default_embedder

    embedder = build_default_embedder()
    embedder.encode_text(["warmup"], batch_size=1)
    typer.secho(
        f"[news] embedder ✓ {embedder.name} (dim={embedder.dim})", fg=typer.colors.GREEN,
    )
    typer.echo(f"[news] vectorizando {len(chunk_dicts)} chunks (batch={batch_size})...")
    vec = vectorize_corpus(chunk_dicts, embedder, batch_size=batch_size)

    # ── persist (base aislada) ────────────────────────────────────────────────
    repo = _context_repo()
    # Dim real: usa la longitud del primer embedding producido (más fiable que
    # embedding_dim, que puede quedar desactualizado si se cambia el modelo).
    dim = detect_embedding_dim(vec.chunks)
    for _c in vec.chunks:
        _emb = _c.get("embedding")
        if _emb:
            dim = len(_emb)
            break

    try:
        existing_dim = repo.existing_embedding_dim()
        needs_reset = existing_dim is not None and existing_dim != dim
        if needs_reset:
            typer.secho(
                f"[news] ⚠ schema tiene vector({existing_dim}), "
                f"embeddings son vector({dim}) → recreando schema...",
                fg=typer.colors.YELLOW,
            )
        setup_corpus(repo, dim=dim, drop_first=needs_reset)
        result = persist_corpus(repo, doc_dicts, vec.chunks, expected_dim=dim)
    except (RuntimeError, ValueError) as e:
        typer.secho(f"[news] ❌ {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    typer.secho(
        f"[news] persist ✓ {result.documents_upserted} noticias, "
        f"{result.chunks_inserted} chunks → '{settings.context_db}'",
        fg=typer.colors.GREEN,
    )
    if result.chunks_skipped_invalid_dim:
        typer.secho(
            f"[news] ⚠ {result.chunks_skipped_invalid_dim} chunks descartados "
            "(dimensión inválida)", fg=typer.colors.YELLOW,
        )
    _print_stats(corpus_stats(repo))


@app.command("setup")
def cmd_setup(
    reset: bool = typer.Option(
        False, "--reset", help="DROP CASCADE + recrear (destructivo).",
    ),
) -> None:
    """Crea/valida la base de contexto aislada + extensión vector + schema."""
    repo = _context_repo()
    # setup no tiene chunks disponibles: usa RAG_EMBEDDING_DIM si está fijado,
    # si no fuerza 4096 (Qwen3-VL-Embedding-8B). Nunca cae al fallback de 1024.
    import os as _os
    dim = int(_os.getenv("RAG_EMBEDDING_DIM") or 4096)
    try:
        setup_corpus(repo, dim=dim, drop_first=reset)
    except RuntimeError as e:
        typer.secho(f"[news] ❌ {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    typer.secho(
        f"[news] ✓ base '{repo.database}' {'reseteada' if reset else 'lista'} "
        f"(dim={dim})", fg=typer.colors.GREEN,
    )


@app.command("stats")
def cmd_stats() -> None:
    """Métricas de la base de contexto actual."""
    repo = _context_repo()
    try:
        _print_stats(corpus_stats(repo))
    except Exception as e:
        typer.secho(f"[news] ❌ no pude consultar '{repo.database}': {e}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


def _print_stats(stats: dict) -> None:
    typer.echo(f"\n[news] STATS '{stats['database']}':")
    typer.echo(f"        noticias (documents): {stats['documents']}")
    typer.echo(f"        chunks:               {stats['chunks']}")
    if stats.get("by_doc_type"):
        for t, n in stats["by_doc_type"]:
            typer.echo(f"          {t:<20} {n}")


if __name__ == "__main__":
    app()
