"""CLI ``banks-ingest`` — orquesta el pipeline de ingesta.

Subcomandos:

  ``banks-ingest extract``    extrae + chunkea PDFs/Excel → JSON
  ``banks-ingest enrich``     enriquecimiento semántico (Fase 1.b)
  ``banks-ingest vectorize``  embeddings (Fase 1.c)
  ``banks-ingest persist``    carga a PostgreSQL (Fase 1.d)
  ``banks-ingest full``       corre 0→3 en orden

Reemplaza al ``run.py`` legacy. Durante Fase 1 algunos subcomandos delegan
a los scripts numerados existentes mediante ``subprocess`` para mantener
paridad funcional.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import typer

from banks_rag.application.ingestion import (
    enrich_corpus,
    extract_corpus,
    vectorize_corpus,
)
from banks_rag.config import paths
from banks_rag.domain.documents import Chunk, ChunkKind, Document

app = typer.Typer(
    name="banks-ingest",
    help="Pipeline de ingesta del corpus RAG (extract → enrich → vectorize → persist).",
    no_args_is_help=True,
)


def _enriched_to_dict(e) -> dict:
    """Serializa EnrichedChunk a dict, convirtiendo ChunkKind enum a string."""
    from dataclasses import asdict
    d = asdict(e)
    if isinstance(d.get("kind"), ChunkKind):
        d["kind"] = d["kind"].value
    return d


@app.command("extract")
def cmd_extract(
    data_root: Path = typer.Option(
        paths.DATA_RAW_DIR,
        "--data-root", "-d",
        help="Directorio raíz con PDFs y Excel.",
    ),
    images_dir: Path = typer.Option(
        paths.IMAGES_DIR,
        "--images-dir",
        help="Dónde guardar los PNGs renderizados de páginas visuales.",
    ),
    output_dir: Path = typer.Option(
        paths.LOGS_DIR,
        "--output-dir",
        help="Dónde guardar documents.json, chunks.json, extraction_report.json.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suprime el log por archivo."),
) -> None:
    """Extrae PDFs/Excel y produce documents.json + chunks.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        typer.secho(f"❌ data_root no existe: {data_root}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[extract] Procesando {data_root}/ → {output_dir}/")

    def _on_complete(rel: str, doc, chunks: list) -> None:
        if quiet:
            return
        suffix = (
            f" [warnings: {','.join(doc.extraction_warnings)}]"
            if doc.extraction_warnings else ""
        )
        typer.echo(f"  ✓ {rel} — {len(chunks)} chunks, {doc.total_pages} páginas{suffix}")

    result = extract_corpus(data_root, images_dir, on_file_complete=_on_complete)

    docs_json = output_dir / "documents.json"
    chunks_json = output_dir / "chunks.json"
    report_json = output_dir / "extraction_report.json"

    docs_json.write_text(
        json.dumps([asdict(d) for d in result.documents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    chunks_json.write_text(
        json.dumps([asdict(c) for c in result.chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result.report:
        report_json.write_text(
            json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    typer.echo("")
    typer.secho(f"[extract] ✓ {docs_json} ({len(result.documents)} documentos)", fg=typer.colors.GREEN)
    typer.secho(f"[extract] ✓ {chunks_json} ({len(result.chunks)} chunks)", fg=typer.colors.GREEN)
    if result.report:
        typer.secho(f"[extract] ✓ {report_json}", fg=typer.colors.GREEN)
        stats = result.report.chunk_char_stats
        typer.echo(
            f"[extract] chunk size: min={stats['min']} "
            f"p50={stats['p50_median']} p90={stats['p90']} max={stats['max']}"
        )
        if not result.report.pymupdf_available:
            typer.secho(
                "[extract] ℹ  PyMuPDF no instalado — extracción visual omitida (pip install pymupdf)",
                fg=typer.colors.YELLOW,
            )
        if result.report.documents_failed:
            typer.secho(
                f"[extract] ⚠ {len(result.report.documents_failed)} archivos fallaron: "
                f"{result.report.documents_failed}",
                fg=typer.colors.YELLOW,
            )


@app.command("enrich")
def cmd_enrich(
    input_dir: Path = typer.Option(
        paths.LOGS_DIR,
        "--input-dir", "-i",
        help="Directorio con documents.json y chunks.json del paso extract.",
    ),
    output_dir: Path = typer.Option(
        paths.LOGS_DIR,
        "--output-dir", "-o",
        help="Dónde guardar chunks_enriched.json + enrichment_report.json.",
    ),
) -> None:
    """Enriquecimiento semántico de chunks (sección, variables, importance, tags)."""
    docs_json = input_dir / "documents.json"
    chunks_json = input_dir / "chunks.json"

    if not docs_json.exists() or not chunks_json.exists():
        typer.secho(
            f"❌ Faltan inputs en {input_dir}/. Corre primero: banks-ingest extract",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    docs_raw = json.loads(docs_json.read_text(encoding="utf-8"))
    chunks_raw = json.loads(chunks_json.read_text(encoding="utf-8"))

    documents = [Document(**{k: v for k, v in d.items() if k in Document.__dataclass_fields__})
                 for d in docs_raw]
    chunks = []
    for c in chunks_raw:
        # ChunkKind viene como string del JSON (o ausente en chunks legacy → default TEXT).
        kind_str = c.get("kind", ChunkKind.TEXT.value)
        chunk_kwargs = {k: v for k, v in c.items() if k in Chunk.__dataclass_fields__ and k != "kind"}
        chunks.append(Chunk(**chunk_kwargs, kind=ChunkKind(kind_str)))

    typer.echo(f"[enrich] Enriqueciendo {len(chunks)} chunks de {len(documents)} documentos...")

    result = enrich_corpus(documents, chunks)
    enriched = result.enriched_chunks
    report = result.report

    out_json = output_dir / "chunks_enriched.json"
    out_json.write_text(
        json.dumps([_enriched_to_dict(e) for e in enriched], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if report:
        report_json = output_dir / "enrichment_report.json"
        report_json.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    typer.secho(f"[enrich] ✓ {out_json} ({len(enriched)} chunks enriquecidos)",
                fg=typer.colors.GREEN)
    if report:
        typer.echo("[enrich] Secciones (top):")
        for section, count in sorted(
            report.section_histogram.items(), key=lambda x: -x[1]
        )[:10]:
            typer.echo(f"           {section:<22} {count}")
        typer.echo("[enrich] Variables económicas (top 10):")
        for var_name, count in sorted(
            report.variable_histogram.items(), key=lambda x: -x[1]
        )[:10]:
            typer.echo(f"           {var_name:<30} {count:>4}")
        pct = 100 * report.high_importance_count // max(1, report.chunks_total)
        typer.echo(
            f"[enrich] Importance: media={report.avg_importance:.3f}, "
            f"chunks ≥0.6: {report.high_importance_count} ({pct}%)"
        )


@app.command("vectorize")
def cmd_vectorize(
    input_dir: Path = typer.Option(
        paths.LOGS_DIR,
        "--input-dir", "-i",
        help="Directorio con chunks_enriched.json del paso enrich.",
    ),
    output_dir: Path = typer.Option(
        paths.LOGS_DIR,
        "--output-dir", "-o",
        help="Dónde guardar chunks_vectorized.json + records separados + cross_references.",
    ),
    batch_size: int = typer.Option(
        4, "--batch-size",
        help="Batch size para encode_text. Qwen3-Embedding-8B: 4–8. E5-small: 32+.",
    ),
) -> None:
    """Vectoriza chunks enriquecidos con el embedder configurado.

    Modelo según ``RAG_EMBEDDING_MODEL`` (default ``Qwen/Qwen3-Embedding``);
    fallback automático a ``intfloat/multilingual-e5-small``. Modelos
    pre-descargados se buscan en ``models/<owner>--<name>/`` (offline).
    """
    enriched_json = input_dir / "chunks_enriched.json"
    if not enriched_json.exists():
        typer.secho(
            f"❌ Falta {enriched_json}. Corre primero: banks-ingest enrich",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = json.loads(enriched_json.read_text(encoding="utf-8"))

    typer.echo(f"[vectorize] Cargando embedder...")
    from banks_rag.infrastructure.embeddings import build_default_embedder

    embedder = build_default_embedder()
    # Fuerza la carga (lazy) y muestra info del modelo.
    embedder.encode_text(["warmup"], batch_size=1)
    typer.secho(
        f"[vectorize] ✓ Modelo: {embedder.name} (dim={embedder.dim}, "
        f"max_seq={embedder.max_seq_length}, multimodal={embedder.is_multimodal})",
        fg=typer.colors.GREEN,
    )

    typer.echo(f"[vectorize] Vectorizando {len(chunks)} chunks (batch={batch_size})...")
    result = vectorize_corpus(chunks, embedder, batch_size=batch_size)

    chunks_out = output_dir / "chunks_vectorized.json"
    chunks_out.write_text(json.dumps(result.chunks, ensure_ascii=False), encoding="utf-8")

    excel_out = output_dir / "excel_daily_records.json"
    pdf_out = output_dir / "pdf_period_records.json"
    cross_out = output_dir / "cross_references.json"
    excel_out.write_text(
        json.dumps(result.excel_daily_records, ensure_ascii=False), encoding="utf-8"
    )
    pdf_out.write_text(
        json.dumps(result.pdf_period_records, ensure_ascii=False), encoding="utf-8"
    )
    cross_out.write_text(
        json.dumps(result.cross_references, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if result.report:
        report_out = output_dir / "vectorization_report.json"
        report_out.write_text(
            json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    size_mb = chunks_out.stat().st_size / (1024 * 1024)
    typer.secho(
        f"[vectorize] ✓ {chunks_out} ({size_mb:.1f} MB)", fg=typer.colors.GREEN
    )
    typer.echo(
        f"[vectorize] ✓ excel_daily={len(result.excel_daily_records)}, "
        f"pdf_period={len(result.pdf_period_records)}"
    )
    if result.report and result.report.image_chunks:
        mode = "VL" if result.report.vl_model else "texto fallback"
        typer.echo(
            f"[vectorize] ✓ {result.report.image_chunks} chunks de imagen embebidos ({mode})"
        )
    if result.report and result.report.token_stats:
        ts = result.report.token_stats
        typer.echo(
            f"[vectorize] tokens por chunk: min={ts['min']} mean={ts['mean']} max={ts['max']}"
        )
    if result.report and result.report.truncated_count:
        pct = 100 * result.report.truncated_count // max(1, result.report.text_chunks)
        typer.secho(
            f"[vectorize] ⚠ {result.report.truncated_count} chunks ({pct}%) "
            f"exceden {result.report.max_seq_length} tokens y serán truncados",
            fg=typer.colors.YELLOW,
        )


@app.command("persist")
def cmd_persist(
    mode: str = typer.Argument("load", help="setup | load | reset | stats"),
) -> None:
    """[Fase 1.d — pendiente] Carga a PostgreSQL.

    Por ahora delega al script legacy ``03_database.py``.
    """
    import subprocess
    typer.echo(f"[persist] Delegando a 03_database.py {mode} (legacy hasta Fase 1.d)")
    rc = subprocess.run([sys.executable, "03_database.py", mode]).returncode
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("full")
def cmd_full() -> None:
    """Corre el pipeline completo: extract → enrich → vectorize → persist (load)."""
    cmd_extract()
    cmd_enrich()
    cmd_vectorize()
    cmd_persist("load")


if __name__ == "__main__":
    app()
