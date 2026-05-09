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

from banks_rag.application.ingestion import extract_corpus
from banks_rag.config import paths

app = typer.Typer(
    name="banks-ingest",
    help="Pipeline de ingesta del corpus RAG (extract → enrich → vectorize → persist).",
    no_args_is_help=True,
)


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
def cmd_enrich() -> None:
    """[Fase 1.b — pendiente] Enriquecimiento semántico de chunks.

    Por ahora delega al script legacy ``01_enrich_metadata.py``.
    """
    import subprocess
    typer.echo("[enrich] Delegando a 01_enrich_metadata.py (legacy hasta Fase 1.b)")
    rc = subprocess.run([sys.executable, "01_enrich_metadata.py"]).returncode
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("vectorize")
def cmd_vectorize() -> None:
    """[Fase 1.c — pendiente] Embeddings (texto + visuales).

    Por ahora delega al script legacy ``02_vectorize.py``.
    """
    import subprocess
    typer.echo("[vectorize] Delegando a 02_vectorize.py (legacy hasta Fase 1.c)")
    rc = subprocess.run([sys.executable, "02_vectorize.py"]).returncode
    if rc != 0:
        raise typer.Exit(code=rc)


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
