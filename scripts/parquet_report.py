#!/usr/bin/env python3
"""Genera el informe descriptivo de datasets parquet (MARKDOWN + HTML).

Análogo a scripts/jarvis_news_report.py pero sobre el catálogo de parquets:
por cada dataset de la selección corre un mini-loop de tool-calling (el LLM
consulta los datos con execute_query / compute_variation / get_series_stats /
compute_composition / detect_anomaly) y redacta UN párrafo descriptivo; luego
una síntesis global sin tools encabeza el informe.

Salidas (subcarpetas de --out, default data/parquet_reports/):
    markdown/  reporte_<selector>_<fecha>.md
    html/      reporte_<selector>_<fecha>.html

Uso:
    python scripts/parquet_report.py --segment ffmm
    python scripts/parquet_report.py --segment "fondos mutuos" --windows 7d,30d,90d
    python scripts/parquet_report.py --datasets flujos_ffmm,duracion_ffmm
    python scripts/parquet_report.py --query "flujos de fondos mutuos"

Requisitos:
    - .env con BANKS_LLM_* (igual que la API). Con BANKS_LLM_BACKEND=openai_compat
      el informe paraleliza el map según BANKS_LLM_SERVER_SLOTS.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.application.reporting import (  # noqa: E402
    generate_parquet_report,
    render_parquet_report_html,
)
from banks_rag.config import get_settings  # noqa: E402


def _setup_logging(out: pathlib.Path) -> pathlib.Path:
    """Redirige todos los loggers a consola Y a un fichero en out/logs/."""
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"parquet_report_{datetime.date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def _write(path: pathlib.Path, data: str) -> pathlib.Path:
    """Escribe ``data`` creando la subcarpeta si hace falta. Devuelve la ruta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


def _build_llm(settings):
    """Engine según el backend del .env (mismo patrón que el lifespan de la API)."""
    if settings.llm_backend == "openai_compat":
        from banks_rag.infrastructure.llm.openai_compat_engine import OpenAICompatEngine

        return OpenAICompatEngine.from_settings(settings)
    from banks_rag.infrastructure.llm.llama_cpp_engine import LlamaCppEngine

    return LlamaCppEngine.from_settings(settings)


async def _run(args: argparse.Namespace) -> None:
    out = pathlib.Path(args.out)
    log_path = _setup_logging(out)
    log = logging.getLogger(__name__)
    log.info("Log de esta corrida: %s", log_path)

    settings = get_settings()
    if settings.llm_family in ("mock", ""):
        print("WARN BANKS_LLM_FAMILY es 'mock'/vacío: configura qwen + el backend en .env.")
        return

    concurrency = args.concurrency
    if concurrency <= 0:  # auto: paralelo solo si el servidor tiene slots reales
        concurrency = settings.llm_server_slots if settings.llm_backend == "openai_compat" else 1

    print(f"Cargando LLM (backend={settings.llm_backend})…", flush=True)
    llm = _build_llm(settings)
    await llm.load()

    dataset_ids = [s.strip() for s in args.datasets.split(",") if s.strip()] if args.datasets else None
    windows = tuple(s.strip() for s in args.windows.split(",") if s.strip())

    print(f"Generando informe descriptivo (ventanas {', '.join(windows)}, concurrencia {concurrency})…", flush=True)
    map_max = args.map_max_tokens
    synth_max = args.synth_max_tokens or settings.synthesis_max_tokens
    log.info(
        "Tokens MAP=%d  SYNTH=%d  concurrencia=%d",
        map_max, synth_max, concurrency,
    )
    report = await generate_parquet_report(
        llm,
        segment=args.segment,
        dataset_ids=dataset_ids,
        query=args.query,
        windows=windows,
        top_k=args.top_k,
        concurrency=concurrency,
        map_max_tokens=map_max,
        synthesis_max_tokens=synth_max,
        think=args.think,
    )

    out = pathlib.Path(args.out)
    stem = f"reporte_{report.selector_label}_{datetime.date.today().isoformat()}"
    counts = report.status_counts()

    md_path = _write(out / "markdown" / f"{stem}.md", report.to_markdown())
    print(
        f"OK markdown -> {md_path}  ({len(report.sections)} datasets: "
        f"ok={counts.get('ok', 0)} sin_datos={counts.get('no_data', 0)} "
        f"error={counts.get('error', 0)})"
    )
    if not args.no_html:
        html_path = _write(out / "html" / f"{stem}.html", render_parquet_report_html(report))
        print(f"OK html     -> {html_path}")

    if args.paragraphs_out:
        # Clave reservada "__sintesis__": la síntesis ejecutiva (puntos del mes y
        # la semana) para que fill_report_texts.py la inyecte arriba del informe.
        paragraphs = {"__sintesis__": report.overview_md}
        paragraphs.update({s.dataset_id: s.paragraph for s in report.sections if s.status == "ok"})
        p_path = pathlib.Path(args.paragraphs_out)
        p_path.parent.mkdir(parents=True, exist_ok=True)
        p_path.write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding="utf-8")  # noqa: ASYNC240
        print(f"OK párrafos -> {p_path}  ({len(paragraphs) - 1} datasets + síntesis)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Informe descriptivo de datasets parquet (markdown + html)."
    )
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--segment", default=None, help="Segmento del catálogo o alias (ej. ffmm, 'fondos mutuos').")
    sel.add_argument("--datasets", default=None, help="CSV de dataset ids o archivos (ej. flujos_ffmm,duracion_ffmm.parquet).")
    sel.add_argument("--query", default=None, help="Texto libre para descubrir datasets en el catálogo.")
    ap.add_argument("--windows", default="7d,30d", help="Ventanas de análisis CSV (Nd/Nw/Nm/Ny; default 7d,30d).")
    ap.add_argument("--out", default="data/parquet_reports", help="Carpeta raíz de salida (subcarpetas markdown/html).")
    ap.add_argument("--top-k", type=int, default=12, help="Máx. datasets cuando la selección es --query.")
    ap.add_argument("--concurrency", type=int, default=0, help="Datasets en paralelo (0 = auto según backend).")
    ap.add_argument("--think", action="store_true", help="Activa el thinking del LLM en el map (más lento, más profundo).")
    ap.add_argument(
        "--map-max-tokens", type=int, default=32768,
        help="Tokens máx por dataset en el MAP (default 32768; sube si el modelo se queda sin espacio para pensar).",
    )
    ap.add_argument(
        "--synth-max-tokens", type=int, default=0,
        help="Tokens máx para la síntesis global (0 = usar BANKS_SYNTHESIS_MAX_TOKENS del .env).",
    )
    ap.add_argument("--no-html", action="store_true", help="No generar el HTML (solo markdown).")
    ap.add_argument(
        "--paragraphs-out", default=None, metavar="PATH",
        help="Ruta .json donde guardar {dataset_id: párrafo} para fill_report_texts.py.",
    )
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
