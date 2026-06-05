#!/usr/bin/env python3
"""Genera el informe analítico de noticias del día (MARKDOWN + HTML + AUDIO JARVIS).

Reusa la infraestructura del agente (el LLM Qwen3.6 vía LlamaCppEngine y el
.env) y el paquete aislado ``jarvis_news``. Pensado para correrse a mano o por
cron; luego la misma lógica se integra como funcionalidad del agente.

Pipeline:
    JSON noticias → informe analítico (Markdown estructurado) → HTML (estilo
    plantilla) → relato hablado (narración) → audio FLAC ES + EN (voz JARVIS).

Salidas (ordenadas en subcarpetas de --out, default data/news_reports/):
    markdown/  reporte_<fecha>.md            informe analítico
               reporte_<fecha>_narracion.md  guion del audio (relato)
    html/      reporte_<fecha>.html
    audio/     reporte_<fecha>_es.flac  reporte_<fecha>_en.flac

Uso:
    python scripts/jarvis_news_report.py                 # markdown + html del JSON más reciente
    python scripts/jarvis_news_report.py --audio         # + narración + audio ES y EN (voz JARVIS)
    python scripts/jarvis_news_report.py --json otro.json --top-n 30 --out data/news_reports

Requisitos:
    - .env con BANKS_LLM_* (igual que la API).
    - Para --audio: BANKS_TTS_ENABLED=true (motor piper por defecto; ver
      docs/SETUP_JARVIS.md). FLAC requiere 'soundfile' (cae a WAV si falta).
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from banks_rag.config import get_settings  # noqa: E402
from banks_rag.infrastructure.llm.llama_cpp_engine import LlamaCppEngine  # noqa: E402
from jarvis_news.audio import synthesize_bilingual  # noqa: E402
from jarvis_news.html_report import render_html_report  # noqa: E402
from jarvis_news.report import generate_news_report, narrate_report  # noqa: E402
from jarvis_news.tts import TTSError  # noqa: E402


def _write(path: pathlib.Path, data: str | bytes) -> pathlib.Path:
    """Escribe ``data`` creando la subcarpeta si hace falta. Devuelve la ruta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    if settings.llm_family in ("mock", ""):
        print("WARN BANKS_LLM_FAMILY es 'mock'/vacío: configura qwen + BANKS_LLM_MODEL_PATH en .env.")
        return

    print(f"Cargando LLM {settings.llm_model_path} …", flush=True)
    llm = LlamaCppEngine.from_settings(settings)
    await llm.load()

    print("Generando informe analítico (map-reduce sobre las noticias)…", flush=True)
    result = await generate_news_report(
        llm, json_path=args.json, top_n=args.top_n,
        report_max_tokens=settings.synthesis_max_tokens,
    )

    out = pathlib.Path(args.out)
    stem = pathlib.Path(result["source_file"]).stem
    report_md = result["report"]

    md_path = _write(out / "markdown" / f"reporte_{stem}.md", report_md)
    print(f"OK markdown -> {md_path}  ({result['n_used']}/{result['n_total']} noticias)")

    if not args.no_html:
        html_path = _write(out / "html" / f"reporte_{stem}.html", render_html_report(report_md))
        print(f"OK html     -> {html_path}")

    if not args.audio:
        return

    try:
        print("Redactando el relato hablado (narración)…", flush=True)
        narration = await narrate_report(llm, report_md, max_tokens=settings.synthesis_max_tokens)
        nar_path = _write(out / "markdown" / f"reporte_{stem}_narracion.md", narration)
        print(f"OK narración-> {nar_path}")

        print(f"Sintetizando audio JARVIS ({args.audio_format.upper()}, ES + EN)…", flush=True)
        audios = await synthesize_bilingual(
            narration, llm, fmt=args.audio_format, max_tokens=settings.synthesis_max_tokens,
        )
        for lang, (data, ext) in audios.items():
            audio_path = _write(out / "audio" / f"reporte_{stem}_{lang}.{ext}", data)
            print(f"OK audio {lang.upper()} -> {audio_path}")
    except TTSError as exc:
        print(f"WARN audio omitido: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Informe analítico de noticias del día (markdown + html + audio JARVIS).")
    ap.add_argument("--json", default=None, help="JSON en Noticias_scrapping/ (default: el más reciente).")
    ap.add_argument("--top-n", type=int, default=25, help="Noticias más relevantes a incluir.")
    ap.add_argument("--out", default="data/news_reports", help="Carpeta raíz de salida (subcarpetas markdown/html/audio).")
    ap.add_argument("--audio", action="store_true", help="Generar también la narración + audio (ES + EN) con voz JARVIS.")
    ap.add_argument("--audio-format", default="flac", choices=["flac", "wav"], help="Formato del audio (default flac).")
    ap.add_argument("--no-html", action="store_true", help="No generar el HTML (solo markdown).")
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
