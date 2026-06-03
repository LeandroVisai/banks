#!/usr/bin/env python3
"""Genera el reporte de noticias del día (TEXTO + AUDIO JARVIS) desde la CLI.

Reusa la infraestructura del agente (el LLM Qwen3.6 vía LlamaCppEngine y el
.env) y el paquete aislado ``jarvis_news``. Pensado para correrse a mano o por
cron; luego la misma lógica se integra como funcionalidad del agente.

Uso:
    python scripts/jarvis_news_report.py                 # texto del JSON más reciente
    python scripts/jarvis_news_report.py --audio         # + audio ES y EN (voz JARVIS)
    python scripts/jarvis_news_report.py --json otro.json --top-n 30 --out data/news_reports

Requisitos:
    - .env con BANKS_LLM_* (igual que la API).
    - Para --audio: BANKS_TTS_ENABLED=true. Por defecto el motor es "sapi" (voz
      del SO + efecto DSP JARVIS, SIN modelos; pip install pyttsx3). Genera ES+EN.
      Alternativa: BANKS_TTS_ENGINE=piper (modelo neural). Ver docs/SETUP_SERVIDOR.txt.
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
from jarvis_news.report import generate_news_report  # noqa: E402
from jarvis_news.tts import TTSError  # noqa: E402


def _save_text(out: pathlib.Path, stem: str, report: str) -> pathlib.Path:
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / f"reporte_{stem}.md"
    txt_path.write_text(report, encoding="utf-8")
    return txt_path


def _save_audio(out: pathlib.Path, stem: str, lang: str, wav: bytes) -> pathlib.Path:
    wav_path = out / f"reporte_{stem}_{lang}.wav"
    wav_path.write_bytes(wav)
    return wav_path


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    if settings.llm_family in ("mock", ""):
        print("WARN BANKS_LLM_FAMILY es 'mock'/vacío: configura qwen + BANKS_LLM_MODEL_PATH en .env.")
        return

    print(f"Cargando LLM {settings.llm_model_path} …", flush=True)
    llm = LlamaCppEngine.from_settings(settings)
    await llm.load()

    print("Generando reporte (map-reduce sobre las noticias)…", flush=True)
    result = await generate_news_report(
        llm, json_path=args.json, top_n=args.top_n,
        report_max_tokens=settings.synthesis_max_tokens,
    )

    out = pathlib.Path(args.out)
    stem = pathlib.Path(result["source_file"]).stem
    txt_path = _save_text(out, stem, result["report"])
    print(f"OK texto -> {txt_path}  ({result['n_used']}/{result['n_total']} noticias)")

    if not args.audio:
        return

    try:
        print("Sintetizando audio JARVIS (ES + EN)…", flush=True)
        audios = await synthesize_bilingual(
            result["report"], llm, max_tokens=settings.synthesis_max_tokens,
        )
        for lang, wav in audios.items():
            wav_path = _save_audio(out, stem, lang, wav)
            print(f"OK audio {lang.upper()} -> {wav_path}")
    except TTSError as exc:
        print(f"WARN audio omitido: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reporte de noticias del día (texto + audio JARVIS).")
    ap.add_argument("--json", default=None, help="JSON en Noticias_scrapping/ (default: el más reciente).")
    ap.add_argument("--top-n", type=int, default=25, help="Noticias más relevantes a incluir.")
    ap.add_argument("--out", default="data/news_reports", help="Carpeta de salida.")
    ap.add_argument("--audio", action="store_true", help="Generar también los .wav (ES + EN) con voz JARVIS.")
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
