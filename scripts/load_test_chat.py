"""Load test de /v1/chat — mide la concurrencia real del stack de inferencia.

Lanza N usuarios virtuales concurrentes, cada uno con una secuencia de preguntas
representativas del dominio BCCh, y reporta latencia p50/p95/max E2E, throughput
y tasa de error. Sirve para comparar backends (in-process vs llama-server) y
quants (Q4 vs Q6) con el MISMO protocolo de medición.

Uso:

    # Baseline (1, 2 y 4 usuarios) contra la API local:
    python scripts/load_test_chat.py --users 1 --out data/bench/u1.json
    python scripts/load_test_chat.py --users 2 --out data/bench/u2.json
    python scripts/load_test_chat.py --users 4 --out data/bench/u4.json

    # Con API key y modo rápido:
    python scripts/load_test_chat.py --users 2 --api-key XXX --thinking-mode off

No depende de banks_rag: solo httpx (stdlib + httpx). Se puede correr desde
cualquier máquina con acceso a la API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

# Preguntas representativas (mezcla cuantitativa/documental/política — alinea
# con el golden set para que el bench ejercite los mismos caminos de tools).
DEFAULT_QUESTIONS: list[str] = [
    "¿Cuál fue la última decisión de TPM y qué argumentos dio el Consejo?",
    "¿Cómo ha variado el tipo de cambio en el último mes?",
    "¿Qué dice el último IPoM sobre el balance de riesgos?",
    "Compara la tenencia de no residentes en renta fija este año contra el anterior.",
    "¿Qué riesgos menciona el research de JPMorgan sobre commodities?",
    "¿Cuál es el spread on-offshore más reciente y su variación semanal?",
]


async def _one_user(
    user_id: int,
    client: httpx.AsyncClient,
    questions: list[str],
    thinking_mode: str | None,
    results: list[dict],
) -> None:
    for i, q in enumerate(questions):
        payload: dict = {"message": q}
        if thinking_mode:
            payload["thinking_mode"] = thinking_mode
        t0 = time.monotonic()
        entry: dict = {"user": user_id, "question_idx": i, "question": q}
        try:
            resp = await client.post("/v1/chat", json=payload)
            elapsed = time.monotonic() - t0
            entry["latency_s"] = round(elapsed, 3)
            entry["status"] = resp.status_code
            if resp.status_code == 200:
                body = resp.json()
                entry["total_tokens"] = body.get("total_tokens", 0)
                entry["iterations"] = body.get("iterations", 0)
                entry["response_chars"] = len(body.get("response", ""))
            else:
                entry["error"] = resp.text[:200]
        except Exception as exc:  # noqa: BLE001
            entry["latency_s"] = round(time.monotonic() - t0, 3)
            entry["status"] = -1
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        results.append(entry)
        print(
            f"  [u{user_id} q{i}] {entry['status']} en {entry['latency_s']:.1f}s",
            flush=True,
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(pct / 100 * (len(values) - 1))))
    return values[k]


async def run(args: argparse.Namespace) -> dict:
    questions = DEFAULT_QUESTIONS[: args.questions]
    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    results: list[dict] = []
    t_start = time.monotonic()
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
        headers=headers,
    ) as client:
        await asyncio.gather(*(
            _one_user(u, client, questions, args.thinking_mode, results)
            for u in range(args.users)
        ))
    wall_s = time.monotonic() - t_start

    ok = [r for r in results if r.get("status") == 200]
    lats = [r["latency_s"] for r in ok]
    total_tokens = sum(r.get("total_tokens", 0) for r in ok)
    summary = {
        "config": {
            "base_url": args.base_url,
            "users": args.users,
            "questions_per_user": len(questions),
            "thinking_mode": args.thinking_mode,
        },
        "wall_clock_s": round(wall_s, 2),
        "requests_total": len(results),
        "requests_ok": len(ok),
        "error_rate": round(1 - len(ok) / len(results), 3) if results else None,
        "latency_s": {
            "p50": round(_percentile(lats, 50), 2),
            "p95": round(_percentile(lats, 95), 2),
            "max": round(max(lats), 2) if lats else 0.0,
            "mean": round(statistics.fmean(lats), 2) if lats else 0.0,
        },
        "throughput_req_per_min": round(len(ok) / wall_s * 60, 2) if wall_s else 0.0,
        "total_completion_tokens": total_tokens,
        "tokens_per_s_aggregate": round(total_tokens / wall_s, 1) if wall_s else 0.0,
        "results": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test de /v1/chat")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--users", type=int, default=2, help="usuarios concurrentes")
    parser.add_argument(
        "--questions", type=int, default=len(DEFAULT_QUESTIONS),
        help="preguntas por usuario (toma las primeras N del set)",
    )
    parser.add_argument("--thinking-mode", choices=["off", "adaptive", "on"], default=None)
    parser.add_argument("--timeout", type=float, default=600.0, help="timeout por request (s)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default=None, help="ruta del JSON de salida")
    args = parser.parse_args()

    print(f"Load test: {args.users} usuario(s) × {args.questions} pregunta(s) → {args.base_url}")
    summary = asyncio.run(run(args))

    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resultados completos → {out}")


if __name__ == "__main__":
    main()
