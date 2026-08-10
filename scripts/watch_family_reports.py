#!/usr/bin/env python3
"""Vigila data_pipeline/parquet/ y reconstruye los informes curados por familia
(ffmm, nr, afp por ahora) cuando detecta parquets más nuevos que la última corrida.

One-shot: NO deja un loop propio corriendo. Pensado para invocarse
periódicamente desde Task Scheduler (Windows) o cron, ej. cada 15 minutos.
La detección de "parquet nuevo" es por mtime: se guarda el mtime máximo visto
en un archivo de estado (--state-file) y solo se reconstruye si algún .parquet
de la carpeta vigilada es más nuevo que esa marca. Reconstruye TODAS las
familias listadas ante cualquier cambio (no distingue qué parquet exacto tocó
a cuál familia).

Cada familia se reconstruye llamando a build_family_report.py como subproceso
(mismos --out/--out-editable/--no-editable que ese script, ver su --help).
Si alguna familia falla, el estado NO avanza: la próxima corrida reintenta
todo (build_family_report.py es idempotente por día, así que reintentar no
duplica nada).

Uso:
    python scripts/watch_family_reports.py                       # chequea y reconstruye si hay novedades
    python scripts/watch_family_reports.py --force                # ignora el estado, reconstruye igual
    python scripts/watch_family_reports.py --families ffmm,nr
    python scripts/watch_family_reports.py --verbose

Task Scheduler (Windows), cada 15 min:
    schtasks /create /tn "BCCh Informes Curados" ^
        /tr "python C:\\ruta\\al\\repo\\scripts\\watch_family_reports.py" ^
        /sc minute /mo 15
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BUILD_SCRIPT = _ROOT / "scripts" / "build_family_report.py"

DEFAULT_PARQUET_DIR = _ROOT / "data_pipeline" / "parquet"
DEFAULT_STATE_FILE = _ROOT / "data" / "parquet_reports" / "curated" / ".watch_state.json"
DEFAULT_FAMILIES = ["ffmm", "nr", "afp"]


def _max_mtime(parquet_dir: pathlib.Path) -> float:
    files = list(parquet_dir.rglob("*.parquet"))
    return max((f.stat().st_mtime for f in files), default=0.0)


def _load_state(state_file: pathlib.Path) -> dict:
    if not state_file.is_file():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state_file: pathlib.Path, mtime: float) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_mtime": mtime,
        "last_run_iso": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vigila parquets nuevos y reconstruye los informes curados por familia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR), help="Carpeta con los .parquet a vigilar.")
    ap.add_argument("--state-file", default=str(DEFAULT_STATE_FILE), help="Dónde persistir el último mtime procesado.")
    ap.add_argument(
        "--families", default=",".join(DEFAULT_FAMILIES),
        help="Familias a reconstruir ante cambios, separadas por coma.",
    )
    ap.add_argument("--force", action="store_true", help="Reconstruye aunque no haya parquets nuevos desde la última corrida.")
    ap.add_argument("--out", default=None, help="Forward a build_family_report.py --out (mismo default si se omite).")
    ap.add_argument("--out-editable", default=None, help="Forward a build_family_report.py --out-editable.")
    ap.add_argument("--no-editable", action="store_true", help="Forward a build_family_report.py --no-editable.")
    ap.add_argument("--verbose", action="store_true", help="Log detallado (propio y forward a build_family_report.py).")
    args = ap.parse_args()

    parquet_dir = pathlib.Path(args.parquet_dir)
    state_file = pathlib.Path(args.state_file)
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    current_mtime = _max_mtime(parquet_dir)
    last_mtime = _load_state(state_file).get("last_mtime", 0.0)

    if not args.force and current_mtime <= last_mtime:
        if args.verbose:
            print(f"Sin parquets nuevos en {parquet_dir} (último procesado: {last_mtime}, actual: {current_mtime}).")
        return

    print(f"Parquets nuevos en {parquet_dir} -> reconstruyendo: {', '.join(families)}", flush=True)

    extra: list[str] = []
    if args.out is not None:
        extra += ["--out", args.out]
    if args.out_editable is not None:
        extra += ["--out-editable", args.out_editable]
    if args.no_editable:
        extra.append("--no-editable")
    if args.verbose:
        extra.append("--verbose")

    all_ok = True
    for fam in families:
        cmd = [sys.executable, str(_BUILD_SCRIPT), "--family", fam, *extra]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            all_ok = False
            print(f"FALLÓ {fam} (exit {result.returncode}); se reintentará en la próxima corrida.", file=sys.stderr, flush=True)

    if all_ok:
        _save_state(state_file, current_mtime)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
