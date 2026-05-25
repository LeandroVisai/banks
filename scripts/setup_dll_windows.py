"""Genera sitecustomize.py en el venv para que llama_cpp encuentre sus DLLs en Windows.

Uso (desde el venv activo):
    python scripts/setup_dll_windows.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig


def main() -> int:
    if sys.platform != "win32":
        print("Este script solo aplica en Windows.")
        return 0

    try:
        import torch
    except ImportError:
        print("ERROR: torch no esta instalado en el venv activo.")
        return 1

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

    spec = importlib.util.find_spec("llama_cpp")
    if spec is None or spec.origin is None:
        print("ERROR: llama_cpp no esta instalado en el venv activo.")
        return 1

    llama_pkg = os.path.dirname(spec.origin)
    llama_lib = os.path.join(llama_pkg, "lib")
    venv_bin = os.path.normpath(os.path.join(llama_pkg, os.pardir, "bin"))

    site_dir = sysconfig.get_path("purelib")
    target = os.path.join(site_dir, "sitecustomize.py")

    lines = [
        "import os",
        "",
        "# add_dll_directory registra paths, pero importar torch primero",
        "# fuerza la carga de las DLLs CUDA en memoria — sin esto, llama.dll",
        "# no resuelve sus dependencias transitivas en Windows.",
        "try:",
        "    import torch  # noqa: F401",
        "except ImportError:",
        "    pass",
        "",
    ]
    for path in (torch_lib, llama_lib, venv_bin):
        if os.path.isdir(path):
            lines.append(f"os.add_dll_directory(r'{path}')")
        else:
            print(f"  WARN: no existe {path}, se omite.")

    content = "\n".join(lines) + "\n"

    with open(target, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"OK: sitecustomize.py escrito en {target}")
    print("Contenido:")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
