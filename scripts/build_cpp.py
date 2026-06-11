"""Build the optional C++ evaluator extension (cmake + ninja + g++) into the package.

Run: ``python scripts/build_cpp.py``. Needs a C++ compiler, CMake, and Ninja on PATH plus
``pybind11`` (installed automatically). The build is entirely optional — the project runs on the
pure-Python evaluator without it. The produced ``_evaluator*`` extension is gitignored.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP_DIR = ROOT / "cpp"
BUILD_DIR = CPP_DIR / "build"
DEST = ROOT / "src" / "alphaforge"


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    _run([sys.executable, "-m", "pip", "install", "--quiet", "pybind11"])
    raw = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"]).decode()
    cmake_dir = raw.strip().strip('"')  # pybind11 quotes the path when it contains spaces

    gpp = shutil.which("g++") or shutil.which("c++")
    generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    configure = [
        "cmake",
        "-S",
        str(CPP_DIR),
        "-B",
        str(BUILD_DIR),
        "-G",
        generator,
        f"-Dpybind11_DIR={cmake_dir}",
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if gpp:
        configure.append(f"-DCMAKE_CXX_COMPILER={gpp}")
    _run(configure)
    _run(["cmake", "--build", str(BUILD_DIR), "--config", "Release"])

    copied = []
    for pattern in ("_evaluator*.pyd", "_evaluator*.so"):
        for artifact in BUILD_DIR.rglob(pattern):
            target = DEST / artifact.name
            shutil.copy2(artifact, target)
            copied.append(target)
            print("copied", artifact, "->", target)
    if not copied:
        print("ERROR: no _evaluator extension produced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
