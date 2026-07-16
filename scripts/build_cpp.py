"""Build the optional C++ evaluator extension (cmake + ninja + g++) into the package.

Run: ``python scripts/build_cpp.py``. Needs a C++ compiler, CMake, and Ninja on PATH plus
``pybind11`` (installed automatically). The build is entirely optional - the project runs on the
pure-Python evaluator without it. The produced ``_evaluator*`` extension is gitignored.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP_DIR = ROOT / "cpp"
BUILD_DIR = CPP_DIR / "build" / (sys.implementation.cache_tag or "current-python")
DEST = ROOT / "src" / "alphalineage"


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-dependency-install",
        action="store_true",
        help="require pybind11 to already be installed (used by reproducible wheel/Docker builds)",
    )
    args = parser.parse_args(argv)
    if not args.skip_dependency_install:
        _run([sys.executable, "-m", "pip", "install", "--quiet", "pybind11"])
    raw = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"]).decode()
    cmake_dir = raw.strip().strip('"')  # pybind11 quotes the path when it contains spaces

    gpp = shutil.which("g++") or shutil.which("c++")
    # A stock Windows wheel runner has Visual Studio installed but may not expose cl.exe on PATH.
    # Let CMake choose its Visual Studio generator in that case; use Ninja with explicit GNU/Clang.
    generator = None if sys.platform == "win32" and not gpp else (
        "Ninja" if shutil.which("ninja") else "Unix Makefiles"
    )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    configure = [
        "cmake",
        "-S",
        str(CPP_DIR),
        "-B",
        str(BUILD_DIR),
        f"-Dpybind11_DIR={cmake_dir}",
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if generator is not None:
        configure[5:5] = ["-G", generator]
    if gpp:
        configure.append(f"-DCMAKE_CXX_COMPILER={gpp}")
    _run(configure)
    _run(["cmake", "--build", str(BUILD_DIR), "--config", "Release"])

    # A cibuildwheel job may build several CPython ABIs from the same checkout. Never let an
    # earlier interpreter's extension leak into the next wheel.
    for pattern in ("_evaluator*.pyd", "_evaluator*.so"):
        for stale in DEST.glob(pattern):
            stale.unlink()

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
