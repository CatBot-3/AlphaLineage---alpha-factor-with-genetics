"""Validation helpers for identifiers used as local storage paths.

The API is normally local, but universe, symbol, factor, workspace, and session identifiers
still cross a trust boundary before becoming filenames.  Keep that boundary in one place so a
separator, drive prefix, or Windows device name can never escape its intended directory.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_SYMBOL = re.compile(r"^[A-Za-z0-9^][A-Za-z0-9._^=-]{0,63}$")
_WINDOWS_DEVICES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _validate_leaf(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{label} cannot be empty or have leading/trailing whitespace")
    if len(value) > 128:
        raise ValueError(f"{label} is too long")
    if value in {".", ".."} or value.endswith((".", " ")):
        raise ValueError(f"invalid {label}")
    if any(ord(char) < 32 for char in value) or any(char in value for char in "/\\:\0"):
        raise ValueError(f"invalid {label}")
    if value.split(".", 1)[0].upper() in _WINDOWS_DEVICES:
        raise ValueError(f"invalid {label}")
    return value


def validate_symbol(symbol: str) -> str:
    """Return a normalized market symbol that is safe to use as a cache filename."""
    clean = symbol.strip().upper() if isinstance(symbol, str) else symbol
    if not isinstance(clean, str) or not _SYMBOL.fullmatch(clean):
        raise ValueError(f"invalid symbol {symbol!r}")
    _validate_leaf(clean, label="symbol")
    return clean


def child_path(root: Path, identifier: str, suffix: str = "", *, label: str = "identifier") -> Path:
    """Build a direct child path and verify its resolved location stays under ``root``."""
    leaf = _validate_leaf(identifier, label=label)
    base = root.resolve()
    candidate = (base / f"{leaf}{suffix}").resolve()
    if candidate.parent != base:
        raise ValueError(f"invalid {label}")
    return candidate


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace a text file using a temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
