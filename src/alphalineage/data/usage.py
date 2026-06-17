"""L2 - local data-usage accounting and an allowlist-only cleaner.

The system accumulates files over time: downloaded prices, custom universes, training sessions,
saved factors, and workspaces. This module reports per-category disk usage and clears a category
on request. Clearing is restricted to a fixed category->directory allowlist (P-S4): it only ever
deletes within known data directories, never an arbitrary path.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alphalineage.data import paths

# category key -> (human label, directory resolver). factors_dir is resolved fresh because it is
# user-configurable; the rest live under the data root.
_CATEGORIES: dict[str, tuple[str, Callable[[], Path]]] = {
    "prices": ("Market data (prices)", paths.prices_dir),
    "universes": ("Custom universes", paths.universe_dir),
    "sessions": ("Training sessions", paths.sessions_dir),
    "factors": ("Saved factors", paths.factors_dir),
    "workspaces": ("Saved workspaces", paths.workspaces_dir),
}


def _dir_stats(directory: Path) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)`` for ``directory`` (recursive); zeros if absent."""
    if not directory.exists():
        return 0, 0
    total = 0
    count = 0
    for path in directory.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
            count += 1
    return total, count


def usage() -> list[dict[str, Any]]:
    """Per-category disk usage, one row each with ``key``, ``label``, ``bytes``, ``count``."""
    rows: list[dict[str, Any]] = []
    for key, (label, resolve) in _CATEGORIES.items():
        total, count = _dir_stats(resolve())
        rows.append({"key": key, "label": label, "bytes": total, "count": count})
    return rows


def clear(category: str) -> dict[str, Any]:
    """Delete every file in ``category``'s directory; recreate it empty. Returns its new usage."""
    entry = _CATEGORIES.get(category)
    if entry is None:
        raise ValueError(f"unknown data category {category!r}")
    label, resolve = entry
    directory = resolve()
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {"key": category, "label": label, "bytes": 0, "count": 0}
