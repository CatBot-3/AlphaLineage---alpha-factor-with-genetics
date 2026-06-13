"""Central configuration for the local data store.

All pulled stock data lives under a single gitignored root (``data_cache/`` by
default, overridable via the ``ALPHALINEAGE_DATA_DIR`` environment variable):

    data_cache/
      prices/{SYMBOL}.parquet            raw OHLCV + div_cash + split_factor
      universe/{name}.parquet            constituents with entry/exit dates
      workspaces/{id}.json               saved UI/backend workspaces
      reports/survivorship_{name}.md     audit reports
      meta/fetch_log.json                provenance + rate-limit accounting
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DIRNAME = "data_cache"
_ENV_VAR = "ALPHALINEAGE_DATA_DIR"


def data_dir() -> Path:
    """Root of the local data store (resolved fresh so tests can redirect it)."""
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else Path.cwd() / _DEFAULT_DIRNAME


def prices_dir() -> Path:
    return data_dir() / "prices"


def universe_dir() -> Path:
    return data_dir() / "universe"


def workspaces_dir() -> Path:
    return data_dir() / "workspaces"


def reports_dir() -> Path:
    return data_dir() / "reports"


def meta_dir() -> Path:
    return data_dir() / "meta"


def ensure_dirs() -> None:
    """Create every subdirectory of the data store if missing."""
    for directory in (prices_dir(), universe_dir(), workspaces_dir(), reports_dir(), meta_dir()):
        directory.mkdir(parents=True, exist_ok=True)
