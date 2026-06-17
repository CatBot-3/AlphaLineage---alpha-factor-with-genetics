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

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_DIRNAME = "data_cache"
_ENV_VAR = "ALPHALINEAGE_DATA_DIR"
_FACTORS_ENV_VAR = "ALPHALINEAGE_FACTORS_DIR"


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


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def reports_dir() -> Path:
    return data_dir() / "reports"


def meta_dir() -> Path:
    return data_dir() / "meta"


# --- settings (user-editable, persisted under meta/) -----------------------------
def settings_path() -> Path:
    return meta_dir() / "settings.json"


def read_settings() -> dict[str, Any]:
    path = settings_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_settings(settings: dict[str, Any]) -> None:
    meta_dir().mkdir(parents=True, exist_ok=True)
    settings_path().write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")


def factors_dir() -> Path:
    """Where saved factors live. Resolution order (P5): env var > settings > default.

    User-customizable so the frontend can relocate the library via ``PUT /settings``.
    """
    override = os.environ.get(_FACTORS_ENV_VAR)
    if override:
        return Path(override)
    configured = read_settings().get("factors_dir")
    if configured:
        return Path(configured)
    return data_dir() / "factors"


def ensure_dirs() -> None:
    """Create every subdirectory of the data store if missing."""
    for directory in (
        prices_dir(),
        universe_dir(),
        workspaces_dir(),
        sessions_dir(),
        reports_dir(),
        meta_dir(),
        factors_dir(),
    ):
        directory.mkdir(parents=True, exist_ok=True)
