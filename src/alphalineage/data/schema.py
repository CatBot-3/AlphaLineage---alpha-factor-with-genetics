"""Canonical normalized price-frame schema shared by every provider.

Keeping one schema (identical column set, dtypes, and index) is what lets the
Parquet cache round-trip exactly and lets Tiingo and yfinance be interchangeable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: Canonical column order. ``div_cash`` / ``split_factor`` carry corporate actions
#: (split_factor == 1.0 means "no split"; div_cash == 0.0 means "no dividend").
PRICE_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "div_cash",
    "split_factor",
]

#: Defaults used when a provider does not supply a corporate-action column.
_DEFAULTS: dict[str, float] = {"div_cash": 0.0, "split_factor": 1.0}

INDEX_NAME = "date"


def _to_naive_dates(index: Any) -> pd.DatetimeIndex:
    """Coerce any date-like index to a tz-naive, midnight-normalized DatetimeIndex."""
    idx = pd.to_datetime(index)
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider frame into the canonical schema.

    Idempotent: ``normalize(normalize(x))`` equals ``normalize(x)``. Missing
    corporate-action columns are filled with their no-op defaults; the index is
    made tz-naive, deduplicated (keeping the last row), and sorted ascending; all
    columns are cast to float64 for deterministic Parquet round-trips.
    """
    out = pd.DataFrame(index=_to_naive_dates(df.index))
    out.index.name = INDEX_NAME

    for col in PRICE_COLUMNS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col].to_numpy(), errors="coerce")
        elif col in _DEFAULTS:
            out[col] = _DEFAULTS[col]
        else:
            raise ValueError(f"missing required price column: {col!r}")

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return validate(out.astype("float64"))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Assert ``df`` conforms to the canonical schema; return it unchanged."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("price frame index must be a DatetimeIndex")
    if df.index.tz is not None:
        raise ValueError("price frame index must be tz-naive")
    if df.index.name != INDEX_NAME:
        raise ValueError(f"price frame index must be named {INDEX_NAME!r}")
    if list(df.columns) != PRICE_COLUMNS:
        raise ValueError(f"price frame columns must be exactly {PRICE_COLUMNS}")
    return df
