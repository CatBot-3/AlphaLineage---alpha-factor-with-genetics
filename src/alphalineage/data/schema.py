"""Canonical normalized price-frame schema shared by every provider.

Keeping one schema (identical column set, dtypes, and index) is what lets the
Parquet cache round-trip exactly and lets Tiingo and yfinance be interchangeable.
"""

from __future__ import annotations

import math
from typing import Any, Literal

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

# Price-level semantics are data, not an implementation detail.  Providers attach these
# attributes before a frame enters the cache and pandas/pyarrow preserves ``DataFrame.attrs``
# in Parquet metadata.  Legacy cache files have no attributes and are classified from their
# split-date continuity by :func:`infer_price_basis`.
PriceBasis = Literal["raw", "split_adjusted", "total_return_adjusted"]
PRICE_BASIS_VALUES = frozenset({"raw", "split_adjusted", "total_return_adjusted"})
PRICE_METADATA_VERSION = 1
PRICE_BASIS_ATTR = "alphalineage_price_basis"
PRICE_PROVIDER_ATTR = "alphalineage_price_provider"
PRICE_METADATA_VERSION_ATTR = "alphalineage_price_metadata_version"


class PriceBasisError(ValueError):
    """The cached price basis is mixed, contradictory, or cannot be used safely."""


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
    attrs = dict(df.attrs)
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
    normalized = validate(out.astype("float64"))
    normalized.attrs.update(attrs)
    return normalized


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


def with_price_metadata(
    df: pd.DataFrame,
    *,
    price_basis: PriceBasis,
    provider: str,
) -> pd.DataFrame:
    """Return ``df`` carrying explicit provider/price-basis cache metadata."""
    if price_basis not in PRICE_BASIS_VALUES:
        raise ValueError(f"unknown price basis {price_basis!r}")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string")
    out = df.copy()
    out.attrs.update(
        {
            PRICE_BASIS_ATTR: price_basis,
            PRICE_PROVIDER_ATTR: provider.strip().lower(),
            PRICE_METADATA_VERSION_ATTR: PRICE_METADATA_VERSION,
        }
    )
    return out


def explicit_price_basis(df: pd.DataFrame) -> PriceBasis | None:
    """Return the provider-declared basis, if this is a versioned cache frame."""
    value = df.attrs.get(PRICE_BASIS_ATTR)
    return value if value in PRICE_BASIS_VALUES else None  # type: ignore[return-value]


def infer_price_basis(df: pd.DataFrame) -> PriceBasis | None:
    """Infer a legacy frame's split basis from continuity around split events.

    A raw 2:1 split has ``close[t] / close[t-1]`` near ``1/2``; an already
    split-adjusted series remains near one.  We compare those two hypotheses in log
    space.  Conflicting events indicate that incremental downloads with different
    semantics were merged and must be resynchronized atomically.

    ``None`` means that the frame contains no decisive split event.  Such a legacy
    frame is backward-compatibly treated as raw because the choice has no effect until
    a split exists.
    """
    validate(df)
    labels: list[PriceBasis] = []
    close = df["close"]
    factors = df["split_factor"]
    # Twenty-five percent is deliberately much smaller than a normal 2:1 split gap but
    # large enough not to classify an ordinary split-day market move as ambiguous.
    margin = math.log(1.25)
    for position in range(1, len(df)):
        factor = float(factors.iloc[position])
        if not math.isfinite(factor) or factor <= 0.0 or math.isclose(factor, 1.0):
            continue
        previous = float(close.iloc[position - 1])
        current = float(close.iloc[position])
        if (
            not math.isfinite(previous)
            or not math.isfinite(current)
            or previous <= 0.0
            or current <= 0.0
        ):
            continue
        ratio = current / previous
        adjusted_error = abs(math.log(ratio))
        raw_error = abs(math.log(ratio * factor))
        if raw_error + margin < adjusted_error:
            labels.append("raw")
        elif adjusted_error + margin < raw_error:
            labels.append("split_adjusted")

    observed = set(labels)
    if len(observed) > 1:
        raise PriceBasisError(
            "price history mixes raw and split-adjusted corporate-action semantics; "
            "refresh the symbol instead of incrementally merging it"
        )
    return next(iter(observed)) if observed else None


def resolve_price_basis(df: pd.DataFrame) -> PriceBasis:
    """Resolve explicit or legacy-inferred basis, rejecting contradictory metadata."""
    declared = explicit_price_basis(df)
    inferred = infer_price_basis(df)
    if declared == "total_return_adjusted":
        return declared
    if declared is not None and inferred is not None and declared != inferred:
        raise PriceBasisError(
            f"declared {declared!r} price basis contradicts split-date continuity "
            f"({inferred!r}); resynchronize the symbol"
        )
    return declared or inferred or "raw"


def price_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-safe effective metadata for capabilities and diagnostics."""
    declared = explicit_price_basis(df)
    effective = resolve_price_basis(df)
    return {
        "price_basis": effective,
        "provider": df.attrs.get(PRICE_PROVIDER_ATTR),
        "metadata_version": df.attrs.get(PRICE_METADATA_VERSION_ATTR),
        "legacy_inferred": declared is None,
    }
