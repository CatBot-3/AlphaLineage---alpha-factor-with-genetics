"""P0-T3 - corporate-action adjustment (back-adjustment).

Produces a continuous, total-return adjusted series from raw OHLCV plus per-day
``split_factor`` and ``div_cash``. The adjusted price for day *t* applies the
cumulative effect of every corporate action **after** *t*:

    cumadj_t = prod_{i > t} (1 / split_factor_i) * (1 - div_cash_i / close_{i-1})

so that across a split or ex-dividend date the adjusted series has no artificial
discontinuity - only the true economic return remains. Volume is adjusted by the
inverse cumulative split factor (more shares after a forward split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalineage.data import schema

_ADJ_PRICE_COLS = {"open": "adj_open", "high": "adj_high", "low": "adj_low", "close": "adj_close"}


def _suffix_excl_self(factor: pd.Series) -> pd.Series:
    """Return ``g`` where ``g_t = prod_{i > t} factor_i`` (last element = 1.0)."""
    suffix_incl = factor[::-1].cumprod()[::-1]  # prod_{i >= t}
    return suffix_incl.shift(-1).fillna(1.0)  # drop self -> prod_{i > t}


def price_adjustment(df: pd.DataFrame) -> pd.Series:
    """Cumulative price adjustment factor (splits + dividends), one per day."""
    schema.validate(df)
    close = df["close"]
    prev_close = close.shift(1)
    div_cash = df["div_cash"]
    split_factor = df["split_factor"].replace(0.0, 1.0)

    # Dividend multiplier 1 - div/prev_close, but only where a dividend was paid and a
    # prior close exists (the first row has no prior close).
    has_div = (div_cash > 0.0) & prev_close.notna() & (prev_close > 0.0)
    div_mult = pd.Series(1.0, index=df.index)
    div_mult[has_div] = 1.0 - div_cash[has_div] / prev_close[has_div]

    day_factor = (1.0 / split_factor) * div_mult
    return _suffix_excl_self(day_factor)


def split_adjustment(df: pd.DataFrame) -> pd.Series:
    """Cumulative split-only adjustment factor (used for volume)."""
    schema.validate(df)
    split_factor = df["split_factor"].replace(0.0, 1.0)
    return _suffix_excl_self(1.0 / split_factor)


def adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with ``adj_open/high/low/close`` and ``adj_volume`` appended."""
    schema.validate(df)
    out = df.copy()
    price_cumadj = price_adjustment(df)
    split_cumadj = split_adjustment(df)

    for raw_col, adj_col in _ADJ_PRICE_COLS.items():
        out[adj_col] = df[raw_col] * price_cumadj
    # Volume scales inversely to the split price factor (forward split -> more shares).
    out["adj_volume"] = df["volume"] / split_cumadj.replace(0.0, np.nan)
    return out


def adjusted_close(df: pd.DataFrame) -> pd.Series:
    """Convenience accessor for the back-adjusted close series."""
    return df["close"] * price_adjustment(df)
