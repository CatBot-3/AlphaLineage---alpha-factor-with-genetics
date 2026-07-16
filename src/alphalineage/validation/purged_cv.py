"""P3-T2 — purged k-fold cross-validation (López de Prado).

Standard k-fold leaks in finance because a train sample's forward-return label overlaps the
test fold, and because serial correlation bleeds across the fold boundary. This **purges**
train dates whose label window (length ``horizon``) overlaps a test fold and **embargoes** a
fraction of dates immediately after it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alphalineage.validation.splits import _positive_integer, _validated_dates


def purged_kfold(
    dates: pd.DatetimeIndex,
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    horizon: int = 1,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield ``(train, test)`` folds with overlapping-label purging and a trailing embargo."""
    idx = _validated_dates(dates)
    n = len(idx)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or not 2 <= n_splits <= n:
        raise ValueError("n_splits must be in [2, len(dates)]")
    if (
        isinstance(embargo_pct, bool)
        or not isinstance(embargo_pct, (int, float))
        or not math.isfinite(float(embargo_pct))
        or not 0.0 <= float(embargo_pct) < 1.0
    ):
        raise ValueError("embargo_pct must be a finite fraction in [0, 1)")
    checked_horizon = _positive_integer("horizon", horizon)
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    embargo = int(n * embargo_pct)

    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for i in range(n_splits):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        keep = np.ones(n, dtype=bool)
        # purge: the test fold itself + train labels that reach into it (t in [lo-horizon, lo))
        keep[max(0, lo - checked_horizon) : hi] = False
        # embargo: train dates immediately after the test fold
        keep[hi : min(n, hi + embargo)] = False
        folds.append((idx[keep], idx[lo:hi]))
    return folds
