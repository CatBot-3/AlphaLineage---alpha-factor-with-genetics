"""P3-T2 — purged k-fold cross-validation (López de Prado).

Standard k-fold leaks in finance because a train sample's forward-return label overlaps the
test fold, and because serial correlation bleeds across the fold boundary. This **purges**
train dates whose label window (length ``horizon``) overlaps a test fold and **embargoes** a
fraction of dates immediately after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def purged_kfold(
    dates: pd.DatetimeIndex,
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    horizon: int = 1,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield ``(train, test)`` folds with overlapping-label purging and a trailing embargo."""
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    if n_splits < 2 or n_splits > n:
        raise ValueError("n_splits must be in [2, len(dates)]")
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    embargo = int(n * embargo_pct)

    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for i in range(n_splits):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        keep = np.ones(n, dtype=bool)
        # purge: the test fold itself + train labels that reach into it (t in [lo-horizon, lo))
        keep[max(0, lo - horizon) : hi] = False
        # embargo: train dates immediately after the test fold
        keep[hi : min(n, hi + embargo)] = False
        folds.append((idx[keep], idx[lo:hi]))
    return folds
