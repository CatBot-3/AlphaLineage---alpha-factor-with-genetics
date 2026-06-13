"""P3-T1 - train / valid / test splits with embargo, and walk-forward windows.

Time is ordered and never shuffled. An **embargo** gap separates adjacent segments so a
trailing-window factor or a forward-return label cannot straddle the boundary and leak.
The **test** segment is the locked out-of-sample split (see
:class:`alphalineage.validation.pipeline.LockedTestSet`): it is never used to select or score
a factor until the final report.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """Ordered train / valid / test date segments (disjoint, embargo-separated)."""

    train: pd.DatetimeIndex
    valid: pd.DatetimeIndex
    test: pd.DatetimeIndex

    @property
    def research(self) -> pd.DatetimeIndex:
        """Train + valid - every date the search is allowed to see."""
        return self.train.union(self.valid)


def time_split(
    dates: pd.DatetimeIndex, *, train: float = 0.6, valid: float = 0.2, embargo: int = 5
) -> Split:
    """Split ordered ``dates`` into train/valid/test with an ``embargo`` gap between each."""
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    if n < 10:
        raise ValueError("need at least 10 dates to form a train/valid/test split")
    n_train = int(n * train)
    n_valid = int(n * valid)

    train_idx = idx[:n_train]
    valid_start = n_train + embargo
    valid_idx = idx[valid_start : valid_start + n_valid]
    test_start = valid_start + n_valid + embargo
    test_idx = idx[test_start:]

    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise ValueError("split produced an empty segment; reduce embargo or fractions")
    return Split(train_idx, valid_idx, test_idx)


def walk_forward(
    dates: pd.DatetimeIndex,
    *,
    n_splits: int,
    train_size: int,
    test_size: int,
    embargo: int = 5,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Rolling ``(train, test)`` windows, each separated by an ``embargo`` gap."""
    idx = pd.DatetimeIndex(dates)
    windows: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    start = 0
    for _ in range(n_splits):
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > len(idx):
            break
        windows.append((idx[start:train_end], idx[test_start:test_end]))
        start += test_size
    return windows
