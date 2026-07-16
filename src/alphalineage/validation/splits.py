"""P3-T1 - train / valid / test splits with embargo, and walk-forward windows.

Time is ordered and never shuffled. An **embargo** gap separates adjacent segments so a
trailing-window factor or a forward-return label cannot straddle the boundary and leak.
The **test** segment is the locked out-of-sample split (see
:class:`alphalineage.validation.pipeline.LockedTestSet`): it is never used to select or score
a factor until the final report.
"""

from __future__ import annotations

import math
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


def _validated_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(dates)
    if idx.hasnans:
        raise ValueError("dates must not contain NaT")
    if not idx.is_monotonic_increasing:
        raise ValueError("dates must be sorted in increasing order")
    if not idx.is_unique:
        raise ValueError("dates must not contain duplicates")
    return idx


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validated_gap(embargo: object, horizon: object) -> tuple[int, int]:
    if isinstance(embargo, bool) or not isinstance(embargo, int) or embargo < 0:
        raise ValueError(f"embargo must be a non-negative integer, got {embargo!r}")
    checked_horizon = _positive_integer("horizon", horizon)
    if embargo < checked_horizon:
        raise ValueError(
            f"embargo ({embargo}) must be at least horizon ({checked_horizon}) "
            "to keep forward-return labels out of the next split"
        )
    return embargo, checked_horizon


def time_split(
    dates: pd.DatetimeIndex,
    *,
    train: float = 0.6,
    valid: float = 0.2,
    embargo: int = 5,
    horizon: int = 1,
) -> Split:
    """Split ordered dates with gaps large enough to contain each forward-return label."""
    idx = _validated_dates(dates)
    n = len(idx)
    if n < 10:
        raise ValueError("need at least 10 dates to form a train/valid/test split")
    for name, value in (("train", train), ("valid", valid)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
        ):
            raise ValueError(f"{name} must be a finite fraction strictly between 0 and 1")
    if train + valid >= 1.0:
        raise ValueError("train + valid must be less than 1 to leave a test segment")
    checked_embargo, _ = _validated_gap(embargo, horizon)
    n_train = int(n * train)
    n_valid = int(n * valid)

    train_idx = idx[:n_train]
    valid_start = n_train + checked_embargo
    valid_idx = idx[valid_start : valid_start + n_valid]
    test_start = valid_start + n_valid + checked_embargo
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
    horizon: int = 1,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Rolling windows separated enough to keep train labels out of each test window."""
    idx = _validated_dates(dates)
    checked_splits = _positive_integer("n_splits", n_splits)
    checked_train = _positive_integer("train_size", train_size)
    checked_test = _positive_integer("test_size", test_size)
    checked_embargo, _ = _validated_gap(embargo, horizon)
    windows: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    start = 0
    for _ in range(checked_splits):
        train_end = start + checked_train
        test_start = train_end + checked_embargo
        test_end = test_start + checked_test
        if test_end > len(idx):
            break
        windows.append((idx[start:train_end], idx[test_start:test_end]))
        start += checked_test
    return windows
