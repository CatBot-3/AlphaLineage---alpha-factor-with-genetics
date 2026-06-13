"""P5-T4 - combination model (mega-alpha).

Combines a diversified factor set into one signal by IC-weighted z-scoring: standardize each
factor cross-sectionally, then add them weighted by their (signed) train IC. The sign aligns
each factor with forward returns and the magnitude down-weights weak factors. No fitting, so it
does not overfit - and combining decorrelated factors lifts the IC information ratio (hence the
deflated metric) above any single factor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import daily_ic
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node


def _zscore(signal: pd.DataFrame) -> pd.DataFrame:
    mean = signal.mean(axis=1)
    std = signal.std(axis=1).replace(0.0, np.nan)
    return signal.sub(mean, axis=0).div(std, axis=0)


def combine(
    signals: Sequence[pd.DataFrame], fwd: pd.DataFrame, *, train_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """IC-weighted z-score combination of factor signals into a single mega-alpha signal."""
    combined: pd.DataFrame | None = None
    for signal in signals:
        ic = daily_ic(
            signal.loc[signal.index.isin(train_dates)],
            fwd.loc[fwd.index.isin(train_dates)],
            "spearman",
        ).mean()
        weight = float(ic) if np.isfinite(ic) else 0.0
        contribution = _zscore(signal) * weight
        combined = contribution if combined is None else combined.add(contribution, fill_value=0.0)
    return combined if combined is not None else pd.DataFrame()


def combine_trees(
    trees: Sequence[Node], panel: Panel, fwd: pd.DataFrame, *, train_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Evaluate factor trees on the panel, then combine their signals."""
    signals = [evaluate(t, panel) for t in trees]
    return combine(
        [s for s in signals if isinstance(s, pd.DataFrame)], fwd, train_dates=train_dates
    )
