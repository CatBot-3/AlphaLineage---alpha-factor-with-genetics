"""Long-short factor returns - the time series the DSR/PBO operate on.

A factor is turned into a daily return by forming a dollar-neutral, unit-gross portfolio:
weight each symbol by the cross-sectionally demeaned factor, normalize so gross exposure is
1, and take the weighted forward return. This is **gross / pre-cost** - Phase 4's backtest
supersedes it with transaction costs and neutralization.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphalineage.core.evaluate import evaluate
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node


def long_short_returns(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily dollar-neutral, unit-gross long-short return of ``factor`` vs forward returns."""
    f, r = factor.align(fwd, join="inner")
    mask = f.notna() & r.notna()
    f = f.where(mask)
    weights = f.sub(f.mean(axis=1), axis=0)  # demean -> dollar neutral
    gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
    weights = weights.div(gross, axis=0)  # unit gross (sum |w| = 1)
    return (weights * r).sum(axis=1, min_count=1)


def tree_returns(tree: Node, panel: Panel, fwd: pd.DataFrame) -> pd.Series:
    """Long-short return series for a factor tree (empty Series if it is not a panel)."""
    factor = evaluate(tree, panel)
    if not isinstance(factor, pd.DataFrame):
        return pd.Series(dtype="float64")
    return long_short_returns(factor, fwd)


def returns_matrix(
    trees: Sequence[Node],
    panel: Panel,
    fwd: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """A ``T x N`` matrix of strategy returns (one column per tree), optional date filter."""
    matrix = pd.DataFrame({i: tree_returns(tree, panel, fwd) for i, tree in enumerate(trees)})
    if dates is not None:
        matrix = matrix.loc[matrix.index.isin(dates)]
    return matrix
