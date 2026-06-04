"""Phase 3 — anti-overfitting validation (the critical gate, invariant 1).

An honest out-of-sample, deflated verdict for any factor: time-ordered splits with embargo
(:mod:`splits`), purged k-fold CV (:mod:`purged_cv`), the Deflated Sharpe Ratio
(:mod:`deflated_sharpe`), Probability of Backtest Overfitting (:mod:`pbo`), a global trial
counter (:mod:`trials`), and a re-judge pipeline with a locked test split (:mod:`pipeline`).
"""

from alphaforge.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from alphaforge.validation.pbo import pbo
from alphaforge.validation.performance import long_short_returns, returns_matrix, tree_returns
from alphaforge.validation.pipeline import LockedTestSet, OverfittingReport, judge
from alphaforge.validation.purged_cv import purged_kfold
from alphaforge.validation.splits import Split, time_split, walk_forward
from alphaforge.validation.trials import TrialCounter

__all__ = [
    "LockedTestSet",
    "OverfittingReport",
    "Split",
    "TrialCounter",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "judge",
    "long_short_returns",
    "pbo",
    "probabilistic_sharpe_ratio",
    "purged_kfold",
    "returns_matrix",
    "sharpe_ratio",
    "time_split",
    "tree_returns",
    "walk_forward",
]
