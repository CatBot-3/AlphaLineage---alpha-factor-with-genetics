"""P3-T6 — re-judge a Phase-2 best factor honestly, out of sample.

Ties the suite together: given the GP's chosen factor and the trials it searched, produce an
:class:`OverfittingReport` whose headline numbers default to out-of-sample / deflated
(invariant 1). The test split is wrapped in a :class:`LockedTestSet` and is only scored at the
very end, after a one-shot ``unlock()`` — any earlier access raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from alphaforge.core.evaluate import evaluate
from alphaforge.core.fitness import forward_returns, mean_ic
from alphaforge.core.panel import Panel
from alphaforge.core.tree import Node
from alphaforge.validation.deflated_sharpe import deflated_sharpe_ratio, sharpe_ratio
from alphaforge.validation.pbo import pbo as compute_pbo
from alphaforge.validation.performance import long_short_returns, returns_matrix
from alphaforge.validation.splits import Split


class LockedTestSet:
    """Guards the locked test split: scoring against it before ``unlock()`` raises."""

    def __init__(self, test_dates: pd.DatetimeIndex) -> None:
        self._dates = pd.DatetimeIndex(test_dates)
        self._unlocked = False

    @property
    def is_locked(self) -> bool:
        return not self._unlocked

    def unlock(self) -> pd.DatetimeIndex:
        """One-shot reveal of the test dates — call only at final reporting."""
        self._unlocked = True
        return self._dates

    @property
    def dates(self) -> pd.DatetimeIndex:
        if not self._unlocked:
            raise RuntimeError(
                "the test split is locked; computing a metric on it before final reporting "
                "is forbidden (invariant 1)"
            )
        return self._dates


@dataclass
class OverfittingReport:
    oos_ic: float  # IC on the locked test split (the honest, default metric)
    deflated_sharpe: float  # P(genuine) after N trials; > 0.95 is significant
    pbo: float  # probability of backtest overfitting; >= 0.5 is a red flag
    train_ic: float
    n_trials: int
    significant: bool


def judge(
    best_tree: Node,
    trials: Sequence[Node],
    split: Split,
    panel: Panel,
    locked_test: LockedTestSet,
    *,
    n_trials: int,
    ic_method: str = "spearman",
    min_names: int = 5,
    n_blocks: int = 16,
) -> OverfittingReport:
    """Re-judge ``best_tree`` out-of-sample with a deflated Sharpe and PBO over ``trials``."""
    fwd = forward_returns(panel)
    factor = evaluate(best_tree, panel)
    if not isinstance(factor, pd.DataFrame):
        raise TypeError("best_tree must evaluate to a panel (SERIES/SIGNAL)")

    def ic_on(dates: pd.DatetimeIndex) -> float:
        return mean_ic(
            factor.loc[factor.index.isin(dates)],
            fwd.loc[fwd.index.isin(dates)],
            ic_method,
            absolute=True,
            min_names=min_names,
        )

    # Deflated Sharpe of the best, with the trial-Sharpe spread as the deflation variance.
    best_research = long_short_returns(factor, fwd)
    best_research = best_research.loc[best_research.index.isin(split.research)]
    trial_returns = returns_matrix(trials, panel, fwd, dates=split.research)
    trial_sharpes = trial_returns.apply(sharpe_ratio, axis=0).dropna()
    var_sr = float(trial_sharpes.var(ddof=1)) if len(trial_sharpes) > 1 else 0.0
    dsr = deflated_sharpe_ratio(best_research, n_trials, var_sr)
    pbo_value = float(compute_pbo(trial_returns, n_blocks=n_blocks)["pbo"])

    train_ic = ic_on(split.train)
    # Final report: unlock the test split exactly once, then score it.
    oos_ic = ic_on(locked_test.unlock())

    return OverfittingReport(
        oos_ic=oos_ic,
        deflated_sharpe=dsr,
        pbo=pbo_value,
        train_ic=train_ic,
        n_trials=n_trials,
        significant=bool(dsr > 0.95 and pbo_value < 0.5),
    )
