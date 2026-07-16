"""P3-T6 - re-judge a Phase-2 best factor honestly, out of sample.

Ties the suite together: given the GP's chosen factor and the trials it searched, produce an
:class:`OverfittingReport` whose headline numbers default to out-of-sample / deflated
(invariant 1). The test split is wrapped in a :class:`LockedTestSet` and is only scored at the
very end, after a one-shot ``unlock()`` - any earlier access raises.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import forward_returns, mean_ic
from alphalineage.core.gp import TrainingCancelled
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node, to_json
from alphalineage.validation.deflated_sharpe import deflated_sharpe_ratio, sharpe_ratio
from alphalineage.validation.pbo import (
    ReportReturnSummary,
    pbo_from_returns,
    pbo_from_summaries,
    summarize_report_returns,
)
from alphalineage.validation.performance import long_short_returns, tree_returns
from alphalineage.validation.splits import Split


class LockedTestSet:
    """Guards the locked test split: scoring against it before ``unlock()`` raises."""

    def __init__(self, test_dates: pd.DatetimeIndex) -> None:
        self._dates = pd.DatetimeIndex(test_dates)
        self._unlocked = False

    @property
    def is_locked(self) -> bool:
        return not self._unlocked

    def unlock(self) -> pd.DatetimeIndex:
        """One-shot reveal of the test dates - call only at final reporting."""
        if self._unlocked:
            raise RuntimeError("the locked test set has already been unlocked")
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
    oos_backtest: dict[str, Any] | None = None


def judge(
    best_tree: Node,
    trials: Sequence[Node],
    split: Split,
    panel: Panel,
    locked_test: LockedTestSet,
    *,
    n_trials: int,
    ic_method: str = "spearman",
    horizon: int = 1,
    fwd: pd.DataFrame | None = None,
    min_names: int = 5,
    n_blocks: int = 16,
    returns_fn: Callable[[Node], pd.Series] | None = None,
    returns_from_factor: Callable[[pd.DataFrame], pd.Series] | None = None,
    n_schemes: int = 1,
    progress: Callable[[int, int], None] | None = None,
    stop: Callable[[], bool] | None = None,
    on_finalizing: Callable[[], None] | None = None,
    summary_cache: MutableMapping[str, ReportReturnSummary] | None = None,
    summary_key: Callable[[Node], str] | None = None,
    holdout_reporter: Callable[[pd.DataFrame, pd.DatetimeIndex], dict[str, Any]] | None = None,
) -> OverfittingReport:
    """Re-judge ``best_tree`` out-of-sample with a deflated Sharpe and PBO over ``trials``.

    ``returns_fn`` injects the per-tree return series (default: gross long-short; pass a
    cost-aware closure to deflate on *net* returns). ``n_schemes`` multiplies the trial count
    - trying K weighting schemes is another overfitting axis, so it deflates harder. ``progress``
    receives ``(completed_trials, total_trials)`` while research returns are streamed. Pass an
    already-computed ``fwd`` target to share it with training/report construction. ``stop`` is
    honored until locked-test finalization begins; ``on_finalizing`` marks that boundary.
    ``returns_from_factor`` lets the best tree reuse the factor frame already evaluated for IC.
    """
    if fwd is None:
        fwd = forward_returns(panel, horizon)
    factor = evaluate(best_tree, panel)
    if not isinstance(factor, pd.DataFrame):
        raise TypeError("best_tree must evaluate to a panel (SERIES/SIGNAL)")
    returns_of = returns_fn or (lambda tree: tree_returns(tree, panel, fwd))
    best_returns = (
        returns_from_factor(factor)
        if returns_from_factor is not None
        else long_short_returns(factor, fwd)
        if returns_fn is None
        else returns_of(best_tree)
    )
    best_key = to_json(best_tree)

    def cached_returns_of(tree: Node) -> pd.Series:
        return best_returns if to_json(tree) == best_key else returns_of(tree)

    def ic_on(dates: pd.DatetimeIndex) -> float:
        return mean_ic(
            factor.loc[factor.index.isin(dates)],
            fwd.loc[fwd.index.isin(dates)],
            ic_method,
            absolute=True,
            min_names=min_names,
        )

    def research(series: pd.Series) -> pd.Series:
        return series.loc[series.index.isin(split.research)]

    # Deflated Sharpe of the best, with the trial-Sharpe spread as the deflation variance.
    best_research = research(best_returns)
    research_index = pd.DatetimeIndex(split.research)
    trial_sharpe_values: list[float] = []

    def streamed_trial_returns() -> Iterator[pd.Series]:
        total = len(trials)
        for completed, tree in enumerate(trials, start=1):
            if stop is not None and stop():
                raise TrainingCancelled("report cancelled before locked-test finalization")
            series = research(cached_returns_of(tree))
            trial_sharpe_values.append(sharpe_ratio(series))
            if progress is not None:
                progress(completed, total)
            yield series

    # Sessions can persist these compact columns and evaluate only newly discovered trials on
    # continuation. A missing/invalid cache retains the streaming reference path; pathological
    # infinities also fall back because their sliced NumPy semantics are not moment-reconstructible.
    pbo_result = None
    if summary_cache is not None:
        summaries: list[ReportReturnSummary] = []
        key_of = summary_key or to_json
        try:
            for completed, tree in enumerate(trials, start=1):
                if stop is not None and stop():
                    raise TrainingCancelled("report cancelled before locked-test finalization")
                key = key_of(tree)
                summary = summary_cache.get(key)
                if summary is None:
                    summary = summarize_report_returns(
                        research(cached_returns_of(tree)), research_index, n_blocks=n_blocks
                    )
                    summary_cache[key] = summary
                summaries.append(summary)
                trial_sharpe_values.append(float(summary["sharpe"]))
                if progress is not None:
                    progress(completed, len(trials))
            pbo_result = pbo_from_summaries(summaries)
        except (KeyError, TypeError, ValueError):
            # Ignore a stale/corrupt summary column and recompute through the exact reference.
            trial_sharpe_values.clear()
            pbo_result = None

    if pbo_result is None:
        pbo_result = pbo_from_returns(
            streamed_trial_returns(),
            research_index,
            n_blocks=n_blocks,
            nonfinite_fallback=lambda: pd.DataFrame(
                {i: research(cached_returns_of(tree)) for i, tree in enumerate(trials)}
            ),
            require_full_index=True,
        )
    trial_sharpes = pd.Series(trial_sharpe_values, dtype="float64").dropna()
    var_sr = float(trial_sharpes.var(ddof=1)) if len(trial_sharpes) > 1 else 0.0
    effective_trials = n_trials * max(1, n_schemes)
    dsr = deflated_sharpe_ratio(best_research, effective_trials, var_sr)
    pbo_value = float(pbo_result["pbo"])

    train_ic = ic_on(split.train)
    # Final report: unlock the test split exactly once, then score it.
    if stop is not None and stop():
        raise TrainingCancelled("report cancelled before locked-test finalization")
    if on_finalizing is not None:
        on_finalizing()
    test_dates = locked_test.unlock()
    oos_ic = ic_on(test_dates)
    # Reuse the already evaluated factor and the dates returned by the single unlock. Richer
    # holdout evidence must never cause another locked-set read or tree evaluation.
    oos_backtest = holdout_reporter(factor, test_dates) if holdout_reporter is not None else None

    return OverfittingReport(
        oos_ic=oos_ic,
        deflated_sharpe=dsr,
        pbo=pbo_value,
        train_ic=train_ic,
        n_trials=effective_trials,
        significant=bool(dsr > 0.95 and pbo_value < 0.5),
        oos_backtest=oos_backtest,
    )
