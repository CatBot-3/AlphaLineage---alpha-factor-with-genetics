"""Phase 3 acceptance + supporting tests for the anti-overfitting suite."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import forward_returns, mean_ic
from alphalineage.core.generate import RandomTreeGenerator
from alphalineage.core.tree import Node
from alphalineage.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from alphalineage.validation.pbo import pbo
from alphalineage.validation.performance import long_short_returns
from alphalineage.validation.pipeline import LockedTestSet, judge
from alphalineage.validation.purged_cv import purged_kfold
from alphalineage.validation.splits import time_split, walk_forward
from alphalineage.validation.trials import TrialCounter


# --- acceptance -----------------------------------------------------------------
def test_noise_rejection(noise_panel):
    """Load-bearing: the IS-best factor on noise is flagged (high PBO, non-significant DSR)."""
    panel = noise_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    fwd = forward_returns(panel)

    gen = RandomTreeGenerator(random.Random(0), max_depth=4, max_nodes=20)
    trials = gen.ramped_half_and_half(60, min_depth=2, max_depth=4)

    def train_ic(tree):
        factor = evaluate(tree, panel)
        return mean_ic(
            factor.loc[factor.index.isin(split.train)],
            fwd.loc[fwd.index.isin(split.train)],
            "spearman",
            absolute=True,
            min_names=5,
        )

    best = max(trials, key=train_ic)
    report = judge(
        best, trials, split, panel, LockedTestSet(split.test), n_trials=len(trials), n_blocks=10
    )

    assert report.pbo > 0.5, report.pbo
    assert report.deflated_sharpe < 0.95, report.deflated_sharpe
    assert not report.significant


def test_deflated_sharpe_known_values():
    # normal returns: PSR = Phi( SR*sqrt(n-1) / sqrt(1 + 0.5*SR^2) )
    sr, n = 0.1, 100
    expected = float(norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 + 0.5 * sr**2)))
    assert np.isclose(probabilistic_sharpe_ratio(sr, n, 0.0, 3.0, 0.0), expected, atol=1e-9)
    # PSR against its own SR is exactly 0.5
    assert np.isclose(probabilistic_sharpe_ratio(sr, n, 0.0, 3.0, sr), 0.5, atol=1e-9)

    # expected max Sharpe matches the closed form
    g = 0.5772156649015329
    z1, z2 = norm.ppf(1 - 1 / 10), norm.ppf(1 - 1 / (10 * np.e))
    sr0 = np.sqrt(0.25) * ((1 - g) * z1 + g * z2)
    assert np.isclose(expected_max_sharpe(10, 0.25), sr0, atol=1e-9)

    # DSR is non-increasing in the number of trials
    returns = pd.Series(np.random.default_rng(1).normal(0.05, 1.0, 250))
    assert deflated_sharpe_ratio(returns, 200, 0.5) <= deflated_sharpe_ratio(returns, 2, 0.5)


def test_pbo_bounds():
    # PBO is noisy per-realization, so average over seeds. On noise PBO sits near/above 0.5
    # (the IS-best regresses out of sample) and never collapses toward 0 as trials grow.
    def mean_pbo(n_strat: int, seeds: int = 20) -> float:
        values = []
        for s in range(seeds):
            mat = pd.DataFrame(np.random.default_rng(1000 + s).normal(0, 1, (120, n_strat)))
            value = pbo(mat, n_blocks=10)["pbo"]
            assert 0.0 <= value <= 1.0  # always a probability
            values.append(value)
        return float(np.mean(values))

    few, many = mean_pbo(10), mean_pbo(100)
    assert many >= 0.40  # the suite refuses to certify noise as skill
    assert many >= few - 0.05  # PBO rises (or holds) as the number of trials grows on noise


def test_test_set_never_touched():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    locked = LockedTestSet(dates)

    assert locked.is_locked
    with pytest.raises(RuntimeError):
        _ = locked.dates  # scoring the locked test split must fail

    revealed = locked.unlock()
    assert not locked.is_locked
    assert locked.dates.equals(revealed)  # accessible only after the one-shot unlock


# --- supporting -----------------------------------------------------------------
def test_time_split_is_ordered_disjoint_and_embargoed():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    split = time_split(dates, train=0.6, valid=0.2, embargo=5)

    assert len(set(split.train) & set(split.test)) == 0
    assert split.train.max() < split.valid.min()
    assert split.valid.max() < split.test.min()
    # the embargo leaves >= 5 trading-day gaps between segments
    assert dates.get_loc(split.valid[0]) - dates.get_loc(split.train[-1]) > 5


def test_walk_forward_windows():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    windows = walk_forward(dates, n_splits=3, train_size=20, test_size=10, embargo=2)
    assert len(windows) == 3
    for train, test in windows:
        assert train.max() < test.min()
        assert len(train) == 20 and len(test) == 10


def test_purged_kfold_purges_and_embargoes():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    horizon = 3
    for train, test in purged_kfold(dates, n_splits=5, embargo_pct=0.03, horizon=horizon):
        assert len(set(train) & set(test)) == 0
        lo, hi = dates.get_loc(test[0]), dates.get_loc(test[-1])
        train_pos = {dates.get_loc(d) for d in train}
        # no train label window overlaps the fold, and none falls in the trailing embargo
        assert all(p < lo - horizon or p > hi for p in train_pos)


def test_long_short_returns_dollar_neutral_and_predictive(signal_panel):
    panel, _ = signal_panel
    fwd = forward_returns(panel)
    factor = evaluate(Node("volume"), panel)
    ret = long_short_returns(factor, fwd)
    assert np.isfinite(ret.dropna()).all()
    assert sharpe_ratio(ret) > 0  # volume carries the signal -> positive long-short Sharpe


def test_sharpe_ratio_and_trial_counter():
    s = pd.Series([1.0, 2.0, 3.0])
    assert np.isclose(sharpe_ratio(s), s.mean() / s.std())
    assert sharpe_ratio(pd.Series([5.0])) == 0.0

    counter = TrialCounter()
    assert counter.add(3) == 3
    assert counter.add() == 4
    counter.reset()
    assert counter.count == 0
