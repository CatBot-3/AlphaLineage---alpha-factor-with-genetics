"""Phase 3 acceptance + supporting tests for the anti-overfitting suite."""

from __future__ import annotations

import json
import random
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import forward_returns, mean_ic
from alphalineage.core.generate import RandomTreeGenerator
from alphalineage.core.gp import TrainingCancelled
from alphalineage.core.tree import Node, to_json
from alphalineage.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from alphalineage.validation.pbo import (
    pbo,
    pbo_from_returns,
    pbo_from_summaries,
    summarize_report_returns,
)
from alphalineage.validation.performance import long_short_returns, tree_returns
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


def _sliced_pbo_reference(returns_matrix: pd.DataFrame, n_blocks: int) -> dict[str, object]:
    data = np.nan_to_num(returns_matrix.to_numpy(dtype="float64"), nan=0.0)
    n_obs, n_strat = data.shape
    s = min(n_blocks, n_obs)
    s -= s % 2
    if s < 2 or n_strat < 2:
        return {"pbo": float("nan"), "logits": [], "n_splits": 0}
    bounds = np.linspace(0, n_obs, s + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(s)]
    logits = []
    for selected in combinations(range(s), s // 2):
        inside = np.concatenate([blocks[i] for i in selected])
        outside = np.concatenate([blocks[i] for i in range(s) if i not in selected])

        def sharpes(rows):
            sample = data[rows]
            mean, std = sample.mean(axis=0), sample.std(axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(std > 0, mean / std, 0.0)

        best = int(np.argmax(sharpes(inside)))
        rank = pd.Series(sharpes(outside)).rank().iloc[best]
        omega = min(max(rank / (n_strat + 1), 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))
    values = np.asarray(logits)
    return {"pbo": float(np.mean(values <= 0.0)), "logits": logits, "n_splits": len(logits)}


def test_pbo_block_statistics_preserve_sliced_cscv_order_and_edge_cases():
    rng = np.random.default_rng(42)
    normal = rng.normal(size=(23, 7))  # uneven block sizes
    normal[rng.random(normal.shape) < 0.12] = np.nan
    normal[:, 1] = normal[:, 0]  # exact average-rank ties
    cases = [normal, np.zeros((23, 7)), np.full((23, 7), 0.1)]
    for values in cases:
        matrix = pd.DataFrame(values)
        expected = _sliced_pbo_reference(matrix, n_blocks=6)
        actual = pbo(matrix, n_blocks=6)
        assert actual["n_splits"] == expected["n_splits"]
        assert actual["pbo"] == expected["pbo"]
        np.testing.assert_allclose(actual["logits"], expected["logits"], rtol=1e-13, atol=1e-13)


def test_streamed_pbo_statistics_match_materialized_matrix():
    rng = np.random.default_rng(7)
    index = pd.date_range("2023-01-01", periods=31, freq="B")
    matrix = pd.DataFrame(rng.normal(size=(31, 9)), index=index)
    matrix.iloc[::5, 2] = np.nan
    columns = [matrix[column].drop(index=index[::7]) for column in matrix]
    expected = pbo(pd.concat(columns, axis=1).reindex(index), n_blocks=8)
    actual = pbo_from_returns(iter(columns), index, n_blocks=8)
    assert actual["pbo"] == expected["pbo"]
    assert actual["n_splits"] == expected["n_splits"]
    np.testing.assert_allclose(actual["logits"], expected["logits"], rtol=1e-13, atol=1e-13)


def test_json_safe_report_return_summaries_preserve_sharpe_and_pbo():
    rng = np.random.default_rng(17)
    index = pd.date_range("2022-01-01", periods=29, freq="B")
    values = rng.normal(size=(29, 7))
    values[::4, 2] = np.nan
    values[:, 1] = values[:, 0]  # tied strategies
    values[:, 6] = 0.1  # non-zero constant exercises NumPy std parity metadata
    matrix = pd.DataFrame(values, index=index)

    summaries = [summarize_report_returns(matrix[column], index, 6) for column in matrix]
    restored = json.loads(json.dumps(summaries, allow_nan=False))
    for column, summary in zip(matrix, restored, strict=True):
        assert summary["sharpe"] == pytest.approx(sharpe_ratio(matrix[column]))

    expected = _sliced_pbo_reference(matrix, n_blocks=6)
    actual = pbo_from_summaries(restored)
    assert actual["pbo"] == expected["pbo"]
    assert actual["n_splits"] == expected["n_splits"]
    np.testing.assert_allclose(actual["logits"], expected["logits"], rtol=1e-13, atol=1e-13)


def test_report_return_summaries_reject_mixed_layouts():
    index = pd.date_range("2022-01-01", periods=20, freq="B")
    series = pd.Series(np.arange(20, dtype=float), index=index)
    a = summarize_report_returns(series, index, 4)
    b = summarize_report_returns(series.iloc[:-1], index[:-1], 4)
    with pytest.raises(ValueError, match="same observation/block layout"):
        pbo_from_summaries([a, b])


def test_test_set_never_touched():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    locked = LockedTestSet(dates)

    assert locked.is_locked
    with pytest.raises(RuntimeError):
        _ = locked.dates  # scoring the locked test split must fail

    revealed = locked.unlock()
    assert not locked.is_locked
    assert locked.dates.equals(revealed)  # accessible only after the one-shot unlock
    with pytest.raises(RuntimeError, match="already been unlocked"):
        locked.unlock()


def test_report_progress_cancellation_and_finalizing_boundary(signal_panel):
    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    trials = [Node("volume"), Node("returns")]

    cancelled_locked = LockedTestSet(split.test)
    progress_calls: list[tuple[int, int]] = []
    stop_polls = 0

    def stop_after_first_trial() -> bool:
        nonlocal stop_polls
        stop_polls += 1
        return stop_polls > 1

    with pytest.raises(TrainingCancelled):
        judge(
            trials[0],
            trials,
            split,
            panel,
            cancelled_locked,
            n_trials=2,
            n_blocks=2,
            progress=lambda done, total: progress_calls.append((done, total)),
            stop=stop_after_first_trial,
        )
    assert progress_calls == [(1, 2)]
    assert cancelled_locked.is_locked

    completed_locked = LockedTestSet(split.test)
    events: list[object] = []

    def finalizing() -> None:
        assert completed_locked.is_locked
        events.append("finalizing")

    judge(
        trials[0],
        trials,
        split,
        panel,
        completed_locked,
        n_trials=2,
        n_blocks=2,
        fwd=forward_returns(panel),
        progress=lambda done, total: events.append((done, total)),
        on_finalizing=finalizing,
    )
    assert events == [(1, 2), (2, 2), "finalizing"]
    assert not completed_locked.is_locked


def test_report_summary_cache_reuses_old_trials(signal_panel):
    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    fwd = forward_returns(panel)
    trials = [Node("volume"), Node("close")]
    cache = {}
    calls = 0

    def returns_of(tree: Node):
        nonlocal calls
        calls += 1
        return tree_returns(tree, panel, fwd)

    judge(
        trials[0],
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=2,
        n_blocks=2,
        fwd=fwd,
        returns_fn=returns_of,
        summary_cache=cache,
        summary_key=to_json,
    )
    first_calls = calls
    assert len(cache) == 2

    judge(
        trials[0],
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=2,
        n_blocks=2,
        fwd=fwd,
        returns_fn=returns_of,
        summary_cache=cache,
        summary_key=to_json,
    )
    # The full best series is retained for DSR skew/kurtosis; cached trial summaries need no
    # factor/portfolio re-evaluation on the second report.
    assert calls - first_calls == 1


# --- supporting -----------------------------------------------------------------
def test_time_split_is_ordered_disjoint_and_embargoed():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    split = time_split(dates, train=0.6, valid=0.2, embargo=5)

    assert len(set(split.train) & set(split.test)) == 0
    assert split.train.max() < split.valid.min()
    assert split.valid.max() < split.test.min()
    # the embargo leaves >= 5 trading-day gaps between segments
    assert dates.get_loc(split.valid[0]) - dates.get_loc(split.train[-1]) > 5


def test_time_split_validates_parameters_and_label_horizon():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    with pytest.raises(ValueError, match=r"train \+ valid"):
        time_split(dates, train=0.8, valid=0.2)
    with pytest.raises(ValueError, match="at least horizon"):
        time_split(dates, embargo=2, horizon=3)
    with pytest.raises(ValueError, match="sorted"):
        time_split(dates[::-1])

    split = time_split(dates, embargo=3, horizon=3)
    train_end = dates.get_loc(split.train[-1])
    valid_start = dates.get_loc(split.valid[0])
    assert train_end + 3 < valid_start


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
