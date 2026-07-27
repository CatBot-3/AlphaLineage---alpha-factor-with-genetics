"""Phase 4 acceptance + supporting tests for the backtest/portfolio (tests/test_backtest.py)."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from alphalineage.backtest.costs import TransactionCostModel, turnover_series
from alphalineage.backtest.engine import backtest, compare_schemes, comparison_frame
from alphalineage.backtest.metrics import max_drawdown, turnover
from alphalineage.backtest.portfolio import QuantileLongShort, RankProportional, neutralize
from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import forward_returns
from alphalineage.core.generate import RandomTreeGenerator
from alphalineage.core.tree import Node
from alphalineage.validation.deflated_sharpe import sharpe_ratio
from alphalineage.validation.pipeline import LockedTestSet, judge
from alphalineage.validation.splits import time_split


# --- acceptance -----------------------------------------------------------------
def test_backtest_matches_independent_recompute(synthetic_panel):
    panel = synthetic_panel
    fwd = forward_returns(panel)
    factor = evaluate(Node("rank", (Node("close"),)), panel)
    scheme = RankProportional()
    costs = TransactionCostModel(commission_bps=1.0, slippage_bps=5.0)
    result = backtest(factor, panel, fwd, scheme, costs)

    # independent vectorized recomputation
    w, r = scheme.weights(factor).align(fwd, join="inner")
    gross = (w * r).sum(axis=1, min_count=1)
    dw = w.diff()
    dw.iloc[0] = w.iloc[0]
    cost = dw.abs().sum(axis=1) * (6.0 / 1e4)
    net = gross - cost

    pd.testing.assert_series_equal(result.gross_returns, gross)
    pd.testing.assert_series_equal(result.net_returns, net)
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(result.realization_dates.iloc[:-1]),
        pd.DatetimeIndex(factor.index[1:]),
    )
    assert pd.isna(result.realization_dates.iloc[-1])
    assert np.isclose(result.net_sharpe, sharpe_ratio(net) * np.sqrt(252))


def test_multi_session_horizon_averages_overlapping_signal_cohorts(synthetic_panel):
    factor = evaluate(Node("rank", (Node("close"),)), synthetic_panel)
    scheme = RankProportional()
    zero_costs = TransactionCostModel(0.0, 0.0)
    horizon = 3

    result = backtest(
        factor,
        synthetic_panel,
        forward_returns(synthetic_panel, horizon),
        scheme,
        zero_costs,
        horizon=horizon,
    )

    signal_weights = scheme.weights(factor)
    cohort_sum = sum(
        (signal_weights.shift(lag).fillna(0.0) for lag in range(1, horizon + 1)),
        start=signal_weights * 0.0,
    )
    active = pd.Series(
        np.minimum(np.arange(len(signal_weights)), horizon),
        index=signal_weights.index,
        dtype="float64",
    ).replace(0.0, np.nan)
    realized_weights = cohort_sum.div(active, axis=0).fillna(0.0)
    expected_realized = (
        realized_weights * synthetic_panel["returns"]
    ).sum(axis=1, min_count=1)
    expected_by_signal = expected_realized.shift(-1)

    pd.testing.assert_series_equal(result.gross_returns, expected_by_signal)
    # The second realized book averages the first two independently formed signal cohorts.
    pd.testing.assert_series_equal(
        realized_weights.iloc[2],
        (signal_weights.iloc[0] + signal_weights.iloc[1]) / 2.0,
        check_names=False,
    )


def test_sliced_multi_horizon_turnover_matches_full_path_cost_transitions(synthetic_panel):
    factor = evaluate(Node("returns"), synthetic_panel)
    scheme = RankProportional()
    costs = TransactionCostModel(commission_bps=3.0, slippage_bps=7.0)
    horizon = 3
    report_dates = pd.DatetimeIndex(factor.index[30:50])

    result = backtest(
        factor,
        synthetic_panel,
        forward_returns(synthetic_panel, horizon),
        scheme,
        costs,
        dates=report_dates,
        horizon=horizon,
    )

    signal_weights = scheme.weights(factor)
    realized_weights = (
        signal_weights.shift(1).rolling(window=horizon, min_periods=1).mean().fillna(0.0)
    )
    expected_turnover = turnover_series(realized_weights).shift(-1).loc[report_dates]
    expected_costs = expected_turnover * costs.rate

    pd.testing.assert_series_equal(result.traded_notional, expected_turnover)
    pd.testing.assert_series_equal(result.transaction_costs, expected_costs)
    pd.testing.assert_series_equal(
        result.net_returns,
        result.gross_returns - result.transaction_costs,
    )
    assert result.turnover == pytest.approx(expected_turnover.mean())

    # Recomputing after slicing drops the real transition into the first report position.
    sliced_weights = realized_weights.shift(-1).loc[report_dates]
    assert result.turnover != pytest.approx(turnover(sliced_weights))


def test_cost_sensitivity(signal_panel):
    panel, _ = signal_panel
    fwd = forward_returns(panel)
    factor = evaluate(Node("volume"), panel)  # carries the signal -> gross profitable
    scheme = QuantileLongShort(0.2)

    cheap = backtest(factor, panel, fwd, scheme, TransactionCostModel(0.0, 0.0))
    pricey = backtest(factor, panel, fwd, scheme, TransactionCostModel(500.0, 500.0))

    assert cheap.usable  # profitable gross
    assert not pricey.usable  # destroyed by realistic-to-extreme costs (invariant 6)
    assert pricey.net_returns.mean() < cheap.net_returns.mean()


def test_turnover_reported(synthetic_panel):
    panel = synthetic_panel
    factor = evaluate(Node("ts_mean", (Node("returns"), Node("window", value=3))), panel)
    weights = QuantileLongShort(0.2).weights(factor)

    t = turnover(weights)
    assert 0.0 <= t <= 2.0  # unit-gross weights bound per-period turnover
    assert t > 0.0  # a changing signal trades


# --- supporting -----------------------------------------------------------------
def test_schemes_are_dollar_neutral_and_unit_gross(synthetic_panel):
    factor = evaluate(Node("rank", (Node("close"),)), synthetic_panel)
    for scheme in (QuantileLongShort(0.2), RankProportional()):
        w = scheme.weights(factor)
        gross = w.abs().sum(axis=1)
        assert np.allclose(w.sum(axis=1).to_numpy(), 0.0, atol=1e-9)
        assert np.allclose(gross[gross > 0].to_numpy(), 1.0, atol=1e-9)


class _FixedWeightingScheme:
    name = "fixed"

    def __init__(self, row: list[float]) -> None:
        self.row = row

    def weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(self.row, (len(factor), 1)),
            index=factor.index,
            columns=factor.columns,
        )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([0.5, -0.5, 0.0, 0.0, 0.0, np.inf], "must be finite"),
        ([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], "must be dollar neutral"),
        ([0.25, -0.25, 0.0, 0.0, 0.0, 0.0], "must have unit gross exposure"),
    ],
)
def test_backtest_rejects_invalid_active_weight_rows(
    synthetic_panel,
    row,
    message,
):
    factor = evaluate(Node("close"), synthetic_panel)

    with pytest.raises(ValueError, match=message):
        backtest(
            factor,
            synthetic_panel,
            forward_returns(synthetic_panel),
            _FixedWeightingScheme(row),
            TransactionCostModel(),
        )


def test_backtest_allows_cash_rows_from_weighting_scheme(synthetic_panel):
    factor = evaluate(Node("close"), synthetic_panel)
    result = backtest(
        factor,
        synthetic_panel,
        forward_returns(synthetic_panel),
        _FixedWeightingScheme([0.0] * factor.shape[1]),
        TransactionCostModel(),
    )

    assert not result.active_exposure.any()


def test_quantile_concentrates_rankproportional_spreads(synthetic_panel):
    factor = evaluate(Node("rank", (Node("close"),)), synthetic_panel)
    n = factor.shape[1]
    q_positions = (QuantileLongShort(0.2).weights(factor) != 0).sum(axis=1).mean()
    rp_positions = (RankProportional().weights(factor) != 0).sum(axis=1).mean()
    assert q_positions < rp_positions
    assert rp_positions >= n - 1  # holds (nearly) all names


def test_quantile_boundary_ties_expand_symmetrically_without_symbol_order_tiebreak():
    factor = pd.DataFrame(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        index=pd.DatetimeIndex(["2024-01-02"]),
        columns=[f"S{index}" for index in range(10)],
    )
    weights = QuantileLongShort(0.2).weights(factor).iloc[0]

    assert (weights.iloc[:5] == -0.1).all()
    assert (weights.iloc[5:] == 0.1).all()
    assert weights.sum() == pytest.approx(0.0)
    assert weights.abs().sum() == pytest.approx(1.0)


def test_quantile_overlapping_boundary_tie_stays_cash_instead_of_partial_gross():
    factor = pd.DataFrame(
        [[0.0, 1.0, 1.0, 1.0, 2.0]],
        index=pd.DatetimeIndex(["2024-01-02"]),
        columns=list("ABCDE"),
    )
    weights = QuantileLongShort(0.4).weights(factor).iloc[0]

    assert (weights == 0.0).all()


def test_rank_proportional_excludes_nonfinite_factor_values():
    factor = pd.DataFrame(
        [[1.0, 2.0, np.inf, -np.inf, np.nan]],
        index=pd.DatetimeIndex(["2024-01-02"]),
        columns=list("ABCDE"),
    )
    weights = RankProportional().weights(factor).iloc[0]

    assert weights["A"] == pytest.approx(-0.5)
    assert weights["B"] == pytest.approx(0.5)
    assert (weights[["C", "D", "E"]] == 0.0).all()


def test_constant_factor_is_reported_as_no_exposure_not_zero_return_evidence(
    synthetic_panel,
):
    from alphalineage.backtest.reporting import backtest_report

    factor = synthetic_panel["close"] * 0.0 - 1.0
    report = backtest_report(
        factor,
        synthetic_panel,
        forward_returns(synthetic_panel),
        QuantileLongShort(),
        TransactionCostModel(),
        synthetic_panel.dates,
    )

    assert report["observations"] == 0
    assert report["calendar_observations"] > 0
    assert report["portfolio_health"]["valid"] is False
    assert report["portfolio_health"]["active_observations"] == 0
    assert report["portfolio_health"]["reason"] == "no_exposure"
    assert report["portfolio_health"]["flat_factor_dates"] > 0
    assert report["metrics"]["gross_sharpe"] is None
    assert report["metrics"]["net_sharpe"] is None
    assert report["metrics"]["max_drawdown"] is None
    assert "no portfolio exposure" in " ".join(report["integrity"]["issues"])
    assert len(report["normalized_equity"]) == 1


def test_neutralize_removes_group_means(synthetic_panel):
    factor = evaluate(Node("close"), synthetic_panel)
    syms = list(factor.columns)
    groups = pd.Series({s: ("A" if i < len(syms) // 2 else "B") for i, s in enumerate(syms)})
    neutral = neutralize(factor, groups=groups)
    for label in ("A", "B"):
        cols = [s for s in syms if groups[s] == label]
        assert np.allclose(neutral[cols].mean(axis=1).to_numpy(), 0.0, atol=1e-9)


def test_max_drawdown():
    # equity 1.1 -> 0.55 -> 0.55: worst drawdown is 0.55/1.1 - 1 = -0.5
    assert np.isclose(max_drawdown(pd.Series([0.1, -0.5, 0.0])), 0.55 / 1.1 - 1.0)
    assert max_drawdown(pd.Series([0.1, 0.2])) == 0.0  # monotone up -> no drawdown


def test_compare_schemes_side_by_side(signal_panel):
    panel, _ = signal_panel
    fwd = forward_returns(panel)
    factor = evaluate(Node("volume"), panel)
    results = compare_schemes(
        factor, panel, fwd, [QuantileLongShort(0.2), RankProportional()], TransactionCostModel()
    )
    table = comparison_frame(results)
    assert list(table["scheme"]) == ["quantile_ls", "rank_proportional"]
    assert table["avg_positions"].nunique() == 2  # distinct position magnitudes


def test_more_schemes_deflate_more(signal_panel):
    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    trials = RandomTreeGenerator(random.Random(0), max_depth=4, max_nodes=20).ramped_half_and_half(
        30, min_depth=2, max_depth=4
    )
    best = Node("volume")

    one = judge(best, trials, split, panel, LockedTestSet(split.test), n_trials=30, n_blocks=8)
    five = judge(
        best, trials, split, panel, LockedTestSet(split.test), n_trials=30, n_schemes=5, n_blocks=8
    )
    assert five.deflated_sharpe <= one.deflated_sharpe  # more schemes -> harder deflation
    assert five.n_trials == 30 * 5
