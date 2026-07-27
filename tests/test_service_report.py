"""Acceptance test: the API report's deflation basis is NET returns (invariant 6).

Kept separate from tests/test_api.py so it can run without importing the FastAPI app
(``alphalineage.api.service`` has no app/HTTP dependency).
"""

from __future__ import annotations

import random

import pytest

from alphalineage.api.service import build_report, user_operator_count
from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.engine import backtest, net_return_fn
from alphalineage.backtest.portfolio import QuantileLongShort
from alphalineage.core.evaluate import evaluate
from alphalineage.core.extensions import ARG, register_operator, unregister_operator
from alphalineage.core.fitness import forward_returns, mean_ic
from alphalineage.core.generate import RandomTreeGenerator
from alphalineage.core.tree import Node
from alphalineage.core.types import DType
from alphalineage.validation.pipeline import LockedTestSet, judge
from alphalineage.validation.splits import time_split


def test_build_report_deflates_on_net_returns(signal_panel):
    """The report judges on NET (after-cost) returns with the default weighting scheme +
    cost model - parity with the CLI verdict in scripts/run_gp.py."""
    panel, _ = signal_panel
    split = time_split(panel.dates)
    gen = RandomTreeGenerator(random.Random(0), max_depth=4, max_nodes=20)
    trials = gen.ramped_half_and_half(12, min_depth=2, max_depth=4)
    best = trials[0]

    report = build_report(best, trials, split, panel, searched_trials=len(trials))

    fwd = forward_returns(panel)
    expected = judge(
        best,
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=len(trials),
        returns_fn=net_return_fn(panel, fwd, QuantileLongShort(), TransactionCostModel()),
    )
    assert report["deflated_sharpe"] == pytest.approx(expected.deflated_sharpe)
    assert report["pbo"] == pytest.approx(expected.pbo)
    assert report["oos_ic"] == pytest.approx(expected.oos_ic)
    assert report["n_trials"] == expected.n_trials == len(trials)  # one scheme => no multiplier
    holdout = report["oos_backtest"]
    assert holdout["start"] == split.test.min().date().isoformat()
    assert holdout["end"] == split.test.max().date().isoformat()
    assert holdout["metrics"]["mean_abs_ic"] == pytest.approx(report["oos_ic"])
    assert len(holdout["returns"]) == len(split.test) - 1
    assert len(holdout["normalized_equity"]) == len(split.test)
    assert holdout["normalized_equity"][0] == {
        "date": split.test.min().date().isoformat(),
        "value": 1.0,
    }
    assert holdout["returns"][0]["signal_date"] == split.test[0].date().isoformat()
    assert holdout["returns"][0]["date"] == split.test[1].date().isoformat()
    factor = evaluate(best, panel)
    assert hasattr(factor, "columns")
    expected_backtest = backtest(
        factor,
        panel,
        fwd,
        QuantileLongShort(),
        TransactionCostModel(),
        dates=split.test,
    )
    assert holdout["metrics"]["gross_sharpe"] == pytest.approx(expected_backtest.gross_sharpe)
    assert holdout["metrics"]["net_sharpe"] == pytest.approx(expected_backtest.net_sharpe)
    assert holdout["metrics"]["max_drawdown"] == pytest.approx(expected_backtest.max_drawdown)
    assert holdout["metrics"]["turnover"] == pytest.approx(expected_backtest.turnover)


def test_build_report_accepts_custom_scheme_and_costs(signal_panel):
    """Costs/scheme are injectable; zero costs reproduce the gross-basis verdict."""
    panel, _ = signal_panel
    split = time_split(panel.dates)
    gen = RandomTreeGenerator(random.Random(1), max_depth=4, max_nodes=20)
    trials = gen.ramped_half_and_half(12, min_depth=2, max_depth=4)
    best = trials[0]

    zero_costs = TransactionCostModel(commission_bps=0.0, slippage_bps=0.0)
    report = build_report(best, trials, split, panel, searched_trials=len(trials), costs=zero_costs)

    fwd = forward_returns(panel)
    expected = judge(
        best,
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=len(trials),
        returns_fn=net_return_fn(panel, fwd, QuantileLongShort(), zero_costs),
    )
    assert report["deflated_sharpe"] == pytest.approx(expected.deflated_sharpe)


def test_build_report_uses_configured_horizon_and_ic_method(signal_panel):
    panel, _ = signal_panel
    horizon = 3
    split = time_split(panel.dates, horizon=horizon)
    gen = RandomTreeGenerator(random.Random(2), max_depth=4, max_nodes=20)
    trials = gen.ramped_half_and_half(12, min_depth=2, max_depth=4)
    # Use the fixture's known non-degenerate predictive field. A randomly generated first tree
    # can be constant/invalid on this split, making both target definitions report IC=0 and
    # weakening the horizon regression into a seed accident.
    best = Node("volume")
    trials[0] = best
    zero_costs = TransactionCostModel(commission_bps=0.0, slippage_bps=0.0)

    report = build_report(
        best,
        trials,
        split,
        panel,
        searched_trials=len(trials),
        horizon=horizon,
        ic_method="pearson",
        costs=zero_costs,
    )

    # Construct the target independently of ``forward_returns`` so this remains a regression
    # test for the horizon contract rather than reproducing the implementation under test.
    cumulative_target = panel["close"].shift(-horizon).div(panel["close"]).sub(1.0)
    legacy_offset_target = panel["returns"].shift(-horizon)
    factor = evaluate(best, panel)
    assert hasattr(factor, "columns")
    expected_oos_ic = mean_ic(
        factor.loc[factor.index.isin(split.test)],
        cumulative_target.loc[cumulative_target.index.isin(split.test)],
        "pearson",
        absolute=True,
        min_names=5,
    )
    expected_train_ic = mean_ic(
        factor.loc[factor.index.isin(split.train)],
        cumulative_target.loc[cumulative_target.index.isin(split.train)],
        "pearson",
        absolute=True,
        min_names=5,
    )
    legacy_oos_ic = mean_ic(
        factor.loc[factor.index.isin(split.test)],
        legacy_offset_target.loc[legacy_offset_target.index.isin(split.test)],
        "pearson",
        absolute=True,
        min_names=5,
    )
    expected = judge(
        best,
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=len(trials),
        horizon=horizon,
        ic_method="pearson",
        fwd=cumulative_target,
        returns_fn=net_return_fn(
            panel,
            cumulative_target,
            QuantileLongShort(),
            zero_costs,
            horizon=horizon,
        ),
    )
    assert expected_oos_ic != pytest.approx(legacy_oos_ic, abs=1e-6)
    assert report["oos_ic"] == pytest.approx(expected_oos_ic)
    assert report["train_ic"] == pytest.approx(expected_train_ic)
    assert expected.oos_ic == pytest.approx(expected_oos_ic)
    assert expected.train_ic == pytest.approx(expected_train_ic)
    assert report["deflated_sharpe"] == pytest.approx(expected.deflated_sharpe)


def test_unrelated_registered_formulas_do_not_change_report_penalty(signal_panel):
    panel, _ = signal_panel
    split = time_split(panel.dates)
    gen = RandomTreeGenerator(random.Random(3), max_depth=3, max_nodes=15)
    trials = gen.ramped_half_and_half(12, min_depth=2, max_depth=3)
    best = trials[0]
    baseline = build_report(best, trials, split, panel, searched_trials=len(trials))

    register_operator(
        "unrelated_formula",
        [DType.SERIES],
        DType.SIGNAL,
        Node("rank", (Node(ARG, value=0),)),
    )
    try:
        unrelated = build_report(best, trials, split, panel, searched_trials=len(trials))
        scoped = build_report(
            best,
            trials,
            split,
            panel,
            searched_trials=len(trials),
            n_user_operators=user_operator_count({"unrelated_formula"}),
        )
    finally:
        unregister_operator("unrelated_formula")

    assert unrelated["n_trials"] == baseline["n_trials"]
    assert scoped["n_trials"] > baseline["n_trials"]
