"""Tests for IC / rank IC / IC IR fitness (P2-T2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalineage.core.fitness import daily_ic, forward_returns, ic_ir, mean_ic, score_tree
from alphalineage.core.tree import Node


def _frame(values: list[list[float]]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.DataFrame(values, index=dates, columns=["A", "B", "C"])


def test_forward_returns_alignment(synthetic_panel):
    fwd = forward_returns(synthetic_panel, horizon=1)
    # forward return at t equals the realized return at t+1
    pd.testing.assert_frame_equal(fwd, synthetic_panel["returns"].shift(-1))
    assert fwd.iloc[-1].isna().all()  # last date has no forward return


def test_daily_ic_perfect_and_inverse():
    factor = _frame([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    fwd = factor.copy()
    ic = daily_ic(factor, fwd, "spearman")
    assert np.allclose(ic.to_numpy(), 1.0)  # factor vs itself -> +1

    ic_inv = daily_ic(factor, -fwd, "spearman")
    assert np.allclose(ic_inv.to_numpy(), -1.0)  # factor vs its negation -> -1


def test_constant_factor_has_zero_ic():
    factor = _frame([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    fwd = _frame([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    # no cross-sectional variation -> undefined corr -> mean_ic falls back to 0
    assert mean_ic(factor, fwd, "spearman", absolute=True) == 0.0


def test_ic_ir():
    daily = pd.Series([0.1, 0.2, 0.3])
    assert np.isclose(ic_ir(daily), daily.mean() / daily.std())
    assert ic_ir(pd.Series([0.5])) == 0.0  # too few points


def test_score_tree_rewards_predictive_factor_and_penalizes_size(signal_panel):
    panel, _ = signal_panel
    fwd = forward_returns(panel)
    predictive, _m = score_tree(Node("volume"), panel, fwd, parsimony=0.0)
    noise, _n = score_tree(Node("returns"), panel, fwd, parsimony=0.0)
    assert predictive > noise  # volume carries the signal; raw returns are ~noise

    # parsimony lowers fitness for a larger but equivalent tree
    big = Node(
        "add", (Node("volume"), Node("mul_scalar", (Node("volume"), Node("const", value=0.0))))
    )
    base_fit, _ = score_tree(Node("volume"), panel, fwd, parsimony=0.1)
    big_fit, _ = score_tree(big, panel, fwd, parsimony=0.1)
    assert big_fit < base_fit
