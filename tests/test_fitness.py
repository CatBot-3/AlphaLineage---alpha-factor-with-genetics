"""Tests for IC / rank IC / IC IR fitness (P2-T2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalineage.core import cpp as cpp_backend
from alphalineage.core.fitness import (
    daily_ic,
    forward_returns,
    ic_ir,
    mean_ic,
    score_tree,
    score_trees,
)
from alphalineage.core.tree import Node


def _frame(values: list[list[float]]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.DataFrame(values, index=dates, columns=["A", "B", "C"])


def test_forward_returns_alignment(synthetic_panel):
    fwd = forward_returns(synthetic_panel, horizon=1)
    # forward return at t equals the realized return at t+1
    pd.testing.assert_frame_equal(fwd, synthetic_panel["returns"].shift(-1))
    assert fwd.iloc[-1].isna().all()  # last date has no forward return


def test_forward_returns_compounds_the_complete_horizon(synthetic_panel):
    horizon = 3
    fwd = forward_returns(synthetic_panel, horizon=horizon)
    expected = (
        synthetic_panel["close"].shift(-horizon).div(synthetic_panel["close"]).sub(1.0)
    )

    pd.testing.assert_frame_equal(fwd, expected)
    assert fwd.iloc[-horizon:].isna().all().all()

    symbol = synthetic_panel.symbols[0]
    daily = synthetic_panel["returns"][symbol].iloc[1 : horizon + 1]
    assert fwd[symbol].iloc[0] == np.prod(1.0 + daily) - 1.0


def test_daily_ic_perfect_and_inverse():
    factor = _frame([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    fwd = factor.copy()
    ic = daily_ic(factor, fwd, "spearman")
    assert np.allclose(ic.to_numpy(), 1.0)  # factor vs itself -> +1

    ic_inv = daily_ic(factor, -fwd, "spearman")
    assert np.allclose(ic_inv.to_numpy(), -1.0)  # factor vs its negation -> -1


def _pandas_daily_ic_reference(
    factor: pd.DataFrame, fwd: pd.DataFrame, method: str, min_names: int
) -> pd.Series:
    factor, fwd = factor.align(fwd, join="inner")
    mask = factor.notna() & fwd.notna()
    a, b = factor.where(mask), fwd.where(mask)
    if method == "spearman":
        a, b = a.rank(axis=1), b.rank(axis=1)
    a_d = a.sub(a.mean(axis=1), axis=0)
    b_d = b.sub(b.mean(axis=1), axis=0)
    num = (a_d * b_d).sum(axis=1, min_count=1)
    den = np.sqrt((a_d**2).sum(axis=1, min_count=1) * (b_d**2).sum(axis=1, min_count=1))
    return (num / den.replace(0.0, np.nan)).where(
        a.notna().sum(axis=1) >= max(2, min_names)
    )


def test_daily_ic_array_kernel_matches_pandas_for_masks_ties_and_infinities():
    dates = pd.date_range("2024-01-01", periods=5)
    factor = pd.DataFrame(
        [
            [1.0, 1.0, 3.0, np.nan],
            [4.0, np.nan, 2.0, 1.0],
            [np.inf, 3.0, 2.0, 1.0],
            [1.0, 2.0, np.nan, np.nan],
            [5.0, 5.0, 5.0, 5.0],
        ],
        index=dates,
        columns=list("ABCD"),
    )
    fwd = pd.DataFrame(
        [
            [4.0, 2.0, 2.0, 0.0],
            [1.0, 3.0, np.nan, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        index=dates + pd.Timedelta(days=1),
        columns=list("BCDE"),
    )
    for method in ("pearson", "spearman"):
        expected = _pandas_daily_ic_reference(factor, fwd, method, min_names=2)
        actual = daily_ic(factor, fwd, method, min_names=2)
        pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-13, atol=1e-13)


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


def test_user_formula_parsimony_uses_expanded_node_count(signal_panel):
    from alphalineage.core.extensions import ARG, expand_all, register_operator, unregister_operator
    from alphalineage.core.types import DType

    panel, _ = signal_panel
    fwd = forward_returns(panel)
    body = Node(
        "add",
        (
            Node(ARG, value=0),
            Node("mul_scalar", (Node(ARG, value=0), Node("const", value=0.0))),
        ),
    )
    register_operator(
        "expanded_penalty_fitness_test",
        [DType.SERIES],
        DType.SERIES,
        body,
    )
    try:
        compact = Node("expanded_penalty_fitness_test", (Node("volume"),))
        expanded = expand_all(compact)
        assert compact.size() == 2
        assert expanded.size() == 5
        compact_score = score_tree(compact, panel, fwd, parsimony=0.1)
        expanded_score = score_tree(expanded, panel, fwd, parsimony=0.1)
        assert compact_score == expanded_score
    finally:
        unregister_operator("expanded_penalty_fitness_test")


def test_score_trees_serial_reference_is_ordered(signal_panel, monkeypatch):
    panel, _ = signal_panel
    fwd = forward_returns(panel)
    trees = [Node("returns"), Node("volume"), Node("close")]
    monkeypatch.setattr(cpp_backend, "supports_native_scoring", lambda _method: False)
    expected = [score_tree(tree, panel, fwd, parsimony=0.01) for tree in trees]
    assert score_trees(trees, panel, fwd, parsimony=0.01, workers=1) == expected


def test_score_trees_native_hook_preserves_order_and_resource_guard(signal_panel, monkeypatch):
    panel, _ = signal_panel
    fwd = forward_returns(panel)
    trees = [Node("returns"), Node("volume"), Node("close")]
    seen: dict[str, int | None] = {}

    def fake_score_many(
        nodes,
        candidate_panel,
        candidate_fwd,
        *,
        method,
        absolute,
        parsimony,
        min_names,
        min_valid_dates,
        workers,
        memory_budget_bytes,
    ):
        assert min_valid_dates == 5
        seen.update(workers=workers, memory_budget_bytes=memory_budget_bytes)
        return [
            score_tree(
                node,
                candidate_panel,
                candidate_fwd,
                method=method,
                absolute=absolute,
                parsimony=parsimony,
                min_names=min_names,
            )
            for node in nodes
        ]

    monkeypatch.setattr(cpp_backend, "supports_native_scoring", lambda method: method == "spearman")
    monkeypatch.setattr(cpp_backend, "score_many", fake_score_many)
    expected = [score_tree(tree, panel, fwd, parsimony=0.01) for tree in trees]
    actual = score_trees(
        trees,
        panel,
        fwd,
        parsimony=0.01,
        workers=3,
        memory_budget_bytes=123_456,
    )
    assert actual == expected
    assert seen == {"workers": 3, "memory_budget_bytes": 123_456}
