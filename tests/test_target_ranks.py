"""The target is ranked once per batch; a tree with its own holes must not borrow those ranks.

``score_factor`` correlates the factor's per-date ranks against the target's. The target is one
fixed frame for the whole run, so ranking it inside the per-tree loop repeats identical work -
but only where the tree is complete. Where the factor has holes of its own, the pair set is a
strict subset of the target's non-NaN symbols and the ranks genuinely differ.

The native code decides between the two by comparing counts, which is sound because pairs are
always a subset of the target's non-NaN set, so equal sizes mean equal sets. These tests hold
that reasoning to account: a panel built so that one symbol's factor is NaN while its forward
return is not, scored against the Python scorer, which has no cache and therefore cannot share
the same mistake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalineage.core import cpp
from alphalineage.core.fitness import forward_returns, score_tree
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node


def node(name: str, *children: Node, value: float | int | None = None) -> Node:
    return Node(name, tuple(children), value)


#: ``ts_std`` of a dead-flat price is exactly zero, so ``0 / 0`` is NaN for that symbol alone
#: while every other symbol adds a harmless zero. The flat symbol's forward return is 0.0, which
#: is present - exactly the shape that makes a pairing partial.
def _holey_factor() -> Node:
    return node(
        "add",
        node("close"),
        node(
            "div",
            node("sub", node("close"), node("close")),
            node("ts_std", node("close"), node("window", value=20)),
        ),
    )


@pytest.fixture
def market_with_a_flat_symbol() -> tuple[Panel, pd.DataFrame]:
    rng = np.random.default_rng(5)
    days, names = 160, 8
    dates = pd.bdate_range("2023-01-02", periods=days)
    symbols = [f"S{index}" for index in range(names)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.012, (days, names)), axis=0), dates, symbols
    )
    close["S3"] = 50.0
    open_ = close.shift(1).bfill()
    panel = Panel.from_prices(
        open=open_,
        high=np.maximum(open_, close) * 1.01,
        low=np.minimum(open_, close) * 0.99,
        close=close,
        volume=pd.DataFrame(rng.uniform(1e6, 5e6, (days, names)), dates, symbols),
    )
    return panel, forward_returns(panel, 1, "next_open")


def test_the_fixture_really_does_produce_partial_pairings(market_with_a_flat_symbol) -> None:
    """Without this, the interesting tests below would pass by never reaching the branch."""
    panel, fwd = market_with_a_flat_symbol
    from alphalineage.core.evaluate import evaluate_python

    values = pd.DataFrame(
        evaluate_python(_holey_factor(), panel).values, panel.dates, panel.symbols
    )
    holes = values.isna() & fwd.notna()
    assert holes.to_numpy().sum() > 0
    partial = (holes.any(axis=1) & values.notna().any(axis=1)).sum()
    assert partial > 50, "the panel should leave most dates pairing on a strict subset"


@pytest.mark.skipif(not cpp.available(), reason="native evaluator is not built")
def test_a_tree_with_its_own_holes_is_scored_on_the_subset_it_paired_on(
    market_with_a_flat_symbol,
) -> None:
    panel, fwd = market_with_a_flat_symbol
    tree = _holey_factor()
    (native,) = cpp.score_many([tree], panel, fwd, workers=1)
    expected_fitness, expected_metrics = score_tree(tree, panel, fwd)

    assert native is not None
    assert native[0] == pytest.approx(expected_fitness, abs=1e-12)
    for key, value in expected_metrics.items():
        assert native[1][key] == pytest.approx(value, abs=1e-12), key


@pytest.mark.skipif(not cpp.available(), reason="native evaluator is not built")
def test_complete_and_holey_trees_in_one_batch_each_get_their_own_answer(
    market_with_a_flat_symbol,
) -> None:
    """One call, one cache, both branches - the mix is what a real population looks like."""
    panel, fwd = market_with_a_flat_symbol
    trees = [
        node("close"),
        _holey_factor(),
        node("sub", node("high"), node("low")),
        node("div", _holey_factor(), node("close")),
        node("ts_mean", node("volume"), node("window", value=10)),
    ]
    scored = cpp.score_many(trees, panel, fwd, workers=2)
    for tree, native in zip(trees, scored, strict=True):
        expected_fitness, expected_metrics = score_tree(tree, panel, fwd)
        assert native is not None
        assert native[0] == pytest.approx(expected_fitness, abs=1e-12)
        assert native[1]["ic"] == pytest.approx(expected_metrics["ic"], abs=1e-12)

    # The two trees differ only in whether they have holes, so a cache leaking across them would
    # show up as the holey one inheriting the complete one's correlation.
    assert scored[1][1]["ic"] != pytest.approx(scored[0][1]["ic"], abs=1e-9)


@pytest.mark.skipif(not cpp.available(), reason="native evaluator is not built")
def test_the_cache_is_per_batch_so_batching_cannot_change_a_score(
    market_with_a_flat_symbol,
) -> None:
    """Scoring alone, together, and at several worker counts must give the same bits."""
    panel, fwd = market_with_a_flat_symbol
    trees = [_holey_factor(), node("close"), node("rank", node("volume"))]
    alone = [cpp.score_many([tree], panel, fwd, workers=1)[0] for tree in trees]
    for workers in (1, 2, 3):
        assert cpp.score_many(trees, panel, fwd, workers=workers) == alone
