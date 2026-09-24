"""Batch CSE: compute each distinct operation once per batch, and prove nothing moved.

The whole justification for sharing is that it changes *how often* a value is computed and
never *what it is*. These tests pin both halves: the sharing is real (instruction counts drop),
and the scores are bit-for-bit what the per-tree path produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalineage.core import cpp
from alphalineage.core.extensions import expand_all
from alphalineage.core.fitness import forward_returns, score_tree
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node


def node(name: str, *children: Node, value: float | int | None = None) -> Node:
    return Node(name, tuple(children), value)


#: Deliberately overlapping, the way a population is: several trees share `high - low`,
#: `delay(close, 1)` and the true-range chain beneath them.
def _overlapping_trees() -> list[Node]:
    span = node("sub", node("high"), node("low"))
    previous_close = node("delay", node("close"), node("window", value=1))
    return [
        span,
        node("div", span, node("close")),
        node("ts_mean", span, node("window", value=10)),
        node("sub", node("close"), previous_close),
        node("abs", node("sub", node("high"), previous_close)),
        node("rank", node("ts_std", node("div", span, node("close")), node("window", value=20))),
        node("where", node("gt", node("close"), node("open")), node("volume"), span),
        node("ts_corr", node("close"), node("volume"), node("window", value=15)),
    ]


@pytest.fixture
def market() -> tuple[Panel, pd.DataFrame]:
    rng = np.random.default_rng(3)
    days, names = 260, 10
    dates = pd.bdate_range("2023-01-02", periods=days)
    symbols = [f"S{index}" for index in range(names)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.012, (days, names)), axis=0), dates, symbols
    )
    open_ = close.shift(1).bfill()
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, (days, names)), dates, symbols)
    panel = Panel.from_prices(open=open_, high=high, low=low, close=close, volume=volume)
    return panel, forward_returns(panel, 1, "next_open")


# --- the compiler, which needs no compiler ----------------------------------------------------
def test_a_batch_emits_each_distinct_operation_once() -> None:
    trees = _overlapping_trees()
    expanded = [expand_all(tree) for tree in trees]
    per_tree = sum(len(cpp._compile_expanded(item).instructions) for item in expanded)

    program = cpp.compile_batch(expanded)
    assert program is not None
    assert len(program.roots) == len(trees)
    assert len(program.instructions) < per_tree
    # Every instruction is distinct, which is the entire point.
    assert len(set(program.instructions)) == len(program.instructions)
    # Roots point at real slots and each tree keeps its own.
    assert all(0 <= root < len(program.instructions) for root in program.roots)


def test_a_root_may_be_reused_as_another_tree_s_subexpression() -> None:
    """`high - low` is both a tree in its own right and a part of two others."""
    trees = _overlapping_trees()[:3]
    program = cpp.compile_batch([expand_all(tree) for tree in trees])
    assert program is not None
    span_root = program.roots[0]
    # The later trees depend on the first tree's result rather than recomputing it.
    dependencies = {
        dependency
        for instruction in program.instructions
        for dependency in cpp._instruction_dependencies(instruction)
    }
    assert span_root in dependencies


def test_identical_trees_in_one_batch_share_a_slot_and_still_get_their_own_result() -> None:
    twice = [expand_all(node("sub", node("high"), node("low")))] * 2
    program = cpp.compile_batch(twice)
    assert program is not None
    assert program.roots[0] == program.roots[1]
    assert len(program.roots) == 2


def test_an_unsupported_operation_disqualifies_the_batch_rather_than_half_sharing() -> None:
    """Every built-in has an opcode today; this guards the day one is added without one."""
    unsupported = Node("not_an_opcode", (Node("close"),))
    assert cpp.compile_batch([unsupported]) is None
    # One unsupported tree spoils the whole program, because a program is all-or-nothing.
    supported = expand_all(node("sub", node("high"), node("low")))
    assert cpp.compile_batch([supported, unsupported]) is None
    assert cpp.compile_batch([]) is None


def test_peak_buffers_accounts_for_roots_being_held_until_scored() -> None:
    expanded = [expand_all(tree) for tree in _overlapping_trees()]
    program = cpp.compile_batch(expanded)
    assert program is not None
    # A shared program holds more at once than one plan, but far less than every tree's peak.
    single_peak = max(cpp._compile_expanded(item).peak_buffers for item in expanded)
    assert program.peak_buffers >= single_peak
    assert program.peak_buffers < len(program.instructions)


def test_partitioning_covers_every_tree_exactly_once() -> None:
    """Grouping by overlap reorders trees, so coverage is the invariant, not position."""
    trees = [expand_all(tree) for tree in _overlapping_trees()] * 10
    for count, workers in ((80, 1), (80, 2), (80, 7), (3, 8), (1, 1)):
        groups = cpp.partition_for_sharing(trees[:count], workers)
        covered = sorted(index for group in groups for index in group)
        assert covered == list(range(count))
        assert len(groups) == max(1, min(workers, count))
        # Balanced, so one perfectly-shared group cannot leave the other workers idle.
        assert max(len(group) for group in groups) - min(len(group) for group in groups) <= 1
    assert cpp.partition_for_sharing([], 4) == []


def test_grouping_by_overlap_shares_more_than_grouping_by_position() -> None:
    """The point of the reordering: at high worker counts, position shares almost nothing."""
    trees = [expand_all(tree) for tree in _overlapping_trees()]
    workers = 4

    def emitted(groups: list[list[int]]) -> int:
        total = 0
        for group in groups:
            program = cpp.compile_batch([trees[index] for index in group])
            assert program is not None
            total += len(program.instructions)
        return total

    by_position = [
        list(range(start, min(start + 2, len(trees)))) for start in range(0, len(trees), 2)
    ]
    assert emitted(cpp.partition_for_sharing(trees, workers)) <= emitted(by_position)


# --- the property everything rests on ---------------------------------------------------------
@pytest.mark.skipif(not cpp.available(), reason="native evaluator is not built")
def test_shared_scoring_is_bit_identical_to_scoring_each_tree_alone(market) -> None:
    panel, fwd = market
    trees = _overlapping_trees()
    assert cpp.supports_shared_scoring(), "this build should expose shared scoring"

    for workers in (1, 2, 3):
        shared = cpp.score_many(trees, panel, fwd, workers=workers)
        per_tree = _without_sharing(
            lambda count=workers: cpp.score_many(trees, panel, fwd, workers=count)
        )
        assert shared == per_tree, f"sharing changed a score at {workers} workers"
        # Not merely close: exactly equal, including every metric in the tuple.
        assert all(entry is not None for entry in shared)


@pytest.mark.skipif(not cpp.available(), reason="native evaluator is not built")
def test_shared_scoring_still_matches_the_python_scorer_within_roundoff(market) -> None:
    panel, fwd = market
    trees = _overlapping_trees()
    shared = cpp.score_many(trees, panel, fwd, workers=2)
    for tree, native in zip(trees, shared, strict=True):
        expected_fitness, expected_metrics = score_tree(tree, panel, fwd)
        assert native is not None
        assert native[0] == pytest.approx(expected_fitness, abs=1e-12)
        assert native[1]["ic"] == pytest.approx(expected_metrics["ic"], abs=1e-12)


@pytest.mark.skipif(not cpp.available(), reason="native evaluator is not built")
def test_a_batch_too_large_for_the_memory_budget_falls_back_instead_of_exceeding_it(
    market,
) -> None:
    panel, fwd = market
    trees = _overlapping_trees()
    # One byte of budget cannot hold a single frame, let alone a shared program's peak.
    tiny = cpp.score_many(trees, panel, fwd, workers=1, memory_budget_bytes=1)
    roomy = cpp.score_many(trees, panel, fwd, workers=1, memory_budget_bytes=1 << 30)
    assert tiny == roomy


@pytest.mark.skipif(not cpp.available(), reason="native evaluator is not built")
def test_an_unsupported_tree_in_the_batch_leaves_the_others_scored(market, monkeypatch) -> None:
    """A batch containing something the backend cannot compile must not lose the rest."""
    panel, fwd = market
    trees = [
        node("sub", node("high"), node("low")),
        node("ts_rank", node("close"), node("window", value=20)),
        node("div", node("sub", node("high"), node("low")), node("close")),
    ]
    expected = cpp.score_many(trees, panel, fwd, workers=1)

    # Withdraw one opcode, as if this build were older than the operator.
    opcodes = {name: code for name, code in cpp.CPP_OPCODES.items() if name != "ts_rank"}
    monkeypatch.setattr(cpp, "CPP_OPCODES", opcodes)
    cpp.clear_plan_cache()
    try:
        scored = cpp.score_many(trees, panel, fwd, workers=2)
    finally:
        cpp.clear_plan_cache()

    assert scored[1] is None, "the withdrawn opcode should make that tree unscorable"
    assert [scored[0], scored[2]] == [expected[0], expected[2]]


def _without_sharing(call):
    """Run ``call`` with the build pretending it predates shared scoring."""
    original = cpp.supports_shared_scoring
    cpp.supports_shared_scoring = lambda: False
    try:
        return call()
    finally:
        cpp.supports_shared_scoring = original
