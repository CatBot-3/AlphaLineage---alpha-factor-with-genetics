"""Parallel report summaries must equal the sequential path exactly."""

from __future__ import annotations

import pandas as pd
import pytest

from alphalineage.api import sessions
from alphalineage.api.service import build_report
from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.portfolio import PortfolioStrategySpec
from alphalineage.core.extensions import ARG, register_operator, unregister_operator
from alphalineage.core.gp import TrainingCancelled
from alphalineage.core.tree import Node
from alphalineage.core.types import DType
from alphalineage.validation import pipeline
from alphalineage.validation.parallel_summaries import SummaryJob, precompute_summaries
from alphalineage.validation.splits import time_split

STRATEGIES = (
    PortfolioStrategySpec("q20", "quantile_ls", 0.2),
    PortfolioStrategySpec("rank", "rank_proportional"),
)
COSTS = TransactionCostModel(commission_bps=1.0, slippage_bps=5.0)


def _trials() -> list[Node]:
    return [
        Node("volume"),
        Node("close"),
        Node("ts_mean", (Node("close"), Node("window", value=5))),
        Node("parallel_test_formula", (Node("volume"),)),  # registered in the app process only
        Node("volume"),  # duplicates are fine
    ]


def _job(panel, split, horizon, execution) -> SummaryJob:
    return SummaryJob(
        panel=panel,
        horizon=horizon,
        execution=execution,
        strategies=tuple(item.to_dict() for item in STRATEGIES),
        commission_bps=COSTS.commission_bps,
        slippage_bps=COSTS.slippage_bps,
        research_dates=pd.DatetimeIndex(split.research),
        n_blocks=pipeline.DEFAULT_PBO_BLOCKS,
    )


@pytest.fixture
def formula():
    register_operator(
        "parallel_test_formula",
        [DType.SERIES],
        DType.SIGNAL,
        Node("rank", (Node(ARG, value=0),)),
    )
    yield
    unregister_operator("parallel_test_formula")


def test_parallel_summaries_equal_the_sequential_report_cache(signal_panel, formula):
    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    trials = _trials()
    horizon, execution = 2, "next_open"

    sequential: dict = {}
    build_report(
        trials[0],
        trials,
        split,
        panel,
        searched_trials=len(trials),
        horizon=horizon,
        execution=execution,
        strategy_specs=list(STRATEGIES),
        costs=COSTS,
        summary_cache=sequential,
        summary_key=sessions._report_tree_key,
    )

    parallel: dict = {}
    events: list[tuple[int, int]] = []
    added = precompute_summaries(
        trials,
        parallel,
        _job(panel, split, horizon, execution),
        workers=2,
        min_trees=1,
        progress=lambda done, total: events.append((done, total)),
    )

    assert parallel == sequential
    assert added == len(sequential) == 4 * len(STRATEGIES)
    assert events[0] == (0, len(trials) * len(STRATEGIES))
    assert events[-1] == (len(trials) * len(STRATEGIES), len(trials) * len(STRATEGIES))


def test_judge_reads_precomputed_summaries_without_evaluating_trials(
    signal_panel, formula, monkeypatch
):
    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    trials = _trials()
    cache: dict = {}
    precompute_summaries(trials, cache, _job(panel, split, 1, "close"), workers=2, min_trees=1)

    calls: list[Node] = []
    real = pipeline.evaluate

    def counting(tree, data):
        calls.append(tree)
        return real(tree, data)

    monkeypatch.setattr(pipeline, "evaluate", counting)
    build_report(
        trials[0],
        trials,
        split,
        panel,
        searched_trials=len(trials),
        strategy_specs=list(STRATEGIES),
        costs=COSTS,
        summary_cache=cache,
        summary_key=sessions._report_tree_key,
    )
    assert calls == [trials[0]]  # only the selected formula, for its IC and holdout


def test_one_worker_leaves_the_work_to_judge(signal_panel):
    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    cache: dict = {}
    job = _job(panel, split, 1, "close")
    assert precompute_summaries([Node("volume")], cache, job, workers=1) == 0
    assert cache == {}


def test_stop_cancels_the_parallel_summaries(signal_panel):
    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    trials = [Node("ts_mean", (Node("close"), Node("window", value=w))) for w in (5, 10, 20, 40)]
    with pytest.raises(TrainingCancelled):
        precompute_summaries(
            trials, {}, _job(panel, split, 1, "close"), workers=2, min_trees=1, stop=lambda: True
        )


def test_report_worker_count_respects_memory_and_override(monkeypatch):
    monkeypatch.delenv(sessions.REPORT_WORKERS_ENV, raising=False)
    gib = 1024**3
    assert sessions.report_worker_count(available=gib, cpus=20) == 1
    assert sessions.report_worker_count(available=4 * gib, cpus=20) == 4
    assert sessions.report_worker_count(available=100 * gib, cpus=20) == sessions.MAX_REPORT_WORKERS
    assert sessions.report_worker_count(available=100 * gib, cpus=4) == 3
    monkeypatch.setenv(sessions.REPORT_WORKERS_ENV, "6")
    assert sessions.report_worker_count(available=2 * gib, cpus=2) == 6


@pytest.fixture
def deep_formula():
    # Each call adds two levels once expanded, so sixteen nested calls are depth 17 as written
    # and depth 33 expanded: one level past the search's default expansion limit.
    register_operator(
        "parallel_test_deep",
        [DType.SERIES],
        DType.SERIES,
        Node(
            "ts_mean",
            (
                Node("ts_mean", (Node(ARG, value=0), Node("window", value=3))),
                Node("window", value=5),
            ),
        ),
    )
    yield
    unregister_operator("parallel_test_deep")


def _deep_tree(levels: int = 16) -> Node:
    tree = Node("volume")
    for _ in range(levels):
        tree = Node("parallel_test_deep", (tree,))
    return tree


def test_trials_deeper_than_the_search_limit_are_cached_by_judge(signal_panel, deep_formula):
    from alphalineage.core.extensions import InvalidOperator, expand_all

    deep = _deep_tree()
    with pytest.raises(InvalidOperator):
        expand_all(deep)  # the default limit rejects it, so it cannot use the native evaluator

    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    trials = [Node("volume"), deep, Node("close")]
    sequential: dict = {}
    build_report(
        trials[0],
        trials,
        split,
        panel,
        searched_trials=len(trials),
        strategy_specs=list(STRATEGIES),
        costs=COSTS,
        summary_cache=sequential,
        summary_key=sessions._report_tree_key,
    )
    # judge keys and caches the deep trial instead of abandoning the cache.
    assert len(sequential) == len(trials) * len(STRATEGIES)

    parallel: dict = {}
    precompute_summaries(trials, parallel, _job(panel, split, 1, "close"), workers=2, min_trees=1)
    deep_key = sessions._report_tree_key(deep)
    assert not any(key.startswith(deep_key) for key in parallel)  # left to the app process
    assert parallel == {key: value for key, value in sequential.items() if key in parallel}
    assert len(parallel) == 2 * len(STRATEGIES)


def test_an_unkeyable_trial_does_not_abandon_the_summary_cache(signal_panel, monkeypatch):
    panel, _ = signal_panel
    split = time_split(panel.dates, train=0.5, valid=0.2, embargo=3)
    trials = [Node("volume"), Node("close"), Node("open")]

    def key(tree: Node) -> str:
        if tree.name == "close":
            raise ValueError("cannot key this tree")
        return sessions._report_tree_key(tree)

    def streamed(*_args, **_kwargs):
        raise AssertionError("the streamed fallback recomputes every trial")

    monkeypatch.setattr(pipeline, "pbo_from_returns", streamed)
    cache: dict = {}
    build_report(
        trials[0],
        trials,
        split,
        panel,
        searched_trials=len(trials),
        strategy_specs=list(STRATEGIES),
        costs=COSTS,
        summary_cache=cache,
        summary_key=key,
    )
    assert len(cache) == 2 * len(STRATEGIES)  # volume and open; close was summarised uncached
