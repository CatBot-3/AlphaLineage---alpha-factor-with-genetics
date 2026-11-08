"""Report summaries for many trials, computed in parallel worker processes.

Finalization needs one compact summary per (trial, strategy) for PBO and the deflated Sharpe
(the summary cache ``judge`` reads). Computing them one trial at a time keeps a single core
busy for days on a large search. Each summary is a pure function of the expanded tree, the
panel, the strategy and the costs, and is computed here with the same functions the
sequential path uses, so the number of worker processes cannot change any value. This fills
the cache before ``judge`` runs; ``judge`` then finds every summary and goes straight to the
locked-test read. Nothing here reads the test split.

Workers receive fully expanded trees, so formulas registered only in the app process need no
registry in the workers: the native evaluator compiles the expanded tree in both places. Any
tree a worker cannot summarise is simply left out of the cache, and ``judge`` computes it in
the app process exactly as before (including its own fallbacks and errors).
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import logging
import multiprocessing
from collections.abc import Callable, MutableMapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.engine import net_returns_for_factor
from alphalineage.backtest.portfolio import PortfolioStrategySpec
from alphalineage.core import cpp
from alphalineage.core.evaluate import evaluate
from alphalineage.core.extensions import (
    MAX_EXPANDED_DEPTH,
    MAX_EXPANDED_NODES,
    InvalidOperator,
    expand_all,
)
from alphalineage.core.fitness import forward_returns
from alphalineage.core.gp import TrainingCancelled
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node, from_json, to_json
from alphalineage.validation.pbo import ReportReturnSummary, summarize_report_returns
from alphalineage.validation.pipeline import summary_cache_key

_LOG = logging.getLogger(__name__)

# Report keys expand trees well past the search's default limits (depth 32, 2,000 nodes). Those
# limits keep the search away from runaway formulas; every report trial has already been built
# and scored. A real 43,346-trial search had 11 trials one level too deep once expanded, and
# failing to key them used to throw away every cached summary. Cycles are still rejected.
REPORT_EXPANSION_MAX_DEPTH = 256
REPORT_EXPANSION_MAX_NODES = 100_000


# Trees past the default limits cannot be compiled by the native evaluator, so they are scored by
# the pure-Python evaluator, which held over 5 GiB for one such tree on the S&P panel. They are
# rare (11 of 43,346 in a real search) and are left to the app process, one at a time, instead of
# letting several workers hit one at once.
def _needs_python_evaluator(expanded: Node) -> bool:
    return (
        expanded.depth() > MAX_EXPANDED_DEPTH
        or expanded.unique_computation_size() > MAX_EXPANDED_NODES
    )


# Starting worker processes costs seconds (each imports the app and receives the panel), so a
# small report is faster in the app process; real reports have tens of thousands of trials.
MIN_TREES_FOR_WORKERS = 64

# Trees per task: small enough that progress and cancellation stay responsive (a task is a
# few seconds per tree on a 500-name, 26-year panel), large enough to amortise the round trip.
TREES_PER_TASK = 4


@dataclass(frozen=True)
class SummaryJob:
    """Everything a worker needs, sent once when the worker starts."""

    panel: Panel
    horizon: int
    execution: str
    strategies: tuple[dict[str, Any], ...]
    commission_bps: float
    slippage_bps: float
    research_dates: pd.DatetimeIndex
    n_blocks: int
    evaluator: str | None = None


def report_expansion(tree: Node) -> Node:
    """The expanded tree a report summary is keyed and evaluated on."""
    return expand_all(
        tree,
        max_depth=REPORT_EXPANSION_MAX_DEPTH,
        max_nodes=REPORT_EXPANSION_MAX_NODES,
    )


def expanded_tree_key(expanded_json: str) -> str:
    """The cache key of a tree, from its expanded JSON (same hash as the session cache)."""
    return hashlib.sha256(expanded_json.encode("utf-8")).hexdigest()


class _WorkerState:
    def __init__(self, job: SummaryJob) -> None:
        if job.evaluator is not None:
            cpp.set_backend(job.evaluator)
        self.job = job
        self.specs = [PortfolioStrategySpec.from_dict(item) for item in job.strategies]
        self.schemes = [spec.weighting_scheme() for spec in self.specs]
        self.costs = TransactionCostModel(
            commission_bps=job.commission_bps,
            slippage_bps=job.slippage_bps,
        )
        self.fwd = forward_returns(job.panel, job.horizon, job.execution)
        self.research_index = pd.DatetimeIndex(job.research_dates)


_STATE: _WorkerState | None = None


def _init_worker(job: SummaryJob) -> None:
    global _STATE
    _STATE = _WorkerState(job)


def summarize_tree(
    expanded_json: str, tree_key: str, state: _WorkerState
) -> list[tuple[str, ReportReturnSummary]]:
    """Summaries of one tree for every strategy, from a single evaluation of the tree."""
    job = state.job
    factor = evaluate(from_json(expanded_json), job.panel)
    count = len(state.specs)
    out: list[tuple[str, ReportReturnSummary]] = []
    for spec, scheme in zip(state.specs, state.schemes, strict=True):
        if isinstance(factor, pd.DataFrame):
            returns = net_returns_for_factor(
                factor,
                state.fwd,
                scheme,
                state.costs,
                panel=job.panel,
                horizon=job.horizon,
                execution=job.execution,
            )
        else:
            # The sequential tree -> returns path yields an empty series for a non-panel tree.
            returns = pd.Series(dtype="float64")
        research = returns.loc[returns.index.isin(state.research_index)]
        out.append(
            (
                summary_cache_key(tree_key, spec.id, count),
                summarize_report_returns(research, state.research_index, n_blocks=job.n_blocks),
            )
        )
    return out


def _summarize_chunk(
    items: list[tuple[str, str]],
) -> tuple[list[tuple[str, ReportReturnSummary]], int]:
    assert _STATE is not None, "worker was not initialised"
    results: list[tuple[str, ReportReturnSummary]] = []
    skipped = 0
    for expanded_json, tree_key in items:
        try:
            results.extend(summarize_tree(expanded_json, tree_key, _STATE))
        except Exception:  # noqa: BLE001 - leave the tree to judge's own sequential handling
            skipped += 1
    return results, skipped


def _stop_workers(executor: cf.ProcessPoolExecutor) -> None:
    executor.shutdown(wait=False, cancel_futures=True)
    for process in list((getattr(executor, "_processes", None) or {}).values()):
        if process.is_alive():
            process.terminate()
    for process in list((getattr(executor, "_processes", None) or {}).values()):
        process.join(timeout=10)


def precompute_summaries(
    trials: Sequence[Node],
    cache: MutableMapping[str, ReportReturnSummary],
    job: SummaryJob,
    *,
    workers: int,
    progress: Callable[[int, int], None] | None = None,
    stop: Callable[[], bool] | None = None,
    min_trees: int = MIN_TREES_FOR_WORKERS,
) -> int:
    """Fill ``cache`` with every missing (trial, strategy) summary; return how many were added.

    With fewer than two workers, or fewer than ``min_trees`` trees to summarise, this does
    nothing and ``judge`` computes the summaries itself.
    A worker failure only stops the parallel part: summaries already received stay in the
    cache and ``judge`` computes the rest in the app process.
    """
    count = len(job.strategies)
    ids = [str(item["id"]) for item in job.strategies]
    occurrences: dict[str, int] = {}
    pending: dict[str, str] = {}
    unkeyed = 0
    for tree in trials:
        try:
            expanded_tree = report_expansion(tree)
        except InvalidOperator:
            unkeyed += 1  # judge summarises it in the app process, without a cache entry
            continue
        if _needs_python_evaluator(expanded_tree):
            unkeyed += 1  # judge summarises (and caches) it in the app process
            continue
        expanded = to_json(expanded_tree)
        key = expanded_tree_key(expanded)
        occurrences[key] = occurrences.get(key, 0) + 1
        if key not in pending and any(
            summary_cache_key(key, strategy_id, count) not in cache for strategy_id in ids
        ):
            pending[key] = expanded
    total = len(trials) * count
    done = total - (sum(occurrences[key] for key in pending) + unkeyed) * count
    if progress is not None:
        progress(done, total)
    if workers < 2 or len(pending) < max(1, min_trees):
        return 0

    items = list(pending.items())
    chunks = [items[i : i + TREES_PER_TASK] for i in range(0, len(items), TREES_PER_TASK)]
    workers = min(workers, len(chunks))
    _LOG.info(
        "computing %d report summaries for %d trials in %d worker processes",
        len(items) * count,
        len(items),
        workers,
    )
    executor = cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(job,),
    )
    added = 0
    skipped = 0
    clean = False
    try:
        queue = iter(chunks)
        inflight: dict[cf.Future, list[tuple[str, str]]] = {}

        def submit_next() -> None:
            chunk = next(queue, None)
            if chunk is not None:
                payload = [(expanded, key) for key, expanded in chunk]
                inflight[executor.submit(_summarize_chunk, payload)] = chunk

        for _ in range(workers * 2):
            submit_next()
        while inflight:
            finished, _ = cf.wait(inflight, timeout=1.0, return_when=cf.FIRST_COMPLETED)
            if stop is not None and stop():
                raise TrainingCancelled("report cancelled before locked-test finalization")
            for future in finished:
                chunk = inflight.pop(future)
                results, chunk_skipped = future.result()
                for key, summary in results:
                    cache[key] = summary
                added += len(results)
                skipped += chunk_skipped
                done += sum(occurrences[key] for key, _expanded in chunk) * count
                if progress is not None:
                    progress(min(done, total), total)
                submit_next()
        clean = True
    except BrokenProcessPool as exc:
        _LOG.warning(
            "report worker processes stopped (%r); %d summaries kept, the rest run in the app",
            exc,
            added,
        )
    finally:
        if clean:
            executor.shutdown(wait=True)
        else:
            _stop_workers(executor)
    if skipped:
        _LOG.warning("%d trials were left for the app process to summarise", skipped)
    return added
