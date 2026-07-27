"""The search service the API dispatches to (kept pure so it is unit-testable).

Runs a GP with a lineage recorder over the train split, then produces the honest net/deflated
verdict (the test split is scored only at the final report). Returns a JSON-serializable result.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from typing import Any

from alphalineage.api.resources import TRAINING_SCHEDULER, ResolvedResources, TrainingScheduler
from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.engine import net_return_fn, net_returns_for_factor
from alphalineage.backtest.portfolio import QuantileLongShort, WeightingScheme
from alphalineage.backtest.reporting import backtest_report
from alphalineage.core.extensions import operator_counts
from alphalineage.core.fitness import forward_returns
from alphalineage.core.gp import GP, GPConfig
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import OPERATORS
from alphalineage.core.tree import Node, to_dict, to_json
from alphalineage.library.store import LineageStore
from alphalineage.validation.pbo import ReportReturnSummary
from alphalineage.validation.pipeline import LockedTestSet, judge
from alphalineage.validation.splits import Split, time_split
from alphalineage.validation.trials import effective_trials


def distinct_trials(trees: Iterable[Node]) -> list[Node]:
    """Keep every structurally distinct scored tree, in first-seen order."""
    seen: set[str] = set()
    distinct: list[Node] = []
    for tree in trees:
        key = to_json(tree)
        if key not in seen:
            seen.add(key)
            distinct.append(tree)
    return distinct


def report_context(split: Split, config: GPConfig, *, embargo: int) -> dict[str, Any]:
    """Stable, user-facing context for interpreting a final report."""
    return {
        "boundaries": {
            "train_end": split.train.max().date().isoformat(),
            "valid_start": split.valid.min().date().isoformat(),
            "valid_end": split.valid.max().date().isoformat(),
            "test_start": split.test.min().date().isoformat(),
            "test_end": split.test.max().date().isoformat(),
            "embargo": embargo,
        },
        "horizon": config.horizon,
        "ic_method": config.ic_method,
        "weighting_scheme": "quantile_ls",
        "quantile": 0.2,
        "commission_bps": 1.0,
        "slippage_bps": 5.0,
    }


def user_operator_count(allowed_operators: set[str] | None) -> int:
    """Count user formulas in this search pool, independent of unrelated registrations."""
    return sum(
        primitive.macro_body is not None
        and (allowed_operators is None or primitive.name in allowed_operators)
        for primitive in OPERATORS.values()
    )


def build_report(
    best_tree: Node,
    trials: Sequence[Node],
    split: Split,
    panel: Panel,
    *,
    searched_trials: int,
    min_names: int = 5,
    horizon: int = 1,
    ic_method: str = "spearman",
    n_user_operators: int = 0,
    scheme: WeightingScheme | None = None,
    costs: TransactionCostModel | None = None,
    progress: Callable[[int, int], None] | None = None,
    stop: Callable[[], bool] | None = None,
    on_finalizing: Callable[[], None] | None = None,
    summary_cache: MutableMapping[str, ReportReturnSummary] | None = None,
    summary_key: Callable[[Node], str] | None = None,
) -> dict[str, Any]:
    """Judge a factor OOS and return its report dict (shared by single runs and sessions).

    ``searched_trials`` is the cumulative distinct-strategy count fed to the deflation; the
    effective count grows with the operator palette so user operators deflate harder
    (invariant 1). A fresh ``LockedTestSet`` is unlocked exactly once per call - so each call
    is one out-of-sample read, which the session counts and surfaces (P3).

    The DSR/PBO are computed on **net** (after-cost) long-short returns via the default
    weighting scheme + cost model (invariant 6; parity with ``scripts/run_gp.py``). Only one
    scheme is exercised on this path, so the trial count carries no scheme multiplier.
    """
    if isinstance(n_user_operators, bool) or not isinstance(n_user_operators, int):
        raise ValueError("n_user_operators must be a non-negative integer")
    if n_user_operators < 0:
        raise ValueError("n_user_operators must be a non-negative integer")
    builtin, _ = operator_counts()
    n_trials = effective_trials(
        searched_trials,
        n_operators=builtin + n_user_operators,
        baseline=builtin,
    )
    fwd = forward_returns(panel, horizon)
    selected_scheme: WeightingScheme = scheme if scheme is not None else QuantileLongShort()
    cost_model = costs if costs is not None else TransactionCostModel()
    report = judge(
        best_tree,
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=n_trials,
        min_names=min_names,
        horizon=horizon,
        fwd=fwd,
        ic_method=ic_method,
        returns_fn=net_return_fn(
            panel,
            fwd,
            selected_scheme,
            cost_model,
            horizon=horizon,
        ),
        returns_from_factor=lambda factor: net_returns_for_factor(
            factor,
            fwd,
            selected_scheme,
            cost_model,
            panel=panel,
            horizon=horizon,
        ),
        progress=progress,
        stop=stop,
        on_finalizing=on_finalizing,
        summary_cache=summary_cache,
        summary_key=summary_key,
        holdout_reporter=lambda factor, dates: backtest_report(
            factor,
            panel,
            fwd,
            selected_scheme,
            cost_model,
            dates,
            ic_method=ic_method,
            min_names=min_names,
            horizon=horizon,
        ),
    )
    payload: dict[str, Any] = {
        "oos_ic": report.oos_ic,
        "deflated_sharpe": report.deflated_sharpe,
        "pbo": report.pbo,
        "train_ic": report.train_ic,
        "n_trials": report.n_trials,
        "significant": report.significant,
    }
    if report.oos_backtest is not None:
        payload["oos_backtest"] = report.oos_backtest
    return payload


def run_search(
    config: GPConfig,
    panel: Panel,
    *,
    train: float = 0.6,
    valid: float = 0.2,
    embargo: int = 5,
    progress: Any = None,
    stop: Callable[[], bool] | None = None,
    allowed_operators: set[str] | None = None,
    resources: ResolvedResources | None = None,
    scheduler: TrainingScheduler = TRAINING_SCHEDULER,
) -> dict[str, Any]:
    """Run a GP search and return the best factor, its OOS/deflated verdict, and the lineage."""
    split = time_split(
        panel.dates, train=train, valid=valid, embargo=embargo, horizon=config.horizon
    )
    train_panel = Panel({f: df.loc[df.index.isin(split.train)] for f, df in panel.fields.items()})

    store = LineageStore()
    recorder: Any = store
    if progress is not None:  # forward recorder calls through the live snapshot (A2)
        progress.attach(store)
        progress.set_target(config.generations)
        recorder = progress

    lease_context = scheduler.acquire(resources, cancel=stop) if resources is not None else None
    started = time.monotonic()
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("queued" if resources is not None else "initializing")

    def execute(lease: Any = None) -> tuple[GP, Any, dict[str, Any], dict[str, float]]:
        workers = int(lease.max_workers) if lease is not None else 1
        memory_budget = int(lease.memory_budget_bytes) if lease is not None else None
        if progress is not None and hasattr(progress, "set_phase"):
            progress.set_phase("initializing")
        training_started = time.monotonic()
        gp = GP(
            config,
            train_panel,
            recorder=recorder,
            allowed_operators=allowed_operators,
            workers=workers,
            memory_budget_bytes=memory_budget,
        )
        best = gp.run(stop=stop)
        training_seconds = time.monotonic() - training_started

        if progress is not None and hasattr(progress, "set_phase"):
            progress.set_phase("validating")
        report_started = time.monotonic()
        trials = distinct_trials(node.tree for node in store.nodes)
        report_progress = (
            progress.set_report_progress
            if progress is not None and hasattr(progress, "set_report_progress")
            else None
        )

        def on_finalizing() -> None:
            if progress is not None and hasattr(progress, "set_phase"):
                progress.set_phase("finalizing")

        report_dict = build_report(
            best.tree,
            trials,
            split,
            panel,
            searched_trials=gp.trial_count,
            min_names=config.min_names,
            horizon=config.horizon,
            ic_method=config.ic_method,
            n_user_operators=user_operator_count(allowed_operators),
            progress=report_progress,
            # A stop that ended evolution still receives the last complete population's report.
            # A new stop requested during validation cancels before the locked test is opened.
            stop=None if gp.termination_reason == "user_stopped" else stop,
            on_finalizing=on_finalizing,
        )
        report_seconds = time.monotonic() - report_started
        return gp, best, report_dict, {
            "training_seconds": training_seconds,
            "reporting_seconds": report_seconds,
            "total_seconds": time.monotonic() - started,
        }

    if lease_context is None:
        gp, best, report_dict, timings = execute()
    else:
        with lease_context as lease:
            gp, best, report_dict, timings = execute(lease)

    lineage_report = {key: value for key, value in report_dict.items() if key != "oos_backtest"}
    store.metadata = {"best_factor": to_dict(best.tree), "report": lineage_report}
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("done")

    return {
        "best_factor": to_json(best.tree),
        "report": report_dict,
        "oos_backtest": report_dict.get("oos_backtest"),
        "generations": gp.generation,
        "history": gp.history,
        "lineage": store.to_dict(),
        "resources": resources.to_dict() if resources is not None else None,
        "timings": timings,
        "context": report_context(split, config, embargo=embargo),
        "termination_reason": gp.termination_reason,
    }
