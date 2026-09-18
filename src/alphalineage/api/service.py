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
from alphalineage.backtest.portfolio import (
    PORTFOLIO_SCHEMA_VERSION,
    PortfolioStrategySpec,
    QuantileLongShort,
    WeightingScheme,
)
from alphalineage.backtest.reporting import backtest_report
from alphalineage.core.extensions import operator_counts
from alphalineage.core.fitness import forward_returns, label_span
from alphalineage.core.gp import GP, GPConfig
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import OPERATORS
from alphalineage.core.tree import Node, to_dict, to_json
from alphalineage.library.store import LineageStore
from alphalineage.validation.pbo import ReportReturnSummary
from alphalineage.validation.pipeline import LockedTestSet, judge
from alphalineage.validation.selection import (
    ValidationSelection,
    orient_training_candidates,
    select_validation_candidate,
)
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
        "execution": config.execution,
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
    execution: str = "close",
    ic_method: str = "spearman",
    n_user_operators: int = 0,
    scheme: WeightingScheme | None = None,
    strategy_specs: Sequence[PortfolioStrategySpec] | None = None,
    primary_strategy_id: str | None = None,
    costs: TransactionCostModel | None = None,
    progress: Callable[[int, int], None] | None = None,
    stop: Callable[[], bool] | None = None,
    on_finalizing: Callable[[], None] | None = None,
    summary_cache: MutableMapping[str, ReportReturnSummary] | None = None,
    summary_key: Callable[[Node], str] | None = None,
    fwd: Any | None = None,
) -> dict[str, Any]:
    """Judge a factor OOS and return its report dict (shared by single runs and sessions).

    ``searched_trials`` is the cumulative distinct-strategy count fed to the deflation; the
    effective count grows with the operator palette so user operators deflate harder
    (invariant 1). A fresh ``LockedTestSet`` is unlocked exactly once per call - so each call
    is one out-of-sample read, which the session counts and surfaces (P3).

    DSR/PBO use **net** (after-cost) returns. When ``strategy_specs`` is supplied,
    every tree/strategy return path is an actual streamed trial variant; the
    selected primary supplies the headline holdout aliases while all strategies
    are retained in ``strategy_results``. Legacy callers still receive one Q20
    strategy unless they inject ``scheme`` explicitly.
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
    resolved_fwd = forward_returns(panel, horizon, execution) if fwd is None else fwd
    if strategy_specs is not None and scheme is not None:
        raise ValueError("pass either scheme or strategy_specs, not both")
    if strategy_specs is None:
        selected_scheme: WeightingScheme = (
            scheme if scheme is not None else QuantileLongShort()
        )
        quantile = (
            float(selected_scheme.quantile)
            if isinstance(selected_scheme, QuantileLongShort)
            else None
        )
        specs = [
            PortfolioStrategySpec(
                "primary",
                selected_scheme.name,
                quantile,
            )
        ]
    else:
        specs = list(strategy_specs)
        if not specs:
            raise ValueError("strategy_specs must contain at least one strategy")
        if len({item.id for item in specs}) != len(specs):
            raise ValueError("strategy ids must be unique")
    resolved_primary_id = primary_strategy_id or specs[0].id
    if resolved_primary_id not in {item.id for item in specs}:
        raise ValueError("primary_strategy_id must reference a strategy")
    cost_model = costs if costs is not None else TransactionCostModel()
    schemes = [item.weighting_scheme() for item in specs]
    return_variants = []
    for spec, resolved_scheme in zip(specs, schemes, strict=True):
        return_variants.append(
            (
                spec.id,
                net_return_fn(
                    panel,
                    resolved_fwd,
                    resolved_scheme,
                    cost_model,
                    horizon=horizon,
                    execution=execution,
                ),
                lambda factor, current_scheme=resolved_scheme: net_returns_for_factor(
                    factor,
                    resolved_fwd,
                    current_scheme,
                    cost_model,
                    panel=panel,
                    horizon=horizon,
                    execution=execution,
                ),
            )
        )

    def holdout_reports(
        factor: Any,
        dates: Any,
    ) -> dict[str, Any]:
        strategy_results: list[dict[str, Any]] = []
        for spec, resolved_scheme in zip(specs, schemes, strict=True):
            tested = backtest_report(
                factor,
                panel,
                resolved_fwd,
                resolved_scheme,
                cost_model,
                dates,
                ic_method=ic_method,
                min_names=min_names,
                horizon=horizon,
                execution=execution,
            )
            strategy_results.append(
                {
                    "strategy_id": spec.id,
                    "spec": spec.to_dict(),
                    "role": (
                        "primary" if spec.id == resolved_primary_id else "comparison"
                    ),
                    "oos_backtest": tested,
                }
            )
        primary = next(
            item["oos_backtest"]
            for item in strategy_results
            if item["strategy_id"] == resolved_primary_id
        )
        return {
            "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
            "primary_strategy_id": resolved_primary_id,
            "strategy_results": strategy_results,
            "primary_oos_backtest": primary,
        }

    report = judge(
        best_tree,
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=n_trials,
        min_names=min_names,
        horizon=horizon,
        fwd=resolved_fwd,
        ic_method=ic_method,
        return_variants=return_variants,
        primary_return_variant=resolved_primary_id,
        progress=progress,
        stop=stop,
        on_finalizing=on_finalizing,
        summary_cache=summary_cache,
        summary_key=summary_key,
        holdout_reporter=holdout_reports,
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
        bundle = report.oos_backtest
        payload["oos_backtest"] = bundle["primary_oos_backtest"]
        payload["strategy_results"] = bundle["strategy_results"]
        payload["primary_strategy_id"] = bundle["primary_strategy_id"]
        payload["portfolio_schema_version"] = bundle["portfolio_schema_version"]
        primary_health = dict(
            payload["oos_backtest"].get("portfolio_health") or {}
        )
        if not bool(primary_health.get("valid", False)):
            payload["significant"] = False
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
        panel.dates,
        train=train,
        valid=valid,
        embargo=embargo,
        horizon=label_span(config.horizon, config.execution),
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

    def execute(
        lease: Any = None,
    ) -> tuple[GP, ValidationSelection, dict[str, Any], dict[str, float]]:
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
        gp.run(stop=stop)
        training_seconds = time.monotonic() - training_started

        if progress is not None and hasattr(progress, "set_phase"):
            progress.set_phase("validating")
        report_started = time.monotonic()
        fwd = forward_returns(panel, config.horizon, config.execution)
        searched = gp.searched_individuals()
        selection = select_validation_candidate(
            searched,
            panel,
            fwd,
            split.valid,
            method=config.ic_method,
            parsimony=config.parsimony,
            complexity_penalty_mode=config.complexity_penalty_mode,
            complexity_penalty_value=config.resolved_complexity_penalty_value,
            max_nodes=config.max_nodes,
            validation_folds=config.validation_folds,
            fold_embargo=embargo,
            min_names=config.min_names,
            workers=workers,
            memory_budget_bytes=memory_budget,
        )
        gp.record_formula_validation_scores(selection.candidate_validation_scores)
        trials = orient_training_candidates(searched)
        report_progress = (
            progress.set_report_progress
            if progress is not None and hasattr(progress, "set_report_progress")
            else None
        )

        def on_finalizing() -> None:
            if progress is not None and hasattr(progress, "set_phase"):
                progress.set_phase("finalizing")

        report_dict = build_report(
            selection.oriented_tree,
            trials,
            split,
            panel,
            searched_trials=gp.trial_count,
            min_names=config.min_names,
            horizon=config.horizon,
            execution=config.execution,
            ic_method=config.ic_method,
            n_user_operators=user_operator_count(allowed_operators),
            progress=report_progress,
            # A stop that ended evolution still receives the last complete population's report.
            # A new stop requested during validation cancels before the locked test is opened.
            stop=None if gp.termination_reason == "user_stopped" else stop,
            on_finalizing=on_finalizing,
            fwd=fwd,
        )
        report_dict["validation_passed"] = selection.validated
        report_dict["validation_reason"] = selection.reason
        if not selection.validated:
            report_dict["significant"] = False
        report_seconds = time.monotonic() - report_started
        return gp, selection, report_dict, {
            "training_seconds": training_seconds,
            "reporting_seconds": report_seconds,
            "total_seconds": time.monotonic() - started,
        }

    if lease_context is None:
        gp, selection, report_dict, timings = execute()
    else:
        with lease_context as lease:
            gp, selection, report_dict, timings = execute(lease)

    lineage_report = {key: value for key, value in report_dict.items() if key != "oos_backtest"}
    store.metadata = {
        "best_factor": to_dict(selection.oriented_tree),
        "selected_source_factor": to_dict(selection.source_tree),
        "selection": selection.metadata(),
        "report": lineage_report,
    }
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("done")

    return {
        "best_factor": to_json(selection.oriented_tree),
        "report": report_dict,
        "oos_backtest": report_dict.get("oos_backtest"),
        "generations": gp.generation,
        "history": gp.history,
        "lineage": store.to_dict(),
        "resources": resources.to_dict() if resources is not None else None,
        "timings": timings,
        "context": report_context(split, config, embargo=embargo),
        "selection": selection.metadata(),
        "termination_reason": gp.termination_reason,
    }
