"""Validation-only candidate selection with training-fixed factor orientation.

The locked holdout must never participate in candidate selection.  Each candidate
is evaluated once on the validation partition and its daily IC observations are
summarized over chronological, embargo-separated folds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, Protocol

import numpy as np
import pandas as pd

from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.portfolio import (
    PORTFOLIO_SCHEMA_VERSION,
    PortfolioStrategySpec,
    QuantileLongShort,
    WeightingScheme,
    validate_portfolio_weights,
)
from alphalineage.core.evaluate import evaluate
from alphalineage.core.extensions import expand_all
from alphalineage.core.fitness import daily_ic, ic_ir
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node, to_json

VALIDATION_SELECTION_VERSION = 2


class Candidate(Protocol):
    tree: Node
    fitness: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class ValidationSelection:
    """A selected source expression and its immutable, training-fixed orientation."""

    source_tree: Node
    oriented_tree: Node
    validated: bool
    reason: str | None
    first_seen: int
    polarity: int
    training_fitness: float
    training_metrics: dict[str, float]
    validation_fitness: float
    validation_metrics: dict[str, Any]
    fold_metrics: tuple[dict[str, Any], ...]
    positive_folds: int
    required_positive_folds: int
    expanded_nodes: int
    expanded_unique_nodes: int
    complexity_penalty_mode: str
    complexity_penalty_value: float
    complexity_deduction: float
    complexity_max_nodes: int
    final_objective: float
    candidate_validation_scores: tuple[tuple[Node, float], ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "validated": self.validated,
            "reason": self.reason,
            "first_seen": self.first_seen,
            "polarity": self.polarity,
            "training_fitness": self.training_fitness,
            "training_metrics": dict(self.training_metrics),
            "validation_fitness": self.validation_fitness,
            "validation_metrics": dict(self.validation_metrics),
            "folds": [dict(item) for item in self.fold_metrics],
            "positive_folds": self.positive_folds,
            "required_positive_folds": self.required_positive_folds,
            "median_oriented_ic": self.validation_metrics.get("median_oriented_ic"),
            "worst_fold_ic": self.validation_metrics.get("worst_fold_ic"),
            "expanded_nodes": self.expanded_nodes,
            "expanded_unique_nodes": self.expanded_unique_nodes,
            "complexity": {
                "expanded_nodes": self.expanded_nodes,
                "expanded_unique_nodes": self.expanded_unique_nodes,
                "mode": self.complexity_penalty_mode,
                "penalty_value": self.complexity_penalty_value,
                "deduction": self.complexity_deduction,
                "max_nodes": self.complexity_max_nodes,
            },
            "final_objective": self.final_objective,
            "source_expression": to_json(self.source_tree),
            "oriented_expression": to_json(self.oriented_tree),
        }


def _oriented_tree(tree: Node, polarity: int) -> Node:
    if polarity >= 0:
        return tree
    # Persist orientation in the expression itself so validation, holdout, PBO, and
    # backtesting cannot accidentally infer a fresh direction from later data.
    return Node(
        "mul_scalar",
        (tree, Node("const", value=-1.0)),
    )


def orient_training_candidates(candidates: Sequence[Candidate]) -> list[Node]:
    """Return first-seen unique trial trees with direction fixed by training only."""
    oriented: list[Node] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = to_json(expand_all(candidate.tree))
        if key in seen:
            continue
        seen.add(key)
        training_signed = candidate.metrics.get(
            "signed_ic",
            candidate.metrics.get("ic", 0.0),
        )
        oriented.append(_oriented_tree(candidate.tree, -1 if training_signed < 0.0 else 1))
    return oriented


def _chronological_folds(
    dates: pd.DatetimeIndex,
    *,
    count: int,
    embargo: int,
) -> list[pd.DatetimeIndex]:
    """Divide ordered dates into folds, leaving ``embargo`` observations between them."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("validation_folds must be a positive integer")
    if isinstance(embargo, bool) or not isinstance(embargo, int) or embargo < 0:
        raise ValueError("validation fold embargo must be a non-negative integer")
    ordered = pd.DatetimeIndex(dates).sort_values().unique()
    usable = len(ordered) - embargo * max(0, count - 1)
    if usable <= 0:
        return [pd.DatetimeIndex([]) for _ in range(count)]
    sizes = [usable // count + (1 if index < usable % count else 0) for index in range(count)]
    folds: list[pd.DatetimeIndex] = []
    cursor = 0
    for index, size in enumerate(sizes):
        folds.append(pd.DatetimeIndex(ordered[cursor : cursor + size]))
        cursor += size
        if index < count - 1:
            cursor += embargo
    return folds


def _complexity_deduction(
    complexity: int,
    *,
    mode: str,
    value: float,
    max_nodes: int,
) -> float:
    if mode == "per_node":
        return float(value * complexity)
    if mode == "normalized_budget":
        return float(value * complexity / max(1, max_nodes))
    raise ValueError(
        "complexity_penalty_mode must be 'per_node' or 'normalized_budget'"
    )


def _candidate_validation(
    tree: Node,
    panel: Panel,
    validation_forward: pd.DataFrame,
    folds: Sequence[pd.DatetimeIndex],
    *,
    polarity: int,
    method: str,
    min_names: int,
    min_valid_dates_per_fold: int,
    minimum_fold_coverage: float,
    weighting_scheme: WeightingScheme,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Evaluate ``tree`` once, then summarize the resulting daily validation IC."""
    expanded = expand_all(tree)
    factor = evaluate(expanded, panel)
    if not isinstance(factor, pd.DataFrame):
        factor = pd.DataFrame(index=validation_forward.index, columns=validation_forward.columns)
    factor, aligned_forward = factor.align(validation_forward, join="inner")
    finite_factor = factor.where(
        np.isfinite(factor.to_numpy(dtype="float64"))
    )
    finite_forward = aligned_forward.where(
        np.isfinite(aligned_forward.to_numpy(dtype="float64"))
    )
    paired = finite_factor.notna() & finite_forward.notna()
    active_names = paired.sum(axis=1).astype("float64")
    daily = daily_ic(
        finite_factor,
        finite_forward,
        method,
        min_names=min_names,
    )
    weights = validate_portfolio_weights(weighting_scheme.weights(finite_factor))
    varying_factor = finite_factor.nunique(axis=1).gt(1)

    summaries: list[dict[str, Any]] = []
    oriented_values: list[float] = []
    for index, fold_dates in enumerate(folds):
        fold_ic = daily.reindex(fold_dates).dropna()
        breadth = active_names.reindex(fold_ic.index).dropna()
        valid_dates = int(len(fold_ic))
        fold_observations = int(len(fold_dates))
        ic_coverage = valid_dates / fold_observations if fold_observations else 0.0
        signed = float(fold_ic.mean()) if valid_dates else 0.0
        if not math.isfinite(signed):
            signed = 0.0
        oriented = float(polarity * signed)
        fold_ir = float(ic_ir(fold_ic))
        min_active = float(breadth.min()) if len(breadth) else 0.0
        fold_weights = weights.reindex(fold_dates).fillna(0.0)
        gross = fold_weights.abs().sum(axis=1)
        positive = fold_weights.clip(lower=0.0).sum(axis=1)
        negative = -fold_weights.clip(upper=0.0).sum(axis=1)
        active_weight_dates = int(gross.gt(1e-12).sum())
        varying_factor_dates = int(
            varying_factor.reindex(fold_dates).fillna(False).sum()
        )
        two_sided_dates = int(
            (positive.gt(1e-12) & negative.gt(1e-12)).sum()
        )
        exposure_coverage = (
            active_weight_dates / fold_observations if fold_observations else 0.0
        )
        varying_factor_coverage = (
            varying_factor_dates / fold_observations if fold_observations else 0.0
        )
        two_sided_coverage = (
            two_sided_dates / fold_observations if fold_observations else 0.0
        )
        coverage_failures: list[str] = []
        if valid_dates < min_valid_dates_per_fold:
            coverage_failures.append("insufficient_valid_dates")
        if ic_coverage < minimum_fold_coverage:
            coverage_failures.append("low_valid_date_coverage")
        if min_active < float(min_names):
            coverage_failures.append("insufficient_cross_sectional_breadth")
        if varying_factor_coverage < minimum_fold_coverage:
            coverage_failures.append("low_varying_factor_coverage")
        if exposure_coverage < minimum_fold_coverage:
            coverage_failures.append("low_exposure_coverage")
        if two_sided_coverage < minimum_fold_coverage:
            coverage_failures.append("low_two_sided_coverage")
        adequate = not coverage_failures
        summaries.append(
            {
                "index": index,
                "start": (
                    fold_dates[0].date().isoformat() if len(fold_dates) else None
                ),
                "end": (
                    fold_dates[-1].date().isoformat() if len(fold_dates) else None
                ),
                "observations": fold_observations,
                "valid_dates": valid_dates,
                "ic_coverage": float(ic_coverage),
                "signed_ic": signed,
                "oriented_ic": oriented,
                "ic_ir": fold_ir,
                "oriented_ic_ir": float(polarity * fold_ir),
                "avg_active_names": (
                    float(breadth.mean()) if len(breadth) else 0.0
                ),
                "min_active_names": min_active,
                "active_weight_dates": active_weight_dates,
                "varying_factor_dates": varying_factor_dates,
                "two_sided_dates": two_sided_dates,
                "varying_factor_coverage": float(varying_factor_coverage),
                "exposure_coverage": float(exposure_coverage),
                "two_sided_coverage": float(two_sided_coverage),
                "avg_gross_exposure": (
                    float(gross.mean()) if fold_observations else 0.0
                ),
                "adequate_coverage": adequate,
                "coverage_failures": coverage_failures,
                "positive": bool(adequate and oriented > 0.0),
            }
        )
        oriented_values.append(oriented)

    validation_observations = sum(len(item) for item in folds)
    validation_dates_index = pd.DatetimeIndex(
        [date for fold in folds for date in fold]
    )
    clean_daily = daily.reindex(validation_dates_index).dropna()
    validation_weights = weights.reindex(validation_dates_index).fillna(0.0)
    validation_gross = validation_weights.abs().sum(axis=1)
    validation_positive = validation_weights.clip(lower=0.0).sum(axis=1)
    validation_negative = -validation_weights.clip(upper=0.0).sum(axis=1)
    median_oriented = float(median(oriented_values)) if oriented_values else 0.0
    worst = float(min(oriented_values)) if oriented_values else 0.0
    metrics: dict[str, Any] = {
        "signed_ic": float(clean_daily.mean()) if len(clean_daily) else 0.0,
        "oriented_ic": (
            float(polarity * clean_daily.mean()) if len(clean_daily) else 0.0
        ),
        "mean_abs_ic": float(clean_daily.abs().mean()) if len(clean_daily) else 0.0,
        "ic_ir": float(ic_ir(clean_daily)),
        "valid_dates": float(len(clean_daily)),
        "valid_date_coverage": (
            float(len(clean_daily) / validation_observations)
            if validation_observations
            else 0.0
        ),
        "varying_factor_coverage": (
            float(
                varying_factor.reindex(validation_dates_index).fillna(False).mean()
            )
            if validation_observations
            else 0.0
        ),
        "exposure_coverage": (
            float(validation_gross.gt(1e-12).mean())
            if validation_observations
            else 0.0
        ),
        "two_sided_coverage": (
            float(
                (
                    validation_positive.gt(1e-12)
                    & validation_negative.gt(1e-12)
                ).mean()
            )
            if validation_observations
            else 0.0
        ),
        "avg_active_names": (
            float(active_names.reindex(clean_daily.index).mean())
            if len(clean_daily)
            else 0.0
        ),
        "min_active_names": (
            float(active_names.reindex(clean_daily.index).min())
            if len(clean_daily)
            else 0.0
        ),
        "median_oriented_ic": median_oriented,
        "worst_fold_ic": worst,
    }
    return tuple(summaries), metrics


def select_validation_candidate(
    candidates: Sequence[Candidate],
    panel: Panel,
    forward: pd.DataFrame,
    validation_dates: pd.DatetimeIndex,
    *,
    method: str = "spearman",
    parsimony: float = 0.0,
    complexity_penalty_mode: str = "per_node",
    complexity_penalty_value: float | None = None,
    max_nodes: int = 40,
    validation_folds: int = 3,
    fold_embargo: int = 5,
    min_valid_dates_per_fold: int = 60,
    minimum_fold_coverage: float = 0.60,
    min_names: int = 5,
    weighting_scheme: WeightingScheme | None = None,
    workers: int = 1,
    memory_budget_bytes: int | None = None,
) -> ValidationSelection:
    """Select a first-seen unique candidate without looking at the locked holdout.

    Candidate polarity comes exclusively from its training metrics. Validation
    ranks the median oriented fold IC minus the resolved complexity deduction.
    Ties prefer the worst fold, training fitness, lower expanded complexity, and
    first-seen order. If no candidate passes every coverage floor and the
    required positive-fold count, the best training candidate is returned as an
    explicitly unvalidated diagnostic.
    """
    if not candidates:
        raise ValueError("validation selection requires at least one candidate")
    del workers, memory_budget_bytes  # one evaluation per candidate; deterministic coordinator
    penalty_value = (
        float(parsimony)
        if complexity_penalty_value is None
        else float(complexity_penalty_value)
    )
    if not math.isfinite(penalty_value) or penalty_value < 0.0:
        raise ValueError("complexity penalty must be finite and non-negative")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ValueError("max_nodes must be a positive integer")
    if (
        isinstance(min_valid_dates_per_fold, bool)
        or not isinstance(min_valid_dates_per_fold, int)
        or min_valid_dates_per_fold <= 0
    ):
        raise ValueError("min_valid_dates_per_fold must be a positive integer")
    if (
        not math.isfinite(float(minimum_fold_coverage))
        or not 0.0 < float(minimum_fold_coverage) <= 1.0
    ):
        raise ValueError("minimum_fold_coverage must be in (0, 1]")
    selected_scheme = weighting_scheme or QuantileLongShort()
    validation_index = pd.DatetimeIndex(validation_dates)
    validation_forward = forward.loc[forward.index.isin(validation_index)]
    folds = _chronological_folds(
        validation_index,
        count=validation_folds,
        embargo=fold_embargo,
    )
    required_positive = validation_folds // 2 + 1

    unique: list[tuple[int, Candidate, str, int, int]] = []
    seen: set[str] = set()
    for first_seen, candidate in enumerate(candidates):
        expanded = expand_all(candidate.tree)
        key = to_json(expanded)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            (
                first_seen,
                candidate,
                key,
                expanded.size(),
                expanded.unique_computation_size(),
            )
        )

    ranked: list[dict[str, Any]] = []
    for first_seen, candidate, _key, complexity, unique_complexity in unique:
        training_signed = candidate.metrics.get(
            "signed_ic",
            candidate.metrics.get("ic", 0.0),
        )
        polarity = -1 if training_signed < 0.0 else 1
        fold_metrics, metrics = _candidate_validation(
            candidate.tree,
            panel,
            validation_forward,
            folds,
            polarity=polarity,
            method=method,
            min_names=min_names,
            min_valid_dates_per_fold=min_valid_dates_per_fold,
            minimum_fold_coverage=float(minimum_fold_coverage),
            weighting_scheme=selected_scheme,
        )
        deduction = _complexity_deduction(
            complexity,
            mode=complexity_penalty_mode,
            value=penalty_value,
            max_nodes=max_nodes,
        )
        metrics["novelty_multiplier"] = candidate.metrics.get("novelty_multiplier", 1.0)
        objective = float(metrics["median_oriented_ic"] * metrics["novelty_multiplier"] - deduction)
        positive = sum(bool(item["positive"]) for item in fold_metrics)
        adequate = all(bool(item["adequate_coverage"]) for item in fold_metrics)
        ranked.append(
            {
                "candidate": candidate,
                "first_seen": first_seen,
                "complexity": complexity,
                "unique_complexity": unique_complexity,
                "polarity": polarity,
                "folds": fold_metrics,
                "metrics": metrics,
                "deduction": deduction,
                "objective": objective,
                "positive_folds": positive,
                "qualified": bool(
                    adequate
                    and positive >= required_positive
                    and objective > 0.0
                ),
            }
        )

    eligible = [item for item in ranked if item["qualified"]]
    reason: str | None = None
    if eligible:
        winner = max(
            eligible,
            key=lambda item: (
                item["objective"],
                item["metrics"]["worst_fold_ic"],
                item["candidate"].fitness,
                -item["complexity"],
                -item["first_seen"],
            ),
        )
        validated = True
    else:
        reason = (
            f"no candidate passed all {validation_folds} validation-fold coverage "
            f"floors with at least {required_positive} positive oriented folds"
        )
        winner = max(
            ranked,
            key=lambda item: (
                item["candidate"].fitness,
                -item["complexity"],
                -item["first_seen"],
            ),
        )
        validated = False

    candidate = winner["candidate"]

    return ValidationSelection(
        source_tree=candidate.tree,
        oriented_tree=_oriented_tree(candidate.tree, winner["polarity"]),
        validated=validated,
        reason=reason,
        first_seen=winner["first_seen"],
        polarity=winner["polarity"],
        training_fitness=float(candidate.fitness),
        training_metrics=dict(candidate.metrics),
        validation_fitness=float(winner["objective"]),
        validation_metrics=dict(winner["metrics"]),
        fold_metrics=tuple(winner["folds"]),
        positive_folds=int(winner["positive_folds"]),
        required_positive_folds=required_positive,
        expanded_nodes=int(winner["complexity"]),
        expanded_unique_nodes=int(winner["unique_complexity"]),
        complexity_penalty_mode=complexity_penalty_mode,
        complexity_penalty_value=penalty_value,
        complexity_deduction=float(winner["deduction"]),
        complexity_max_nodes=max_nodes,
        final_objective=float(winner["objective"]),
        candidate_validation_scores=tuple(
            (item["candidate"].tree, float(item["objective"])) for item in ranked
        ),
    )


def compare_validation_strategies(
    tree: Node,
    panel: Panel,
    forward: pd.DataFrame,
    validation_dates: pd.DatetimeIndex,
    strategies: Sequence[PortfolioStrategySpec],
    *,
    costs: TransactionCostModel,
    horizon: int = 1,
    execution: str = "close",
    method: str = "spearman",
    min_names: int = 5,
    validation_folds: int = 3,
    fold_embargo: int = 5,
    min_valid_dates_per_fold: int = 60,
    minimum_fold_coverage: float = 0.60,
) -> list[dict[str, Any]]:
    """Compare pinned portfolio strategies without touching the locked holdout.

    The formula and its training-derived polarity are already frozen in ``tree``.
    This helper evaluates it once, reports each strategy on the same validation
    dates, and applies the same fold-level exposure floor used during candidate
    selection.
    """
    # Local import avoids a package-initialization cycle through
    # backtest.engine -> backtest.metrics -> validation.
    from alphalineage.backtest.reporting import backtest_report

    if not strategies:
        raise ValueError("at least one strategy is required")
    if len({item.id for item in strategies}) != len(strategies):
        raise ValueError("strategy ids must be unique")
    factor = evaluate(expand_all(tree), panel)
    if not isinstance(factor, pd.DataFrame):
        raise TypeError("strategy comparison formula must evaluate to a panel")
    factor = factor.where(np.isfinite(factor.to_numpy(dtype="float64")))
    clean_forward = forward.where(
        np.isfinite(forward.to_numpy(dtype="float64"))
    )
    folds = _chronological_folds(
        pd.DatetimeIndex(validation_dates),
        count=validation_folds,
        embargo=fold_embargo,
    )
    validation_ic = daily_ic(factor, clean_forward, method, min_names=min_names)
    validation_varying = factor.where(
        np.isfinite(factor.to_numpy(dtype="float64"))
    ).nunique(axis=1).gt(1)
    results: list[dict[str, Any]] = []
    for spec in strategies:
        scheme = spec.weighting_scheme()
        aggregate = backtest_report(
            factor,
            panel,
            clean_forward,
            scheme,
            costs,
            pd.DatetimeIndex(validation_dates),
            ic_method=method,
            min_names=min_names,
            horizon=horizon,
            execution=execution,
        )
        fold_results: list[dict[str, Any]] = []
        for index, fold_dates in enumerate(folds):
            report = backtest_report(
                factor,
                panel,
                clean_forward,
                scheme,
                costs,
                fold_dates,
                ic_method=method,
                min_names=min_names,
                horizon=horizon,
                execution=execution,
            )
            health = dict(report.get("portfolio_health") or {})
            valid_ic_dates = int(len(validation_ic.reindex(fold_dates).dropna()))
            varying_factor_dates = int(
                validation_varying.reindex(fold_dates).fillna(False).sum()
            )
            # IC coverage is already enforced for the selected expression. For a
            # portfolio strategy, use realized calendar coverage plus actual
            # exposure; both must cover the same frozen fold.
            observations = int(len(fold_dates))
            ic_coverage = valid_ic_dates / observations if observations else 0.0
            varying_factor_coverage = (
                varying_factor_dates / observations if observations else 0.0
            )
            realized_coverage = (
                int(report.get("calendar_observations", 0)) / observations
                if observations
                else 0.0
            )
            coverage_failures: list[str] = []
            if observations < min_valid_dates_per_fold:
                coverage_failures.append("insufficient_fold_dates")
            if valid_ic_dates < min_valid_dates_per_fold:
                coverage_failures.append("insufficient_valid_dates")
            if ic_coverage < minimum_fold_coverage:
                coverage_failures.append("low_valid_date_coverage")
            if varying_factor_coverage < minimum_fold_coverage:
                coverage_failures.append("low_varying_factor_coverage")
            if realized_coverage < minimum_fold_coverage:
                coverage_failures.append("low_realized_return_coverage")
            if (
                float(health.get("exposure_coverage", 0.0))
                < minimum_fold_coverage
            ):
                coverage_failures.append("low_exposure_coverage")
            if (
                float(health.get("two_sided_coverage", 0.0))
                < minimum_fold_coverage
            ):
                coverage_failures.append("low_two_sided_coverage")
            adequate = not coverage_failures
            fold_results.append(
                {
                    "index": index,
                    "start": (
                        fold_dates[0].date().isoformat() if len(fold_dates) else None
                    ),
                    "end": (
                        fold_dates[-1].date().isoformat() if len(fold_dates) else None
                    ),
                    "observations": observations,
                    "valid_ic_dates": valid_ic_dates,
                    "ic_coverage": float(ic_coverage),
                    "varying_factor_dates": varying_factor_dates,
                    "varying_factor_coverage": float(varying_factor_coverage),
                    "realized_coverage": float(realized_coverage),
                    "adequate_coverage": adequate,
                    "coverage_failures": coverage_failures,
                    "metrics": dict(report.get("metrics") or {}),
                    "portfolio_health": health,
                }
            )
        results.append(
            {
                "strategy_id": spec.id,
                "spec": spec.to_dict(),
                "strategy": spec.to_dict(),  # compatibility alias
                "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
                "validation_backtest": aggregate,
                "validation": aggregate,  # compatibility alias
                "folds": fold_results,
                "eligible": all(item["adequate_coverage"] for item in fold_results),
            }
        )
    return results
