"""The backtest: factor -> weights -> gross/net returns -> metrics.

Supersedes the Phase-3 gross ``long_short_returns`` for the verdict: it charges transaction
costs and exposes a ``usable`` flag (gross-profitable but net-negative -> not usable,
invariant 6). ``net_return_fn`` packages the cost-aware return series so the validation
pipeline can deflate on *net* returns.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphalineage.backtest.costs import TransactionCostModel, turnover_series
from alphalineage.backtest.metrics import (
    annualized_sharpe,
    max_drawdown,
    position_magnitude,
)
from alphalineage.backtest.portfolio import (
    WeightingScheme,
    validate_portfolio_weights,
)
from alphalineage.core.evaluate import evaluate
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node


@dataclass
class BacktestResult:
    scheme: str
    gross_returns: pd.Series
    net_returns: pd.Series
    realization_dates: pd.Series
    traded_notional: pd.Series
    transaction_costs: pd.Series
    gross_sharpe: float
    net_sharpe: float
    max_drawdown: float
    turnover: float
    position: dict[str, float]
    usable: bool
    insolvent: bool
    missing_return_observations: int
    integrity_issues: tuple[str, ...]
    active_exposure: pd.Series
    portfolio_health: dict[str, Any]


def portfolio_health(
    weights: pd.DataFrame,
    *,
    eligible: pd.Series | None = None,
    factor: pd.DataFrame | None = None,
    minimum_coverage: float = 0.60,
) -> dict[str, Any]:
    """Describe whether a factor actually formed an investable portfolio.

    A string of zero cash returns is not performance evidence.  Health is based on
    realized portfolio weights, independent from whatever the market happened to
    return, and requires both non-zero and two-sided exposure on a useful share of
    the requested dates.
    """
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1")
    eligible_mask = (
        pd.Series(True, index=weights.index)
        if eligible is None
        else eligible.reindex(weights.index).fillna(False).astype(bool)
    )
    gross = weights.abs().sum(axis=1)
    positive = weights.clip(lower=0.0).sum(axis=1)
    negative = -weights.clip(upper=0.0).sum(axis=1)
    active = eligible_mask & gross.gt(1e-12)
    two_sided = active & positive.gt(1e-12) & negative.gt(1e-12)
    eligible_count = int(eligible_mask.sum())
    if factor is None:
        eligible_dates = eligible_count
        flat_factor_dates = 0
    else:
        aligned_factor = factor.reindex(index=weights.index, columns=weights.columns)
        finite_factor = aligned_factor.where(
            np.isfinite(aligned_factor.to_numpy(dtype="float64"))
        )
        finite_names = finite_factor.notna().sum(axis=1)
        varying = finite_factor.nunique(axis=1).gt(1)
        eligible_dates = int((eligible_mask & finite_names.ge(2)).sum())
        flat_factor_dates = int((eligible_mask & ~varying).sum())
    active_count = int(active.sum())
    two_sided_count = int(two_sided.sum())
    exposure_coverage = active_count / eligible_count if eligible_count else 0.0
    two_sided_coverage = two_sided_count / eligible_count if eligible_count else 0.0
    issues: list[str] = []
    reason: str | None = None
    if eligible_count == 0:
        issues.append("no realized observations are available")
        reason = "no_calendar_observations"
    elif active_count == 0:
        issues.append("formula produced no portfolio exposure")
        reason = "no_exposure"
    elif exposure_coverage < minimum_coverage:
        issues.append(
            f"portfolio exposure covers only {exposure_coverage:.1%} of realized observations"
        )
        reason = "low_exposure_coverage"
    if active_count and two_sided_coverage < minimum_coverage:
        issues.append(
            f"two-sided exposure covers only {two_sided_coverage:.1%} of realized observations"
        )
        reason = reason or "low_two_sided_coverage"
    return {
        "valid": not issues,
        "reason": reason,
        "eligible_dates": eligible_dates,
        "calendar_observations": eligible_count,
        "eligible_observations": eligible_count,
        "active_observations": active_count,
        "two_sided_observations": two_sided_count,
        "exposure_coverage": float(exposure_coverage),
        "two_sided_coverage": float(two_sided_coverage),
        "flat_factor_dates": flat_factor_dates,
        "minimum_coverage": float(minimum_coverage),
        "issues": issues,
    }


def _staggered_portfolio_path(
    factor: pd.DataFrame,
    panel: Panel,
    scheme: WeightingScheme,
    costs: TransactionCostModel,
    horizon: int,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return signal-indexed P&L from next-session, overlapping holdings.

    A signal observed at session ``t`` can first earn the close-to-close return labelled
    ``t + 1`` in :class:`Panel`. For a horizon of ``h``, each signal cohort remains active
    for ``h`` sessions and the live book is the equal-weight average of available cohorts.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")

    signal_weights = validate_portfolio_weights(scheme.weights(factor))
    signal_weights, realized_returns = signal_weights.align(panel["returns"], join="inner")
    realized_weights = (
        signal_weights.shift(1).rolling(window=horizon, min_periods=1).mean().fillna(0.0)
    )

    active = realized_weights.ne(0.0)
    unavailable = active & (
        realized_returns.isna()
        | ~pd.DataFrame(
            np.isfinite(realized_returns.to_numpy(dtype="float64")),
            index=realized_returns.index,
            columns=realized_returns.columns,
        )
    )
    gross_realized = (realized_weights * realized_returns).sum(axis=1, min_count=1)
    # Never silently reallocate away from a held name because its realized return is
    # missing.  The portfolio path is unknown on that date and all downstream evidence
    # must remain visibly incomplete.
    gross_realized = gross_realized.mask(unavailable.any(axis=1))
    traded_realized = turnover_series(realized_weights)
    # Reuse the exact transition series exposed in the report. DSR/PBO evaluates this path for
    # many candidates, so recomputing the same DataFrame diff inside ``costs.cost`` is material.
    costs_realized = traded_realized * costs.rate
    net_realized = gross_realized - costs_realized

    # Validation partitions are defined on signal dates. Retain that stable index while carrying
    # the actual market realization date explicitly for reporting and benchmark alignment.
    gross_by_signal = gross_realized.shift(-1)
    net_by_signal = net_realized.shift(-1)
    weights_by_signal = realized_weights.shift(-1)
    traded_by_signal = traded_realized.shift(-1)
    costs_by_signal = costs_realized.shift(-1)
    realization_dates = pd.Series(
        pd.DatetimeIndex(signal_weights.index),
        index=signal_weights.index,
        dtype="datetime64[ns]",
    ).shift(-1)
    return (
        gross_by_signal,
        net_by_signal,
        weights_by_signal,
        realization_dates,
        traded_by_signal,
        costs_by_signal,
    )


def backtest(
    factor: pd.DataFrame,
    panel: Panel,
    fwd: pd.DataFrame,
    scheme: WeightingScheme,
    costs: TransactionCostModel,
    *,
    periods: int = 252,
    dates: pd.DatetimeIndex | None = None,
    horizon: int = 1,
) -> BacktestResult:
    """Run ``scheme`` over ``factor``; optionally report metrics on signal ``dates``.

    ``fwd`` remains explicit for compatibility and documents the IC target beside this report.
    P&L itself uses next-session realized returns. Weights and costs are built over the complete
    timeline so rolling state and overlapping cohorts cross split boundaries honestly.
    """
    del fwd
    (
        gross,
        net,
        weights,
        realization_dates,
        traded_notional,
        transaction_costs,
    ) = _staggered_portfolio_path(factor, panel, scheme, costs, horizon)

    if dates is not None:
        mask = gross.index.isin(dates)
        gross, net = gross[mask], net[mask]
        weights, realization_dates = weights.loc[mask], realization_dates.loc[mask]
        traded_notional = traded_notional.loc[mask]
        transaction_costs = transaction_costs.loc[mask]

    realized = realization_dates.notna()
    metric_gross = gross.loc[realized]
    metric_net = net.loc[realized]
    metric_weights = weights.loc[realized]
    metric_turnover = traded_notional.loc[realized]
    missing_return_observations = int(metric_net.isna().sum())
    insolvent = bool(metric_net.le(-1.0).fillna(False).any())
    issues: list[str] = []
    if missing_return_observations:
        issues.append(
            f"{missing_return_observations} realized portfolio returns are unavailable"
        )
    if insolvent:
        issues.append("portfolio return reached or crossed -100%; the equity path is insolvent")
    health = portfolio_health(
        metric_weights,
        factor=factor.reindex(metric_weights.index),
    )
    issues.extend(str(issue) for issue in health["issues"])
    metric_valid = not issues
    active_exposure = weights.abs().sum(axis=1).gt(1e-12)

    return BacktestResult(
        scheme=scheme.name,
        gross_returns=gross,
        net_returns=net,
        realization_dates=realization_dates,
        traded_notional=traded_notional,
        transaction_costs=transaction_costs,
        gross_sharpe=annualized_sharpe(metric_gross, periods) if metric_valid else float("nan"),
        net_sharpe=annualized_sharpe(metric_net, periods) if metric_valid else float("nan"),
        max_drawdown=(
            -1.0
            if insolvent
            else max_drawdown(metric_net)
            if metric_valid
            else float("nan")
        ),
        turnover=float(metric_turnover.mean()) if len(metric_turnover) else 0.0,
        position=position_magnitude(metric_weights),
        usable=bool(metric_valid and metric_net.mean() > 0),
        insolvent=insolvent,
        missing_return_observations=missing_return_observations,
        integrity_issues=tuple(issues),
        active_exposure=active_exposure,
        portfolio_health=health,
    )


def compare_schemes(
    factor: pd.DataFrame,
    panel: Panel,
    fwd: pd.DataFrame,
    schemes: Sequence[WeightingScheme],
    costs: TransactionCostModel,
    *,
    periods: int = 252,
    dates: pd.DatetimeIndex | None = None,
    horizon: int = 1,
) -> list[BacktestResult]:
    """Run every scheme on the same factor + window for a side-by-side comparison."""
    return [
        backtest(
            factor,
            panel,
            fwd,
            scheme,
            costs,
            periods=periods,
            dates=dates,
            horizon=horizon,
        )
        for scheme in schemes
    ]


def comparison_frame(results: Sequence[BacktestResult]) -> pd.DataFrame:
    """Tabulate scheme results (net/gross Sharpe, drawdown, turnover, position size, usable)."""
    return pd.DataFrame(
        [
            {
                "scheme": r.scheme,
                "net_sharpe": r.net_sharpe,
                "gross_sharpe": r.gross_sharpe,
                "max_drawdown": r.max_drawdown,
                "turnover": r.turnover,
                "avg_positions": r.position["avg_positions"],
                "max_position": r.position["max_position"],
                "usable": r.usable,
                "insolvent": r.insolvent,
                "missing_return_observations": r.missing_return_observations,
                "exposure_coverage": r.portfolio_health["exposure_coverage"],
                "active_observations": r.portfolio_health["active_observations"],
            }
            for r in results
        ]
    )


def net_return_fn(
    panel: Panel,
    fwd: pd.DataFrame,
    scheme: WeightingScheme,
    costs: TransactionCostModel,
    *,
    horizon: int = 1,
) -> Callable[[Node], pd.Series]:
    """A ``tree -> net (after-cost) return Series`` closure for the validation verdict."""

    def fn(tree: Node) -> pd.Series:
        factor = evaluate(tree, panel)
        if not isinstance(factor, pd.DataFrame):
            return pd.Series(dtype="float64")
        return net_returns_for_factor(
            factor,
            fwd,
            scheme,
            costs,
            panel=panel,
            horizon=horizon,
        )

    return fn


def net_returns_for_factor(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    scheme: WeightingScheme,
    costs: TransactionCostModel,
    *,
    panel: Panel | None = None,
    horizon: int = 1,
) -> pd.Series:
    """Net returns for an already-evaluated factor, avoiding duplicate evaluation.

    Passing ``panel`` selects the staggered next-session implementation. The original
    forward-target calculation remains as a compatibility fallback for direct callers.
    """
    if panel is not None:
        _, net, _, _, _, _ = _staggered_portfolio_path(
            factor, panel, scheme, costs, horizon
        )
        return net
    weights = validate_portfolio_weights(scheme.weights(factor))
    weights, returns = weights.align(fwd, join="inner")
    gross = (weights * returns).sum(axis=1, min_count=1)
    return gross - costs.cost(weights)
