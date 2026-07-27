"""The backtest: factor -> weights -> gross/net returns -> metrics.

Supersedes the Phase-3 gross ``long_short_returns`` for the verdict: it charges transaction
costs and exposes a ``usable`` flag (gross-profitable but net-negative -> not usable,
invariant 6). ``net_return_fn`` packages the cost-aware return series so the validation
pipeline can deflate on *net* returns.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from alphalineage.backtest.costs import TransactionCostModel, turnover_series
from alphalineage.backtest.metrics import (
    annualized_sharpe,
    max_drawdown,
    position_magnitude,
)
from alphalineage.backtest.portfolio import WeightingScheme
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

    signal_weights, realized_returns = scheme.weights(factor).align(
        panel["returns"], join="inner"
    )
    realized_weights = (
        signal_weights.shift(1).rolling(window=horizon, min_periods=1).mean().fillna(0.0)
    )

    gross_realized = (realized_weights * realized_returns).sum(axis=1, min_count=1)
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

    return BacktestResult(
        scheme=scheme.name,
        gross_returns=gross,
        net_returns=net,
        realization_dates=realization_dates,
        traded_notional=traded_notional,
        transaction_costs=transaction_costs,
        gross_sharpe=annualized_sharpe(metric_gross, periods),
        net_sharpe=annualized_sharpe(metric_net, periods),
        max_drawdown=max_drawdown(metric_net),
        turnover=float(metric_turnover.mean()) if len(metric_turnover) else 0.0,
        position=position_magnitude(metric_weights),
        usable=bool(metric_net.mean() > 0),
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
    weights = scheme.weights(factor)
    weights, returns = weights.align(fwd, join="inner")
    gross = (weights * returns).sum(axis=1, min_count=1)
    return gross - costs.cost(weights)
