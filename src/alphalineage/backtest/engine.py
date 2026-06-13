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

from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.metrics import (
    annualized_sharpe,
    max_drawdown,
    position_magnitude,
    turnover,
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
    gross_sharpe: float
    net_sharpe: float
    max_drawdown: float
    turnover: float
    position: dict[str, float]
    usable: bool


def backtest(
    factor: pd.DataFrame,
    panel: Panel,
    fwd: pd.DataFrame,
    scheme: WeightingScheme,
    costs: TransactionCostModel,
    *,
    periods: int = 252,
    dates: pd.DatetimeIndex | None = None,
) -> BacktestResult:
    """Run ``scheme`` over ``factor``; optionally report metrics on a ``dates`` sub-window.

    Weights and costs are computed over the full timeline (so trailing windows warm up and
    turnover stays continuous across a split boundary); metrics are then taken on ``dates``.
    """
    weights = scheme.weights(factor)
    weights, returns = weights.align(fwd, join="inner")
    gross = (weights * returns).sum(axis=1, min_count=1)
    net = gross - costs.cost(weights)

    if dates is not None:
        mask = gross.index.isin(dates)
        gross, net, weights = gross[mask], net[mask], weights.loc[mask]

    return BacktestResult(
        scheme=scheme.name,
        gross_returns=gross,
        net_returns=net,
        gross_sharpe=annualized_sharpe(gross, periods),
        net_sharpe=annualized_sharpe(net, periods),
        max_drawdown=max_drawdown(net),
        turnover=turnover(weights),
        position=position_magnitude(weights),
        usable=bool(net.mean() > 0),
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
) -> list[BacktestResult]:
    """Run every scheme on the same factor + window for a side-by-side comparison."""
    return [backtest(factor, panel, fwd, s, costs, periods=periods, dates=dates) for s in schemes]


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
) -> Callable[[Node], pd.Series]:
    """A ``tree -> net (after-cost) return Series`` closure for the validation verdict."""

    def fn(tree: Node) -> pd.Series:
        factor = evaluate(tree, panel)
        if not isinstance(factor, pd.DataFrame):
            return pd.Series(dtype="float64")
        weights = scheme.weights(factor)
        weights, returns = weights.align(fwd, join="inner")
        gross = (weights * returns).sum(axis=1, min_count=1)
        return gross - costs.cost(weights)

    return fn
