"""P4-T3 - backtest metrics: Sharpe, max drawdown, turnover, position magnitude, IC decay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalineage.backtest.costs import turnover_series
from alphalineage.core.fitness import daily_ic, forward_returns
from alphalineage.core.panel import Panel
from alphalineage.validation.deflated_sharpe import sharpe_ratio


def annualized_sharpe(returns: pd.Series, periods: int = 252) -> float:
    return float(sharpe_ratio(returns) * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown of the compounded equity curve (<= 0)."""
    r = pd.Series(returns, dtype="float64").fillna(0.0)
    if r.empty:
        return 0.0
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def turnover(weights: pd.DataFrame) -> float:
    """Mean per-rebalance turnover ``Î£|Î”w|`` (excluding the initial establishment)."""
    series = turnover_series(weights).iloc[1:]
    return float(series.mean()) if len(series) else 0.0


def position_magnitude(weights: pd.DataFrame) -> dict[str, float]:
    """Average gross exposure, number of held names, and largest single position."""
    abs_w = weights.abs()
    return {
        "avg_gross": float(abs_w.sum(axis=1).mean()),
        "avg_positions": float((weights != 0).sum(axis=1).mean()),
        "max_position": float(abs_w.max(axis=1).mean()),
    }


def ic_decay(
    factor: pd.DataFrame, panel: Panel, horizons: tuple[int, ...] = (1, 2, 3, 5, 10)
) -> dict[int, float]:
    """Mean |rank IC| of ``factor`` against forward returns at growing horizons."""
    out: dict[int, float] = {}
    for h in horizons:
        ic = daily_ic(factor, forward_returns(panel, h), "spearman")
        out[h] = float(ic.abs().mean()) if ic.notna().any() else 0.0
    return out
