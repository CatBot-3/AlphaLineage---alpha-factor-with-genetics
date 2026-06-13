"""P4-T2 — transaction-cost + slippage model (invariant 6: costs are real).

Each rebalance pays a proportional cost on the notional traded:
``cost_t = rate * turnover_t`` where ``turnover_t = Σ_i |w_{t,i} - w_{t-1,i}|`` and the first
period charges for establishing the book from cash. Net return = gross return - cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    """Per-date traded notional ``Σ|Δw|``; the first row is the initial establishment."""
    delta = weights.diff()
    delta.iloc[0] = weights.iloc[0]
    return delta.abs().sum(axis=1)


@dataclass
class TransactionCostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 5.0

    @property
    def rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4

    def cost(self, weights: pd.DataFrame) -> pd.Series:
        """Per-date cost charged on traded notional."""
        return turnover_series(weights) * self.rate
