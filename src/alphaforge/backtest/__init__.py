"""Phase 4 — backtest + portfolio with costs (invariant 6).

Turn a factor into a realistic long/short portfolio (:mod:`portfolio`), charge transaction
costs and slippage (:mod:`costs`), measure it (:mod:`metrics`), and run it end to end
(:mod:`engine`) — producing net-of-cost returns and a ``usable`` flag, and feeding the net
returns into the deflated verdict.
"""

from alphaforge.backtest.costs import TransactionCostModel, turnover_series
from alphaforge.backtest.engine import (
    BacktestResult,
    backtest,
    compare_schemes,
    comparison_frame,
    net_return_fn,
)
from alphaforge.backtest.metrics import (
    annualized_sharpe,
    ic_decay,
    max_drawdown,
    position_magnitude,
    turnover,
)
from alphaforge.backtest.portfolio import (
    SCHEMES,
    QuantileLongShort,
    RankProportional,
    WeightingScheme,
    get_scheme,
    neutralize,
)

__all__ = [
    "SCHEMES",
    "BacktestResult",
    "QuantileLongShort",
    "RankProportional",
    "TransactionCostModel",
    "WeightingScheme",
    "annualized_sharpe",
    "backtest",
    "compare_schemes",
    "comparison_frame",
    "get_scheme",
    "ic_decay",
    "max_drawdown",
    "net_return_fn",
    "neutralize",
    "position_magnitude",
    "turnover",
    "turnover_series",
]
