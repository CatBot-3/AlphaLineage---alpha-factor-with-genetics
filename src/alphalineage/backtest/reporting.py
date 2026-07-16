"""Shared, JSON-safe backtest reporting for formula tests and locked holdouts.

Keeping this serialization beside the backtest engine prevents the exploratory formula-test
path and the final locked-test path from quietly drifting to different metric definitions.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.engine import backtest
from alphalineage.backtest.portfolio import WeightingScheme
from alphalineage.core.fitness import daily_ic, ic_ir
from alphalineage.core.panel import Panel


def json_number(value: Any) -> float | None:
    """Return a finite JSON number, otherwise ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def backtest_report(
    factor: pd.DataFrame,
    panel: Panel,
    forward: pd.DataFrame,
    scheme: WeightingScheme,
    costs: TransactionCostModel,
    dates: pd.DatetimeIndex,
    *,
    ic_method: str = "spearman",
    min_names: int = 5,
) -> dict[str, Any]:
    """Compute the canonical metric/time-series payload over ``dates``.

    Rolling state, weights, and costs are built on the complete panel and only reporting is
    sliced to ``dates``.  This preserves warm-up and boundary turnover semantics.
    """
    report_dates = pd.DatetimeIndex(dates)
    tested = backtest(factor, panel, forward, scheme, costs, dates=report_dates)
    ic = daily_ic(factor, forward, ic_method, min_names=min_names).reindex(report_dates)
    clean_ic = ic.dropna()
    signed_ic = clean_ic.mean() if len(clean_ic) else float("nan")
    mean_abs_ic = clean_ic.abs().mean() if len(clean_ic) else float("nan")

    gross, net = tested.gross_returns.align(tested.net_returns, join="outer")
    dated_returns = [
        {
            "date": pd.Timestamp(date).date().isoformat(),
            "gross": json_number(gross.loc[date]),
            "net": json_number(net.loc[date]),
        }
        for date in gross.index
    ]
    equity = (1.0 + net.fillna(0.0)).cumprod()
    normalized_equity = [
        {"date": pd.Timestamp(date).date().isoformat(), "value": json_number(value)}
        for date, value in equity.items()
    ]
    return {
        "start": (
            pd.Timestamp(report_dates.min()).date().isoformat() if len(report_dates) else None
        ),
        "end": pd.Timestamp(report_dates.max()).date().isoformat() if len(report_dates) else None,
        "observations": int(net.notna().sum()),
        "metrics": {
            "signed_ic": json_number(signed_ic),
            "mean_abs_ic": json_number(mean_abs_ic),
            # ``ic`` remains as a compatibility alias for formula-result consumers.
            "ic": json_number(signed_ic),
            "ic_ir": json_number(ic_ir(ic)),
            "gross_sharpe": json_number(tested.gross_sharpe),
            "net_sharpe": json_number(tested.net_sharpe),
            "max_drawdown": json_number(tested.max_drawdown),
            "turnover": json_number(tested.turnover),
            "avg_gross": json_number(tested.position.get("avg_gross")),
            "avg_positions": json_number(tested.position.get("avg_positions")),
            "max_position": json_number(tested.position.get("max_position")),
            "usable": tested.usable,
        },
        "returns": dated_returns,
        "normalized_equity": normalized_equity,
    }
