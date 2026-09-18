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
    horizon: int = 1,
    execution: str = "close",
) -> dict[str, Any]:
    """Compute the canonical metric/time-series payload over ``dates``.

    Rolling state, weights, and costs are built on the complete panel and only reporting is
    sliced to ``dates``.  This preserves warm-up and boundary turnover semantics.
    """
    report_dates = pd.DatetimeIndex(dates)
    tested = backtest(
        factor,
        panel,
        forward,
        scheme,
        costs,
        dates=report_dates,
        horizon=horizon,
        execution=execution,
    )
    ic = daily_ic(factor, forward, ic_method, min_names=min_names).reindex(report_dates)
    clean_ic = ic.dropna()
    signed_ic = clean_ic.mean() if len(clean_ic) else float("nan")
    mean_abs_ic = clean_ic.abs().mean() if len(clean_ic) else float("nan")

    gross, net = tested.gross_returns.align(tested.net_returns, join="outer")
    realization_dates = tested.realization_dates.reindex(gross.index)
    active_exposure = tested.active_exposure.reindex(gross.index).fillna(False)
    dated_returns = []
    for signal_date in gross.index:
        realization_date = realization_dates.loc[signal_date]
        if pd.isna(realization_date):
            continue
        dated_returns.append(
            {
                "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                "date": pd.Timestamp(realization_date).date().isoformat(),
                "gross": json_number(gross.loc[signal_date]),
                "net": json_number(net.loc[signal_date]),
                "active": bool(active_exposure.loc[signal_date]),
            }
        )

    normalized_equity: list[dict[str, Any]] = []
    equity_terminated_reason: str | None = None
    if dated_returns:
        # The explicit pre-return baseline prevents the first realized return from disappearing
        # when the chart converts levels to cumulative percentage performance.
        normalized_equity.append({"date": dated_returns[0]["signal_date"], "value": 1.0})
        level = 1.0
        # An all-cash path is not an equity curve. Keep only the explicit
        # pre-return baseline and let portfolio_health explain why no evidence exists.
        equity_points = (
            dated_returns
            if int(tested.portfolio_health["active_observations"]) > 0
            else []
        )
        for point in equity_points:
            daily_net = point["net"]
            if daily_net is None:
                equity_terminated_reason = "missing_realized_return"
                break
            if daily_net <= -1.0:
                # Equity cannot become negative and then resume compounding as if the
                # strategy had received fresh capital.  Pin the terminal insolvency level.
                level = 0.0
                normalized_equity.append({"date": point["date"], "value": level})
                equity_terminated_reason = "insolvent"
                break
            level *= 1.0 + daily_net
            normalized_equity.append({"date": point["date"], "value": json_number(level)})
    return {
        "start": (
            normalized_equity[0]["date"] if normalized_equity else None
        ),
        "end": normalized_equity[-1]["date"] if normalized_equity else None,
        # Compatibility field now means observations with actual portfolio
        # exposure. Calendar/cash observations remain explicit below.
        "observations": int((net.notna() & active_exposure).sum()),
        "calendar_observations": int(realization_dates.notna().sum()),
        "active_observations": int(tested.portfolio_health["active_observations"]),
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
            "insolvent": tested.insolvent,
            "missing_return_observations": tested.missing_return_observations,
            "exposure_coverage": json_number(
                tested.portfolio_health["exposure_coverage"]
            ),
            "two_sided_coverage": json_number(
                tested.portfolio_health["two_sided_coverage"]
            ),
        },
        "returns": dated_returns,
        "normalized_equity": normalized_equity,
        "integrity": {
            "valid": not tested.integrity_issues,
            "issues": list(tested.integrity_issues),
            "equity_terminated_reason": equity_terminated_reason,
        },
        "portfolio_health": dict(tested.portfolio_health),
    }
