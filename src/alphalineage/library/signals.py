"""Signals: what a discovered factor actually says about today's universe.

Everything else in this project measures a factor. This module *uses* one: it evaluates the
frozen expression on the most recent cached prices and reports the cross-sectional ranking it
produces, plus the raw number behind each rank.

Three things make that honest rather than merely useful:

* **A stored formula already carries its direction.** ``validation.selection`` orients every
  trial by its training-window sign before a round is written, so the saved tree is the version
  whose *higher* values predicted *higher* forward returns. Rank 1 is therefore the largest
  value, with no per-view sign convention to get wrong.
* **A snapshot names the bar it was computed from and the bar it could be traded at.** A signal
  from the close of day *t* is tradable at the session's execution timing, never at the close
  that produced it - the same ``execution`` that defined the label the factor was scored on.
* **Missing is reported, not filled.** A symbol whose cached prices stop before the snapshot
  date is excluded and listed as stale with its own last date; a symbol that simply lacks the
  history the formula's windows need is excluded as warming up. Neither is quietly ranked on
  an older number, because a stale rank is indistinguishable from a current one on screen.

Not investment advice. Research output only (invariant 8).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphalineage.core.fitness import execution_delay
from alphalineage.core.panel import Panel

SIGNALS_VERSION = 1
#: Extra bars kept beyond a formula's effective lookback when trimming history for a snapshot.
LOOKBACK_MARGIN = 40
#: Bars of history kept when a formula's lookback is unbounded (expanding windows) or unknown.
UNBOUNDED_HISTORY_BARS = 2_000


@dataclass(frozen=True)
class SignalRow:
    """One ranked symbol in a snapshot."""

    symbol: str
    rank: int
    percentile: float
    value: float
    previous_rank: int | None
    rank_change: int | None
    close: float | None
    day_change: float | None
    last_price_date: str | None


@dataclass(frozen=True)
class ExcludedSymbol:
    """A universe member that could not be ranked on the snapshot date, and why."""

    symbol: str
    reason: str
    detail: str
    last_price_date: str | None
    stale_bars: int | None


def _finite(value: Any) -> float | None:
    """Coerce to a JSON-safe float, turning NaN/inf into ``None`` rather than invalid JSON."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def history_bars_needed(lookback: int, *, unbounded: bool) -> int:
    """How many trailing bars a snapshot needs to reproduce the full-history value.

    Trimming is what makes a snapshot interactive: evaluating twenty years to read one date
    costs seconds per refresh. An unbounded formula (expanding mean, cumulative sums) has no
    such bound, so it keeps a large fixed window and the caller says the value is approximate.
    """
    if unbounded:
        return UNBOUNDED_HISTORY_BARS
    return max(int(lookback), 0) + LOOKBACK_MARGIN


def trim_panel(panel: Panel, bars: int) -> Panel:
    """The last ``bars`` rows of a panel, or the panel itself when it is already shorter."""
    dates = pd.DatetimeIndex(panel.dates)
    if bars <= 0 or len(dates) <= bars:
        return panel
    keep = dates >= dates[len(dates) - bars]
    return Panel({name: frame.loc[keep] for name, frame in panel.fields.items()})


def _latest_dense_date(factor: pd.DataFrame, min_names: int) -> pd.Timestamp | None:
    """The most recent date whose cross-section is wide enough to rank.

    A single symbol reporting one day early would otherwise define a "snapshot" of one name.
    """
    counts = factor.notna().sum(axis=1)
    eligible = counts.index[counts >= max(int(min_names), 1)]
    return pd.Timestamp(eligible[-1]) if len(eligible) else None


def _ranks_on(values: pd.Series) -> pd.Series:
    """Dense competition ranks with 1 as the largest value; ties share the better rank."""
    return values.rank(ascending=False, method="min").astype(int)


def snapshot(
    factor: pd.DataFrame,
    panel: Panel,
    *,
    min_names: int = 5,
    execution: str = "close",
    approximate: bool = False,
    snapshot_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Rank a universe on the latest date the factor covers, and say what is missing.

    ``factor`` is the evaluated expression over ``panel``; both are indexed by date. The result
    is JSON-safe and self-describing: the caller can render it without re-deriving anything.
    """
    close = panel["close"]
    dates = pd.DatetimeIndex(factor.index)
    as_of = snapshot_date if snapshot_date is not None else _latest_dense_date(factor, min_names)
    if as_of is None:
        raise ValueError(
            f"no date has at least {min_names} symbols with a value; "
            "the formula may need more history than the cache holds"
        )

    earlier = dates[dates < as_of]
    previous = _latest_dense_date(factor.loc[earlier], min_names) if len(earlier) else None

    current = factor.reindex([as_of]).iloc[0].dropna()
    current = current[np.isfinite(current.to_numpy(dtype="float64"))]
    ranks = _ranks_on(current)
    # Percentile of the *ranking*, so a two-name cross-section still reads 100/0 rather than
    # collapsing on a divide-by-zero.
    percentiles = current.rank(ascending=True, pct=True) * 100.0
    previous_ranks: pd.Series | None = None
    if previous is not None:
        prior = factor.loc[previous].dropna()
        prior = prior[np.isfinite(prior.to_numpy(dtype="float64"))]
        if len(prior):
            previous_ranks = _ranks_on(prior)

    last_price_dates: dict[str, pd.Timestamp | None] = {}
    for symbol in panel.symbols:
        series = close[symbol].dropna() if symbol in close.columns else pd.Series(dtype="float64")
        last_price_dates[symbol] = pd.Timestamp(series.index[-1]) if len(series) else None

    day_changes = close.pct_change(fill_method=None)
    rows: list[SignalRow] = []
    for symbol, value in current.items():
        name = str(symbol)
        rank = int(ranks[symbol])
        prior_rank = (
            int(previous_ranks[symbol])
            if previous_ranks is not None and symbol in previous_ranks.index
            else None
        )
        last_seen = last_price_dates.get(name)
        rows.append(
            SignalRow(
                symbol=name,
                rank=rank,
                percentile=round(float(percentiles[symbol]), 2),
                value=float(value),
                previous_rank=prior_rank,
                # Positive means the symbol climbed: a move from rank 9 to rank 4 is +5.
                rank_change=None if prior_rank is None else prior_rank - rank,
                close=_finite(close.at[as_of, symbol]) if symbol in close.columns else None,
                day_change=(
                    _finite(day_changes.at[as_of, symbol]) if symbol in close.columns else None
                ),
                last_price_date=None if last_seen is None else last_seen.date().isoformat(),
            )
        )
    rows.sort(key=lambda row: (row.rank, row.symbol))

    panel_end = pd.Timestamp(dates[-1]) if len(dates) else as_of
    excluded: list[ExcludedSymbol] = []
    for symbol in panel.symbols:
        if symbol in current.index:
            continue
        last_seen = last_price_dates.get(symbol)
        has_price_today = (
            symbol in close.columns
            and as_of in close.index
            and bool(pd.notna(close.at[as_of, symbol]))
        )
        if last_seen is None:
            reason, detail, stale = "no_prices", "no cached prices for this symbol", None
        elif has_price_today:
            reason = "warming_up"
            detail = "priced on this date, but the formula has no value yet"
            stale = 0
        else:
            reason = "stale_prices"
            stale = int((dates > last_seen).sum())
            detail = f"cached prices stop {stale} trading day(s) before the snapshot"
        excluded.append(
            ExcludedSymbol(
                symbol=str(symbol),
                reason=reason,
                detail=detail,
                last_price_date=None if last_seen is None else last_seen.date().isoformat(),
                stale_bars=stale,
            )
        )
    excluded.sort(key=lambda item: (item.reason, item.symbol))

    return {
        "signals_version": SIGNALS_VERSION,
        "as_of": as_of.date().isoformat(),
        "previous_as_of": None if previous is None else previous.date().isoformat(),
        "panel_end": panel_end.date().isoformat(),
        # A snapshot one or more bars behind the cache's own last date means the formula could
        # not be computed on the newest rows; saying so beats a date the reader must compare.
        "bars_behind_panel": int((dates > as_of).sum()),
        "execution": execution,
        "execution_delay": execution_delay(execution),
        "ranked_count": len(rows),
        "excluded_count": len(excluded),
        "min_names": int(min_names),
        "approximate": bool(approximate),
        "rows": [asdict(row) for row in rows],
        "excluded": [asdict(item) for item in excluded],
    }
