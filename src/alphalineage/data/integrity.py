"""Deterministic market-data integrity checks used before scoring or reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphalineage.data import schema
from alphalineage.data.adjust import adjust


@dataclass(frozen=True)
class PriceIntegrityIssue:
    code: str
    message: str
    date: str | None = None


class PriceIntegrityError(ValueError):
    """A symbol cannot safely enter a research panel."""

    def __init__(self, symbol: str, issues: list[PriceIntegrityIssue]) -> None:
        self.symbol = symbol
        self.issues = tuple(issues)
        details = "; ".join(
            f"{issue.code}{f' on {issue.date}' if issue.date else ''}: {issue.message}"
            for issue in issues
        )
        super().__init__(f"{symbol}: {details}")


def audit_price_frame(df: pd.DataFrame) -> list[PriceIntegrityIssue]:
    """Return blocking integrity issues for one canonical price frame.

    The audit deliberately avoids arbitrary global daily-return caps.  Large genuine moves
    remain legal; only impossible bars and unresolved split-date discontinuities are blocked.
    """
    schema.validate(df)
    issues: list[PriceIntegrityIssue] = []
    ohlc = df[["open", "high", "low", "close"]]
    values = ohlc.to_numpy(dtype="float64")
    observed = ~np.isnan(values)
    if np.any(observed & (~np.isfinite(values) | (values <= 0.0))):
        issues.append(
            PriceIntegrityIssue(
                "invalid_price",
                "OHLC observations must be finite and strictly positive",
            )
        )

    volume = df["volume"].to_numpy(dtype="float64")
    if np.any(~np.isnan(volume) & (~np.isfinite(volume) | (volume < 0.0))):
        issues.append(
            PriceIntegrityIssue(
                "invalid_volume",
                "volume observations must be finite and non-negative",
            )
        )

    complete = ohlc.notna().all(axis=1)
    invalid_bar = complete & (
        (df["high"] < df[["open", "low", "close"]].max(axis=1))
        | (df["low"] > df[["open", "high", "close"]].min(axis=1))
    )
    if invalid_bar.any():
        first = pd.Timestamp(invalid_bar.index[invalid_bar][0]).date().isoformat()
        issues.append(
            PriceIntegrityIssue(
                "invalid_ohlc",
                "high/low do not contain the open and close",
                first,
            )
        )

    try:
        adjusted = adjust(df)
    except schema.PriceBasisError as exc:
        issues.append(PriceIntegrityIssue("mixed_price_basis", str(exc)))
        return issues

    adjusted_close = adjusted["adj_close"]
    adjusted_return = adjusted_close.pct_change(fill_method=None)
    split_events = df["split_factor"].notna() & ~np.isclose(
        df["split_factor"].fillna(1.0), 1.0
    )
    # A 50% residual move precisely on an advertised action is overwhelmingly likely to
    # be an unhandled/double-applied split.  Ordinary large moves away from action dates
    # are not censored.
    residual = split_events & adjusted_return.abs().gt(0.5)
    if residual.any():
        first_index = residual.index[residual][0]
        value = float(adjusted_return.loc[first_index])
        issues.append(
            PriceIntegrityIssue(
                "split_discontinuity",
                f"adjusted close still jumps {value:+.1%} across a split event",
                pd.Timestamp(first_index).date().isoformat(),
            )
        )

    if adjusted_close.notna().any():
        finite = adjusted_close.dropna().map(math.isfinite)
        if not bool(finite.all()):
            issues.append(
                PriceIntegrityIssue(
                    "nonfinite_adjusted_price",
                    "corporate-action adjustment produced a nonfinite close",
                )
            )
    return issues


def require_price_integrity(symbol: str, df: pd.DataFrame) -> None:
    issues = audit_price_frame(df)
    if issues:
        raise PriceIntegrityError(symbol, issues)
