"""Tests for corporate-action adjustment beyond the split-continuity acceptance test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data.adjust import adjust, adjusted_close
from alphaforge.data.schema import normalize


def _frame(close, *, div_cash=None, split_factor=None):
    n = len(close)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return normalize(
        pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.full(n, 1e6),
                "div_cash": np.zeros(n) if div_cash is None else div_cash,
                "split_factor": np.ones(n) if split_factor is None else split_factor,
            },
            index=idx,
        )
    )


def test_dividend_continuity():
    # $10 dividend on day 2; ex-div close drops 100 -> 90 (total return flat).
    df = _frame([100.0, 100.0, 90.0, 90.0], div_cash=[0.0, 0.0, 10.0, 0.0])
    adj_logret = np.log(adjusted_close(df)).diff().dropna()
    raw_logret = np.log(df["close"]).diff().dropna()

    assert abs(raw_logret.iloc[1]) > 0.1  # raw shows the ex-div drop
    assert (adj_logret.abs() < 1e-9).all()  # adjusted is continuous


def test_no_actions_leaves_close_unchanged():
    df = _frame([10.0, 11.0, 12.0, 11.5])
    assert np.allclose(adjusted_close(df).to_numpy(), df["close"].to_numpy())


def test_volume_adjusted_for_split():
    # 2:1 split on day 2: pre-split volume should be scaled up (more shares).
    df = _frame([100.0, 50.0, 50.0], split_factor=[1.0, 2.0, 1.0])
    adj = adjust(df)
    assert adj["adj_volume"].iloc[0] > df["volume"].iloc[0]
    assert np.isclose(adj["adj_volume"].iloc[0], df["volume"].iloc[0] * 2.0)
    assert np.isclose(adj["adj_volume"].iloc[-1], df["volume"].iloc[-1])
