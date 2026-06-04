"""Tests for the cache->panel loader."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.core.panel import Panel
from alphaforge.data.cache import ParquetCache
from alphaforge.data.schema import normalize


def _price_frame(dates: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    n = len(dates)
    close = base + np.arange(n) * 0.1
    return normalize(
        pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1e6),
                "div_cash": np.zeros(n),
                "split_factor": np.ones(n),
            },
            index=dates,
        )
    )


def test_panel_from_cache_aligns_symbols_and_derives_fields():
    cache = ParquetCache()
    cache.store("AAA", _price_frame(pd.date_range("2020-01-01", periods=10, freq="B"), 100.0))
    cache.store("BBB", _price_frame(pd.date_range("2020-01-03", periods=10, freq="B"), 50.0))

    panel = Panel.from_cache(["AAA", "BBB"])

    assert list(panel.symbols) == ["AAA", "BBB"]  # sorted
    for field in ("open", "high", "low", "close", "volume", "vwap", "returns"):
        assert field in panel
        assert panel[field].shape == (len(panel.dates), 2)

    # Derived fields follow their definitions.
    pd.testing.assert_frame_equal(
        panel["vwap"], (panel["high"] + panel["low"] + panel["close"]) / 3.0
    )
    assert panel["returns"].iloc[0].isna().all()


def test_panel_skips_uncached_symbols():
    cache = ParquetCache()
    cache.store("AAA", _price_frame(pd.date_range("2020-01-01", periods=5, freq="B"), 100.0))

    panel = Panel.from_cache(["AAA", "MISSING"])
    assert list(panel.symbols) == ["AAA"]
