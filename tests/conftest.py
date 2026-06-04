"""Shared fixtures. Every test runs against a temp data dir and never hits the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path, monkeypatch):
    """Redirect the local data store to a per-test temp directory."""
    data_dir = tmp_path / "data_cache"
    monkeypatch.setenv("ALPHAFORGE_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """A clean 30-day canonical-schema price frame with mild upward drift, no actions."""
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    close = 100.0 + np.arange(30) * 0.1
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(30, 1_000_000.0),
            "div_cash": np.zeros(30),
            "split_factor": np.ones(30),
        },
        index=idx,
    )


@pytest.fixture
def synthetic_panel():
    """A deterministic 60-day x 6-symbol panel with all operand fields populated."""
    from alphaforge.core.panel import Panel

    rng = np.random.default_rng(42)
    n_days, n_syms = 60, 6
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    symbols = [f"S{i}" for i in range(n_syms)]

    rets = rng.normal(0.0005, 0.02, size=(n_days, n_syms))
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=symbols)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.DataFrame(
        np.maximum(open_.to_numpy(), close.to_numpy()) * (1.0 + rng.uniform(0, 0.01, close.shape)),
        index=dates,
        columns=symbols,
    )
    low = pd.DataFrame(
        np.minimum(open_.to_numpy(), close.to_numpy()) * (1.0 - rng.uniform(0, 0.01, close.shape)),
        index=dates,
        columns=symbols,
    )
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, close.shape), index=dates, columns=symbols)
    return Panel.from_prices(open=open_, high=high, low=low, close=close, volume=volume)
