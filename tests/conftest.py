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
