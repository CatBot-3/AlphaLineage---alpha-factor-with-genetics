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


@pytest.fixture
def signal_panel():
    """A panel whose forward returns are driven by a known latent signal carried in volume.

    Returns ``(panel, injected)`` where ``injected`` is the cross-sectional z-score of the
    latent. ``volume = exp(0.3*latent)`` is a strictly monotone (per-date) carrier, so any
    volume-based factor has |rank IC| ~ 1.0 against ``injected`` — the GP should recover it.
    Forward returns are ``beta*injected[t] + noise``, so volume also predicts returns.
    """
    from alphaforge.core.panel import Panel

    rng = np.random.default_rng(11)
    n_days, n_syms = 180, 12
    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")
    symbols = [f"S{i}" for i in range(n_syms)]

    latent = rng.normal(0.0, 1.0, (n_days, n_syms))
    lat = pd.DataFrame(latent, index=dates, columns=symbols)
    injected = lat.sub(lat.mean(axis=1), axis=0).div(lat.std(axis=1).replace(0.0, np.nan), axis=0)

    beta, noise_sd = 0.02, 0.02
    noise = rng.normal(0.0, noise_sd, (n_days, n_syms))
    ret = beta * injected.shift(1).fillna(0.0).to_numpy() + noise
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + ret, axis=0), index=dates, columns=symbols)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.DataFrame(
        np.maximum(open_.to_numpy(), close.to_numpy()) * (1.0 + rng.uniform(0, 0.005, close.shape)),
        index=dates,
        columns=symbols,
    )
    low = pd.DataFrame(
        np.minimum(open_.to_numpy(), close.to_numpy()) * (1.0 - rng.uniform(0, 0.005, close.shape)),
        index=dates,
        columns=symbols,
    )
    volume = pd.DataFrame(1e6 * np.exp(0.3 * latent), index=dates, columns=symbols)
    panel = Panel.from_prices(open=open_, high=high, low=low, close=close, volume=volume)
    return panel, injected


@pytest.fixture
def noise_panel():
    """A pure-noise panel: random returns and independent random volume, no signal at all."""
    from alphaforge.core.panel import Panel

    rng = np.random.default_rng(123)
    n_days, n_syms = 160, 12
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    symbols = [f"S{i}" for i in range(n_syms)]

    ret = rng.normal(0.0, 0.02, (n_days, n_syms))
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + ret, axis=0), index=dates, columns=symbols)
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
