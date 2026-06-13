"""Phase 0 acceptance tests (DEVELOPMENT_PLAN §6, Phase 0)."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd

from alphalineage.data.adjust import adjust
from alphalineage.data.cache import ParquetCache
from alphalineage.data.schema import normalize
from alphalineage.data.survivorship import survivorship_report
from alphalineage.data.universe import Membership, Universe, sample_universe


def test_cache_roundtrip(synthetic_prices):
    """First call fetches (mocked) and writes Parquet; second call serves from disk."""
    cache = ParquetCache()
    fetch = Mock(return_value=synthetic_prices)

    first = cache.get_or_fetch("AAPL", fetch)
    second = cache.get_or_fetch("AAPL", fetch)

    fetch.assert_called_once()  # second read never touches the provider
    pd.testing.assert_frame_equal(first, second)
    assert cache.has("AAPL")


def test_split_continuity():
    """Across a 2:1 split, the adjusted close has no discontinuity (raw does)."""
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    raw_close = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]  # economically flat
    split_factor = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0]  # 2:1 split on day index 3
    df = normalize(
        pd.DataFrame(
            {
                "open": raw_close,
                "high": raw_close,
                "low": raw_close,
                "close": raw_close,
                "volume": np.full(6, 1e6),
                "div_cash": np.zeros(6),
                "split_factor": split_factor,
            },
            index=idx,
        )
    )

    adj = adjust(df)
    adj_logret = np.log(adj["adj_close"]).diff().dropna()
    raw_logret = np.log(df["close"]).diff().dropna()

    # Raw close shows the artificial ~-0.69 split jump; adjusted close does not.
    assert abs(raw_logret.iloc[2]) > 0.6
    assert (adj_logret.abs() < 1e-9).all()


def test_universe_point_in_time():
    """A symbol that entered later is excluded when querying an earlier as-of date."""
    u = Universe(
        "t",
        [
            Membership("EARLY", pd.Timestamp("2018-01-01"), None),
            Membership("LATE", pd.Timestamp("2020-06-01"), None),
            Membership("GONE", pd.Timestamp("2015-01-01"), pd.Timestamp("2019-01-01")),
        ],
    )

    asof = u.members_asof("2019-06-01")

    assert "EARLY" in asof  # active before and after
    assert "LATE" not in asof  # entered after the query date
    assert "GONE" not in asof  # already delisted by the query date


def test_survivorship_report():
    """The report enumerates active vs delisted coverage and is non-empty."""
    universe = sample_universe("sp500-lite")
    report = survivorship_report(universe)

    assert report.strip()
    assert "active" in report.lower()
    assert "delisted" in report.lower()
    assert len(universe.delisted()) >= 1  # sample includes a delisted name
    # Every delisted symbol is named in the report.
    for member in universe.delisted():
        assert member.symbol in report
