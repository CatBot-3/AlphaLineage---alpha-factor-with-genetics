"""Phase 0 acceptance tests: the data layer's point-in-time and adjustment guarantees."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from alphalineage.data.adjust import adjust, adjusted_close, split_adjusted_close
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


def test_split_adjusted_close_excludes_dividends_but_normalizes_splits():
    """Price-return levels keep ex-dividend drops while removing split discontinuities."""
    idx = pd.date_range("2025-01-02", periods=4, freq="B")
    raw = normalize(
        pd.DataFrame(
            {
                "open": [100.0, 90.0, 45.0, 47.0],
                "high": [100.0, 90.0, 45.0, 47.0],
                "low": [100.0, 90.0, 45.0, 47.0],
                "close": [100.0, 90.0, 45.0, 47.0],
                "volume": np.full(4, 1e6),
                "div_cash": [0.0, 10.0, 0.0, 0.0],
                "split_factor": [1.0, 1.0, 2.0, 1.0],
            },
            index=idx,
        )
    )

    price_levels = split_adjusted_close(raw)
    total_return_levels = adjusted_close(raw)

    assert price_levels.tolist() == pytest.approx([50.0, 45.0, 45.0, 47.0])
    assert total_return_levels.tolist() == pytest.approx([45.0, 45.0, 45.0, 47.0])
    assert price_levels.iloc[1] / price_levels.iloc[0] - 1.0 == pytest.approx(-0.1)
    assert price_levels.iloc[2] / price_levels.iloc[1] - 1.0 == pytest.approx(0.0)


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


@pytest.mark.parametrize(
    "name",
    [" spaced ", "../outside", "..\\outside", "nested/universe", "NUL", ""],
)
def test_universe_rejects_unsafe_or_noncanonical_names(name):
    with pytest.raises(ValueError):
        Universe(name, [Membership("AAA", pd.Timestamp("2020-01-01"))])


def test_universe_normalizes_aware_membership_and_query_dates_to_utc_naive():
    universe = Universe(
        "timezone-safe",
        [
            Membership(
                "AAA",
                pd.Timestamp("2020-01-01T23:30:00-05:00"),  # 2020-01-02 UTC
                pd.Timestamp("2020-01-04T00:30:00+02:00"),  # 2020-01-03 UTC
            )
        ],
    )
    membership = universe.memberships[0]
    assert membership.entry == pd.Timestamp("2020-01-02")
    assert membership.exit == pd.Timestamp("2020-01-03")
    assert membership.entry.tzinfo is None
    assert membership.exit is not None and membership.exit.tzinfo is None

    assert universe.members_asof("2020-01-01T20:00:00-05:00") == ["AAA"]
    assert universe.members_asof("2020-01-02T20:00:00-05:00") == []

    aware_dates = pd.DatetimeIndex(["2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"])
    mask = universe.membership_mask(aware_dates, ["AAA"])
    assert mask.index.equals(aware_dates)
    assert mask["AAA"].tolist() == [True, False]


def test_survivorship_report():
    """The sample distinguishes membership evidence from security-status claims."""
    universe = sample_universe("sp500-lite")
    report = survivorship_report(universe)

    assert report.strip()
    assert "open membership intervals" in report.lower()
    assert "declared membership exits" in report.lower()
    assert "does not establish that the security was delisted" in report.lower()
    assert universe.mode == "static_snapshot"
    assert len(universe.active()) == 15
    assert universe.exited() == []
    assert "LEH" not in universe.all_symbols()
    assert {"AAPL", "MSFT", "NVDA", "BRK-B", "COST"} <= set(universe.all_symbols())
    assert "likely survivorship-biased" in report
