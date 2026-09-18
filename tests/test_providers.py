"""Tests for the Tiingo / yfinance providers and the fallback orchestration."""

from __future__ import annotations

import pandas as pd
import pytest

from alphalineage.data.provider import FallbackProvider, ProviderError
from alphalineage.data.schema import PRICE_COLUMNS
from alphalineage.data.tiingo_client import TiingoError, TiingoProvider
from alphalineage.data.yfinance_provider import YFinanceProvider


def test_tiingo_provider_parses_payload(requests_mock):
    payload = [
        {
            "date": "2020-01-02T00:00:00.000Z",
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100,
            "divCash": 0.0,
            "splitFactor": 1.0,
        }
    ]
    requests_mock.get("https://api.tiingo.com/tiingo/daily/AAPL/prices", json=payload)

    provider = TiingoProvider(api_key="k")
    frame = provider.get_prices("AAPL")

    assert list(frame.columns) == PRICE_COLUMNS
    assert frame.index.tz is None
    assert frame["close"].iloc[0] == 1.5
    assert frame["div_cash"].iloc[0] == 0.0


def test_tiingo_retries_on_429_then_succeeds(requests_mock):
    requests_mock.get(
        "https://api.tiingo.com/tiingo/daily/AAPL/prices",
        [
            {"status_code": 429, "headers": {"Retry-After": "0"}, "json": {}},
            {
                "status_code": 200,
                "json": [
                    {"date": "2020-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                ],
            },
        ],
    )
    provider = TiingoProvider(api_key="k", sleep=lambda _s: None)
    frame = provider.get_prices("AAPL")
    assert len(frame) == 1


def test_tiingo_requires_key():
    provider = TiingoProvider(api_key="")
    with pytest.raises(TiingoError):
        provider.get_prices("AAPL")


def test_yfinance_maps_splits_and_dividends():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0],
            "High": [1.0, 1.0, 1.0],
            "Low": [1.0, 1.0, 1.0],
            "Close": [1.0, 1.0, 1.0],
            "Volume": [10, 10, 10],
            "Dividends": [0.0, 0.5, 0.0],
            "Stock Splits": [0.0, 0.0, 2.0],
        },
        index=idx,
    )
    provider = YFinanceProvider(download=lambda *_a: raw, warn=False)
    frame = provider.get_prices("X")

    assert frame["split_factor"].tolist() == [1.0, 1.0, 2.0]  # 0.0 -> 1.0
    assert frame["div_cash"].tolist() == [0.0, 0.5, 0.0]


def test_fallback_uses_second_provider_on_failure(synthetic_prices):
    class _Boom:
        name = "tiingo"

        def get_prices(self, *_a, **_k):
            raise RuntimeError("no key")

    class _Ok:
        name = "yfinance"

        def get_prices(self, *_a, **_k):
            return synthetic_prices

    fallback = FallbackProvider([_Boom(), _Ok()])
    frame = fallback.get_prices("AAPL")

    assert len(frame) == len(synthetic_prices)
    assert fallback.sources["AAPL"] == "yfinance"


def test_fallback_raises_when_all_fail():
    class _Boom:
        name = "x"

        def get_prices(self, *_a, **_k):
            raise RuntimeError("down")

    with pytest.raises(ProviderError):
        FallbackProvider([_Boom()]).get_prices("AAPL")


# --- quota handling ------------------------------------------------------------------------
_MONTHLY_TEXT = (
    "You have run over your 500 symbol look up for this month. Please upgrade at "
    "https://api.tiingo.com/pricing to have your limits increased."
)


def test_tiingo_monthly_symbol_quota_plain_text_200_is_not_retried(requests_mock):
    from alphalineage.data.provider import QuotaExceededError

    adapter = requests_mock.get(
        "https://api.tiingo.com/tiingo/daily/AAPL/prices",
        status_code=200,
        text=_MONTHLY_TEXT,
    )
    provider = TiingoProvider(api_key="k", sleep=lambda _s: None)
    with pytest.raises(QuotaExceededError) as caught:
        provider.get_prices("AAPL")

    # A quota rejection must stop immediately: every retry would spend more of the allowance.
    assert adapter.call_count == 1
    assert isinstance(caught.value, TiingoError)
    assert caught.value.scope == "monthly"
    assert "k" not in str(caught.value).split()


def test_tiingo_hourly_429_without_short_retry_after_raises_quota(requests_mock):
    from alphalineage.data.provider import QuotaExceededError

    adapter = requests_mock.get(
        "https://api.tiingo.com/tiingo/daily/AAPL/prices",
        status_code=429,
        json={"detail": "Error: You have run over your hourly request allocation."},
    )
    provider = TiingoProvider(api_key="k", sleep=lambda _s: None)
    with pytest.raises(QuotaExceededError) as caught:
        provider.get_prices("AAPL")
    assert adapter.call_count == 1
    assert caught.value.scope == "hourly"


def test_tiingo_non_json_body_is_a_readable_non_retried_error(requests_mock):
    adapter = requests_mock.get(
        "https://api.tiingo.com/tiingo/daily/AAPL/prices",
        status_code=200,
        text="<html>gateway hiccup</html>",
    )
    provider = TiingoProvider(api_key="k", sleep=lambda _s: None)
    with pytest.raises(TiingoError, match="non-JSON"):
        provider.get_prices("AAPL")
    assert adapter.call_count == 1


def test_fallback_does_not_mask_a_quota_stop_with_another_provider(synthetic_prices):
    from alphalineage.data.provider import QuotaExceededError

    class _Quota:
        name = "tiingo"

        def get_prices(self, *_a, **_k):
            raise QuotaExceededError("hourly allowance used", scope="hourly")

    class _Ok:
        name = "yfinance"

        def get_prices(self, *_a, **_k):
            return synthetic_prices

    fallback = FallbackProvider([_Quota(), _Ok()])
    with pytest.raises(QuotaExceededError):
        fallback.get_prices("AAPL")
    assert "AAPL" not in fallback.sources
