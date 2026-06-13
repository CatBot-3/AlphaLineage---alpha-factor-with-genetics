"""P0-T1 - Tiingo EOD price provider with retry/backoff and rate-limit awareness.

Uses the daily prices endpoint (``/tiingo/daily/{ticker}/prices``), which returns raw
OHLCV plus ``divCash`` and ``splitFactor`` per day - exactly the corporate-action
inputs the adjuster needs. The HTTP session is injectable so tests mock it.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

from alphalineage.data import schema
from alphalineage.data.retry import RateLimiter, with_retry

_BASE_URL = "https://api.tiingo.com"

#: Tiingo daily field -> canonical schema column.
_FIELD_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "divCash": "div_cash",
    "splitFactor": "split_factor",
}


class TiingoError(RuntimeError):
    """A non-retryable Tiingo API or transport error."""


class RateLimitError(TiingoError):
    """HTTP 429 from Tiingo; carries the server's Retry-After (seconds) if present."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("Tiingo rate limit exceeded (HTTP 429)")
        self.retry_after = retry_after


def _parse_retry_after(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _tiingo_retry_after(exc: BaseException) -> float | None:
    return exc.retry_after if isinstance(exc, RateLimitError) else None


class TiingoProvider:
    """Fetch normalized daily price frames from Tiingo."""

    name = "tiingo"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: requests.Session | None = None,
        base_url: str = _BASE_URL,
        rate_limiter: RateLimiter | None = None,
        sleep: object = None,
        max_attempts: int = 5,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("TIINGO_API_KEY")
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.max_attempts = max_attempts
        # ``sleep`` is threaded into with_retry so tests avoid real delays.
        self._sleep = sleep

    def get_prices(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if not self.api_key:
            raise TiingoError("TIINGO_API_KEY is not set")

        url = f"{self.base_url}/tiingo/daily/{symbol}/prices"
        params: dict[str, str] = {"format": "json"}
        if start:
            params["startDate"] = start
        if end:
            params["endDate"] = end
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

        def _request() -> list[dict[str, object]]:
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 429:
                raise RateLimitError(retry_after=_parse_retry_after(response))
            if response.status_code >= 400:
                raise TiingoError(f"Tiingo HTTP {response.status_code}: {response.text[:200]}")
            payload = response.json()
            if not isinstance(payload, list):
                raise TiingoError(f"unexpected Tiingo payload for {symbol!r}")
            return payload

        retry_kwargs: dict[str, object] = {
            "max_attempts": self.max_attempts,
            "retry_on": (RateLimitError, requests.RequestException),
            "get_retry_after": _tiingo_retry_after,
        }
        if self._sleep is not None:
            retry_kwargs["sleep"] = self._sleep
        rows = with_retry(_request, **retry_kwargs)  # type: ignore[arg-type]
        return _to_frame(rows)


def _to_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        # Empty but schema-valid frame.
        empty = pd.DataFrame(columns=schema.PRICE_COLUMNS)
        empty.index = pd.DatetimeIndex([], name=schema.INDEX_NAME)
        return schema.validate(empty.astype("float64"))

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns:
        raise TiingoError("Tiingo payload missing 'date' field")
    frame = frame.set_index("date")
    renamed = frame.rename(columns=_FIELD_MAP)
    keep = [c for c in _FIELD_MAP.values() if c in renamed.columns]
    return schema.normalize(renamed[keep])
