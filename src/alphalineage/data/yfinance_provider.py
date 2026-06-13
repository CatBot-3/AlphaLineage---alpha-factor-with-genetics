"""yfinance fallback provider.

Zero-key prototype source. yfinance data is **survivorship-biased** and of variable
quality, so this provider emits a warning (invariant 7) and should only back-stop
Tiingo for prototyping. The raw download is wrapped in an injectable callable so
tests never hit the network.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pandas as pd

from alphalineage.data import schema

#: yfinance column -> canonical schema column. "Stock Splits" / "Dividends" come from
#: ``history(actions=True)``; "Stock Splits" is 0.0 on non-split days (mapped to 1.0).
_FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Dividends": "div_cash",
}

DownloadFn = Callable[[str, str | None, str | None], pd.DataFrame]


def _default_download(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # local import: heavy, and tests inject a fake instead

    ticker = yf.Ticker(symbol)
    return ticker.history(start=start, end=end, auto_adjust=False, actions=True)


class YFinanceProvider:
    """Fetch normalized daily price frames from yfinance."""

    name = "yfinance"

    def __init__(self, *, download: DownloadFn = _default_download, warn: bool = True) -> None:
        self._download = download
        self._warn = warn

    def get_prices(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if self._warn:
            warnings.warn(
                "yfinance data is survivorship-biased and of variable quality; "
                "treat it as prototype-grade only.",
                stacklevel=2,
            )
        raw = self._download(symbol, start, end)
        return _to_frame(raw)


def _to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        empty = pd.DataFrame(columns=schema.PRICE_COLUMNS)
        empty.index = pd.DatetimeIndex([], name=schema.INDEX_NAME)
        return schema.validate(empty.astype("float64"))

    mapped = pd.DataFrame(index=raw.index)
    for src, dst in _FIELD_MAP.items():
        if src in raw.columns:
            mapped[dst] = raw[src]

    # yfinance "Stock Splits": ratio on a split day (e.g. 2.0 for a 2:1), 0.0 otherwise.
    # Our convention: split_factor == 1.0 means "no split".
    if "Stock Splits" in raw.columns:
        splits = pd.to_numeric(raw["Stock Splits"], errors="coerce").fillna(0.0)
        mapped["split_factor"] = splits.where(splits > 0.0, 1.0)

    return schema.normalize(mapped)
