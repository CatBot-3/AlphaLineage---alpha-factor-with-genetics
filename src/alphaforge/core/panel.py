"""Panel: the aligned cross-sectional price data the evaluator runs over.

A panel holds one wide DataFrame per field (``index = dates``, ``columns = symbols``),
all sharing the same index and columns. ``Panel.from_cache`` assembles it from the
Phase 0 Parquet cache: it loads each symbol, back-adjusts it for corporate actions
(reusing :func:`alphaforge.data.adjust.adjust`), and derives the operand fields the DSL
exposes. The GP loop reads panels — never the data API (invariant: fetch once, cache).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.core.primitives import OPERAND_FIELDS
from alphaforge.data.adjust import adjust
from alphaforge.data.cache import ParquetCache

# Operand field -> adjusted source column produced by adjust().
_PRICE_SOURCE = {
    "open": "adj_open",
    "high": "adj_high",
    "low": "adj_low",
    "close": "adj_close",
    "volume": "adj_volume",
}


class Panel:
    """Aligned wide frames keyed by operand field name."""

    def __init__(self, fields: dict[str, pd.DataFrame]) -> None:
        missing = set(OPERAND_FIELDS) - set(fields)
        if missing:
            raise ValueError(f"panel is missing required fields: {sorted(missing)}")
        ref = fields["close"]
        for name, frame in fields.items():
            if not frame.index.equals(ref.index) or not frame.columns.equals(ref.columns):
                raise ValueError(f"panel field {name!r} is not aligned with 'close'")
        self.fields = fields

    def __getitem__(self, field: str) -> pd.DataFrame:
        return self.fields[field]

    def __contains__(self, field: str) -> bool:
        return field in self.fields

    @property
    def dates(self) -> pd.Index:
        return self.fields["close"].index

    @property
    def symbols(self) -> pd.Index:
        return self.fields["close"].columns

    # --- constructors ------------------------------------------------------------
    @classmethod
    def from_prices(
        cls,
        *,
        open: pd.DataFrame,  # noqa: A002 - matches the OHLCV field name
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> Panel:
        """Build a panel from aligned wide OHLCV frames, deriving vwap and returns."""
        vwap = (high + low + close) / 3.0
        returns = close.pct_change(fill_method=None)
        return cls(
            {
                "open": open,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "vwap": vwap,
                "returns": returns,
            }
        )

    @classmethod
    def from_cache(
        cls,
        symbols: list[str],
        *,
        cache: ParquetCache | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> Panel:
        """Assemble a panel from cached, corporate-action-adjusted symbols."""
        store = cache or ParquetCache()
        adjusted: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if not store.has(symbol):
                continue
            adjusted[symbol] = adjust(store.load(symbol))
        if not adjusted:
            raise ValueError("no cached symbols found to build a panel")

        cols = sorted(adjusted)
        wide = {
            field: pd.DataFrame({s: adjusted[s][src] for s in cols}).sort_index()
            for field, src in _PRICE_SOURCE.items()
        }
        panel = cls.from_prices(
            open=wide["open"],
            high=wide["high"],
            low=wide["low"],
            close=wide["close"],
            volume=wide["volume"],
        )
        if start is not None or end is not None:
            dates = panel.dates
            keep = np.ones(len(dates), dtype=bool)
            if start is not None:
                keep &= np.asarray(dates >= pd.Timestamp(start))
            if end is not None:
                keep &= np.asarray(dates <= pd.Timestamp(end))
            panel = cls({f: df.loc[keep] for f, df in panel.fields.items()})
        return panel
