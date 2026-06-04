"""P0-T2 — Parquet local cache.

Fetch a symbol once, persist it to ``data_cache/prices/{SYMBOL}.parquet``, and serve
every subsequent read from disk. This enforces the invariant that the data API is
never called inside the GP loop: the loop only ever reads the cache.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from alphaforge.data import paths, schema

FetchFn = Callable[[str], pd.DataFrame]


class ParquetCache:
    """A directory of per-symbol Parquet price frames."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        # Resolved lazily so an ALPHAFORGE_DATA_DIR override set after construction
        # (as tests do) is still honored.
        return self._root if self._root is not None else paths.prices_dir()

    def path_for(self, symbol: str) -> Path:
        return self.root / f"{symbol.upper()}.parquet"

    def has(self, symbol: str) -> bool:
        return self.path_for(symbol).exists()

    def load(self, symbol: str) -> pd.DataFrame:
        frame = pd.read_parquet(self.path_for(symbol))
        return schema.validate(frame)

    def store(self, symbol: str, frame: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        schema.validate(frame).to_parquet(self.path_for(symbol))

    def get_or_fetch(self, symbol: str, fetch_fn: FetchFn) -> pd.DataFrame:
        """Return the cached frame, fetching and persisting it on first miss.

        Both the miss path and the hit path return the *loaded-from-disk* frame, so
        repeated calls yield byte-identical results regardless of fetch ordering.
        """
        if not self.has(symbol):
            frame = schema.normalize(fetch_fn(symbol))
            self.store(symbol, frame)
        return self.load(symbol)
