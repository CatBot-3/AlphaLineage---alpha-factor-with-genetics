"""P0-T2 - Parquet local cache.

Fetch a symbol once, persist it to ``data_cache/prices/{SYMBOL}.parquet``, and serve
every subsequent read from disk. This enforces the invariant that the data API is
never called inside the GP loop: the loop only ever reads the cache.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from alphalineage.data import paths, schema
from alphalineage.data.identifiers import child_path, validate_symbol

FetchFn = Callable[[str], pd.DataFrame]


def merge_price_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge incremental provider frames without losing or mixing price semantics.

    All explicitly labelled pieces must use one basis.  Legacy pieces are classified
    after the merge, which both migrates unambiguous histories and rejects a cache made
    from incompatible raw/split-adjusted downloads.
    """
    if not frames:
        raise ValueError("at least one price frame is required")
    validated = [schema.validate(frame) for frame in frames]
    explicit = {
        basis
        for frame in validated
        if (basis := schema.explicit_price_basis(frame)) is not None
    }
    if len(explicit) > 1:
        raise schema.PriceBasisError(
            "incremental price frames use different adjustment bases; refresh the symbol"
        )
    providers = {
        str(frame.attrs.get(schema.PRICE_PROVIDER_ATTR))
        for frame in validated
        if frame.attrs.get(schema.PRICE_PROVIDER_ATTR)
    }
    if len(providers) > 1:
        raise schema.PriceBasisError(
            "incremental price frames came from different providers; refresh the symbol"
        )

    merged = pd.concat(validated).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = schema.validate(merged[schema.PRICE_COLUMNS].astype("float64"))
    declared = next(iter(explicit)) if explicit else None
    inferred = schema.infer_price_basis(merged)
    if declared is not None and inferred is not None and declared != inferred:
        raise schema.PriceBasisError(
            "incremental price history contradicts its provider adjustment basis; "
            "refresh the symbol"
        )
    return schema.with_price_metadata(
        merged,
        price_basis=declared or inferred or "raw",
        provider=next(iter(providers), "legacy"),
    )


class ParquetCache:
    """A directory of per-symbol Parquet price frames."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        # Resolved lazily so an ALPHALINEAGE_DATA_DIR override set after construction
        # (as tests do) is still honored.
        return self._root if self._root is not None else paths.prices_dir()

    def path_for(self, symbol: str) -> Path:
        clean = validate_symbol(symbol)
        return child_path(self.root, clean, ".parquet", label="symbol")

    def has(self, symbol: str) -> bool:
        return self.path_for(symbol).exists()

    def load(self, symbol: str) -> pd.DataFrame:
        frame = pd.read_parquet(self.path_for(symbol))
        return schema.validate(frame)

    def store(self, symbol: str, frame: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        validated = schema.validate(frame)
        basis = schema.resolve_price_basis(validated)
        provider = str(validated.attrs.get(schema.PRICE_PROVIDER_ATTR) or "legacy")
        persisted = schema.with_price_metadata(
            validated,
            price_basis=basis,
            provider=provider,
        )
        persisted.to_parquet(self.path_for(symbol))

    def metadata(self, symbol: str) -> dict[str, object]:
        """Effective provider/basis metadata, including legacy inference state."""
        return schema.price_metadata(self.load(symbol))

    def get_or_fetch(self, symbol: str, fetch_fn: FetchFn) -> pd.DataFrame:
        """Return the cached frame, fetching and persisting it on first miss.

        Both the miss path and the hit path return the *loaded-from-disk* frame, so
        repeated calls yield byte-identical results regardless of fetch ordering.
        """
        if not self.has(symbol):
            frame = schema.normalize(fetch_fn(symbol))
            self.store(symbol, frame)
        return self.load(symbol)
