"""Phase 0 - data foundation.

Reliable, cached, corporate-action-adjusted, point-in-time universe data behind a
provider abstraction (Tiingo primary, yfinance fallback). Everything downstream reads
from the local Parquet cache; the data API is never called inside the GP loop.
"""

from alphalineage.data.cache import ParquetCache
from alphalineage.data.provider import FallbackProvider, PriceProvider, ProviderError
from alphalineage.data.tiingo_client import TiingoProvider
from alphalineage.data.universe import Membership, Universe, sample_universe
from alphalineage.data.yfinance_provider import YFinanceProvider

__all__ = [
    "FallbackProvider",
    "Membership",
    "ParquetCache",
    "PriceProvider",
    "ProviderError",
    "TiingoProvider",
    "Universe",
    "YFinanceProvider",
    "sample_universe",
]
