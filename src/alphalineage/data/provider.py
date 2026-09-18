"""Provider abstraction and Tiingo -> yfinance fallback orchestration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PriceProvider(Protocol):
    """A source of normalized daily price frames (see :mod:`alphalineage.data.schema`)."""

    name: str

    def get_prices(
        self, symbol: str, start: str | None = ..., end: str | None = ...
    ) -> pd.DataFrame: ...


class ProviderError(RuntimeError):
    """Raised when every provider in a :class:`FallbackProvider` fails for a symbol."""


class QuotaExceededError(RuntimeError):
    """A provider refused the request because an account allowance is used up.

    This is a *stop* signal, not a per-symbol failure: every further request in the same batch
    spends more of the allowance (or is refused outright), so callers end the batch and report
    what was fetched. ``scope`` is ``"hourly"``, ``"daily"``, ``"monthly"`` or ``"unknown"``.
    """

    def __init__(self, message: str, *, scope: str = "unknown", provider: str = "") -> None:
        super().__init__(message)
        self.scope = scope
        self.provider = provider


class FallbackProvider:
    """Try each provider in order; return the first success.

    Records which provider served each symbol in :attr:`sources` so the survivorship
    and provenance reporting can flag yfinance-sourced (prototype-grade) data.
    """

    name = "fallback"

    def __init__(self, providers: list[PriceProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers
        self.sources: dict[str, str] = {}

    def get_prices(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        errors: list[str] = []
        for provider in self.providers:
            try:
                frame = provider.get_prices(symbol, start, end)
            except QuotaExceededError:
                # Never mask an allowance stop by silently switching sources: the next provider
                # has different adjustment/survivorship semantics and the batch must end anyway.
                raise
            except Exception as exc:  # noqa: BLE001 - we intentionally fall through
                errors.append(f"{provider.name}: {exc!r}")
                continue
            self.sources[symbol] = provider.name
            return frame
        raise ProviderError(f"all providers failed for {symbol!r}: {'; '.join(errors)}")
