"""Fetch a universe's price history into the local Parquet cache.

Usage:
    python scripts/download_universe.py --universe sp500-lite --years 15
    python scripts/download_universe.py --universe sp500-lite --years 1 --provider yfinance

Fetches each symbol once via the (Tiingo -> yfinance) fallback provider and caches it
to ``data_cache/prices/``. Re-runs are served from disk, honoring the invariant that
the data API is never called twice for the same cached symbol.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running as a plain script (no install) by adding ``src`` to the path.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402

from alphaforge.data import paths  # noqa: E402
from alphaforge.data.cache import ParquetCache  # noqa: E402
from alphaforge.data.provider import FallbackProvider, PriceProvider  # noqa: E402
from alphaforge.data.tiingo_client import TiingoProvider  # noqa: E402
from alphaforge.data.universe import sample_universe  # noqa: E402
from alphaforge.data.yfinance_provider import YFinanceProvider  # noqa: E402


def _build_provider(choice: str, *, have_key: bool) -> PriceProvider:
    tiingo = TiingoProvider()
    yfin = YFinanceProvider()
    if choice == "tiingo":
        return tiingo
    if choice == "yfinance":
        return yfin
    # auto: prefer Tiingo when a key is present, else fall straight to yfinance.
    providers: list[PriceProvider] = [tiingo, yfin] if have_key else [yfin]
    return FallbackProvider(providers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a universe into the local cache.")
    parser.add_argument("--universe", default="sp500-lite", help="sample universe name")
    parser.add_argument("--years", type=int, default=15, help="years of history to fetch")
    parser.add_argument("--provider", choices=["auto", "tiingo", "yfinance"], default="auto")
    args = parser.parse_args(argv)

    load_dotenv()
    import os

    have_key = bool(os.environ.get("TIINGO_API_KEY"))
    if args.provider in {"auto", "tiingo"} and not have_key:
        print("TODO(human): TIINGO_API_KEY not set — set it in .env to use Tiingo.")
        if args.provider == "tiingo":
            return 1
        print("Falling back to yfinance (survivorship-biased, prototype-grade).")

    universe = sample_universe(args.universe)
    provider = _build_provider(args.provider, have_key=have_key)
    cache = ParquetCache()
    paths.ensure_dirs()

    end = date.today()
    start = end - timedelta(days=365 * args.years + 5)
    start_s, end_s = start.isoformat(), end.isoformat()

    log: dict[str, object] = {"universe": args.universe, "fetched": {}, "errors": {}}
    fetched: dict[str, int] = {}
    errors: dict[str, str] = {}
    for symbol in universe.all_symbols():
        try:
            frame = cache.get_or_fetch(symbol, lambda s: provider.get_prices(s, start_s, end_s))
            fetched[symbol] = len(frame)
            print(f"  {symbol:6s} {len(frame):6d} rows")
        except Exception as exc:  # noqa: BLE001 - log and continue the batch
            errors[symbol] = repr(exc)
            print(f"  {symbol:6s} FAILED: {exc!r}")

    log["fetched"] = fetched
    log["errors"] = errors
    if isinstance(provider, FallbackProvider):
        log["sources"] = provider.sources

    meta_path = paths.meta_dir() / "fetch_log.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"Fetched {len(fetched)}/{len(universe.all_symbols())} symbols; log -> {meta_path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
