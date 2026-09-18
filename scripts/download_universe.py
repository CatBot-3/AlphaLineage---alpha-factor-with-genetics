"""Fetch one or more universes' price history into the local Parquet cache.

Usage:
    python scripts/download_universe.py --universe sp500-lite --years 15
    python scripts/download_universe.py --universe sp500-energy --universe sp500-semiconductors
    python scripts/download_universe.py --group sectors --years 20
    python scripts/download_universe.py --universe sp500-lite --years 1 --provider yfinance

Universes may be the bundled sample, any bundled snapshot id or alias (``builtin-sp500-current``,
``sp500-energy``, ``builtin-sp500-theme-banks`` ...) or a saved custom universe. ``--group`` expands
to every bundled S&P 500 sector (``sectors``), theme (``themes``) or both (``subsets``).

Symbols already in ``data_cache/prices/`` are served from disk and spend no provider request, so
re-running after a stop resumes where the previous run ended.

Provider allowances: Tiingo's free plan allows roughly 50 requests/hour, 1,000/day and 500 unique
symbols/month. When the provider reports an exhausted allowance the script stops immediately
(exit code 3) instead of spending further requests or silently switching data sources; run it
again once the window resets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running as a plain script (no install) by adding ``src`` to the path.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402

from alphalineage.data import paths  # noqa: E402
from alphalineage.data.cache import ParquetCache  # noqa: E402
from alphalineage.data.provider import (  # noqa: E402
    FallbackProvider,
    PriceProvider,
    QuotaExceededError,
)
from alphalineage.data.tiingo_client import TiingoProvider  # noqa: E402
from alphalineage.data.universe import (  # noqa: E402
    Universe,
    bundled_snapshot_name,
    bundled_snapshot_specs,
    bundled_universe,
    sample_universe,
)
from alphalineage.data.yfinance_provider import YFinanceProvider  # noqa: E402

EXIT_OK = 0
EXIT_ERRORS = 2
EXIT_QUOTA = 3


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


def resolve_universe(name: str) -> Universe:
    """Sample, bundled (id or alias) or saved custom universe, in that order."""
    try:
        return sample_universe(name)
    except KeyError:
        pass
    canonical = bundled_snapshot_name(name)
    if canonical is not None:
        return bundled_universe(canonical)
    try:
        return Universe.load(name)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"unknown universe {name!r}: {exc}") from exc


def group_universes(group: str) -> list[str]:
    prefixes = {
        "sectors": ("builtin-sp500-sector-",),
        "themes": ("builtin-sp500-theme-",),
        "subsets": ("builtin-sp500-sector-", "builtin-sp500-theme-"),
    }[group]
    return [
        str(spec["id"]) for spec in bundled_snapshot_specs() if str(spec["id"]).startswith(prefixes)
    ]


def ordered_symbols(universes: list[Universe]) -> list[tuple[str, str]]:
    """``(cache symbol, provider symbol)`` pairs, de-duplicated in universe order."""
    seen: dict[str, str] = {}
    for universe in universes:
        # Universe symbols are already normalized to the vendor spelling (``BRK.B`` -> ``BRK-B``);
        # an explicit alias, when a definition declares one for the symbol, wins.
        for symbol in universe.all_symbols():
            seen.setdefault(symbol, universe.aliases.get(symbol, symbol))
    return list(seen.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download universes into the local cache.")
    parser.add_argument(
        "--universe",
        action="append",
        default=[],
        help="universe id, alias or saved name (repeatable)",
    )
    parser.add_argument(
        "--group",
        choices=["sectors", "themes", "subsets"],
        action="append",
        default=[],
        help="add every bundled S&P 500 sector/theme universe (repeatable)",
    )
    parser.add_argument("--years", type=int, default=15, help="years of history to fetch")
    parser.add_argument("--start", default=None, help="explicit start date (YYYY-MM-DD)")
    parser.add_argument("--provider", choices=["auto", "tiingo", "yfinance"], default="auto")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="stop after this many provider requests (leave headroom in your allowance)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    have_key = bool(os.environ.get("TIINGO_API_KEY"))
    if args.provider in {"auto", "tiingo"} and not have_key:
        print("TODO(human): TIINGO_API_KEY not set - set it in .env to use Tiingo.")
        if args.provider == "tiingo":
            return 1
        print("Falling back to yfinance (survivorship-biased, prototype-grade).")

    names = list(args.universe)
    for group in args.group:
        names.extend(group_universes(group))
    if not names:
        names = ["sp500-lite"]
    universes = [resolve_universe(name) for name in dict.fromkeys(names)]
    pairs = ordered_symbols(universes)

    provider = _build_provider(args.provider, have_key=have_key)
    cache = ParquetCache()
    paths.ensure_dirs()

    end = date.today()
    start = (
        date.fromisoformat(args.start) if args.start else end - timedelta(days=365 * args.years + 5)
    )
    start_s, end_s = start.isoformat(), end.isoformat()

    fetched: dict[str, int] = {}
    errors: dict[str, str] = {}
    quota: dict[str, str] | None = None
    requests_made = 0
    remaining: list[str] = []
    cached_at_start = sum(cache.has(symbol) for symbol, _ in pairs)
    print(
        f"{len(universes)} universe(s), {len(pairs)} unique symbols, "
        f"{cached_at_start} already cached"
    )
    for index, (symbol, provider_symbol) in enumerate(pairs):
        if cache.has(symbol):
            continue
        if args.max_requests is not None and requests_made >= args.max_requests:
            remaining = [item for item, _ in pairs[index:] if not cache.has(item)]
            print(f"Reached --max-requests {args.max_requests}; stopping.")
            break
        requests_made += 1
        try:
            frame = cache.get_or_fetch(
                symbol, lambda _s, ps=provider_symbol: provider.get_prices(ps, start_s, end_s)
            )
        except QuotaExceededError as exc:
            quota = {"scope": exc.scope, "provider": exc.provider, "message": str(exc)}
            remaining = [item for item, _ in pairs[index:] if not cache.has(item)]
            print(f"  {symbol:6s} STOPPED: {exc}")
            break
        except Exception as exc:  # noqa: BLE001 - log and continue the batch
            errors[symbol] = repr(exc)
            print(f"  {symbol:6s} FAILED: {exc!r}")
            continue
        fetched[symbol] = len(frame)
        print(f"  {symbol:6s} {len(frame):6d} rows")

    log: dict[str, object] = {
        "universes": [universe.name for universe in universes],
        "start": start_s,
        "end": end_s,
        "requests": requests_made,
        "fetched": fetched,
        "already_cached": cached_at_start,
        "errors": errors,
        "quota": quota,
        "remaining": remaining,
    }
    if isinstance(provider, FallbackProvider):
        log["sources"] = provider.sources
    meta_path = paths.meta_dir() / "fetch_log.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    print(
        f"Fetched {len(fetched)}, already cached {cached_at_start}, failed {len(errors)}, "
        f"not yet fetched {len(remaining)}; log -> {meta_path}"
    )
    if quota is not None:
        print(
            f"Provider {quota['scope']} allowance exhausted. Re-run the same command after the "
            "window resets; cached symbols are skipped."
        )
        return EXIT_QUOTA
    return EXIT_OK if not errors else EXIT_ERRORS


if __name__ == "__main__":
    raise SystemExit(main())
