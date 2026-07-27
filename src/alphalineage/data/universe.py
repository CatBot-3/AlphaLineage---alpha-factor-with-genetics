"""P0-T4 - point-in-time universe builder.

A universe is a set of membership intervals ``[entry, exit)`` per symbol. Querying
``members_asof(date)`` returns only symbols whose interval contains ``date`` - so a
symbol that entered the index later, or left the selected universe earlier, is excluded.
This is what prevents survivorship and look-ahead bias at the universe level.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from alphalineage.data import paths
from alphalineage.data.identifiers import child_path, validate_symbol

UNIVERSE_MODES = {"point_in_time", "static_snapshot"}
MARKET_SYMBOL_ALIASES = {"BRK.B": "BRK-B", "BF.B": "BF-B"}


def normalize_market_symbol(symbol: str) -> str:
    """Normalize stable vendor spelling differences used by bundled index definitions."""
    clean = validate_symbol(symbol)
    return MARKET_SYMBOL_ALIASES.get(clean, clean)


def _definition_fingerprint(
    name: str,
    mode: str,
    memberships: Iterable[Membership],
    definition: Mapping[str, Any],
) -> str:
    payload = {
        "name": name,
        "mode": mode,
        "definition": dict(definition),
        "memberships": [
            {
                "symbol": item.symbol,
                "entry": item.entry.date().isoformat(),
                "exit": item.exit.date().isoformat() if item.exit is not None else None,
            }
            for item in memberships
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Membership:
    """One symbol's point-in-time interval; ``exit is None`` means membership is open."""

    symbol: str
    entry: pd.Timestamp
    exit: pd.Timestamp | None = None

    def contains(self, date: pd.Timestamp) -> bool:
        if date < self.entry:
            return False
        return self.exit is None or date < self.exit

    def overlaps(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Whether this ``[entry, exit)`` interval overlaps inclusive ``start..end``."""
        return self.entry <= end and (self.exit is None or self.exit > start)

    @property
    def is_active(self) -> bool:
        return self.exit is None


def _ts(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _required_ts(value: object, *, label: str) -> pd.Timestamp:
    timestamp = _ts(value)
    if timestamp is None:
        raise ValueError(f"{label} is required")
    return timestamp


def _comparison_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC-naive dates for comparisons while callers retain their original index."""
    index = pd.DatetimeIndex(dates)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index.normalize()


class Universe:
    """A named collection of point-in-time memberships."""

    def __init__(
        self,
        name: str,
        memberships: Iterable[Membership],
        *,
        mode: str = "point_in_time",
        definition: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        # Validate the exact identifier instead of normalizing it: silently stripping a name
        # could overwrite another universe's file while retaining a different in-memory key.
        child_path(paths.universe_dir(), name, ".parquet", label="universe name")
        if mode not in UNIVERSE_MODES:
            raise ValueError(f"unknown universe mode {mode!r}")
        self.name = name
        self.mode = mode
        normalized: list[Membership] = []
        for membership in memberships:
            entry = _ts(membership.entry)
            exit_ = _ts(membership.exit)
            if entry is None:
                raise ValueError("membership entry date is required")
            if exit_ is not None and exit_ <= entry:
                raise ValueError(f"membership exit must be after entry for {membership.symbol!r}")
            normalized.append(
                Membership(normalize_market_symbol(membership.symbol), entry=entry, exit=exit_)
            )
        if not normalized:
            raise ValueError("universe must contain at least one membership")

        # Re-entry is supported, but two simultaneous intervals for one symbol are ambiguous
        # and would make provenance dependent on row order. Reject those at the boundary.
        by_symbol: dict[str, list[Membership]] = {}
        for membership in normalized:
            by_symbol.setdefault(membership.symbol, []).append(membership)
        for symbol, intervals in by_symbol.items():
            ordered = sorted(intervals, key=lambda item: item.entry)
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if previous.exit is None or current.entry < previous.exit:
                    raise ValueError(f"overlapping membership intervals for {symbol!r}")

        self.memberships = normalized
        self.definition = {
            "id": name,
            "display_name": name,
            "snapshot_date": None,
            "interval_semantics": "dated half-open [entry, exit) membership intervals",
            "member_count": len({item.symbol for item in normalized}),
            **dict(definition or {}),
        }
        self.provenance = dict(provenance or {})
        self.aliases = dict(aliases or {})
        self.fingerprint = _definition_fingerprint(
            self.name, self.mode, self.memberships, self.definition
        )

    # --- queries -----------------------------------------------------------------
    def members_asof(self, date: object) -> list[str]:
        """Symbols whose membership interval contains ``date`` (sorted)."""
        as_of = _required_ts(date, label="as-of date")
        if self.mode == "static_snapshot":
            return self.all_symbols()
        return sorted({m.symbol for m in self.memberships if m.contains(as_of)})

    def memberships_overlapping(self, start: object, end: object) -> list[Membership]:
        """Membership rows that overlap an inclusive research date range."""
        start_ts = _required_ts(start, label="range start")
        end_ts = _required_ts(end, label="range end")
        if end_ts < start_ts:
            raise ValueError("universe range end must not precede its start")
        if self.mode == "static_snapshot":
            return list(self.memberships)
        return [m for m in self.memberships if m.overlaps(start_ts, end_ts)]

    def members_overlapping(self, start: object, end: object) -> list[str]:
        """Unique symbols with any membership during an inclusive research date range."""
        return sorted({m.symbol for m in self.memberships_overlapping(start, end)})

    def members_through(self, date: object) -> list[str]:
        """All symbols that had entered on or before ``date``, including exited names."""
        end = _required_ts(date, label="as-of date")
        if self.mode == "static_snapshot":
            return self.all_symbols()
        return sorted({m.symbol for m in self.memberships if m.entry <= end})

    def membership_mask(self, dates: pd.DatetimeIndex, symbols: Iterable[str]) -> pd.DataFrame:
        """Return the date x symbol point-in-time membership mask for a panel."""
        index = pd.DatetimeIndex(dates)
        comparable = _comparison_dates(index)
        columns = list(dict.fromkeys(normalize_market_symbol(symbol) for symbol in symbols))
        mask = pd.DataFrame(False, index=index, columns=columns, dtype=bool)
        if self.mode == "static_snapshot":
            # A snapshot means exactly one fixed constituent set. Applying it to earlier market
            # rows is intentionally possible for exploratory work, but the API labels the result
            # survivorship-biased and never presents it as point-in-time history.
            mask.loc[:, [symbol for symbol in columns if symbol in self.all_symbols()]] = True
            return mask
        for membership in self.memberships:
            if membership.symbol not in mask.columns:
                continue
            active = comparable >= membership.entry
            if membership.exit is not None:
                active &= comparable < membership.exit
            mask.loc[:, membership.symbol] |= active
        return mask

    def all_symbols(self) -> list[str]:
        return sorted({m.symbol for m in self.memberships})

    def active(self) -> list[Membership]:
        return [m for m in self.memberships if m.is_active]

    def exited(self) -> list[Membership]:
        """Membership intervals with a declared exit.

        An index-membership exit does not by itself mean that the security was delisted.
        """
        return [m for m in self.memberships if not m.is_active]

    def delisted(self) -> list[Membership]:
        """Deprecated alias for :meth:`exited`; no security-status inference is performed."""
        return self.exited()

    # --- persistence -------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": [m.symbol for m in self.memberships],
                "entry_date": [m.entry for m in self.memberships],
                "exit_date": [m.exit for m in self.memberships],
            }
        )

    @classmethod
    def from_frame(cls, name: str, frame: pd.DataFrame) -> Universe:
        memberships = [
            Membership(
                symbol=str(row.symbol),
                entry=pd.Timestamp(row.entry_date).normalize(),
                exit=_ts(row.exit_date),
            )
            for row in frame.itertuples(index=False)
        ]
        return cls(name, memberships)

    def save(self, path: Path | None = None) -> Path:
        target = (
            child_path(paths.universe_dir(), self.name, ".parquet", label="universe name")
            if path is None
            else child_path(path.parent, path.stem, path.suffix, label="universe filename")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(target)
        return target

    @classmethod
    def load(cls, name: str, path: Path | None = None) -> Universe:
        source = (
            child_path(paths.universe_dir(), name, ".parquet", label="universe name")
            if path is None
            else child_path(path.parent, path.stem, path.suffix, label="universe filename")
        )
        return cls.from_frame(name, pd.read_parquet(source))


# --- immutable offline snapshots ------------------------------------------------
_SNAPSHOT_RESOURCE_DIR = "universe_snapshots"


@lru_cache(maxsize=1)
def _snapshot_manifest() -> tuple[dict[str, Any], ...]:
    source = resources.files("alphalineage.data").joinpath(
        _SNAPSHOT_RESOURCE_DIR, "manifest.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("snapshots"), list):
        raise ValueError("invalid bundled universe snapshot manifest")
    retrieved_at = str(payload.get("retrieved_at") or "")
    return tuple({"retrieved_at": retrieved_at, **dict(item)} for item in payload["snapshots"])


@cache
def _snapshot_asset(asset: str) -> tuple[tuple[str, ...], dict[str, str]]:
    source = resources.files("alphalineage.data").joinpath(_SNAPSHOT_RESOURCE_DIR, asset)
    payload = json.loads(source.read_text(encoding="utf-8"))
    symbols = tuple(validate_symbol(str(symbol)) for symbol in payload.get("symbols", []))
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError(f"invalid or duplicate symbols in bundled snapshot {asset!r}")
    aliases = {
        validate_symbol(str(source_symbol)): validate_symbol(str(provider_symbol))
        for source_symbol, provider_symbol in dict(payload.get("symbol_aliases", {})).items()
    }
    return symbols, aliases


def bundled_snapshot_specs() -> list[dict[str, Any]]:
    """Immutable manifest entries for every loadable offline static snapshot."""
    return [dict(item) for item in _snapshot_manifest()]


def bundled_snapshot_name(name: str) -> str | None:
    """Resolve a canonical snapshot id or a stable user-facing alias."""
    wanted = name.strip().casefold()
    for item in _snapshot_manifest():
        names = {str(item["id"]).casefold(), *(str(alias).casefold() for alias in item["aliases"])}
        if wanted in names:
            return str(item["id"])
    return None


def bundled_universe(name: str) -> Universe:
    """Load a packaged, dated constituent snapshot without network or mutable local state."""
    canonical = bundled_snapshot_name(name)
    if canonical is None:
        raise KeyError(f"unknown bundled universe {name!r}")
    item = next(spec for spec in _snapshot_manifest() if spec["id"] == canonical)
    symbols, symbol_aliases = _snapshot_asset(str(item["asset"]))
    if len(symbols) != int(item["source_symbol_count"]):
        raise ValueError(f"bundled universe {canonical!r} symbol count does not match its manifest")
    checksum_payload = json.dumps(list(symbols), separators=(",", ":"))
    checksum = hashlib.sha256(checksum_payload.encode("utf-8")).hexdigest()
    if checksum != item["source_checksum"]:
        raise ValueError(f"bundled universe {canonical!r} failed its checksum")
    snapshot_date = pd.Timestamp(item["snapshot_date"]).normalize()
    return Universe(
        canonical,
        [Membership(symbol, snapshot_date) for symbol in symbols],
        mode="static_snapshot",
        definition={
            "id": canonical,
            "display_name": str(item["display_name"]),
            "snapshot_date": snapshot_date.date().isoformat(),
            "interval_semantics": (
                "fixed constituent set captured on snapshot_date and applied unchanged to "
                "historical rows (survivorship-biased)"
            ),
            "member_count": len(symbols),
            "benchmark": str(item["benchmark"]),
        },
        provenance={
            "provider": str(item["source_name"]),
            "source_url": str(item["source_url"]),
            "retrieved_at": str(item["retrieved_at"]),
            "attribution": str(item["attribution"]),
            "license": str(item["license"]),
            "license_url": str(item["license_url"]),
            "terms_url": str(item["terms_url"]),
            "note": str(item["note"]),
        },
        aliases=symbol_aliases,
    )


# --- sample data -----------------------------------------------------------------
# This is a deliberately small *current* demonstration basket, not historical index
# membership.  The stable ``sp500-lite`` id is retained for existing workspaces, but the
# definition and UI describe what it actually is.  Every name was present in the bundled
# 2026-07-15 S&P 500 snapshot; no exit date is inferred from price availability.
_SAMPLE_UNIVERSES: dict[str, list[tuple[str, str, str | None]]] = {
    "sp500-lite": [
        ("AAPL", "2026-07-15", None),
        ("MSFT", "2026-07-15", None),
        ("NVDA", "2026-07-15", None),
        ("AMZN", "2026-07-15", None),
        ("GOOGL", "2026-07-15", None),
        ("META", "2026-07-15", None),
        ("AVGO", "2026-07-15", None),
        ("TSLA", "2026-07-15", None),
        ("JPM", "2026-07-15", None),
        ("BRK.B", "2026-07-15", None),
        ("V", "2026-07-15", None),
        ("WMT", "2026-07-15", None),
        ("XOM", "2026-07-15", None),
        ("JNJ", "2026-07-15", None),
        ("COST", "2026-07-15", None),
    ],
}


_PRESET_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "sp500-lite",
        "display_name": "Popular US stocks sample",
        "benchmark": "US large-cap demonstration basket",
        "status": "bundled_sample",
        "available": True,
        "snapshot_available": True,
        "snapshot_universe": "sp500-lite",
        "pit_import_name": "sp500-lite-pit",
        "mode": "static_snapshot",
        "membership_history": "static_snapshot",
        "coverage": "15 recognizable, liquid US stocks in a dated offline sample",
        "research_ready": False,
        "provenance": "Selected from the bundled 2026-07-15 S&P 500 snapshot",
        "warning": (
            "This is a current demonstration basket, not historical index membership."
        ),
    },
    {
        "id": "sp500",
        "display_name": "S&P 500",
        "benchmark": "S&P 500",
        "status": "bundled_snapshot",
        "available": True,
        "snapshot_available": True,
        "snapshot_universe": "builtin-sp500-current",
        "pit_import_name": "sp500",
        "mode": "static_snapshot",
        "membership_history": "static_snapshot",
        "coverage": "503 securities in an offline snapshot dated 2026-07-15",
        "research_ready": False,
        "provenance": "Bundled dated public constituent snapshot",
        "warning": "Static constituents are survivorship-biased; import PIT history for research.",
    },
    {
        "id": "djia",
        "display_name": "Dow Jones Industrial Average",
        "benchmark": "DJIA",
        "status": "bundled_snapshot",
        "available": True,
        "snapshot_available": True,
        "snapshot_universe": "builtin-djia-current",
        "pit_import_name": "djia",
        "mode": "static_snapshot",
        "membership_history": "static_snapshot",
        "coverage": "30 securities in an offline snapshot dated 2026-07-15",
        "research_ready": False,
        "provenance": "Bundled dated public constituent snapshot",
        "warning": "Static constituents are survivorship-biased; import PIT history for research.",
    },
    {
        "id": "nasdaq100",
        "display_name": "Nasdaq-100",
        "benchmark": "Nasdaq-100",
        "status": "bundled_snapshot",
        "available": True,
        "snapshot_available": True,
        "snapshot_universe": "builtin-nasdaq100-current",
        "pit_import_name": "nasdaq100",
        "mode": "static_snapshot",
        "membership_history": "static_snapshot",
        "coverage": "103 securities in an offline snapshot dated 2026-07-15",
        "research_ready": False,
        "provenance": "Bundled dated public constituent snapshot",
        "warning": "Static constituents are survivorship-biased; import PIT history for research.",
    },
)


def universe_presets() -> list[dict[str, Any]]:
    """Honest dual-mode descriptors for bundled snapshots and separate PIT imports."""
    return [dict(item) for item in _PRESET_CATALOG]


def universe_integrity(name: str, *, source: str) -> dict[str, Any]:
    """Integrity metadata suitable for universe-list and detail API responses."""
    preset = next(
        (
            item
            for item in _PRESET_CATALOG
            if item["id"] == name or item.get("snapshot_universe") == name
        ),
        None,
    )
    if preset is not None and source in {"sample", "bundled"}:
        return {
            key: value
            for key, value in preset.items()
            if key
            in {
                "membership_history",
                "coverage",
                "research_ready",
                "provenance",
                "warning",
            }
        }
    return {
        "membership_history": "user_supplied",
        "coverage": "Defined by the saved membership intervals",
        "research_ready": False,
        "provenance": source,
        "warning": "Verify the membership source and cached price coverage before research use.",
    }


def sample_universe(name: str = "sp500-lite") -> Universe:
    """Return the small bundled current-stock demonstration basket."""
    try:
        rows = _SAMPLE_UNIVERSES[name]
    except KeyError as exc:
        raise KeyError(f"unknown sample universe {name!r}") from exc
    memberships = [
        Membership(symbol=sym, entry=pd.Timestamp(entry).normalize(), exit=_ts(exit_))
        for sym, entry, exit_ in rows
    ]
    return Universe(
        name,
        memberships,
        mode="static_snapshot",
        definition={
            "id": name,
            "display_name": "Popular US stocks sample",
            "snapshot_date": "2026-07-15",
            "interval_semantics": (
                "fixed 15-stock demonstration basket captured on snapshot_date and applied "
                "unchanged to historical rows"
            ),
            "member_count": len({membership.symbol for membership in memberships}),
            "benchmark": "US large-cap demonstration basket",
        },
        provenance={
            "provider": "Wikipedia: List of S&P 500 companies",
            "source_url": (
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            ),
            "retrieved_at": "2026-07-15",
            "attribution": "Wikipedia contributors",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "terms_url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
            "note": (
                "Selected from the bundled dated S&P 500 constituent asset; current-stock "
                "sample only, not point-in-time index history."
            ),
        },
    )
