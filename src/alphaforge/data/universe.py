"""P0-T4 — point-in-time universe builder.

A universe is a set of membership intervals ``[entry, exit)`` per symbol. Querying
``members_asof(date)`` returns only symbols whose interval contains ``date`` — so a
symbol that entered the index later, or was delisted earlier, is correctly excluded.
This is what prevents survivorship and look-ahead bias at the universe level.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alphaforge.data import paths


@dataclass(frozen=True)
class Membership:
    """One symbol's point-in-time membership interval; ``exit is None`` means active."""

    symbol: str
    entry: pd.Timestamp
    exit: pd.Timestamp | None = None

    def contains(self, date: pd.Timestamp) -> bool:
        if date < self.entry:
            return False
        return self.exit is None or date < self.exit

    @property
    def is_active(self) -> bool:
        return self.exit is None


def _ts(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if pd.isna(ts):
        return None
    return ts.normalize()


class Universe:
    """A named collection of point-in-time memberships."""

    def __init__(self, name: str, memberships: Iterable[Membership]) -> None:
        self.name = name
        self.memberships: list[Membership] = list(memberships)

    # --- queries -----------------------------------------------------------------
    def members_asof(self, date: object) -> list[str]:
        """Symbols whose membership interval contains ``date`` (sorted)."""
        as_of = pd.Timestamp(date).normalize()  # type: ignore[arg-type]
        return sorted(m.symbol for m in self.memberships if m.contains(as_of))

    def all_symbols(self) -> list[str]:
        return sorted({m.symbol for m in self.memberships})

    def active(self) -> list[Membership]:
        return [m for m in self.memberships if m.is_active]

    def delisted(self) -> list[Membership]:
        return [m for m in self.memberships if not m.is_active]

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
        target = path or (paths.universe_dir() / f"{self.name}.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(target)
        return target

    @classmethod
    def load(cls, name: str, path: Path | None = None) -> Universe:
        source = path or (paths.universe_dir() / f"{name}.parquet")
        return cls.from_frame(name, pd.read_parquet(source))


# --- sample data -----------------------------------------------------------------
# TODO(human): real, survivorship-bias-free index constituent history requires a paid
# data source (e.g. CRSP / index vendor). This small hand-built sample exists only so
# the point-in-time logic and survivorship report are exercisable without that data.
# It deliberately includes a delisted name (LEH) so the audit report is non-trivial.
_SAMPLE_UNIVERSES: dict[str, list[tuple[str, str, str | None]]] = {
    "sp500-lite": [
        ("AAPL", "2000-01-03", None),
        ("MSFT", "2000-01-03", None),
        ("AMZN", "2000-01-03", None),
        ("JPM", "2000-01-03", None),
        ("XOM", "2000-01-03", None),
        ("NVDA", "2000-01-03", None),
        ("GOOGL", "2004-08-19", None),  # entered at IPO, not earlier
        ("META", "2012-05-18", None),  # entered at IPO, not earlier
        ("LEH", "2000-01-03", "2008-09-15"),  # delisted (bankruptcy)
    ],
}


def sample_universe(name: str = "sp500-lite") -> Universe:
    """Return a small hand-built sample universe (prototype data only)."""
    try:
        rows = _SAMPLE_UNIVERSES[name]
    except KeyError as exc:
        raise KeyError(f"unknown sample universe {name!r}") from exc
    memberships = [
        Membership(symbol=sym, entry=pd.Timestamp(entry).normalize(), exit=_ts(exit_))
        for sym, entry, exit_ in rows
    ]
    return Universe(name, memberships)
