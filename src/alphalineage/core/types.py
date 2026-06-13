"""P1-T1 — the strongly-typed DSL's type system.

Four types: SCALAR (a float constant), WINDOW (an int lookback), SERIES (a date x symbol
panel of values), and SIGNAL (a panel intended as the final cross-sectional alpha score).

SERIES and SIGNAL are *mutually substitutable* (each is a subtype of the other): any raw
series may serve as a signal, and a normalized signal (e.g. ``rank(close)``) may feed
further operators (e.g. ``ts_mean(rank(close), 5)``). SCALAR and WINDOW are leaf-only
parameter types. Because every argument type has at least one terminal that can produce
it, random generation can always close any open hole — there are no dead ends.
"""

from __future__ import annotations

from enum import Enum


class DType(Enum):
    """A value type in the typed expression DSL."""

    SCALAR = "scalar"
    WINDOW = "window"
    SERIES = "series"
    SIGNAL = "signal"


# The structurally-identical panel types; substitutable for one another.
_PANEL: frozenset[DType] = frozenset({DType.SERIES, DType.SIGNAL})


def is_subtype(sub: DType, sup: DType) -> bool:
    """True if a value of type ``sub`` can fill a hole expecting type ``sup``."""
    if sub is sup:
        return True
    return sub in _PANEL and sup in _PANEL
