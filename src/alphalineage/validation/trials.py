"""P3-T5 — the global trial counter.

The number of strategies tried is the deflation's ``N``: the more configurations searched
(including future user-added primitives that enlarge the search space, Phase 7), the higher
the Sharpe expected from luck alone. The GP already memoizes every distinct tree it scores,
so its ``trial_count`` is the natural source; this counter generalizes that across a session.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrialCounter:
    """A running total of distinct strategy evaluations fed into the deflated Sharpe."""

    count: int = 0

    def add(self, n: int = 1) -> int:
        if n < 0:
            raise ValueError("trial increment must be non-negative")
        self.count += n
        return self.count

    def reset(self) -> None:
        self.count = 0


def effective_trials(
    searched: int,
    *,
    n_operators: int,
    baseline: int,
    n_schemes: int = 1,
    exponent: float = 2.0,
) -> int:
    """Trial count for the deflation, grown for the search-space size (invariant 1, P7-T3).

    Adding operators enlarges the per-node choice set, so the number of distinct strategies the
    search could have produced grows. We scale the searched count by ``(n_operators/baseline) **
    exponent`` (a conservative polynomial proxy) and by the number of weighting schemes tried.
    Strictly increasing in ``n_operators`` — user-added operators deflate harder.
    """
    if baseline <= 0 or n_operators <= 0:
        return max(0, searched) * max(1, n_schemes)
    growth = (n_operators / baseline) ** exponent
    return int(round(max(0, searched) * max(1, n_schemes) * growth))
