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
