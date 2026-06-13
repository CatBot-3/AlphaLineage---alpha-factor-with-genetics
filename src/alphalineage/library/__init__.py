"""Phase 5 - the factor library: lineage, diversity pruning, and the mega-alpha.

Persist a run's lineage so it replays (:mod:`store`), prune the population to a decorrelated
set (:mod:`diversity`), and combine that set into a mega-alpha that beats the best single
factor's deflated metric (:mod:`combine`).
"""

from alphalineage.library.combine import combine, combine_trees
from alphalineage.library.diversity import prune, signal_correlation
from alphalineage.library.store import LineageNode, LineageStore

__all__ = [
    "LineageNode",
    "LineageStore",
    "combine",
    "combine_trees",
    "prune",
    "signal_correlation",
]
