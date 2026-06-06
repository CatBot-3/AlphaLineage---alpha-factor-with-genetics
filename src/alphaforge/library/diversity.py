"""P5-T3 — diversity pruning by pairwise correlation.

A GP population converges to many near-duplicate factors. Pruning keeps the highest-fitness
factor and admits a new one only if it is sufficiently *decorrelated* from everything kept, so
the surviving set is genuinely diverse (and a better, more independent input to the combination
model and to PBO).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from alphaforge.core.evaluate import evaluate
from alphaforge.core.panel import Panel
from alphaforge.core.tree import Node


def _rank_vector(signal: pd.DataFrame) -> pd.Series:
    """Cross-sectionally rank per date and flatten to a (date, symbol) vector."""
    return signal.rank(axis=1).stack()


def signal_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Rank correlation of two factor signals over their common (date, symbol) support."""
    va, vb = _rank_vector(a), _rank_vector(b)
    common = va.index.intersection(vb.index)
    if len(common) < 3:
        return 0.0
    va, vb = va.loc[common], vb.loc[common]
    if va.std() == 0 or vb.std() == 0:
        return 0.0
    return float(np.corrcoef(va.to_numpy(), vb.to_numpy())[0, 1])


def prune(
    trees: Sequence[Node],
    panel: Panel,
    *,
    fitness: Callable[[Node], float],
    threshold: float = 0.7,
) -> list[Node]:
    """Greedily keep high-fitness factors that are pairwise |corr| < ``threshold``."""
    ordered = sorted(trees, key=fitness, reverse=True)
    kept: list[Node] = []
    kept_signals: list[pd.DataFrame] = []
    for tree in ordered:
        signal = evaluate(tree, panel)
        if not isinstance(signal, pd.DataFrame):
            continue
        if all(abs(signal_correlation(signal, k)) < threshold for k in kept_signals):
            kept.append(tree)
            kept_signals.append(signal)
    return kept
