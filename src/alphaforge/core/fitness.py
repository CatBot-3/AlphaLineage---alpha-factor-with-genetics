"""P2-T2 — fitness: the information coefficient (IC / rank IC / IC IR).

Fitness is the cross-sectional IC of a factor against *forward* returns, not PnL
(invariant 4). For each date we correlate the factor across symbols with the next
period's return; the mean daily IC is the IC and mean/std is the IC IR. The GP maximizes
mean ``|rank IC|`` (sign-indifferent), minus a small node-count penalty (anti-bloat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.core.evaluate import evaluate
from alphaforge.core.panel import Panel
from alphaforge.core.tree import Node


def forward_returns(panel: Panel, horizon: int = 1) -> pd.DataFrame:
    """Next-``horizon`` simple return per date/symbol (the prediction target)."""
    return panel["returns"].shift(-horizon)


def _rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int) -> pd.Series:
    """Per-row (per-date) Pearson correlation; ``a`` and ``b`` share their NaN mask.

    Dates with fewer than ``min_names`` paired points are dropped (NaN). The floor matters:
    a 2-point cross-section is *always* perfectly rank-correlated, so a too-low floor lets the
    GP win by emitting a factor that is non-NaN on only a couple of symbols per date.
    """
    a_d = a.sub(a.mean(axis=1), axis=0)
    b_d = b.sub(b.mean(axis=1), axis=0)
    num = (a_d * b_d).sum(axis=1, min_count=1)
    den = np.sqrt((a_d**2).sum(axis=1, min_count=1) * (b_d**2).sum(axis=1, min_count=1))
    out = num / den.replace(0.0, np.nan)
    enough = a.notna().sum(axis=1) >= max(2, min_names)
    return out.where(enough)


def daily_ic(
    factor: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman", *, min_names: int = 2
) -> pd.Series:
    """Per-date cross-sectional IC between ``factor`` and forward returns ``fwd``."""
    factor, fwd = factor.align(fwd, join="inner")
    mask = factor.notna() & fwd.notna()
    a = factor.where(mask)
    b = fwd.where(mask)
    if method == "spearman":
        a = a.rank(axis=1)
        b = b.rank(axis=1)
    elif method != "pearson":
        raise ValueError(f"unknown IC method {method!r}")
    return _rowwise_corr(a, b, min_names)


def mean_ic(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    method: str = "spearman",
    *,
    absolute: bool = False,
    min_names: int = 2,
) -> float:
    ic = daily_ic(factor, fwd, method, min_names=min_names)
    value = ic.abs().mean() if absolute else ic.mean()
    return float(value) if np.isfinite(value) else 0.0


def ic_ir(daily: pd.Series) -> float:
    """Information ratio of the daily IC series (mean / std)."""
    clean = daily.dropna()
    if len(clean) < 2:
        return 0.0
    std = clean.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(clean.mean() / std)


#: A factor must score IC on at least this many dates, else it is treated as degenerate.
_MIN_VALID_DATES = 5


def score_tree(
    tree: Node,
    panel: Panel,
    fwd: pd.DataFrame,
    *,
    method: str = "spearman",
    absolute: bool = True,
    parsimony: float = 0.0,
    min_names: int = 5,
) -> tuple[float, dict[str, float]]:
    """Return ``(fitness, metrics)`` for ``tree``: mean |IC| minus a node-count penalty.

    ``min_names`` (the cross-section breadth floor) and a minimum number of valid dates guard
    against factors that earn a spuriously perfect IC on a near-empty cross-section.
    """
    factor = evaluate(tree, panel)
    if not isinstance(factor, pd.DataFrame):  # a degenerate non-panel result
        return -parsimony * tree.size(), {"ic": 0.0, "ic_ir": 0.0}
    ic = daily_ic(factor, fwd, method, min_names=min_names)
    if ic.notna().sum() < _MIN_VALID_DATES:
        return -parsimony * tree.size(), {"ic": 0.0, "ic_ir": 0.0}
    raw = ic.abs().mean() if absolute else ic.mean()
    raw = float(raw) if np.isfinite(raw) else 0.0
    fitness = raw - parsimony * tree.size()
    return fitness, {"ic": raw, "ic_ir": ic_ir(ic)}
