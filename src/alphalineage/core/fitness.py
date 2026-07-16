"""P2-T2 - fitness: the information coefficient (IC / rank IC / IC IR).

Fitness is the cross-sectional IC of a factor against *forward* returns, not PnL
(invariant 4). For each date we correlate the factor across symbols with the next
period's return; the mean daily IC is the IC and mean/std is the IC IR. The GP maximizes
mean ``|rank IC|`` (sign-indifferent), minus a small node-count penalty (anti-bloat).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from alphalineage.core.evaluate import evaluate
from alphalineage.core.extensions import expand_all
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node


def forward_returns(panel: Panel, horizon: int = 1) -> pd.DataFrame:
    """Next-``horizon`` simple return per date/symbol (the prediction target)."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    return panel["returns"].shift(-horizon)


def _rowwise_corr_values(a: np.ndarray, b: np.ndarray, min_names: int) -> np.ndarray:
    """Vectorized row-wise Pearson correlation for equally shaped, paired arrays."""
    paired = ~np.isnan(a) & ~np.isnan(b)
    counts = paired.sum(axis=1)
    x = np.where(paired, a, 0.0)
    y = np.where(paired, b, 0.0)
    out = np.full(a.shape[0], np.nan, dtype=np.float64)

    # The moment identity removes the centered DataFrame-sized temporaries from the hot path.
    # Numerically delicate rows take the centered fallback below, so constants, huge offsets and
    # infinities keep the pandas reference behavior rather than suffering cancellation.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        sum_a = x.sum(axis=1)
        sum_b = y.sum(axis=1)
        sum_ab = (x * y).sum(axis=1)
        sum_aa = (x * x).sum(axis=1)
        sum_bb = (y * y).sum(axis=1)
        numerator = sum_ab - sum_a * sum_b / counts
        ss_a = sum_aa - sum_a * sum_a / counts
        ss_b = sum_bb - sum_b * sum_b / counts

    tolerance = np.finfo(np.float64).eps * 16.0
    unstable = (
        ~np.isfinite(sum_a + sum_b + sum_ab + sum_aa + sum_bb)
        | (ss_a <= tolerance * np.maximum(sum_aa, 1.0))
        | (ss_b <= tolerance * np.maximum(sum_bb, 1.0))
    )
    enough = counts >= max(2, min_names)
    fast = enough & ~unstable
    with np.errstate(invalid="ignore", divide="ignore"):
        out[fast] = numerator[fast] / np.sqrt(ss_a[fast] * ss_b[fast])

    slow = enough & unstable
    if np.any(slow):
        slow_a = np.where(paired[slow], a[slow], np.nan)
        slow_b = np.where(paired[slow], b[slow], np.nan)
        slow_counts = counts[slow]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            a_mean = np.nansum(slow_a, axis=1) / slow_counts
            b_mean = np.nansum(slow_b, axis=1) / slow_counts
            a_d = slow_a - a_mean[:, None]
            b_d = slow_b - b_mean[:, None]
            products = a_d * b_d
            num = np.nansum(products, axis=1)
            den = np.sqrt(np.nansum(a_d * a_d, axis=1) * np.nansum(b_d * b_d, axis=1))
            slow_out = num / np.where(den == 0.0, np.nan, den)
        # pandas ``sum(min_count=1)`` returns NaN when every centered product is NaN.
        slow_out[~np.any(~np.isnan(products), axis=1)] = np.nan
        out[slow] = slow_out

    out[counts < max(2, min_names)] = np.nan
    return out


def _rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int) -> pd.Series:
    """Per-row (per-date) Pearson correlation; ``a`` and ``b`` share their NaN mask.

    Dates with fewer than ``min_names`` paired points are dropped (NaN). The floor matters:
    a 2-point cross-section is *always* perfectly rank-correlated, so a too-low floor lets the
    GP win by emitting a factor that is non-NaN on only a couple of symbols per date.
    """
    av = a.to_numpy(dtype="float64", na_value=np.nan)
    bv = b.to_numpy(dtype="float64", na_value=np.nan)
    return pd.Series(_rowwise_corr_values(av, bv, min_names), index=a.index)


def daily_ic(
    factor: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman", *, min_names: int = 2
) -> pd.Series:
    """Per-date cross-sectional IC between ``factor`` and forward returns ``fwd``."""
    factor, fwd = factor.align(fwd, join="inner")
    a = factor.to_numpy(dtype="float64", na_value=np.nan)
    b = fwd.to_numpy(dtype="float64", na_value=np.nan)
    paired = ~np.isnan(a) & ~np.isnan(b)
    a = np.where(paired, a, np.nan)
    b = np.where(paired, b, np.nan)
    if method == "spearman":
        # SciPy's average-tie, omit-NaN rank is the array equivalent of DataFrame.rank(axis=1).
        a = rankdata(a, method="average", axis=1, nan_policy="omit")
        b = rankdata(b, method="average", axis=1, nan_policy="omit")
    elif method != "pearson":
        raise ValueError(f"unknown IC method {method!r}")
    return pd.Series(_rowwise_corr_values(a, b, min_names), index=factor.index)


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


def _score_factor(
    complexity: int,
    factor: object,
    fwd: pd.DataFrame,
    *,
    method: str,
    absolute: bool,
    parsimony: float,
    min_names: int,
) -> tuple[float, dict[str, float]]:
    """Score an already-evaluated factor with the same contract as :func:`score_tree`."""
    if not isinstance(factor, pd.DataFrame):
        return -parsimony * complexity, {"ic": 0.0, "ic_ir": 0.0}
    ic = daily_ic(factor, fwd, method, min_names=min_names)
    if ic.notna().sum() < _MIN_VALID_DATES:
        return -parsimony * complexity, {"ic": 0.0, "ic_ir": 0.0}
    raw = ic.abs().mean() if absolute else ic.mean()
    raw = float(raw) if np.isfinite(raw) else 0.0
    fitness = raw - parsimony * complexity
    return fitness, {"ic": raw, "ic_ir": ic_ir(ic)}


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
    expanded = expand_all(tree)
    return _score_factor(
        expanded.size(),
        evaluate(expanded, panel),
        fwd,
        method=method,
        absolute=absolute,
        parsimony=parsimony,
        min_names=min_names,
    )


def score_trees(
    trees: Sequence[Node],
    panel: Panel,
    fwd: pd.DataFrame,
    *,
    method: str = "spearman",
    absolute: bool = True,
    parsimony: float = 0.0,
    min_names: int = 5,
    workers: int = 1,
    memory_budget_bytes: int | None = None,
) -> list[tuple[float, dict[str, float]]]:
    """Score trees in input order, using fused native scoring when it is available.

    One worker remains serial, but uses the same versioned native scoring kernel as larger worker
    counts so changing the resource policy cannot change a trajectory.  The fallback stays serial:
    Python threads make pandas evaluation slower and add nondeterministic cache completion.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    ordered = list(trees)
    if not ordered:
        return []

    # Import lazily so the optional extension does not become a requirement of fitness.py.
    from alphalineage.core import cpp as cpp_backend

    native_results: list[tuple[float, dict[str, float]] | None] | None = None
    score_many = getattr(cpp_backend, "score_many", None)
    supports_scoring = getattr(cpp_backend, "supports_native_scoring", None)
    if callable(score_many) and callable(supports_scoring) and supports_scoring(method):
        native_results = score_many(
            ordered,
            panel,
            fwd,
            method=method,
            absolute=absolute,
            parsimony=parsimony,
            min_names=min_names,
            min_valid_dates=_MIN_VALID_DATES,
            workers=workers,
            memory_budget_bytes=memory_budget_bytes,
        )

    if native_results is None or len(native_results) != len(ordered):
        native_results = [None] * len(ordered)

    # Unsupported/custom expressions and non-parity-safe methods are intentionally scored by the
    # ordinary evaluator on the coordinator thread, preserving deterministic fallback ordering.
    results: list[tuple[float, dict[str, float]]] = []
    for tree, native_result in zip(ordered, native_results, strict=True):
        results.append(
            native_result
            if native_result is not None
            else score_tree(
                tree,
                panel,
                fwd,
                method=method,
                absolute=absolute,
                parsimony=parsimony,
                min_names=min_names,
            )
        )
    return results
