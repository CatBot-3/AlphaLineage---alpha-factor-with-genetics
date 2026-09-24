"""P2-T2 - fitness: the information coefficient (IC / rank IC / IC IR).

Fitness is the cross-sectional IC of a factor against *forward* returns, not PnL
(invariant 4). For each date we correlate the factor across symbols with the next
period's return.  Sign-indifference is applied once to the aggregate:
``abs(mean(daily IC))``.  Applying ``abs`` per day rewards a directionless factor whose
relationship continually flips, which is not a tradeable signal.
"""

from __future__ import annotations

from typing import Any

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from alphalineage.core.evaluate import evaluate
from alphalineage.core.extensions import expand_all
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node

COMPLEXITY_PENALTY_MODES = frozenset({"per_node", "normalized_budget"})
DEFAULT_NORMALIZED_COMPLEXITY_PENALTY = 0.005


def complexity_penalty_rate(
    *,
    parsimony: float = 0.0,
    complexity_penalty_mode: str = "per_node",
    complexity_penalty_value: float | None = None,
    max_nodes: int | None = None,
) -> float:
    """Return the per-expanded-node deduction for the configured penalty semantics.

    ``parsimony`` remains the compatibility input for old callers and checkpoints. New
    normalized-budget callers specify the maximum total deduction at ``max_nodes``.
    """
    if complexity_penalty_mode not in COMPLEXITY_PENALTY_MODES:
        raise ValueError(
            "complexity_penalty_mode must be 'per_node' or 'normalized_budget'"
        )
    value = (
        float(complexity_penalty_value)
        if complexity_penalty_value is not None
        else (
            DEFAULT_NORMALIZED_COMPLEXITY_PENALTY
            if complexity_penalty_mode == "normalized_budget"
            else float(parsimony)
        )
    )
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("complexity penalty must be a finite non-negative number")
    if complexity_penalty_mode == "per_node":
        return value
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ValueError("max_nodes must be a positive integer for normalized_budget")
    return value / float(max_nodes)


def complexity_deduction(
    complexity: int,
    *,
    parsimony: float = 0.0,
    complexity_penalty_mode: str = "per_node",
    complexity_penalty_value: float | None = None,
    max_nodes: int | None = None,
) -> float:
    """Return the exact complexity deduction applied to one expanded expression."""
    return float(complexity) * complexity_penalty_rate(
        parsimony=parsimony,
        complexity_penalty_mode=complexity_penalty_mode,
        complexity_penalty_value=complexity_penalty_value,
        max_nodes=max_nodes,
    )


def expanded_complexities(tree: Node) -> tuple[int, int]:
    """Return expanded occurrence nodes and distinct executable computations.

    Formula expansion memoizes repeated dependencies as a DAG for evaluation, but each textual
    occurrence still contributes to search complexity. The distinct-computation count remains
    useful for the hard feasibility budget and memory diagnostics.
    """
    expanded = expand_all(tree)
    return expanded.size(), expanded.unique_computation_size()


#: When a signal computed from session ``t`` data (including the ``t`` close) is traded.
#:
#: * ``close``      - in the closing auction of ``t`` itself. Optimistic: it assumes the order
#:   can use the very close that produced the signal. Legacy meaning of every stored session.
#: * ``next_open``  - at the opening auction of ``t + 1`` (orders sent overnight).
#: * ``next_close`` - in the closing auction of ``t + 1`` (a full one-session delay).
EXECUTION_TIMINGS: tuple[str, ...] = ("close", "next_open", "next_close")
LEGACY_EXECUTION = "close"


def validate_execution(execution: object) -> str:
    if not isinstance(execution, str) or execution not in EXECUTION_TIMINGS:
        raise ValueError(
            f"execution must be one of {', '.join(EXECUTION_TIMINGS)}, got {execution!r}"
        )
    return execution


def execution_delay(execution: str) -> int:
    """Sessions between the signal date and the first bar whose price the trade uses."""
    return 0 if validate_execution(execution) == "close" else 1


def label_span(horizon: int, execution: str = LEGACY_EXECUTION) -> int:
    """How many sessions past the signal date its forward-return label reaches.

    Embargoes and purges must be at least this long, or a label near a split boundary would
    read prices from the next segment.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    return horizon + execution_delay(execution)


def forward_returns(
    panel: Panel, horizon: int = 1, execution: str = LEGACY_EXECUTION
) -> pd.DataFrame:
    """The ``horizon``-session return a signal dated ``t`` can actually earn.

    ``close``: ``close[t + h] / close[t] - 1`` (one session: the next close-to-close return).
    ``next_open``: ``open[t + 1 + h] / open[t + 1] - 1``.
    ``next_close``: ``close[t + 1 + h] / close[t + 1] - 1``.

    Longer horizons are cumulative returns, not the one-day return observed ``h`` rows later.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    execution = validate_execution(execution)
    if execution == "close":
        if horizon == 1:
            # Preserve the established default scorer bit-for-bit.
            return panel["returns"].shift(-1)
        close = panel["close"]
        return close.shift(-horizon).div(close).sub(1.0)
    price = panel["open"] if execution == "next_open" else panel["close"]
    entry = price.shift(-1)
    return price.shift(-(1 + horizon)).div(entry).sub(1.0)


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
    signed = ic.mean()
    value = abs(signed) if absolute else signed
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


def _empty_metrics() -> dict[str, float]:
    # Preserve the compact legacy shape for degenerate factors.  Valid factors carry the
    # richer diagnostics below; consumers must already treat a missing metric as unavailable.
    return {"ic": 0.0, "ic_ir": 0.0}


def _ic_metrics(
    ic: pd.Series,
    active_names: pd.Series,
    *,
    absolute: bool,
) -> dict[str, float]:
    clean = ic.dropna()
    signed = float(clean.mean())
    mean_abs = float(clean.abs().mean())
    polarity = -1.0 if signed < 0.0 else 1.0
    oriented = abs(signed)
    signed_ir = ic_ir(clean)
    objective = oriented if absolute else signed
    same_direction = (
        (clean * polarity > 0.0).mean() if len(clean) else 0.0
    )
    breadth = active_names.reindex(clean.index).dropna()
    return {
        # ``ic`` remains the optimization metric for checkpoint/UI compatibility.
        "ic": float(objective),
        "signed_ic": signed,
        "oriented_ic": oriented,
        "mean_abs_ic": mean_abs,
        "polarity": polarity,
        "ic_ir": float(signed_ir),
        "oriented_ic_ir": float(polarity * signed_ir),
        "sign_consistency": float(same_direction),
        "valid_dates": float(len(clean)),
        "avg_active_names": float(breadth.mean()) if len(breadth) else 0.0,
        "min_active_names": float(breadth.min()) if len(breadth) else 0.0,
    }


def _score_factor(
    complexity: int,
    unique_complexity: int,
    factor: object,
    fwd: pd.DataFrame,
    *,
    method: str,
    absolute: bool,
    parsimony: float,
    complexity_penalty_mode: str,
    complexity_penalty_value: float | None,
    max_nodes: int | None,
    min_names: int,
) -> tuple[float, dict[str, float]]:
    """Score an already-evaluated factor with the same contract as :func:`score_tree`."""
    penalty = complexity_deduction(
        complexity,
        parsimony=parsimony,
        complexity_penalty_mode=complexity_penalty_mode,
        complexity_penalty_value=complexity_penalty_value,
        max_nodes=max_nodes,
    )
    if not isinstance(factor, pd.DataFrame):
        return -penalty, _empty_metrics()
    aligned_factor, aligned_fwd = factor.align(fwd, join="inner")
    paired = aligned_factor.notna() & aligned_fwd.notna()
    active_names = paired.sum(axis=1).astype("float64")
    ic = daily_ic(aligned_factor, aligned_fwd, method, min_names=min_names)
    if ic.notna().sum() < _MIN_VALID_DATES:
        return -penalty, _empty_metrics()
    metrics = _ic_metrics(ic, active_names, absolute=absolute)
    raw = metrics["ic"]
    metrics.update(
        {
            "raw_objective": float(raw),
            "expanded_complexity": float(complexity),
            "expanded_unique_nodes": float(unique_complexity),
            "complexity_penalty": float(penalty),
        }
    )
    fitness = raw - penalty
    return fitness, metrics


def score_tree(
    tree: Node,
    panel: Panel,
    fwd: pd.DataFrame,
    *,
    method: str = "spearman",
    absolute: bool = True,
    parsimony: float = 0.0,
    complexity_penalty_mode: str = "per_node",
    complexity_penalty_value: float | None = None,
    max_nodes: int | None = None,
    min_names: int = 5,
    observer: Any = None,
) -> tuple[float, dict[str, float]]:
    """Return ``(fitness, metrics)``: ``abs(mean(IC))`` minus node penalty.

    ``min_names`` (the cross-section breadth floor) and a minimum number of valid dates guard
    against factors that earn a spuriously perfect IC on a near-empty cross-section.
    """
    expanded = expand_all(tree)
    factor = evaluate(expanded, panel)
    result = _score_factor(
        expanded.size(),
        expanded.unique_computation_size(),
        factor,
        fwd,
        method=method,
        absolute=absolute,
        parsimony=parsimony,
        complexity_penalty_mode=complexity_penalty_mode,
        complexity_penalty_value=complexity_penalty_value,
        max_nodes=max_nodes,
        min_names=min_names,
    )
    return observer(tree, factor.to_numpy(), result) if observer is not None and isinstance(factor, pd.DataFrame) else result



def score_trees(
    trees: Sequence[Node],
    panel: Panel,
    fwd: pd.DataFrame,
    *,
    method: str = "spearman",
    absolute: bool = True,
    parsimony: float = 0.0,
    complexity_penalty_mode: str = "per_node",
    complexity_penalty_value: float | None = None,
    max_nodes: int | None = None,
    min_names: int = 5,
    workers: int = 1,
    memory_budget_bytes: int | None = None,
    observer: Any = None,
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

    observations: dict[int, np.ndarray] = {}
    native_results: list[tuple[float, dict[str, float]] | None] | None = None
    score_many = getattr(cpp_backend, "score_many", None)
    supports_scoring = getattr(cpp_backend, "supports_native_scoring", None)
    if callable(score_many) and callable(supports_scoring) and supports_scoring(method):
        native_kwargs = {
            "method": method,
            "absolute": absolute,
            "parsimony": parsimony,
            "min_names": min_names,
            "min_valid_dates": _MIN_VALID_DATES,
            "workers": workers,
            "memory_budget_bytes": memory_budget_bytes,
        }
        if (
            complexity_penalty_mode != "per_node"
            or complexity_penalty_value is not None
            or max_nodes is not None
        ):
            native_kwargs.update(
                {
                    "complexity_penalty_mode": complexity_penalty_mode,
                    "complexity_penalty_value": complexity_penalty_value,
                    "max_nodes": max_nodes,
                }
            )
        if observer is not None:
            native_kwargs["observations"] = observations
        native_results = score_many(ordered, panel, fwd, **native_kwargs)

    if native_results is None or len(native_results) != len(ordered):
        native_results = [None] * len(ordered)

    # Unsupported/custom expressions and non-parity-safe methods are intentionally scored by the
    # ordinary evaluator on the coordinator thread, preserving deterministic fallback ordering.
    results: list[tuple[float, dict[str, float]]] = []
    for index, (tree, native_result) in enumerate(zip(ordered, native_results, strict=True)):
        if native_result is not None and observer is not None:
            native_result = observer(tree, observations[index], native_result)
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
                complexity_penalty_mode=complexity_penalty_mode,
                complexity_penalty_value=complexity_penalty_value,
                max_nodes=max_nodes,
                min_names=min_names,
                observer=observer,
            )
        )
    return results
