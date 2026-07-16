"""P3-T4 — Probability of Backtest Overfitting via CSCV (Bailey et al., 2014).

Combinatorially Symmetric Cross-Validation splits the return history into ``S`` blocks and,
over every way of choosing ``S/2`` blocks as in-sample (the rest out-of-sample), asks: does
the in-sample-best strategy stay above the OOS median? PBO is the fraction of splits where it
does **not** — i.e. the probability that picking the in-sample winner gives you a below-median
strategy out of sample. PBO near 0.5+ means the selection is indistinguishable from luck.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import combinations, islice
from typing import TypedDict

import numpy as np
import pandas as pd

from alphalineage.validation.deflated_sharpe import sharpe_ratio


class PBOResult(TypedDict):
    pbo: float
    logits: list[float]
    n_splits: int


class ReportReturnSummary(TypedDict):
    """JSON-safe per-strategy statistics reused by report/session caches."""

    n_obs: int
    block_counts: list[int]
    block_sums: list[float]
    block_sum_squares: list[float]
    sharpe: float
    constant_value: float | None


def _sharpe_per_strategy(block: np.ndarray) -> np.ndarray:
    """Per-column Sharpe of a (rows x strategies) return block."""
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(std > 0, mean / std, 0.0)


def _block_layout(n_obs: int, n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Return CSCV block bounds and row counts, using the historical split policy."""
    s = min(n_blocks, n_obs)
    s -= s % 2
    if s < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    bounds = np.linspace(0, n_obs, s + 1).astype(int)
    return bounds, np.diff(bounds).astype(np.float64)


def _block_statistics(data: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compact per-block sums and squared sums for every strategy."""
    squared = data * data
    sums = np.stack([data[bounds[i] : bounds[i + 1]].sum(axis=0) for i in range(len(bounds) - 1)])
    sum_squares = np.stack(
        [squared[bounds[i] : bounds[i + 1]].sum(axis=0) for i in range(len(bounds) - 1)]
    )
    return sums, sum_squares


def _sharpe_from_statistics(
    counts: np.ndarray, sums: np.ndarray, sum_squares: np.ndarray
) -> np.ndarray:
    """Per-strategy sample Sharpe for one or more aggregate block selections."""
    counts = np.asarray(counts, dtype=np.float64)
    row_counts = counts[:, None] if counts.ndim == 1 else counts
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mean = sums / row_counts
        centered_ss = sum_squares - sums * sums / row_counts
        # Round-off can put a mathematically zero variance a few ulps below zero.
        variance = np.maximum(centered_ss, 0.0) / (row_counts - 1.0)
        std = np.sqrt(variance)
        return np.where((row_counts > 1.0) & (std > 0.0), mean / std, 0.0)


def _chunks(items: Iterator[tuple[int, ...]], size: int = 512) -> Iterator[list[tuple[int, ...]]]:
    while chunk := list(islice(items, size)):
        yield chunk


def _pbo_from_statistics(
    block_counts: np.ndarray,
    block_sums: np.ndarray,
    block_sum_squares: np.ndarray,
    constant_values: np.ndarray | None = None,
) -> PBOResult:
    """Compute CSCV from compact block statistics in canonical combination order."""
    s, n_strat = block_sums.shape
    if s < 2 or n_strat < 2:
        return {"pbo": float("nan"), "logits": [], "n_splits": 0}

    logits: list[float] = []
    combo_iter = combinations(range(s), s // 2)
    for combo_chunk in _chunks(combo_iter):
        membership = np.zeros((len(combo_chunk), s), dtype=np.float64)
        for row, selected in enumerate(combo_chunk):
            membership[row, list(selected)] = 1.0
        complement = 1.0 - membership

        is_counts = membership @ block_counts
        oos_counts = complement @ block_counts
        is_sharpes = _sharpe_from_statistics(
            is_counts, membership @ block_sums, membership @ block_sum_squares
        )
        oos_sharpes = _sharpe_from_statistics(
            oos_counts, complement @ block_sums, complement @ block_sum_squares
        )
        if constant_values is not None:
            constant_columns = np.flatnonzero(~np.isnan(constant_values))
            for count in np.unique(np.concatenate([is_counts, oos_counts])).astype(int):
                if count < 2:
                    constant_sharpes = np.zeros(n_strat, dtype=np.float64)
                else:
                    # NumPy's axis-0 reduction round-off depends on the matrix stride/column
                    # count. Reconstruct that shape so cached non-zero constants retain the
                    # sliced implementation's observable tie/selection behavior.
                    synthetic = np.zeros((count, n_strat), dtype=np.float64)
                    synthetic[:, constant_columns] = constant_values[constant_columns]
                    constant_sharpes = _sharpe_per_strategy(synthetic)
                for column in constant_columns:
                    is_sharpes[is_counts == count, column] = constant_sharpes[column]
                    oos_sharpes[oos_counts == count, column] = constant_sharpes[column]

        best = np.argmax(is_sharpes, axis=1)
        chosen = oos_sharpes[np.arange(len(combo_chunk)), best]
        # pandas Series.rank defaults to ascending average ranks.  Computing only the selected
        # strategy's rank avoids sorting or materializing a full rank matrix.
        less = (oos_sharpes < chosen[:, None]).sum(axis=1)
        equal = (oos_sharpes == chosen[:, None]).sum(axis=1)
        oos_rank = less + (equal + 1.0) / 2.0
        omega = np.clip(oos_rank / (n_strat + 1.0), 1e-6, 1.0 - 1e-6)
        logits.extend(np.log(omega / (1.0 - omega)).astype(float).tolist())

    arr = np.asarray(logits, dtype=np.float64)
    return {"pbo": float(np.mean(arr <= 0.0)), "logits": logits, "n_splits": len(logits)}


def summarize_report_returns(
    returns: pd.Series, index: pd.Index, n_blocks: int = 16
) -> ReportReturnSummary:
    """Return a compact JSON-safe Sharpe/CSCV summary for one strategy.

    NaN means no position and contributes zero PnL to CSCV, while Sharpe retains its historical
    drop-NaN behavior. Infinities are rejected because neither JSON nor block moment identities
    can preserve their sliced NumPy semantics.
    """
    aligned_index = pd.Index(index)
    aligned = returns.reindex(aligned_index)
    values = aligned.to_numpy(dtype="float64", na_value=np.nan)
    strategy_sharpe = sharpe_ratio(values)
    values = np.nan_to_num(values, nan=0.0)
    if not np.isfinite(values).all():
        raise ValueError("report returns must contain only finite values or NaN")

    bounds, block_counts = _block_layout(len(aligned_index), n_blocks)
    if len(bounds) == 0:
        sums = np.empty(0, dtype=np.float64)
        sum_squares = np.empty(0, dtype=np.float64)
    else:
        sums, sum_squares = _block_statistics(values[:, None], bounds)
        sums, sum_squares = sums[:, 0], sum_squares[:, 0]
    constant_value = float(values[0]) if len(values) and np.ptp(values) == 0.0 else None
    return {
        "n_obs": len(aligned_index),
        "block_counts": block_counts.astype(int).tolist(),
        "block_sums": sums.astype(float).tolist(),
        "block_sum_squares": sum_squares.astype(float).tolist(),
        "sharpe": float(strategy_sharpe),
        "constant_value": constant_value,
    }


def pbo_from_summaries(summaries: Iterable[ReportReturnSummary]) -> PBOResult:
    """Combine compatible cached return summaries into a PBO result."""
    materialized = list(summaries)
    if len(materialized) < 2:
        return {"pbo": float("nan"), "logits": [], "n_splits": 0}

    first = materialized[0]
    n_obs = int(first["n_obs"])
    block_counts = np.asarray(first["block_counts"], dtype=np.float64)
    expected_blocks = len(block_counts)
    if expected_blocks < 2 or int(block_counts.sum()) != n_obs:
        return {"pbo": float("nan"), "logits": [], "n_splits": 0}

    for summary in materialized:
        if (
            int(summary["n_obs"]) != n_obs
            or summary["block_counts"] != first["block_counts"]
            or len(summary["block_sums"]) != expected_blocks
            or len(summary["block_sum_squares"]) != expected_blocks
        ):
            raise ValueError("PBO summaries must use the same observation/block layout")
    block_sums = np.column_stack(
        [np.asarray(summary["block_sums"], dtype=np.float64) for summary in materialized]
    )
    block_sum_squares = np.column_stack(
        [np.asarray(summary["block_sum_squares"], dtype=np.float64) for summary in materialized]
    )
    if not np.isfinite(block_sums).all() or not np.isfinite(block_sum_squares).all():
        raise ValueError("PBO summary moments must be finite")
    constant_values = np.asarray(
        [
            np.nan if summary["constant_value"] is None else summary["constant_value"]
            for summary in materialized
        ],
        dtype=np.float64,
    )
    if not np.isfinite(constant_values[~np.isnan(constant_values)]).all():
        raise ValueError("PBO summary constant values must be finite or None")
    return _pbo_from_statistics(
        block_counts,
        block_sums,
        block_sum_squares,
        constant_values=constant_values,
    )


def _pbo_sliced(data: np.ndarray, bounds: np.ndarray) -> PBOResult:
    """Historical sliced CSCV path for non-finite or numerically degenerate inputs."""
    s, n_strat = len(bounds) - 1, data.shape[1]
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(s)]
    logits: list[float] = []
    for is_blocks in combinations(range(s), s // 2):
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(s) if b not in is_blocks])
        best = int(np.argmax(_sharpe_per_strategy(data[is_rows])))
        oos_rank = pd.Series(_sharpe_per_strategy(data[oos_rows])).rank().iloc[best]
        omega = min(max(oos_rank / (n_strat + 1), 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))
    arr = np.asarray(logits)
    return {"pbo": float(np.mean(arr <= 0.0)), "logits": logits, "n_splits": len(logits)}


def pbo_from_returns(
    returns: Iterable[pd.Series],
    index: pd.Index,
    n_blocks: int = 16,
    *,
    nonfinite_fallback: Callable[[], pd.DataFrame] | None = None,
    require_full_index: bool = False,
) -> PBOResult:
    """Stream strategy returns into CSCV block statistics without a ``T x N`` DataFrame.

    Series are aligned to ``index``; absent/NaN observations mean no position and contribute
    zero PnL, exactly as in :func:`pbo`.
    """
    aligned_index = pd.Index(index)
    bounds, block_counts = _block_layout(len(aligned_index), n_blocks)
    if len(bounds) == 0:
        # Consume the iterator because callers may compute other streaming statistics alongside
        # PBO (as the reporting pipeline does).
        for _series in returns:
            pass
        return {"pbo": float("nan"), "logits": [], "n_splits": 0}

    sum_columns: list[np.ndarray] = []
    square_columns: list[np.ndarray] = []
    constant_values: list[float] = []
    needs_materialized_index_union = False
    for series in returns:
        needs_materialized_index_union = needs_materialized_index_union or bool(
            require_full_index and not series.index.equals(aligned_index)
        )
        values = series.reindex(aligned_index).to_numpy(dtype="float64", na_value=np.nan)
        values = np.nan_to_num(values, nan=0.0)
        squared = values * values
        constant_values.append(
            float(values[0])
            if len(values) and np.isfinite(values).all() and np.ptp(values) == 0.0
            else np.nan
        )
        column_sums = np.array(
            [values[bounds[i] : bounds[i + 1]].sum() for i in range(len(bounds) - 1)]
        )
        column_squares = np.array(
            [squared[bounds[i] : bounds[i + 1]].sum() for i in range(len(bounds) - 1)]
        )
        sum_columns.append(column_sums)
        square_columns.append(column_squares)

    if len(sum_columns) < 2:
        return {"pbo": float("nan"), "logits": [], "n_splits": 0}
    block_sums = np.stack(sum_columns, axis=1)
    block_sum_squares = np.stack(square_columns, axis=1)
    if (
        not np.isfinite(block_sums).all()
        or not np.isfinite(block_sum_squares).all()
        or needs_materialized_index_union
    ):
        # Preserve the sliced reference for infinities or a non-canonical index union. Normal
        # return streams stay on the compact-statistics path.
        if nonfinite_fallback is None:
            raise ValueError("these returns require a materialized PBO fallback")
        return pbo(nonfinite_fallback(), n_blocks=n_blocks)
    return _pbo_from_statistics(
        block_counts,
        block_sums,
        block_sum_squares,
        constant_values=np.asarray(constant_values, dtype=np.float64),
    )


def pbo(returns_matrix: pd.DataFrame, n_blocks: int = 16) -> PBOResult:
    """Probability of backtest overfitting for a ``T x N`` strategy-returns matrix."""
    data = returns_matrix.to_numpy(dtype="float64")
    data = np.nan_to_num(data, nan=0.0)  # a NaN return = no position that day = 0 PnL
    n_obs, n_strat = data.shape

    bounds, block_counts = _block_layout(n_obs, n_blocks)
    if len(bounds) == 0 or n_strat < 2:
        return {"pbo": float("nan"), "logits": [], "n_splits": 0}
    if not np.isfinite(data).all():
        # Block identities are undefined for infinities; retain the sliced reference.
        return _pbo_sliced(data, bounds)

    block_sums, block_sum_squares = _block_statistics(data, bounds)
    constant_values = np.where(np.ptp(data, axis=0) == 0.0, data[0], np.nan)
    return _pbo_from_statistics(
        block_counts, block_sums, block_sum_squares, constant_values=constant_values
    )
