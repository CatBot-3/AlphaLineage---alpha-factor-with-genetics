"""P3-T4 — Probability of Backtest Overfitting via CSCV (Bailey et al., 2014).

Combinatorially Symmetric Cross-Validation splits the return history into ``S`` blocks and,
over every way of choosing ``S/2`` blocks as in-sample (the rest out-of-sample), asks: does
the in-sample-best strategy stay above the OOS median? PBO is the fraction of splits where it
does **not** — i.e. the probability that picking the in-sample winner gives you a below-median
strategy out of sample. PBO near 0.5+ means the selection is indistinguishable from luck.
"""

from __future__ import annotations

from itertools import combinations
from typing import TypedDict

import numpy as np
import pandas as pd


class PBOResult(TypedDict):
    pbo: float
    logits: list[float]
    n_splits: int


def _sharpe_per_strategy(block: np.ndarray) -> np.ndarray:
    """Per-column Sharpe of a (rows x strategies) return block."""
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(std > 0, mean / std, 0.0)


def pbo(returns_matrix: pd.DataFrame, n_blocks: int = 16) -> PBOResult:
    """Probability of backtest overfitting for a ``T x N`` strategy-returns matrix."""
    data = returns_matrix.to_numpy(dtype="float64")
    data = np.nan_to_num(data, nan=0.0)  # a NaN return = no position that day = 0 PnL
    n_obs, n_strat = data.shape

    s = min(n_blocks, n_obs)
    s -= s % 2  # CSCV needs an even number of blocks
    if s < 2 or n_strat < 2:
        return {"pbo": float("nan"), "logits": [], "n_splits": 0}

    bounds = np.linspace(0, n_obs, s + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(s)]

    logits: list[float] = []
    for is_blocks in combinations(range(s), s // 2):
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(s) if b not in is_blocks])

        best = int(np.argmax(_sharpe_per_strategy(data[is_rows])))
        oos_rank = pd.Series(_sharpe_per_strategy(data[oos_rows])).rank().iloc[best]
        omega = min(max(oos_rank / (n_strat + 1), 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))

    arr = np.array(logits)
    return {"pbo": float(np.mean(arr <= 0.0)), "logits": logits, "n_splits": len(logits)}
