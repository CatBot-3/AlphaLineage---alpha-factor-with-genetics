"""P3-T3 — the Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

A high Sharpe means little after many trials. The Probabilistic Sharpe Ratio (PSR) is the
probability the true Sharpe exceeds a benchmark, given sample length and non-normality. The
Deflated Sharpe Ratio (DSR) sets that benchmark to the Sharpe you'd expect as the *maximum*
of ``N`` independent trials — so DSR is the probability the selected strategy is genuinely
better than the best of ``N`` random ones. DSR is a probability in [0, 1]; > 0.95 is
"significant".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: pd.Series | np.ndarray) -> float:
    """Per-period Sharpe ratio (mean / std, ddof=1). 0 if undefined."""
    r = pd.Series(returns, dtype="float64").dropna()
    if len(r) < 2:
        return 0.0
    std = r.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(r.mean() / std)


def _moments(returns: pd.Series | np.ndarray) -> tuple[float, int, float, float]:
    r = pd.Series(returns, dtype="float64").dropna()
    n = len(r)
    if n < 3:
        return sharpe_ratio(r), n, 0.0, 3.0
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # pandas reports *excess* kurtosis; make it non-excess
    return sharpe_ratio(r), n, skew, kurt


def probabilistic_sharpe_ratio(
    sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0
) -> float:
    """P(true Sharpe > ``sr_benchmark``) given the estimate ``sr`` over ``n`` observations."""
    if n < 2:
        return 0.5
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """The Sharpe expected as the maximum of ``n_trials`` trials with Sharpe variance ``var_sr``."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    gamma = EULER_MASCHERONI
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    return float(np.sqrt(var_sr) * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(returns: pd.Series | np.ndarray, n_trials: int, var_sr: float) -> float:
    """DSR = PSR of ``returns`` against the expected-max-of-``n_trials`` benchmark."""
    sr, n, skew, kurt = _moments(returns)
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)
