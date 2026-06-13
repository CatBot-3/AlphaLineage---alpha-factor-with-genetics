"""P4-T1 — rank -> long/short portfolio weights (pluggable schemes) + neutralization.

A factor becomes a portfolio via a swappable :class:`WeightingScheme`. Two ship built in,
both **dollar-neutral** (Σw ≈ 0) and **unit-gross** (Σ|w| = 1): quantile long/short and
rank-proportional. The same factor can be run through several schemes for a side-by-side
comparison, so the number of schemes is another search axis (it feeds the deflation's trials).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class WeightingScheme(Protocol):
    """Maps a factor (date x symbol) to portfolio weights (date x symbol)."""

    name: str

    def weights(self, factor: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class QuantileLongShort:
    """Long the top ``quantile`` of names, short the bottom, equal-weight within each side."""

    quantile: float = 0.2

    @property
    def name(self) -> str:
        return "quantile_ls"

    def weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        ranks = factor.rank(axis=1, pct=True)
        longs = ranks >= (1.0 - self.quantile)
        shorts = ranks <= self.quantile
        n_long = longs.sum(axis=1).replace(0, np.nan)
        n_short = shorts.sum(axis=1).replace(0, np.nan)
        w = longs.div(n_long, axis=0) * 0.5 - shorts.div(n_short, axis=0) * 0.5
        return w.where(factor.notna(), 0.0).fillna(0.0)


@dataclass
class RankProportional:
    """Weight every name by its cross-sectionally demeaned rank (dollar-neutral, unit-gross)."""

    @property
    def name(self) -> str:
        return "rank_proportional"

    def weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        ranks = factor.rank(axis=1)
        centered = ranks.sub(ranks.mean(axis=1), axis=0)
        gross = centered.abs().sum(axis=1).replace(0.0, np.nan)
        w = centered.div(gross, axis=0)
        return w.where(factor.notna(), 0.0).fillna(0.0)


SCHEMES: dict[str, type] = {
    "quantile_ls": QuantileLongShort,
    "rank_proportional": RankProportional,
}


def get_scheme(name: str, **kwargs: object) -> WeightingScheme:
    try:
        return SCHEMES[name](**kwargs)  # type: ignore[no-any-return]
    except KeyError as exc:
        raise KeyError(f"unknown weighting scheme {name!r}") from exc


# --- neutralization --------------------------------------------------------------
def _group_demean(factor: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract each group's per-date mean (e.g. sector-neutralize)."""
    out = factor.copy()
    labels = pd.Series(groups)
    for label in labels.unique():
        cols = [c for c in labels.index[labels == label] if c in factor.columns]
        if cols:
            block = factor[cols]
            out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out


def _regress_out(factor: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional residual of ``factor`` after removing the ``size`` exposure per date."""
    fd = factor.sub(factor.mean(axis=1), axis=0)
    sd = size.sub(size.mean(axis=1), axis=0)
    beta = (fd * sd).sum(axis=1) / (sd * sd).sum(axis=1).replace(0.0, np.nan)
    return factor.sub(sd.mul(beta, axis=0))


def neutralize(
    factor: pd.DataFrame,
    *,
    groups: pd.Series | None = None,
    size: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sector- and/or size-neutralize a factor.

    ``groups`` maps symbol -> label (sector). Real sector labels need a data source
    (``TODO(human)``); ``size`` is any date x symbol exposure (e.g. log dollar volume).
    """
    out = factor
    if groups is not None:
        out = _group_demean(out, groups)
    if size is not None:
        out = _regress_out(out, size)
    return out
