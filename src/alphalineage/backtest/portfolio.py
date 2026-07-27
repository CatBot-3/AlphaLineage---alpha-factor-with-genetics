"""P4-T1 — rank -> long/short portfolio weights (pluggable schemes) + neutralization.

A factor becomes a portfolio via a swappable :class:`WeightingScheme`. Two ship built in,
both **dollar-neutral** (Σw ≈ 0) and **unit-gross** (Σ|w| = 1): quantile long/short and
rank-proportional. The same factor can be run through several schemes for a side-by-side
comparison, so the number of schemes is another search axis (it feeds the deflation's trials).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

PORTFOLIO_SCHEMA_VERSION = 2


@runtime_checkable
class WeightingScheme(Protocol):
    """Maps a factor (date x symbol) to portfolio weights (date x symbol)."""

    @property
    def name(self) -> str: ...

    def weights(self, factor: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class QuantileLongShort:
    """Long the top ``quantile`` of names, short the bottom, equal-weight within each side."""

    quantile: float = 0.2

    def __post_init__(self) -> None:
        if (
            isinstance(self.quantile, bool)
            or not np.isfinite(float(self.quantile))
            or not 0.0 < float(self.quantile) < 0.5
        ):
            raise ValueError("quantile must be finite and strictly between 0 and 0.5")
        self.quantile = float(self.quantile)

    @property
    def name(self) -> str:
        return "quantile_ls"

    def weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        """Return deterministic, tie-aware long/short weights.

        Percentile ranks with average ties can put every member of a binary
        cross-section between both cutoffs, producing an all-cash portfolio.
        Exact order-statistic boundaries include the complete boundary tie group
        without a symbol-order tie-break. Overlapping boundaries and fully
        constant rows remain cash.
        """
        values = factor.to_numpy(dtype="float64")
        finite_mask = np.isfinite(values)
        counts = finite_mask.sum(axis=1)
        # Replace missing/nonfinite cells with +inf so the first ``n`` entries of
        # every partitioned row are exactly the finite observations. Grouping by
        # ``n`` keeps the order-statistic calculation vectorized for normal panels.
        sortable = np.where(finite_mask, values, np.inf)
        lower = np.full(len(factor), np.nan, dtype="float64")
        upper = np.full(len(factor), np.nan, dtype="float64")
        for count in np.unique(counts[counts >= 2]):
            row_indices = np.flatnonzero(counts == count)
            tail_size = max(1, int(np.floor(float(count) * self.quantile)))
            partitioned = np.partition(
                sortable[row_indices],
                (tail_size - 1, int(count) - tail_size),
                axis=1,
            )
            lower[row_indices] = partitioned[:, tail_size - 1]
            upper[row_indices] = partitioned[:, int(count) - tail_size]
        separated = lower < upper
        shorts = pd.DataFrame(
            finite_mask & (values <= lower[:, None]) & separated[:, None],
            index=factor.index,
            columns=factor.columns,
        )
        longs = pd.DataFrame(
            finite_mask & (values >= upper[:, None]) & separated[:, None],
            index=factor.index,
            columns=factor.columns,
        )
        n_long = longs.sum(axis=1).replace(0, np.nan)
        n_short = shorts.sum(axis=1).replace(0, np.nan)
        w = longs.div(n_long, axis=0) * 0.5 - shorts.div(n_short, axis=0) * 0.5
        return w.fillna(0.0)


@dataclass
class RankProportional:
    """Weight every name by its cross-sectionally demeaned rank (dollar-neutral, unit-gross)."""

    @property
    def name(self) -> str:
        return "rank_proportional"

    def weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        finite = factor.where(np.isfinite(factor))
        ranks = finite.rank(axis=1)
        centered = ranks.sub(ranks.mean(axis=1), axis=0)
        gross = centered.abs().sum(axis=1).replace(0.0, np.nan)
        w = centered.div(gross, axis=0)
        return w.where(finite.notna(), 0.0).fillna(0.0)


SCHEMES: dict[str, type] = {
    "quantile_ls": QuantileLongShort,
    "rank_proportional": RankProportional,
}


def get_scheme(name: str, **kwargs: object) -> WeightingScheme:
    try:
        return SCHEMES[name](**kwargs)  # type: ignore[no-any-return]
    except KeyError as exc:
        raise KeyError(f"unknown weighting scheme {name!r}") from exc


@dataclass(frozen=True)
class PortfolioStrategySpec:
    """Immutable weighting strategy used by comparisons and holdout plans.

    Transaction costs deliberately do not live here. A training round freezes
    those once, while a strategy comparison varies only portfolio construction.
    """

    id: str
    scheme: str
    quantile: float | None = None

    def __post_init__(self) -> None:
        strategy_id = self.id.strip()
        if not strategy_id or len(strategy_id) > 80:
            raise ValueError("strategy id must contain 1 through 80 characters")
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown weighting scheme {self.scheme!r}")
        if self.scheme == "quantile_ls":
            value = 0.2 if self.quantile is None else float(self.quantile)
            QuantileLongShort(value)  # central validation
            object.__setattr__(self, "quantile", value)
        elif self.quantile is not None:
            raise ValueError("quantile is only valid for the quantile_ls scheme")
        object.__setattr__(self, "id", strategy_id)

    def weighting_scheme(self) -> WeightingScheme:
        if self.scheme == "quantile_ls":
            assert self.quantile is not None
            return QuantileLongShort(float(self.quantile))
        return get_scheme(self.scheme)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "scheme": self.scheme}
        if self.quantile is not None:
            payload["quantile"] = self.quantile
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PortfolioStrategySpec:
        return cls(
            id=str(payload.get("id") or ""),
            scheme=str(payload.get("scheme") or ""),
            quantile=(
                None
                if payload.get("quantile") is None
                else float(payload["quantile"])
            ),
        )


def default_strategy_spec() -> PortfolioStrategySpec:
    """Backward-compatible finalization plan used when no pinned plan is supplied."""
    return PortfolioStrategySpec("quantile-20", "quantile_ls", 0.2)


def validate_portfolio_weights(
    weights: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Enforce the weighting-scheme contract on every active row.

    Cash rows are allowed. Every non-cash row must contain only finite weights,
    be dollar neutral, and have unit gross exposure.
    """
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weighting scheme must return a pandas DataFrame")
    try:
        values = weights.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("portfolio weights must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("portfolio weights must be finite")
    normalized = pd.DataFrame(values, index=weights.index, columns=weights.columns)
    gross = normalized.abs().sum(axis=1)
    active = gross.gt(tolerance)
    net = normalized.sum(axis=1)
    nonneutral = active & net.abs().gt(tolerance)
    if bool(nonneutral.any()):
        first = nonneutral[nonneutral].index[0]
        raise ValueError(f"active portfolio weights must be dollar neutral (row {first})")
    nonunit = active & gross.sub(1.0).abs().gt(tolerance)
    if bool(nonunit.any()):
        first = nonunit[nonunit].index[0]
        raise ValueError(f"active portfolio weights must have unit gross exposure (row {first})")
    return normalized


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
