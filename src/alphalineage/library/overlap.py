"""Overlap check: is a discovered factor new, or a known factor in disguise?

A GP search often rediscovers a published anomaly or a technical indicator with a few cosmetic
operators around it. Its IC is real, but it is not new information: a portfolio that already
holds the known factor gains little. This module measures, on the *training window only*:

* **Rank correlation** with every reference factor: the mean over dates of the cross-sectional
  Spearman correlation between the two signals. The sign is kept (a strongly negative value is
  the same idea inverted), and ranking uses its absolute value.
* **Unique IC**: on each date, the candidate's cross-sectional ranks are regressed (OLS) on the
  ranks of its most correlated reference factors. The candidate's rank IC splits exactly into the
  part carried by the fitted known factors and the part carried by the residual; the residual's
  part is the edge that is plausibly new. ``unique_share = unique IC / IC`` (see
  :func:`ic_decomposition`).

Nothing here reads validation or holdout rows: callers pass a panel truncated at the frozen
training boundary, so labels near that boundary are missing rather than borrowed from the
future. The check selects nothing and spends no holdout read.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from alphalineage.core.fitness import daily_ic
from alphalineage.core.tree import Node

OVERLAP_VERSION = 1
#: |mean rank correlation| at or above which a reference factor is a near-duplicate.
DUPLICATE_THRESHOLD = 0.7
#: |mean rank correlation| at or above which a reference factor counts as related and is used to
#: explain the candidate.
RELATED_THRESHOLD = 0.3
#: Two reference factors this correlated with each other count as one when explaining a candidate.
REDUNDANT_THRESHOLD = 0.9
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class ReferenceFactor:
    """A known factor with fixed parameters, expressed as an evaluable tree."""

    key: str
    name: str
    display_name: str
    group: str
    family: str
    tree: Node


def _default_children(inputs: Sequence[dict[str, Any]]) -> tuple[Node, ...] | None:
    children: list[Node] = []
    for item in inputs:
        kind = str(item.get("type") or "")
        default = item.get("default")
        if default is None or isinstance(default, bool):
            return None
        if kind == "window":
            children.append(Node("window", value=int(default)))
        elif kind == "scalar":
            children.append(Node("const", value=float(default)))
        else:
            # Series inputs need an explicit binding; there is no neutral default to assume.
            return None
    return tuple(children)


def references_from_formulas(formulas: Iterable[dict[str, Any]]) -> list[ReferenceFactor]:
    """Active formulas whose every input has a numeric default, called with those defaults.

    ``formulas`` are formula-store entries (latest revision per family). Managed starter
    formulas become the ``catalog`` group; the user's own formulas become ``your_formulas``.
    """
    references: list[ReferenceFactor] = []
    for formula in formulas:
        if formula.get("status", "active") != "active":
            continue
        if formula.get("registered") is False or formula.get("error"):
            continue
        if str(formula.get("out_type")) not in {"series", "signal"}:
            continue
        children = _default_children(list(formula.get("inputs") or []))
        if children is None:
            continue
        runtime = str(formula.get("runtime_name") or formula["name"])
        catalog = formula.get("origin") == "catalog_formula"
        references.append(
            ReferenceFactor(
                key=f"formula:{formula['name']}",
                name=str(formula["name"]),
                display_name=str(formula.get("display_name") or formula["name"]),
                group="catalog" if catalog else "your_formulas",
                family=str(
                    formula.get("family")
                    or formula.get("category")
                    or ("catalog" if catalog else "custom")
                ),
                tree=Node(runtime, children),
            )
        )
    return references


def _rank(frame: pd.DataFrame) -> np.ndarray:
    return frame.rank(axis=1, pct=True).to_numpy(dtype="float64", na_value=np.nan)


def residualize(
    candidate: pd.DataFrame,
    explanatory: Sequence[pd.DataFrame],
    *,
    min_names: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-date OLS residual of candidate ranks on explanatory ranks, plus each date's R²."""
    y_all = _rank(candidate)
    index, columns = candidate.index, candidate.columns
    residual = np.full(y_all.shape, np.nan)
    r_squared = np.full(len(index), np.nan)
    if not explanatory:
        return candidate.copy(), pd.Series(0.0, index=index)
    x_all = np.stack(
        [_rank(frame.reindex(index=index, columns=columns)) for frame in explanatory], axis=2
    )
    k = x_all.shape[2]
    for row in range(len(index)):
        y = y_all[row]
        x = x_all[row]
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        count = int(mask.sum())
        if count < max(min_names, k + 3):
            continue
        yy = y[mask] - y[mask].mean()
        xx = x[mask] - x[mask].mean(axis=0)
        beta, *_ = np.linalg.lstsq(xx, yy, rcond=None)
        resid = yy - xx @ beta
        residual[row, mask] = resid
        total = float(yy @ yy)
        r_squared[row] = 1.0 - float(resid @ resid) / total if total > 0 else np.nan
    return (
        pd.DataFrame(residual, index=index, columns=columns),
        pd.Series(r_squared, index=index),
    )


def ic_decomposition(
    candidate: pd.DataFrame,
    explanatory: Sequence[pd.DataFrame],
    forward: pd.DataFrame,
    *,
    min_names: int,
) -> pd.DataFrame:
    """Split each date's rank IC into the part explained by known factors and the unique rest.

    On a date, with ``y`` the candidate's centered ranks, ``r`` the forward return's centered
    ranks and ``y = X b + e`` the OLS fit on the known factors' ranks (all over the names every
    input covers)::

        IC = cov(y, r) / (sd(y) sd(r)) = cov(Xb, r) / (sd(y) sd(r)) + cov(e, r) / (sd(y) sd(r))

    The second term is the **unique IC**. It is scaled by the candidate's own dispersion, not
    the residual's, so a near-duplicate whose residual is tiny noise gets a unique IC near zero
    instead of a re-ranked residual that can look arbitrarily predictive. The two parts add up
    to the date's IC exactly.
    """
    index, columns = candidate.index, candidate.columns
    frames = [candidate, forward.reindex(index=index, columns=columns)] + [
        frame.reindex(index=index, columns=columns) for frame in explanatory
    ]
    values = np.stack([frame.to_numpy(dtype="float64", na_value=np.nan) for frame in frames], 2)
    k = len(explanatory)
    out = np.full((len(index), 3), np.nan)
    for row in range(len(index)):
        block = values[row]
        mask = np.all(np.isfinite(block), axis=1)
        count = int(mask.sum())
        if count < max(min_names, k + 3, 3):
            continue
        ranks = rankdata(block[mask], method="average", axis=0)
        centered = ranks - ranks.mean(axis=0)
        y, r = centered[:, 0], centered[:, 1]
        scale = math.sqrt(float(y @ y) * float(r @ r))
        if scale == 0.0:
            continue
        total_ic = float(y @ r) / scale
        if k:
            x = centered[:, 2:]
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            e = y - x @ beta
            unique_ic = float(e @ r) / scale
            r2 = 1.0 - float(e @ e) / float(y @ y)
        else:
            unique_ic, r2 = total_ic, 0.0
        out[row] = (total_ic, unique_ic, r2)
    return pd.DataFrame(out, index=index, columns=["ic", "unique_ic", "r_squared"])


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(series: pd.Series) -> float | None:
    clean = series.dropna()
    return _finite(clean.mean()) if len(clean) else None


def _t_stat(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 3:
        return None
    std = clean.std()
    if not std or not math.isfinite(std):
        return None
    return _finite(clean.mean() / std * math.sqrt(len(clean)))


def verdict(max_abs_corr: float | None, unique_share: float | None) -> str:
    """Plain-language classification used by the UI badge."""
    if max_abs_corr is None:
        return "unmeasured"
    if max_abs_corr >= DUPLICATE_THRESHOLD:
        return "near_duplicate"
    if unique_share is not None and unique_share < 0.5:
        return "mostly_explained"
    if max_abs_corr >= RELATED_THRESHOLD:
        return "related"
    return "novel"


def overlap_report(
    candidate: pd.DataFrame,
    references: Sequence[tuple[ReferenceFactor, pd.DataFrame | None, str | None]],
    forward: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    min_names: int = 5,
    top_k: int = DEFAULT_TOP_K,
    related_threshold: float = RELATED_THRESHOLD,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Compare an evaluated candidate with evaluated reference factors over ``dates``.

    ``references`` holds ``(reference, evaluated frame or None, error or None)``. Every metric is
    restricted to ``dates``; callers pass training dates on a training-truncated panel.
    """
    window = pd.DatetimeIndex(dates)
    cand = candidate.reindex(window)
    fwd = forward.reindex(index=window, columns=cand.columns)
    candidate_ic_series = daily_ic(cand, fwd, "spearman", min_names=min_names)
    candidate_ic = _mean(candidate_ic_series)

    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    total = len(references)
    for done, (reference, frame, error) in enumerate(references, start=1):
        row: dict[str, Any] = {
            "key": reference.key,
            "name": reference.name,
            "display_name": reference.display_name,
            "group": reference.group,
            "family": reference.family,
        }
        if frame is None:
            row.update({"status": "unavailable", "error": error or "did not evaluate to a panel"})
        else:
            aligned = frame.reindex(index=window, columns=cand.columns)
            corr = daily_ic(cand, aligned, "spearman", min_names=min_names)
            valid = int(corr.notna().sum())
            mean_corr = _mean(corr)
            if valid == 0 or mean_corr is None:
                row.update(
                    {"status": "no_overlap", "error": "no dates with a shared cross-section"}
                )
            else:
                own_ic = daily_ic(aligned, fwd, "spearman", min_names=min_names)
                row.update(
                    {
                        "status": "ok",
                        "mean_rank_corr": mean_corr,
                        "abs_mean_rank_corr": abs(mean_corr),
                        "share_dates_abs_corr_above_half": _finite(
                            (corr.abs() > 0.5).sum() / valid
                        ),
                        "dates": valid,
                        "reference_ic": _mean(own_ic),
                        "reference_ic_t": _t_stat(own_ic),
                    }
                )
                frames[reference.key] = aligned
        rows.append(row)
        if progress is not None:
            progress(done, total)

    measured = [row for row in rows if row["status"] == "ok"]
    measured.sort(key=lambda row: row["abs_mean_rank_corr"], reverse=True)
    unmeasured = [row for row in rows if row["status"] != "ok"]
    # Pick the most similar known factors greedily, skipping any that is itself a near-copy of
    # one already picked (SMA vs Bollinger middle band): a redundant regressor explains nothing
    # new and would crowd out a genuinely different known factor.
    explanatory_rows: list[dict[str, Any]] = []
    for row in measured:
        if len(explanatory_rows) >= max(0, int(top_k)):
            break
        if row["abs_mean_rank_corr"] < related_threshold:
            break
        redundant = None
        for chosen in explanatory_rows:
            mutual = _mean(
                daily_ic(frames[row["key"]], frames[chosen["key"]], "spearman", min_names=min_names)
            )
            if mutual is not None and abs(mutual) >= REDUNDANT_THRESHOLD:
                redundant = chosen["key"]
                break
        if redundant is None:
            explanatory_rows.append(row)
        else:
            row["redundant_with"] = redundant
    decomposition = ic_decomposition(
        cand,
        [frames[row["key"]] for row in explanatory_rows],
        fwd,
        min_names=min_names,
    )
    decomposed_ic = _mean(decomposition["ic"])
    unique_ic = _mean(decomposition["unique_ic"])
    unique_share = (
        _finite(unique_ic * math.copysign(1.0, decomposed_ic) / abs(decomposed_ic))
        if unique_ic is not None and decomposed_ic not in (None, 0.0)
        else None
    )
    max_abs = measured[0]["abs_mean_rank_corr"] if measured else None
    return {
        "overlap_version": OVERLAP_VERSION,
        "window": {
            "start": window.min().date().isoformat() if len(window) else None,
            "end": window.max().date().isoformat() if len(window) else None,
            "dates": len(window),
            "label": "training window",
        },
        "thresholds": {
            "related": related_threshold,
            "near_duplicate": DUPLICATE_THRESHOLD,
            "top_k": int(top_k),
        },
        "candidate": {
            "ic": candidate_ic,
            "ic_t": _t_stat(candidate_ic_series),
        },
        "residual": {
            # ``ic`` is the unique IC: the additive share of the candidate's IC that the known
            # factors cannot account for, measured over names every input covers.
            "ic": unique_ic,
            "ic_t": _t_stat(decomposition["unique_ic"]),
            "explained_ic": _finite(decomposed_ic - unique_ic)
            if decomposed_ic is not None and unique_ic is not None
            else None,
            "decomposed_ic": decomposed_ic,
            "unique_share": unique_share,
            "mean_r_squared": _mean(decomposition["r_squared"]) if explanatory_rows else 0.0,
            "explained_by": [row["key"] for row in explanatory_rows],
        },
        "max_abs_rank_corr": max_abs,
        "verdict": verdict(max_abs, unique_share),
        "references": measured + unmeasured,
        "reference_count": len(rows),
        "measured_count": len(measured),
    }
