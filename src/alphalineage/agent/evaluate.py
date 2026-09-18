"""P10-T2 - scoring an agent's candidate on an inner split carved out of the training window.

The design problem this solves: an agent needs a generalization signal, or it will hill-climb
straight into a train-overfit expression. But if it sees the real validation IC on every
iteration, N sequential proposals become N *informed* selections against the validation window —
a far more efficient overfitting process per trial than the GP's single champion selection, and
it silently spends the evidence the session depends on.

So the training window is split again, internally. The agent gets an inner-train / inner-holdout
pair with the same purge-and-embargo discipline the real split uses, and the real validation
window is never touched by the loop. Two consequences follow, and both are stated in the UI
rather than left implicit:

* inner-holdout IC is **not** validation IC. It is a within-train generalization proxy computed
  from data the session already considers in-sample.
* the real validation read still happens exactly once, later, when a human promotes a candidate
  through the existing session pipeline, and is counted there.

Nothing here takes a date, a universe, or a split argument. The window comes from
``AgentContext``, whose panel is already truncated at ``train_end``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alphalineage.agent.context import AgentContext
from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.engine import backtest
from alphalineage.backtest.portfolio import get_scheme
from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import forward_returns, label_span, score_tree
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node

#: Fraction of the training window held out as the inner generalization check.
INNER_HOLDOUT_FRACTION = 0.25
#: Minimum dates each side needs before a number from it means anything.
MIN_INNER_DATES = 40


@dataclass(frozen=True)
class InnerSplit:
    """The agent's private train/holdout pair, entirely inside the session's training window."""

    train: pd.DatetimeIndex
    holdout: pd.DatetimeIndex
    embargo: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_train_start": self.train.min().date().isoformat(),
            "inner_train_end": self.train.max().date().isoformat(),
            "inner_train_dates": len(self.train),
            "inner_holdout_start": self.holdout.min().date().isoformat(),
            "inner_holdout_end": self.holdout.max().date().isoformat(),
            "inner_holdout_dates": len(self.holdout),
            "embargo": self.embargo,
        }


def inner_split(
    dates: pd.DatetimeIndex,
    *,
    horizon: int,
    embargo: int,
    holdout_fraction: float = INNER_HOLDOUT_FRACTION,
) -> InnerSplit:
    """Carve a chronological inner holdout with an embargo gap wide enough for the label window.

    Chronological, never shuffled, and the gap is at least ``horizon`` bars so a forward-return
    label computed on the last inner-train date cannot reach into the inner holdout.
    """
    idx = pd.DatetimeIndex(dates)
    gap = max(int(embargo), int(horizon))
    n = len(idx)
    n_holdout = int(n * holdout_fraction)
    n_train = n - n_holdout - gap
    if n_train < MIN_INNER_DATES or n_holdout < MIN_INNER_DATES:
        raise ValueError(
            f"training window is too short to carve an inner split: {n} dates would leave "
            f"{max(n_train, 0)} inner-train and {n_holdout} inner-holdout"
        )
    return InnerSplit(train=idx[:n_train], holdout=idx[n_train + gap :], embargo=gap)


def _slice(panel: Panel, dates: pd.DatetimeIndex) -> Panel:
    """A panel restricted to ``dates`` for reporting, keeping earlier rows as warm-up.

    Lookback operators need history before the first reported date, so the returned panel keeps
    everything up to the window's end; only the *metrics* are computed on the window itself.
    """
    idx = pd.DatetimeIndex(panel.dates)
    keep = idx <= dates.max()
    return Panel({name: frame.loc[keep] for name, frame in panel.fields.items()})


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class CandidateScore:
    """What the agent is told back about one candidate. Split labels are never omitted."""

    expression: str
    inner_train: dict[str, float | None] = field(default_factory=dict)
    inner_holdout: dict[str, float | None] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    feasible: bool = True
    issues: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "inner_train": self.inner_train,
            "inner_holdout": self.inner_holdout,
            "backtest_inner_holdout": self.backtest,
            "split": self.split,
            "feasible": self.feasible,
            "issues": list(self.issues),
            "note": self.note,
            "measured_on": (
                "An inner holdout carved out of the TRAINING window. This is not the session's "
                "validation IC and it is not out-of-sample evidence. The real validation read "
                "happens once, when a human promotes this candidate."
            ),
        }

    @property
    def generalization_gap(self) -> float | None:
        """Inner-train IC minus inner-holdout IC: how much of the edge did not survive."""
        train = self.inner_train.get("oriented_ic")
        holdout = self.inner_holdout.get("oriented_ic")
        if train is None or holdout is None:
            return None
        return round(train - holdout, 6)


def _ic_block(metrics: dict[str, float]) -> dict[str, float | None]:
    keys = (
        "oriented_ic",
        "signed_ic",
        "ic_ir",
        "sign_consistency",
        "valid_dates",
        "avg_active_names",
        "min_active_names",
    )
    return {key: _finite(metrics.get(key)) for key in keys}


def score_candidate(
    tree: Node,
    context: AgentContext,
    *,
    note: str = "",
    expression: str = "",
    run_backtest: bool = True,
) -> CandidateScore:
    """Score a candidate on the inner split. The only place agent evaluation touches data.

    Asserts the split guard on every window it reads, so a future refactor that widens the panel
    fails loudly here rather than quietly spending evidence.
    """
    context.guard.assert_within(context.panel.dates, what="agent candidate scoring")
    split = inner_split(
        context.train_dates,
        horizon=label_span(context.horizon, context.execution),
        embargo=int(context.boundaries.embargo),
    )
    context.guard.assert_within(split.holdout, what="agent inner holdout")

    scored: dict[str, dict[str, float]] = {}
    for name, window in (("inner_train", split.train), ("inner_holdout", split.holdout)):
        window_panel = _slice(context.panel, window)
        context.guard.assert_within(window_panel.dates, what=f"agent {name} panel")
        fwd = forward_returns(window_panel, context.horizon, context.execution).reindex(window)
        _, metrics = score_tree(
            tree,
            window_panel,
            fwd,
            method=context.ic_method,
            absolute=True,
            parsimony=float(context.config.get("parsimony", 0.0) or 0.0),
            complexity_penalty_mode=str(context.config.get("complexity_penalty_mode", "per_node")),
            complexity_penalty_value=context.config.get("complexity_penalty_value"),
            max_nodes=context.config.get("max_nodes"),
            min_names=int(context.config.get("min_names", 5) or 5),
        )
        scored[name] = metrics

    result = CandidateScore(
        expression=expression,
        inner_train=_ic_block(scored["inner_train"]),
        inner_holdout=_ic_block(scored["inner_holdout"]),
        split=split.to_dict(),
        note=note,
    )

    valid_dates = scored["inner_holdout"].get("valid_dates", 0.0)
    if not valid_dates:
        result.feasible = False
        result.issues.append(
            "the factor produced no usable cross-section on the inner holdout "
            "(constant, all-NaN, or below the min_names breadth floor)"
        )
    if run_backtest and result.feasible:
        result.backtest = _inner_backtest(tree, context, split)
        if result.backtest.get("integrity_issues"):
            result.feasible = False
            result.issues.extend(result.backtest["integrity_issues"])
    return result


def _inner_backtest(tree: Node, context: AgentContext, split: InnerSplit) -> dict[str, Any]:
    """Cost-aware portfolio path over the inner holdout (invariant 6: costs are real)."""
    window_panel = _slice(context.panel, split.holdout)
    context.guard.assert_within(window_panel.dates, what="agent inner backtest")
    factor = evaluate(tree, window_panel)
    if not isinstance(factor, pd.DataFrame):
        return {"integrity_issues": ["the expression did not evaluate to a panel"]}
    fwd = forward_returns(window_panel, context.horizon, context.execution)
    scheme = get_scheme(
        context.weighting_scheme,
        **({"quantile": context.quantile} if context.weighting_scheme == "quantile_ls" else {}),
    )
    report = backtest(
        factor,
        window_panel,
        fwd,
        scheme,
        TransactionCostModel(context.commission_bps, context.slippage_bps),
        dates=split.holdout,
        horizon=context.horizon,
        execution=context.execution,
    )
    return {
        "scheme": report.scheme,
        "net_sharpe": _finite(report.net_sharpe),
        "gross_sharpe": _finite(report.gross_sharpe),
        "max_drawdown": _finite(report.max_drawdown),
        "turnover": _finite(report.turnover),
        "usable": bool(report.usable),
        "integrity_issues": list(report.integrity_issues),
        "costs_bps": {
            "commission": context.commission_bps,
            "slippage": context.slippage_bps,
        },
    }


def correlation_with_target(
    tree: Node, context: AgentContext, *, sample_dates: int = 250
) -> float | None:
    """Rank correlation of a candidate with the factor being improved, on the inner window.

    A variant that scores well but is 0.98 correlated with the original has not added anything —
    it is the same bet with different arithmetic. Cheap to compute and the single most useful
    number for deciding whether a proposal is worth keeping.
    """
    if context.target_expanded is None:
        return None
    dates = context.train_dates[-sample_dates:]
    if len(dates) == 0:
        return None
    window_panel = _slice(context.panel, dates)
    context.guard.assert_within(window_panel.dates, what="agent correlation check")
    try:
        left = evaluate(tree, window_panel)
        right = evaluate(context.target_expanded, window_panel)
    except (KeyError, ValueError, TypeError):
        return None
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        return None
    left, right = left.reindex(dates), right.reindex(dates)
    per_date = []
    for date in dates:
        a, b = left.loc[date], right.loc[date]
        paired = a.notna() & b.notna()
        if int(paired.sum()) < 5:
            continue
        rho = a[paired].rank().corr(b[paired].rank())
        if rho is not None and np.isfinite(rho):
            per_date.append(float(rho))
    if not per_date:
        return None
    return round(float(np.mean(per_date)), 4)
