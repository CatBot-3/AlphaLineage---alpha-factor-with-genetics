"""P10-T1 - ``AgentContext``: the only world an agent tool can see.

Assembling a context is where the split clamp physically happens. The panel stored here is
truncated at the session's frozen ``train_end`` **before** any tool exists to ask for it, so the
validation window and the locked holdout are not merely forbidden — they are absent. That is a
much stronger property than a permission check, because it survives a bug in the checking.

The context is built once per agent run, is immutable, and is the sole argument every tool
handler receives besides its own parameters. No tool reads global state, the settings file, or
the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from alphalineage.agent.guards import BudgetLedger, GuardViolation, SplitGuard, TrialLedger
from alphalineage.api.sessions import Boundaries
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node
from alphalineage.core.tree import from_dict as tree_from_dict

#: The agent needs enough training history for an inner split to mean anything. Below this the
#: run refuses to start rather than producing a confident number from 40 observations.
MIN_TRAIN_DATES = 120


def truncate_panel(panel: Panel, guard: SplitGuard) -> Panel:
    """Return a panel containing only dates the agent may read.

    Deliberately not lazy and not a view: the frames are sliced now so nothing downstream can
    address a validation or holdout row even by accident.
    """
    dates = pd.DatetimeIndex(panel.dates)
    keep = dates <= guard.train_end
    if not bool(keep.any()):
        raise GuardViolation(
            "the session's training window contains no cached dates; sync data before "
            "running an agent"
        )
    truncated = Panel({name: frame.loc[keep] for name, frame in panel.fields.items()})
    guard.assert_within(truncated.dates, what="agent panel construction")
    return truncated


@dataclass(frozen=True)
class AgentContext:
    """Everything an agent run may touch, and nothing else."""

    session_id: str
    session_name: str
    universe: str
    boundaries: Boundaries
    guard: SplitGuard
    #: Truncated at ``train_end``. This is the agent's entire view of the market.
    panel: Panel
    config: dict[str, Any]
    #: The factor the run starts from (surface tree), plus its macro-expanded form.
    target_tree: Node | None = None
    target_expanded: Node | None = None
    target_label: str = ""
    #: Metrics already recorded for the target, labelled by split. Read-only context.
    target_metrics: dict[str, Any] = field(default_factory=dict)
    horizon: int = 1
    #: Session execution timing (``fitness.EXECUTION_TIMINGS``); frozen, never agent-editable.
    execution: str = "close"
    ic_method: str = "spearman"
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    quantile: float = 0.2
    weighting_scheme: str = "quantile_ls"
    budget: BudgetLedger = field(default_factory=BudgetLedger)
    trials: TrialLedger = field(default_factory=TrialLedger)
    #: Operator specs pinned by the session, so expansion does not depend on registry state.
    pinned_formulas: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if len(self.panel.dates) < MIN_TRAIN_DATES:
            raise GuardViolation(
                f"the agent needs at least {MIN_TRAIN_DATES} training dates to carve a "
                f"meaningful inner split; this session has {len(self.panel.dates)}"
            )

    @property
    def train_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.panel.dates)

    def describe(self) -> dict[str, Any]:
        """The banner the UI shows and the header the model is given."""
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "universe": self.universe,
            "horizon": self.horizon,
            "execution": self.execution,
            "ic_method": self.ic_method,
            "train_dates": len(self.panel.dates),
            "symbols": len(self.panel.symbols),
            "reach": self.guard.describe(),
            "costs": {
                "commission_bps": self.commission_bps,
                "slippage_bps": self.slippage_bps,
            },
        }


def context_from_session(
    session: dict[str, Any],
    *,
    panel: Panel,
    target_tree: Node | None = None,
    target_expanded: Node | None = None,
    target_label: str = "",
    target_metrics: dict[str, Any] | None = None,
    budget: BudgetLedger | None = None,
) -> AgentContext:
    """Build a context from a stored ``session.json`` payload and a full panel.

    The panel passed in may span the whole history; it is truncated here. Callers must not
    pre-slice it to anything wider than the training window and must not pass a panel built for
    a different universe.
    """
    boundaries = Boundaries.from_dict(session["boundaries"])
    guard = SplitGuard(boundaries)
    config = dict(session.get("config") or {})
    # Cost and portfolio settings are not part of GPConfig; a session records them per round.
    # Defaults mirror the run form so an agent's backtest is priced the same way a user's is.
    commission_bps, slippage_bps, quantile = 1.0, 5.0, 0.2
    weighting_scheme = "quantile_ls"
    return AgentContext(
        session_id=str(session.get("id") or ""),
        session_name=str(session.get("name") or session.get("id") or ""),
        universe=str(session.get("universe") or ""),
        boundaries=boundaries,
        guard=guard,
        panel=truncate_panel(panel, guard),
        config=config,
        target_tree=target_tree,
        target_expanded=target_expanded or target_tree,
        target_label=target_label,
        target_metrics=dict(target_metrics or {}),
        horizon=int(config.get("horizon", 1) or 1),
        execution=str(config.get("execution") or "close"),
        ic_method=str(config.get("ic_method", "spearman") or "spearman"),
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        quantile=quantile,
        weighting_scheme=weighting_scheme,
        budget=budget or BudgetLedger(),
        pinned_formulas=tuple(session.get("formula_revisions") or ()),
    )


def target_from_round(result: dict[str, Any]) -> tuple[Node | None, dict[str, Any]]:
    """Pull the selected factor and its split-labelled metrics out of a stored segment."""
    import json

    raw = result.get("best_factor")
    tree: Node | None = None
    if isinstance(raw, str):
        try:
            tree = tree_from_dict(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            tree = None
    elif isinstance(raw, dict):
        try:
            tree = tree_from_dict(raw)
        except (ValueError, KeyError, TypeError):
            tree = None
    selection = result.get("selection") or {}
    metrics = {
        "training": dict(selection.get("training_metrics") or {}),
        "validation": dict(selection.get("validation_metrics") or {}),
        "evidence_status": result.get("evidence_status"),
        "cumulative_trials": result.get("cumulative_trials"),
        "test_reads": result.get("test_reads"),
    }
    return tree, metrics
