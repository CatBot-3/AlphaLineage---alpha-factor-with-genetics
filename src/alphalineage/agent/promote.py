"""P11-T5 - promoting a staged factor into a real, finalizable round.

The decision this implements: an agent-authored factor becomes a full round, numbered alongside
the GP's and eligible for holdout finalization like any other. That is what makes "improve this"
mean something — an improvement you cannot take to the holdout is a suggestion, not a result.

Two consequences follow, and both are handled here rather than left implicit.

**Promotion runs the real validation pass.** The agent only ever saw an inner holdout carved out
of training data. If a promoted factor were filed with only that number it would sit in the round
list looking like a peer of rounds that have been validated on the session's actual validation
window. So promotion evaluates it through ``select_validation_candidate`` — the same machinery
the GP champion goes through — and the round carries genuine validation metrics.

**Trials fold in before the round exists.** A round's deflated statistics read the session's
``cumulative_trials``. The moment an agent round exists it is finalizable, so its search cost has
to already be in that number; otherwise the first finalization would deflate against a count that
omits the search which produced the factor. An agent's trials are *informed* — it proposes what it
expects to score well — which carries more selection bias per trial than the GP's near-random
draws, so under-counting here would flatter every downstream number.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from alphalineage.backtest.portfolio import PORTFOLIO_SCHEMA_VERSION, get_scheme
from alphalineage.core.fitness import forward_returns, score_tree
from alphalineage.core.gp import GPConfig
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node
from alphalineage.core.tree import from_dict as tree_from_dict
from alphalineage.core.tree import to_json as tree_to_json
from alphalineage.validation.selection import (
    VALIDATION_SELECTION_VERSION,
    select_validation_candidate,
)

#: Marks a round the agent authored. The UI colours on this and the model finds its own work by it.
AGENT_ORIGIN = "agent"


class PromotionError(ValueError):
    """A staged factor could not be promoted into a round."""


@dataclass
class _Candidate:
    """The ``Candidate`` protocol ``select_validation_candidate`` expects."""

    tree: Node
    fitness: float
    metrics: dict[str, float]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def promote_factor(
    session_id: str,
    *,
    tree_payload: dict[str, Any],
    name: str,
    rationale: str,
    panel: Panel,
    conversation_id: str,
    turn_index: int,
    agent_trials: int,
    parent_round_index: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Validate a staged factor and write it as a new round. Returns the round summary.

    Runs on a background job: validation over the session's validation window is real work, not
    a file write.
    """
    from alphalineage.api import sessions as sessions_module

    started = time.monotonic()
    if not sessions_module.exists(session_id):
        raise PromotionError(f"session {session_id!r} not found")
    session = sessions_module.load_session(session_id)
    boundaries = sessions_module.Boundaries.from_dict(session["boundaries"])
    config = GPConfig.from_dict(session["config"])

    try:
        tree = tree_from_dict(tree_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionError(f"the staged expression is malformed: {exc}") from None

    split = sessions_module.split_from_boundaries(pd.DatetimeIndex(panel.dates), boundaries)
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("validating")

    # Training metrics first: polarity is fixed by training, never by validation.
    fwd = forward_returns(panel, config.horizon)
    train_panel = _slice(panel, split.train)
    train_fitness, train_metrics = score_tree(
        tree,
        train_panel,
        fwd.reindex(split.train),
        method=config.ic_method,
        parsimony=config.parsimony,
        complexity_penalty_mode=config.complexity_penalty_mode,
        complexity_penalty_value=config.complexity_penalty_value,
        max_nodes=config.max_nodes,
        min_names=config.min_names,
    )

    selection = select_validation_candidate(
        [_Candidate(tree=tree, fitness=train_fitness, metrics=train_metrics)],
        panel,
        fwd,
        split.valid,
        method=config.ic_method,
        parsimony=config.parsimony,
        complexity_penalty_mode=config.complexity_penalty_mode,
        complexity_penalty_value=config.complexity_penalty_value,
        max_nodes=config.max_nodes,
        validation_folds=config.validation_folds,
        fold_embargo=boundaries.embargo,
        min_names=config.min_names,
        weighting_scheme=get_scheme("quantile_ls", quantile=0.2),
    )

    # Fold the search cost in BEFORE the round exists, because the round is finalizable the
    # moment it does. See the module docstring.
    session = sessions_module.load_session(session_id)
    baseline = int(session.get("cumulative_trials", 0) or 0)
    cumulative = baseline + max(0, int(agent_trials))
    session["cumulative_trials"] = cumulative
    agent_spend = dict(session.get("agent_trials") or {})
    agent_spend[f"{conversation_id}:{turn_index}"] = int(agent_trials)
    session["agent_trials"] = agent_spend

    index = _next_round_index(session)
    result = {
        "round_index": index,
        "segment": index,
        "best_factor": tree_to_json(selection.oriented_tree),
        "report": None,
        "oos_backtest": None,
        "validation_only": True,
        "generations": 0,
        "history": [],
        "lineage": {"run_id": session_id, "metadata": {}, "nodes": []},
        "session_id": session_id,
        "test_reads": int(session.get("test_reads", 0) or 0),
        "session_holdout_reads": int(session.get("session_holdout_reads", 0) or 0),
        "cumulative_trials": cumulative,
        "searched_trials": max(1, int(agent_trials)),
        "report_trials": [tree_to_json(selection.oriented_tree)],
        "n_user_operators": len(session.get("operators") or []),
        "evidence_status": "validation_only",
        "termination_reason": "completed",
        "repeated_oos_warning": False,
        "test_read_index": None,
        "selection": selection.metadata(),
        "context": _round_context(session, boundaries, config),
        "timings": {"validation_seconds": round(time.monotonic() - started, 3)},
        "resources": {"profile": "agent", "workers": 1},
        "round_metadata": {
            "schema_version": 3,
            "index": index,
            "segment_index": index,
            "origin": AGENT_ORIGIN,
            "agent_conversation_id": conversation_id,
            "agent_turn_index": turn_index,
            "agent_trials": int(agent_trials),
            "agent_name": name[:120],
            "agent_rationale": rationale[:2000],
            "agent_expression": _text(selection.oriented_tree),
            "parent_round_index": parent_round_index,
            "validation_only": True,
            "report_available": False,
            "finalization_available": True,
            "evidence_status": "validation_only",
            "status": "done",
            "started_at": _now(),
            "completed_at": _now(),
            "requested_generations": 0,
            "gen_start": 0,
            "gen_end": 0,
            "termination_reason": "completed",
            "panel_fingerprint": sessions_module.panel_fingerprint(panel),
            "scorer_version": sessions_module.SCORER_VERSION,
            "evolution_version": sessions_module.EVOLUTION_VERSION,
            "adjustment_version": sessions_module._adjustment_version(),
            "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
            "validation_selection_version": VALIDATION_SELECTION_VERSION,
            "config": session["config"],
        },
    }

    path = sessions_module.round_path(session_id, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    summary = sessions_module._round_summary_from_result(result, index=index)
    summary["origin"] = AGENT_ORIGIN
    summary["agent_expression"] = result["round_metadata"]["agent_expression"]
    summary["parent_round_index"] = parent_round_index
    session.setdefault("rounds", []).append(summary)
    session["updated_at"] = _now()
    sessions_module.save_session(session)
    return summary


def _next_round_index(session: dict[str, Any]) -> int:
    indices = [
        int(item.get("index", -1)) for item in session.get("rounds") or [] if isinstance(item, dict)
    ]
    return (max(indices) + 1) if indices else 0


def _slice(panel: Panel, dates: pd.DatetimeIndex) -> Panel:
    """Keep every row up to the window's end so lookback operators have their warm-up."""
    index = pd.DatetimeIndex(panel.dates)
    keep = index <= dates.max()
    return Panel({name: frame.loc[keep] for name, frame in panel.fields.items()})


def _text(node: Node) -> str:
    from alphalineage.explain.anatomy import formula_text

    return formula_text(node)


def _round_context(session: dict[str, Any], boundaries: Any, config: GPConfig) -> dict[str, Any]:
    return {
        "boundaries": boundaries.to_dict(),
        "horizon": config.horizon,
        "ic_method": config.ic_method,
        "weighting_scheme": "quantile_ls",
        "quantile": 0.2,
        "commission_bps": 1.0,
        "slippage_bps": 5.0,
        "universe": session.get("universe"),
        "as_of": session.get("as_of"),
        "universe_definition": session.get("universe_definition") or {},
    }
