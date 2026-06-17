"""The search service the API dispatches to (kept pure so it is unit-testable).

Runs a GP with a lineage recorder over the train split, then produces the honest net/deflated
verdict (the test split is scored only at the final report). Returns a JSON-serializable result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from alphalineage.core.extensions import operator_counts
from alphalineage.core.gp import GP, GPConfig
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node, to_dict, to_json
from alphalineage.library.store import LineageStore
from alphalineage.validation.pipeline import LockedTestSet, judge
from alphalineage.validation.splits import Split, time_split
from alphalineage.validation.trials import effective_trials


def build_report(
    best_tree: Node,
    trials: Sequence[Node],
    split: Split,
    panel: Panel,
    *,
    searched_trials: int,
    min_names: int = 5,
) -> dict[str, Any]:
    """Judge a factor OOS and return its report dict (shared by single runs and sessions).

    ``searched_trials`` is the cumulative distinct-strategy count fed to the deflation; the
    effective count grows with the operator palette so user operators deflate harder
    (invariant 1). A fresh ``LockedTestSet`` is unlocked exactly once per call - so each call
    is one out-of-sample read, which the session counts and surfaces (P3).
    """
    builtin, user = operator_counts()
    n_trials = effective_trials(searched_trials, n_operators=builtin + user, baseline=builtin)
    report = judge(
        best_tree,
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=n_trials,
        min_names=min_names,
    )
    return {
        "oos_ic": report.oos_ic,
        "deflated_sharpe": report.deflated_sharpe,
        "pbo": report.pbo,
        "train_ic": report.train_ic,
        "n_trials": report.n_trials,
        "significant": report.significant,
    }


def run_search(
    config: GPConfig,
    panel: Panel,
    *,
    train: float = 0.6,
    valid: float = 0.2,
    embargo: int = 5,
    progress: Any = None,
    stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run a GP search and return the best factor, its OOS/deflated verdict, and the lineage."""
    split = time_split(panel.dates, train=train, valid=valid, embargo=embargo)
    train_panel = Panel({f: df.loc[df.index.isin(split.train)] for f, df in panel.fields.items()})

    store = LineageStore()
    recorder: Any = store
    if progress is not None:  # forward recorder calls through the live snapshot (A2)
        progress.attach(store)
        progress.set_target(config.generations)
        recorder = progress
    gp = GP(config, train_panel, recorder=recorder)
    best = gp.run(stop=stop)

    trials = [ind.tree for ind in gp.population]
    report_dict = build_report(
        best.tree, trials, split, panel, searched_trials=gp.trial_count, min_names=config.min_names
    )
    store.metadata = {"best_factor": to_dict(best.tree), "report": report_dict}

    return {
        "best_factor": to_json(best.tree),
        "report": report_dict,
        "generations": gp.generation,
        "history": gp.history,
        "lineage": store.to_dict(),
    }
