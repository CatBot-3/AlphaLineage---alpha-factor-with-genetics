"""The search service the API dispatches to (kept pure so it is unit-testable).

Runs a GP with a lineage recorder over the train split, then produces the honest net/deflated
verdict (the test split is scored only at the final report). Returns a JSON-serializable result.
"""

from __future__ import annotations

from typing import Any

from alphaforge.core.extensions import operator_counts
from alphaforge.core.gp import GP, GPConfig
from alphaforge.core.panel import Panel
from alphaforge.core.tree import to_dict, to_json
from alphaforge.library.store import LineageStore
from alphaforge.validation.pipeline import LockedTestSet, judge
from alphaforge.validation.splits import time_split
from alphaforge.validation.trials import effective_trials


def run_search(
    config: GPConfig, panel: Panel, *, train: float = 0.6, valid: float = 0.2, embargo: int = 5
) -> dict[str, Any]:
    """Run a GP search and return the best factor, its OOS/deflated verdict, and the lineage."""
    split = time_split(panel.dates, train=train, valid=valid, embargo=embargo)
    train_panel = Panel({f: df.loc[df.index.isin(split.train)] for f, df in panel.fields.items()})

    store = LineageStore()
    gp = GP(config, train_panel, recorder=store)
    best = gp.run()

    trials = [ind.tree for ind in gp.population]
    # The deflation's trial count grows with the search space — including any user-added
    # operators (Phase 7) — so adding primitives makes a lucky factor harder to certify.
    builtin, user = operator_counts()
    n_trials = effective_trials(gp.trial_count, n_operators=builtin + user, baseline=builtin)
    report = judge(
        best.tree,
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=n_trials,
        min_names=config.min_names,
    )
    report_dict = {
        "oos_ic": report.oos_ic,
        "deflated_sharpe": report.deflated_sharpe,
        "pbo": report.pbo,
        "train_ic": report.train_ic,
        "n_trials": report.n_trials,
        "significant": report.significant,
    }
    store.metadata = {"best_factor": to_dict(best.tree), "report": report_dict}

    return {
        "best_factor": to_json(best.tree),
        "report": report_dict,
        "generations": gp.generation,
        "history": gp.history,
        "lineage": store.to_dict(),
    }
