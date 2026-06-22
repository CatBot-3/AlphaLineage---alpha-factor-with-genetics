"""A4/A5 - iterative training sessions: create, run a segment, continue, inspect.

A *session* is a long-lived search a user can grow over many *segments*: start with a
universe and GP config, then continue from the evolved population with changed parameters,
universe, or operators, and seed new sessions from saved factors. State lives under
``data_cache/sessions/{id}/``:

    session.json     boundaries, config, operators, seeds, cumulative trial/test-read counts
    checkpoint.json  the GP checkpoint (exact resume), carrying the trial count
    lineage.json     the cumulative lineage across every segment (continuous numbering)
    result.json      the latest segment's report/result payload

Honesty invariants this module enforces (see the plan's P1-P3):
  - the train/valid/test **time boundary** is frozen at session creation; every segment,
    even with a changed universe, rebuilds its split against those frozen dates, so the GP
    never sees a date >= ``test_start`` (asserted in :func:`run_segment`);
  - the trial count is cumulative and monotone, carried through the checkpoint;
  - each segment is one out-of-sample read; ``test_reads`` is counted and surfaced.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alphalineage.api.service import build_report
from alphalineage.core.gp import GP, GPConfig
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node, to_json
from alphalineage.data import paths
from alphalineage.library.store import LineageStore
from alphalineage.validation.splits import Split, time_split


@dataclass(frozen=True)
class Boundaries:
    """Frozen train/valid/test time boundaries (ISO date strings) plus the embargo gap."""

    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    embargo: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Boundaries:
        return cls(
            train_end=data["train_end"],
            valid_start=data["valid_start"],
            valid_end=data["valid_end"],
            test_start=data["test_start"],
            embargo=int(data["embargo"]),
        )


def derive_boundaries(
    dates: pd.DatetimeIndex, *, train: float = 0.6, valid: float = 0.2, embargo: int = 5
) -> Boundaries:
    """Freeze boundaries from a session's initial panel (one canonical split)."""
    split = time_split(dates, train=train, valid=valid, embargo=embargo)
    return Boundaries(
        train_end=split.train[-1].isoformat(),
        valid_start=split.valid[0].isoformat(),
        valid_end=split.valid[-1].isoformat(),
        test_start=split.test[0].isoformat(),
        embargo=embargo,
    )


def split_from_boundaries(dates: pd.DatetimeIndex, boundaries: Boundaries) -> Split:
    """Rebuild a split for ``dates`` against frozen boundaries (P1).

    Train = dates <= train_end; valid = [valid_start, valid_end]; test = dates >= test_start.
    Dates falling in the embargo gaps belong to no segment, so the gaps are preserved.
    """
    idx = pd.DatetimeIndex(dates)
    train_end = pd.Timestamp(boundaries.train_end)
    valid_start = pd.Timestamp(boundaries.valid_start)
    valid_end = pd.Timestamp(boundaries.valid_end)
    test_start = pd.Timestamp(boundaries.test_start)

    train = idx[idx <= train_end]
    valid = idx[(idx >= valid_start) & (idx <= valid_end)]
    test = idx[idx >= test_start]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("panel has no dates in the locked train/test segments")
    return Split(train, valid, test)


# --- session file I/O ------------------------------------------------------------
def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip().lower()).strip("-._")
    return f"{cleaned or 'session'}-{uuid.uuid4().hex[:8]}"


def session_dir(session_id: str) -> Path:
    return paths.sessions_dir() / session_id


def exists(session_id: str) -> bool:
    return (session_dir(session_id) / "session.json").exists()


def load_session(session_id: str) -> dict[str, Any]:
    return json.loads((session_dir(session_id) / "session.json").read_text(encoding="utf-8"))


def save_session(session: dict[str, Any]) -> None:
    directory = session_dir(session["id"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "session.json").write_text(
        json.dumps(session, indent=2, sort_keys=True), encoding="utf-8"
    )


def new_session(
    *,
    name: str,
    universe: str,
    as_of: str,
    config: GPConfig,
    operators: Sequence[dict[str, Any]],
    seed_factor_ids: Sequence[str],
    boundaries: Boundaries,
    trial_baseline: int,
    test_reads_baseline: int,
    created_at: str,
) -> dict[str, Any]:
    """Create and persist a new session's ``session.json`` (no segment run yet)."""
    session = {
        "id": _slug(name),
        "name": name,
        "created_at": created_at,
        "universe": universe,
        "as_of": as_of,
        "boundaries": boundaries.to_dict(),
        "config": config.to_dict(),
        "operators": list(operators),
        "seed_factor_ids": list(seed_factor_ids),
        "trial_baseline": int(trial_baseline),
        "cumulative_trials": 0,
        "test_reads": int(test_reads_baseline),
        "segments": [],
        "last_job_id": None,
    }
    save_session(session)
    return session


# --- the segment runner ----------------------------------------------------------
def _last_generation(store: LineageStore) -> int:
    return max((n.generation for n in store.nodes), default=0)


def _inject_seeds(gp: GP, seeds: Sequence[Node]) -> None:
    """Replace the lowest-fitness individuals with validated seed trees (warm continue)."""
    validated = [gp._validate_seed(s) for s in seeds]
    order = sorted(range(len(gp.population)), key=lambda i: gp.population[i].fitness)
    for tree, idx in zip(validated, order, strict=False):
        gp.population[idx] = gp._individual(tree)


def run_segment(
    session_id: str,
    *,
    job_id: str,
    panel: Panel,
    config: GPConfig,
    generations: int,
    seeds: Sequence[Node] = (),
    extra_seeds: Sequence[Node] = (),
    rescore: bool = False,
    progress: Any = None,
    stop: Callable[[], bool] | None = None,
    allowed_operators: set[str] | None = None,
) -> dict[str, Any]:
    """Run one segment (fresh start or warm continue) and persist all session state."""
    directory = session_dir(session_id)
    session = load_session(session_id)
    boundaries = Boundaries.from_dict(session["boundaries"])

    split = split_from_boundaries(panel.dates, boundaries)  # ValueError if locked segs empty (P1)
    train_panel = Panel({f: df.loc[df.index.isin(split.train)] for f, df in panel.fields.items()})
    test_start = pd.Timestamp(boundaries.test_start)
    if len(train_panel.dates) and train_panel.dates.max() >= test_start:
        raise RuntimeError("train panel leaked a locked test date (invariant 1)")

    checkpoint = directory / "checkpoint.json"
    lineage_path = directory / "lineage.json"
    warm = checkpoint.exists()

    store = (
        LineageStore.load(lineage_path)
        if lineage_path.exists()
        else LineageStore(run_id=session_id)
    )
    recorder: Any = store
    if progress is not None:
        progress.attach(store)
        recorder = progress

    prev_cumulative = int(session["cumulative_trials"])
    if warm:
        store.continue_from(_last_generation(store))
        gp = GP.from_checkpoint(
            checkpoint, train_panel, recorder=recorder, allowed_operators=allowed_operators
        )
        gp.config = config  # apply any continue-time overrides
        if rescore:
            gp.rescore_population()
        if extra_seeds:
            _inject_seeds(gp, extra_seeds)
        gen_start = gp.generation
        target = gp.generation + generations
        if progress is not None:
            progress.set_target(target)
        best = gp.run(generations=target, stop=stop)
    else:
        gp = GP(config, train_panel, recorder=recorder, allowed_operators=allowed_operators)
        gen_start = 0
        target = generations
        if progress is not None:
            progress.set_target(target)
        best = gp.run(generations=target, seeds=list(seeds), stop=stop)

    gp.save_checkpoint(checkpoint)
    store.save(lineage_path)

    trials = [ind.tree for ind in gp.population]
    searched = int(session["trial_baseline"]) + gp.trial_count
    report = build_report(
        best.tree, trials, split, panel, searched_trials=searched, min_names=gp.config.min_names
    )

    session["cumulative_trials"] = gp.trial_count
    session["test_reads"] = int(session["test_reads"]) + 1
    session["last_job_id"] = job_id
    session["segments"].append(
        {
            "index": len(session["segments"]),
            "universe": session["universe"],
            "config": config.to_dict(),
            "gen_start": gen_start,
            "gen_end": gp.generation,
            "new_trials": gp.trial_count - prev_cumulative,
            "status": "done",
        }
    )
    save_session(session)

    result = {
        "best_factor": to_json(best.tree),
        "report": report,
        "generations": gp.generation,
        "history": gp.history,
        "lineage": store.to_dict(),
        "session_id": session_id,
        "segment": len(session["segments"]) - 1,
        "test_reads": session["test_reads"],
        "cumulative_trials": session["cumulative_trials"],
        "repeated_oos_warning": session["test_reads"] > 1,
    }
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result
