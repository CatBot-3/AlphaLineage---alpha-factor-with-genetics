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

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphalineage.api.resources import TRAINING_SCHEDULER, ResolvedResources, TrainingScheduler
from alphalineage.api.service import (
    build_report,
    distinct_trials,
    report_context,
    user_operator_count,
)
from alphalineage.core.extensions import expand_all
from alphalineage.core.gp import GP, SCORER_VERSION, GPConfig, TrainingCancelled
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node, to_json
from alphalineage.data import paths
from alphalineage.data.identifiers import atomic_write_text, child_path
from alphalineage.library.store import LineageStore
from alphalineage.validation.pbo import ReportReturnSummary
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
    dates: pd.DatetimeIndex,
    *,
    train: float = 0.6,
    valid: float = 0.2,
    embargo: int = 5,
    horizon: int = 1,
) -> Boundaries:
    """Freeze boundaries from a session's initial panel (one canonical split)."""
    split = time_split(dates, train=train, valid=valid, embargo=embargo, horizon=horizon)
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
_SESSION_IO_LOCK = threading.RLock()
_REPORT_CACHE_VERSION = 1


def _panel_fingerprint(panel: Panel) -> str:
    """Content fingerprint used to prevent report-stat reuse after cached market data changes."""
    digest = hashlib.sha256()
    digest.update("\0".join(map(str, panel.dates)).encode("utf-8"))
    digest.update("\0".join(map(str, panel.symbols)).encode("utf-8"))
    for field in sorted(panel.fields):
        digest.update(field.encode("utf-8"))
        values = np.ascontiguousarray(panel[field].to_numpy(dtype="float64"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _report_cache_context(
    session: dict[str, Any], panel: Panel, boundaries: Boundaries, config: GPConfig
) -> str:
    payload = {
        "version": _REPORT_CACHE_VERSION,
        "scorer_version": SCORER_VERSION,
        "universe": session["universe"],
        "universe_fingerprint": (session.get("universe_definition") or {}).get("fingerprint"),
        "as_of": session["as_of"],
        "boundaries": boundaries.to_dict(),
        "horizon": config.horizon,
        "weighting": {"name": "quantile_ls", "quantile": 0.2},
        "costs": {"commission_bps": 1.0, "slippage_bps": 5.0},
        "panel": _panel_fingerprint(panel),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _report_tree_key(tree: Node) -> str:
    expanded = to_json(expand_all(tree))
    return hashlib.sha256(expanded.encode("utf-8")).hexdigest()


def _load_report_cache(
    path: Path, context: str
) -> dict[str, ReportReturnSummary]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if (
        payload.get("version") != _REPORT_CACHE_VERSION
        or payload.get("context") != context
        or not isinstance(payload.get("entries"), dict)
    ):
        return {}
    return dict(payload["entries"])


def _save_report_cache(
    path: Path, context: str, entries: dict[str, ReportReturnSummary]
) -> None:
    payload = {
        "version": _REPORT_CACHE_VERSION,
        "context": context,
        "entries": entries,
    }
    atomic_write_text(path, json.dumps(payload, separators=(",", ":"), allow_nan=False))


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip().lower()).strip("-._")
    return f"{cleaned or 'session'}-{uuid.uuid4().hex[:8]}"


def session_dir(session_id: str) -> Path:
    return child_path(paths.sessions_dir(), session_id, label="session id")


def exists(session_id: str) -> bool:
    return (session_dir(session_id) / "session.json").exists()


def load_session(session_id: str) -> dict[str, Any]:
    with _SESSION_IO_LOCK:
        return json.loads((session_dir(session_id) / "session.json").read_text(encoding="utf-8"))


def save_session(session: dict[str, Any]) -> None:
    with _SESSION_IO_LOCK:
        directory = session_dir(session["id"])
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(directory / "session.json", json.dumps(session, indent=2, sort_keys=True))


def claim_job(session_id: str, job_id: str) -> bool:
    """Atomically reserve a session for one queued/running segment."""
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        if session.get("active_job_id"):
            return False
        now = datetime.now(UTC).isoformat()
        session["active_job_id"] = job_id
        session["last_job_id"] = job_id
        session["last_job"] = {
            "id": job_id,
            "status": "queued",
            "queued_at": now,
            "updated_at": now,
            "error": None,
        }
        save_session(session)
        return True


def update_job_status(session_id: str, job_id: str, status: str) -> None:
    """Persist the active segment status without letting an old worker clobber a new job."""
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        if session.get("active_job_id") != job_id:
            return
        last_job = dict(session.get("last_job") or {"id": job_id})
        last_job.update({"status": status, "updated_at": datetime.now(UTC).isoformat()})
        session["last_job"] = last_job
        save_session(session)


def finish_job(session_id: str, job_id: str, status: str, *, error: str | None = None) -> None:
    """Release a segment reservation and retain its final persisted status."""
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        if session.get("active_job_id") == job_id:
            session["active_job_id"] = None
        last_job = dict(session.get("last_job") or {"id": job_id})
        if last_job.get("id") == job_id:
            last_job.update(
                {
                    "status": status,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "error": error,
                }
            )
            session["last_job"] = last_job
        save_session(session)


def new_session(
    *,
    name: str,
    universe: str,
    as_of: str,
    config: GPConfig,
    operators: Sequence[dict[str, Any]],
    formula_revisions: Sequence[dict[str, Any]] = (),
    universe_definition: dict[str, Any] | None = None,
    seed_factor_ids: Sequence[str],
    boundaries: Boundaries,
    trial_baseline: int,
    test_reads_baseline: int,
    created_at: str,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and persist a new session's ``session.json`` (no segment run yet)."""
    session = {
        "id": _slug(name),
        "name": name,
        "created_at": created_at,
        "universe": universe,
        "universe_definition": dict(universe_definition or {}),
        "as_of": as_of,
        "boundaries": boundaries.to_dict(),
        "config": config.to_dict(),
        "resources": dict(resources or {"profile": "auto", "cpu_budget_percent": None}),
        "operators": list(operators),
        "formula_revisions": list(formula_revisions),
        "seed_factor_ids": list(seed_factor_ids),
        "trial_baseline": int(trial_baseline),
        "cumulative_trials": 0,
        "test_reads": int(test_reads_baseline),
        "segments": [],
        "last_job_id": None,
        "active_job_id": None,
        "last_job": None,
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
    resources: ResolvedResources | None = None,
    scheduler: TrainingScheduler = TRAINING_SCHEDULER,
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
    started = time.monotonic()
    lease_context = scheduler.acquire(resources, cancel=stop) if resources is not None else None
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("queued" if resources is not None else "initializing")

    def execute(
        lease: Any = None,
    ) -> tuple[GP, Any, dict[str, Any] | None, int, int, dict[str, float]]:
        workers = int(lease.max_workers) if lease is not None else 1
        memory_budget = int(lease.memory_budget_bytes) if lease is not None else None
        if progress is not None and hasattr(progress, "set_phase"):
            progress.set_phase("initializing")
        training_started = time.monotonic()
        if warm:
            store.continue_from(_last_generation(store))
            gp = GP.from_checkpoint(
                checkpoint,
                train_panel,
                recorder=recorder,
                allowed_operators=allowed_operators,
                workers=workers,
                memory_budget_bytes=memory_budget,
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
            gp = GP(
                config,
                train_panel,
                recorder=recorder,
                allowed_operators=allowed_operators,
                workers=workers,
                memory_budget_bytes=memory_budget,
            )
            gen_start = 0
            target = generations
            if progress is not None:
                progress.set_target(target)
            best = gp.run(generations=target, seeds=list(seeds), stop=stop)
        training_seconds = time.monotonic() - training_started

        gp.save_checkpoint(checkpoint)
        store.save(lineage_path)
        if progress is not None and hasattr(progress, "set_phase"):
            progress.set_phase("validating")
        reporting_started = time.monotonic()
        trials = distinct_trials(node.tree for node in store.nodes)
        searched = int(session["trial_baseline"]) + gp.trial_count
        report_cache_path = directory / "report_stats.json"
        report_context = _report_cache_context(session, panel, boundaries, gp.config)
        report_summaries = _load_report_cache(report_cache_path, report_context)
        report_progress = (
            progress.set_report_progress
            if progress is not None and hasattr(progress, "set_report_progress")
            else None
        )

        def on_finalizing() -> None:
            if progress is not None and hasattr(progress, "set_phase"):
                progress.set_phase("finalizing")

        try:
            report = build_report(
                best.tree,
                trials,
                split,
                panel,
                searched_trials=searched,
                min_names=gp.config.min_names,
                horizon=gp.config.horizon,
                ic_method=gp.config.ic_method,
                n_user_operators=user_operator_count(allowed_operators),
                progress=report_progress,
                stop=None if gp.termination_reason == "user_stopped" else stop,
                on_finalizing=on_finalizing,
                summary_cache=report_summaries,
                summary_key=_report_tree_key,
            )
        except TrainingCancelled:
            # Research-only validation is cancellable. The checkpoint/lineage above remain a
            # complete generation, and the locked test was not opened, so the session can resume.
            gp.termination_reason = "user_stopped"
            report = None
        finally:
            _save_report_cache(report_cache_path, report_context, report_summaries)
        return gp, best, report, gen_start, target, {
            "training_seconds": training_seconds,
            "reporting_seconds": time.monotonic() - reporting_started,
            "total_seconds": time.monotonic() - started,
        }

    if lease_context is None:
        gp, best, report, gen_start, target, timings = execute()
    else:
        with lease_context as lease:
            gp, best, report, gen_start, target, timings = execute(lease)

    session["cumulative_trials"] = gp.trial_count
    session["test_reads"] = int(session["test_reads"]) + (1 if report is not None else 0)
    session["last_job_id"] = job_id
    termination_reason = gp.termination_reason
    stopped_early = termination_reason == "user_stopped"
    resolved_resources = resources.to_dict() if resources is not None else None
    session["segments"].append(
        {
            "index": len(session["segments"]),
            "universe": session["universe"],
            "config": config.to_dict(),
            "gen_start": gen_start,
            "gen_end": gp.generation,
            "new_trials": gp.trial_count - prev_cumulative,
            "status": "stopped" if stopped_early else "done",
            "resources": resolved_resources,
            "termination_reason": termination_reason,
            "report_cancelled": report is None,
        }
    )
    save_session(session)

    result = {
        "best_factor": to_json(best.tree),
        "report": report,
        "oos_backtest": report.get("oos_backtest") if report is not None else None,
        "generations": gp.generation,
        "history": gp.history,
        "lineage": store.to_dict(),
        "session_id": session_id,
        "segment": len(session["segments"]) - 1,
        "test_reads": session["test_reads"],
        "cumulative_trials": session["cumulative_trials"],
        "repeated_oos_warning": session["test_reads"] > 1,
        "resources": resolved_resources,
        "termination_reason": termination_reason,
        "timings": timings,
        "context": {
            **report_context(split, gp.config, embargo=boundaries.embargo),
            "universe": session["universe"],
            "universe_revision": (session.get("universe_definition") or {}).get("fingerprint"),
            "universe_definition": session.get("universe_definition"),
            "as_of": session["as_of"],
        },
        "report_cancelled": report is None,
    }
    if report is not None:
        # A cancelled research-only report must not erase the previous complete OOS result.
        atomic_write_text(directory / "result.json", json.dumps(result))
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("done")
    return result
