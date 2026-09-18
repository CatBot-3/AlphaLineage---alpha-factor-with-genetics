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
    report_context,
    user_operator_count,
)
from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.portfolio import (
    PORTFOLIO_SCHEMA_VERSION,
    PortfolioStrategySpec,
    default_strategy_spec,
)
from alphalineage.core.extensions import expand_all
from alphalineage.core.fitness import forward_returns
from alphalineage.core.gp import (
    EVOLUTION_VERSION,
    GP,
    SCORER_VERSION,
    GPConfig,
    TrainingCancelled,
)
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node, from_json, to_json
from alphalineage.data import paths
from alphalineage.data.identifiers import atomic_write_text, child_path
from alphalineage.library.store import LineageStore
from alphalineage.validation.pbo import ReportReturnSummary
from alphalineage.validation.selection import (
    VALIDATION_SELECTION_VERSION,
    ValidationSelection,
    compare_validation_strategies,
    orient_training_candidates,
    select_validation_candidate,
)
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
    """Freeze boundaries from a session's initial panel (one canonical split).

    ``horizon`` here is the label span (``fitness.label_span``): forward horizon plus any
    execution delay, which the embargo must cover.
    """
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
SESSION_SCHEMA_VERSION = 3
ROUND_SCHEMA_VERSION = 3
FINALIZATION_SCHEMA_VERSION = 2
STRATEGY_COMPARISON_SCHEMA_VERSION = 1
FINALIZATION_PLAN_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
    session: dict[str, Any],
    panel: Panel,
    boundaries: Boundaries,
    config: GPConfig,
    *,
    panel_fingerprint: str | None = None,
) -> str:
    payload = {
        "version": _REPORT_CACHE_VERSION,
        "scorer_version": SCORER_VERSION,
        "evolution_version": EVOLUTION_VERSION,
        "universe": session["universe"],
        "universe_fingerprint": (session.get("universe_definition") or {}).get("fingerprint"),
        "as_of": session["as_of"],
        "boundaries": boundaries.to_dict(),
        "horizon": config.horizon,
        **({} if config.execution == "close" else {"execution": config.execution}),
        "weighting": {"name": "quantile_ls", "quantile": 0.2},
        "costs": {"commission_bps": 1.0, "slippage_bps": 5.0},
        "panel": panel_fingerprint or _panel_fingerprint(panel),
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
        session["schema_version"] = SESSION_SCHEMA_VERSION
        session["updated_at"] = _utc_now()
        atomic_write_text(directory / "session.json", json.dumps(session, indent=2, sort_keys=True))


def rounds_dir(session_id: str) -> Path:
    return session_dir(session_id) / "rounds"


def round_path(session_id: str, index: int) -> Path:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("round index must be a non-negative integer")
    return rounds_dir(session_id) / f"{index:04d}.json"


def finalizations_dir(session_id: str) -> Path:
    return session_dir(session_id) / "finalizations"


def finalization_path(session_id: str, evaluation_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", evaluation_id):
        raise ValueError("invalid finalization evaluation id")
    return finalizations_dir(session_id) / f"{evaluation_id}.json"


def strategy_comparisons_dir(session_id: str) -> Path:
    return session_dir(session_id) / "strategy_comparisons"


def strategy_comparison_path(session_id: str, comparison_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", comparison_id):
        raise ValueError("invalid strategy comparison id")
    return strategy_comparisons_dir(session_id) / f"{comparison_id}.json"


def finalization_plans_dir(session_id: str) -> Path:
    return session_dir(session_id) / "finalization_plans"


def finalization_plan_path(session_id: str, strategy_plan_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", strategy_plan_id):
        raise ValueError("invalid finalization plan id")
    return finalization_plans_dir(session_id) / f"{strategy_plan_id}.json"


def _read_result(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _int_or(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _round_summary_from_result(
    result: dict[str, Any],
    *,
    index: int,
    legacy: bool = False,
) -> dict[str, Any]:
    metadata = dict(result.get("round_metadata") or {})
    raw_test_read_index = metadata.get("test_read_index", result.get("test_read_index"))
    test_read_index = (
        _int_or(raw_test_read_index, index + 1)
        if raw_test_read_index is not None
        else None
    )
    report_available = result.get("report") is not None
    return _with_round_validity({
        "index": index,
        "segment_index": _int_or(
            metadata.get("segment_index", result.get("segment")), index
        ),
        "report_available": report_available,
        "validation_only": bool(
            metadata.get("validation_only", result.get("validation_only", not report_available))
        ),
        "finalization_available": bool(metadata.get("finalization_available", report_available)),
        "latest_finalization_id": metadata.get("latest_finalization_id"),
        "latest_strategy_comparison_id": metadata.get(
            "latest_strategy_comparison_id"
        ),
        "latest_strategy_plan_id": metadata.get("latest_strategy_plan_id"),
        "test_read_index": test_read_index,
        "evidence_status": metadata.get(
            "evidence_status",
            (
                "locked"
                if test_read_index == 1
                else "exploratory_repeat"
                if test_read_index is not None
                else "validation_only"
            ),
        ),
        "status": metadata.get(
            "status",
            "stopped" if result.get("termination_reason") == "user_stopped" else "done",
        ),
        "started_at": metadata.get("started_at"),
        "completed_at": metadata.get("completed_at"),
        "requested_generations": metadata.get("requested_generations"),
        "gen_start": metadata.get("gen_start"),
        "gen_end": metadata.get("gen_end", result.get("generations")),
        "termination_reason": metadata.get(
            "termination_reason", result.get("termination_reason")
        ),
        "selected_lineage_node_id": metadata.get("selected_lineage_node_id"),
        "resources": metadata.get("resources", result.get("resources")),
        "timings": metadata.get("timings", result.get("timings")),
        "panel_fingerprint": metadata.get("panel_fingerprint"),
        "scorer_version": metadata.get("scorer_version"),
        "evolution_version": metadata.get("evolution_version"),
        "adjustment_version": metadata.get("adjustment_version"),
        "portfolio_schema_version": metadata.get("portfolio_schema_version"),
        "validation_selection_version": metadata.get(
            "validation_selection_version"
        ),
        "selection": metadata.get("selection", result.get("selection")),
        "legacy": legacy,
    })


def _with_round_validity(summary: dict[str, Any]) -> dict[str, Any]:
    """Label reports created under the known-invalid pre-fix semantics."""
    scorer = summary.get("scorer_version")
    evolution = summary.get("evolution_version")
    adjustment = summary.get("adjustment_version")
    portfolio = summary.get("portfolio_schema_version")
    selection = summary.get("validation_selection_version")
    valid = (
        isinstance(scorer, int)
        and scorer >= SCORER_VERSION
        and isinstance(evolution, int)
        and evolution >= EVOLUTION_VERSION
        and isinstance(adjustment, int)
        and adjustment >= _adjustment_version()
        and isinstance(portfolio, int)
        and portfolio >= PORTFOLIO_SCHEMA_VERSION
        and isinstance(selection, int)
        and selection >= VALIDATION_SELECTION_VERSION
    )
    summary["validity"] = "valid" if valid else "invalid_legacy_semantics"
    summary["restart_required"] = not valid
    if valid:
        summary.pop("invalid_reason", None)
    else:
        summary["invalid_reason"] = (
            "This report predates the corrected price-adjustment, directional-fitness, "
            "portfolio-construction, validation-selection, or evolution semantics."
        )
    return summary


def _legacy_round_summary(
    session: dict[str, Any], result: dict[str, Any], *, index: int = 0
) -> dict[str, Any]:
    summary = _round_summary_from_result(result, index=index, legacy=True)
    segment_index = _int_or(result.get("segment"), summary["segment_index"])
    segments = session.get("segments") or []
    segment = next(
        (
            item
            for item in segments
            if isinstance(item, dict)
            and _int_or(item.get("index"), -1) == segment_index
        ),
        {},
    )
    for key in (
        "status",
        "started_at",
        "completed_at",
        "requested_generations",
        "gen_start",
        "gen_end",
        "termination_reason",
        "selected_lineage_node_id",
        "resources",
        "timings",
        "panel_fingerprint",
        "scorer_version",
        "evolution_version",
        "adjustment_version",
    ):
        if summary.get(key) is None and segment.get(key) is not None:
            summary[key] = segment[key]
    if summary.get("requested_generations") is None:
        summary["requested_generations"] = max(
            0,
            _int_or(summary.get("gen_end") or result.get("generations"), 0)
            - _int_or(summary.get("gen_start"), 0),
        )
    summary["gen_start"] = _int_or(summary.get("gen_start"), 0)
    summary["gen_end"] = _int_or(
        summary.get("gen_end") or result.get("generations"), summary["gen_start"]
    )
    return _with_round_validity(summary)


def list_rounds(session_id: str) -> list[dict[str, Any]]:
    """Return lightweight immutable completed-round summaries.

    Pre-round-schema sessions retain only their latest ``result.json``.  Expose that
    result as a synthetic round zero until the next continuation migrates it.
    """
    session = load_session(session_id)
    stored = session.get("rounds")
    if isinstance(stored, list) and stored:
        return [
            _with_round_validity(dict(item))
            for item in stored
            if isinstance(item, dict)
        ]
    legacy = _read_result(session_dir(session_id) / "result.json")
    if legacy is None or legacy.get("report") is None:
        return []
    return [_legacy_round_summary(session, legacy)]


def load_round(session_id: str, index: int) -> dict[str, Any] | None:
    """Load one completed round, with a read-only legacy fallback."""
    payload = _read_result(round_path(session_id, index))
    if payload is not None:
        payload = dict(payload)
        summary = _round_summary_from_result(payload, index=index)
        stored_summary = next(
            (
                item
                for item in load_session(session_id).get("rounds", [])
                if isinstance(item, dict) and _int_or(item.get("index"), -1) == index
            ),
            {},
        )
        payload["round_metadata"] = {
            **dict(payload.get("round_metadata") or {}),
            **summary,
            **dict(stored_summary),
        }
        payload["validity"] = summary["validity"]
        payload["restart_required"] = summary["restart_required"]
        return payload
    session = load_session(session_id)
    if index == 0 and not session.get("rounds"):
        legacy = _read_result(session_dir(session_id) / "result.json")
        if legacy is not None and legacy.get("report") is not None:
            legacy = dict(legacy)
            legacy.setdefault("round_index", 0)
            legacy.setdefault(
                "round_metadata",
                _legacy_round_summary(session, legacy),
            )
            summary = _legacy_round_summary(session, legacy)
            legacy["validity"] = summary["validity"]
            legacy["restart_required"] = summary["restart_required"]
            return legacy
    return None


def _round_cost_model(round_payload: dict[str, Any]) -> TransactionCostModel:
    context = dict(round_payload.get("context") or {})
    return TransactionCostModel(
        commission_bps=float(context.get("commission_bps", 1.0)),
        slippage_bps=float(context.get("slippage_bps", 5.0)),
    )


def _comparison_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison_id": str(artifact["comparison_id"]),
        "round_index": int(artifact["round_index"]),
        "job_id": artifact.get("job_id"),
        "status": str(artifact.get("status", "queued")),
        "evidence_status": artifact.get("evidence_status"),
        "prior_holdout_reads": int(artifact.get("prior_holdout_reads", 0)),
        "strategy_ids": [
            str(item.get("id"))
            for item in artifact.get("strategies", [])
            if isinstance(item, dict)
        ],
        "created_at": artifact.get("created_at"),
        "completed_at": artifact.get("completed_at"),
        "error": artifact.get("error"),
    }


def list_strategy_comparisons(session_id: str) -> list[dict[str, Any]]:
    session = load_session(session_id)
    stored = session.get("strategy_comparisons")
    if isinstance(stored, list):
        return [dict(item) for item in stored if isinstance(item, dict)]
    return []


def load_strategy_comparison(
    session_id: str,
    comparison_id: str,
) -> dict[str, Any] | None:
    return _read_result(strategy_comparison_path(session_id, comparison_id))


def claim_strategy_comparison(
    session_id: str,
    *,
    job_id: str,
    comparison_id: str,
    round_index: int,
    strategies: Sequence[PortfolioStrategySpec],
    confirm_repeat: bool = False,
) -> tuple[bool, str | None]:
    """Reserve one validation-only comparison and persist its immutable inputs."""
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        _ensure_evidence_fields(session)
        if session.get("active_strategy_comparison_job_id"):
            return False, "a strategy comparison is already running"
        prior_holdout_reads = int(session.get("session_holdout_reads", 0))
        if prior_holdout_reads > 0 and not confirm_repeat:
            return (
                False,
                "holdout evidence has already been viewed; set confirm_repeat=true "
                "to compare a changed strategy set as post-holdout exploratory work",
            )
        round_payload = load_round(session_id, round_index)
        if round_payload is None:
            return False, "unknown round"
        if not strategies:
            return False, "at least one strategy is required"
        if len(strategies) > 4:
            return False, "at most four strategies may be compared"
        if len({item.id for item in strategies}) != len(strategies):
            return False, "strategy ids must be unique"
        target = strategy_comparison_path(session_id, comparison_id)
        if target.exists():
            return False, "strategy comparison id already exists"
        created_at = _utc_now()
        costs = _round_cost_model(round_payload)
        best_factor = str(round_payload.get("best_factor") or "")
        artifact = {
            "schema_version": STRATEGY_COMPARISON_SCHEMA_VERSION,
            "comparison_id": comparison_id,
            "session_id": session_id,
            "round_index": round_index,
            "job_id": job_id,
            "status": "queued",
            "evidence_status": (
                "post_holdout_adaptive"
                if prior_holdout_reads > 0
                else "validation_only"
            ),
            "prior_holdout_reads": prior_holdout_reads,
            "created_at": created_at,
            "completed_at": None,
            "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
            "costs": {
                "commission_bps": costs.commission_bps,
                "slippage_bps": costs.slippage_bps,
            },
            "strategies": [item.to_dict() for item in strategies],
            "strategy_results": [],
            "formula_fingerprint": hashlib.sha256(
                best_factor.encode("utf-8")
            ).hexdigest(),
            "holdout_fingerprint": (
                round_payload.get("holdout_fingerprint")
                or (round_payload.get("round_metadata") or {}).get(
                    "holdout_fingerprint"
                )
            ),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, json.dumps(artifact))
        summary = _comparison_summary(artifact)
        session.setdefault("strategy_comparisons", []).append(summary)
        session["active_strategy_comparison_job_id"] = job_id
        session["active_strategy_comparison"] = summary
        for round_summary in session.get("rounds", []):
            if int(round_summary.get("index", -1)) == round_index:
                round_summary["latest_strategy_comparison_id"] = comparison_id
        save_session(session)
        return True, None


def update_strategy_comparison_status(
    session_id: str,
    comparison_id: str,
    job_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    with _SESSION_IO_LOCK:
        artifact = load_strategy_comparison(session_id, comparison_id)
        if artifact is None or artifact.get("job_id") != job_id:
            return
        artifact["status"] = status
        artifact["error"] = error
        if status in {"done", "failed", "stopped"}:
            artifact["completed_at"] = _utc_now()
        atomic_write_text(
            strategy_comparison_path(session_id, comparison_id),
            json.dumps(artifact),
        )
        session = load_session(session_id)
        for index, item in enumerate(session.get("strategy_comparisons", [])):
            if item.get("comparison_id") == comparison_id:
                session["strategy_comparisons"][index] = _comparison_summary(artifact)
        if status in {"done", "failed", "stopped"}:
            session["active_strategy_comparison_job_id"] = None
            session["active_strategy_comparison"] = None
        else:
            session["active_strategy_comparison"] = _comparison_summary(artifact)
        save_session(session)


def run_strategy_comparison(
    session_id: str,
    round_index: int,
    *,
    comparison_id: str,
    job_id: str,
    panel: Panel,
    progress: Any = None,
    stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Compute an immutable validation-only strategy comparison artifact."""
    artifact = load_strategy_comparison(session_id, comparison_id)
    if artifact is None or artifact.get("job_id") != job_id:
        raise RuntimeError("strategy comparison reservation no longer exists")
    if stop is not None and stop():
        raise TrainingCancelled("strategy comparison cancelled")
    round_payload = load_round(session_id, round_index)
    if round_payload is None:
        raise ValueError("unknown round")
    metadata = dict(round_payload.get("round_metadata") or {})
    session = load_session(session_id)
    config = GPConfig.from_dict(metadata.get("config") or session["config"])
    boundaries = Boundaries.from_dict(session["boundaries"])
    split = split_from_boundaries(panel.dates, boundaries)
    best_factor = round_payload.get("best_factor")
    if not isinstance(best_factor, str):
        raise ValueError("round has no selected formula")
    strategies = [
        PortfolioStrategySpec.from_dict(item)
        for item in artifact.get("strategies", [])
        if isinstance(item, dict)
    ]
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("validating")
    results = compare_validation_strategies(
        from_json(best_factor),
        panel,
        forward_returns(panel, config.horizon, config.execution),
        split.valid,
        strategies,
        costs=_round_cost_model(round_payload),
        horizon=config.horizon,
        execution=config.execution,
        method=config.ic_method,
        min_names=config.min_names,
        validation_folds=getattr(config, "validation_folds", 3),
        fold_embargo=boundaries.embargo,
    )
    if stop is not None and stop():
        raise TrainingCancelled("strategy comparison cancelled")
    normalized_results = [
        {
            "strategy_id": item["strategy_id"],
            "spec": item["spec"],
            "role": None,
            "validation_backtest": item["validation_backtest"],
            # Explicit null prevents clients from mistaking validation evidence
            # for a locked-holdout report while retaining a stable result shape.
            "oos_backtest": None,
            "folds": item["folds"],
            "eligible": item["eligible"],
        }
        for item in results
    ]
    with _SESSION_IO_LOCK:
        current = load_strategy_comparison(session_id, comparison_id)
        if current is None or current.get("job_id") != job_id:
            raise RuntimeError("strategy comparison reservation was released")
        current["status"] = "done"
        current["completed_at"] = _utc_now()
        current["strategy_results"] = normalized_results
        atomic_write_text(
            strategy_comparison_path(session_id, comparison_id),
            json.dumps(current),
        )
        session = load_session(session_id)
        for index, item in enumerate(session.get("strategy_comparisons", [])):
            if item.get("comparison_id") == comparison_id:
                session["strategy_comparisons"][index] = _comparison_summary(current)
        session["active_strategy_comparison_job_id"] = None
        session["active_strategy_comparison"] = None
        save_session(session)
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("done")
    return current


def _plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_plan_id": str(plan["strategy_plan_id"]),
        "round_index": int(plan["round_index"]),
        "comparison_id": (
            str(plan["comparison_id"])
            if plan.get("comparison_id") is not None
            else None
        ),
        "primary_strategy_id": str(plan["primary_strategy_id"]),
        "strategy_ids": [
            str(item.get("id"))
            for item in plan.get("strategies", [])
            if isinstance(item, dict)
        ],
        "created_at": plan.get("created_at"),
    }


def list_finalization_plans(session_id: str) -> list[dict[str, Any]]:
    session = load_session(session_id)
    stored = session.get("finalization_plans")
    if isinstance(stored, list):
        return [dict(item) for item in stored if isinstance(item, dict)]
    return []


def load_finalization_plan(
    session_id: str,
    strategy_plan_id: str,
) -> dict[str, Any] | None:
    return _read_result(finalization_plan_path(session_id, strategy_plan_id))


def create_finalization_plan(
    session_id: str,
    round_index: int,
    *,
    comparison_id: str,
    primary_strategy_id: str,
    strategy_plan_id: str | None = None,
) -> dict[str, Any]:
    """Pin one completed validation comparison for a future holdout read."""
    with _SESSION_IO_LOCK:
        comparison = load_strategy_comparison(session_id, comparison_id)
        if comparison is None or int(comparison.get("round_index", -1)) != round_index:
            raise ValueError("unknown strategy comparison for this round")
        if comparison.get("status") != "done":
            raise ValueError("strategy comparison has not completed")
        strategies = [
            PortfolioStrategySpec.from_dict(item)
            for item in comparison.get("strategies", [])
            if isinstance(item, dict)
        ]
        if primary_strategy_id not in {item.id for item in strategies}:
            raise ValueError("primary_strategy_id is not in the comparison")
        primary_result = next(
            (
                item
                for item in comparison.get("strategy_results", [])
                if isinstance(item, dict)
                and item.get("strategy_id") == primary_strategy_id
            ),
            None,
        )
        if primary_result is None or not bool(primary_result.get("eligible")):
            raise ValueError(
                "primary strategy is ineligible because validation coverage or "
                "portfolio exposure checks failed"
            )
        plan_id = strategy_plan_id or uuid.uuid4().hex
        target = finalization_plan_path(session_id, plan_id)
        if target.exists():
            raise ValueError("strategy finalization plan id already exists")
        plan = {
            "schema_version": FINALIZATION_PLAN_SCHEMA_VERSION,
            "strategy_plan_id": plan_id,
            "session_id": session_id,
            "round_index": round_index,
            "comparison_id": comparison_id,
            "comparison_fingerprint": hashlib.sha256(
                json.dumps(
                    comparison,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "source_comparison_evidence_status": comparison.get(
                "evidence_status", "validation_only"
            ),
            "primary_strategy_id": primary_strategy_id,
            "strategies": [item.to_dict() for item in strategies],
            "costs": dict(comparison.get("costs") or {}),
            "portfolio_schema_version": comparison.get(
                "portfolio_schema_version", PORTFOLIO_SCHEMA_VERSION
            ),
            "formula_fingerprint": comparison.get("formula_fingerprint"),
            "holdout_fingerprint": comparison.get("holdout_fingerprint"),
            "created_at": _utc_now(),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, json.dumps(plan))
        session = load_session(session_id)
        session.setdefault("finalization_plans", []).append(_plan_summary(plan))
        for round_summary in session.get("rounds", []):
            if int(round_summary.get("index", -1)) == round_index:
                round_summary["latest_strategy_plan_id"] = plan_id
        save_session(session)
        return plan


def _ensure_compatibility_finalization_plan(
    session: dict[str, Any],
    round_payload: dict[str, Any],
    round_index: int,
    holdout_fingerprint_value: str,
) -> dict[str, Any]:
    """Persist the legacy no-body request as an explicit immutable Q20 plan."""
    plan_id = f"compat-q20-r{round_index}-{holdout_fingerprint_value[:16]}"
    existing = load_finalization_plan(str(session["id"]), plan_id)
    if existing is not None:
        return existing
    spec = default_strategy_spec()
    costs = _round_cost_model(round_payload)
    best_factor = str(round_payload.get("best_factor") or "")
    plan = {
        "schema_version": FINALIZATION_PLAN_SCHEMA_VERSION,
        "strategy_plan_id": plan_id,
        "session_id": str(session["id"]),
        "round_index": round_index,
        "comparison_id": None,
        "comparison_fingerprint": None,
        "source_comparison_evidence_status": (
            "post_holdout_adaptive"
            if int(session.get("session_holdout_reads", 0)) > 0
            else "validation_only"
        ),
        "primary_strategy_id": spec.id,
        "strategies": [spec.to_dict()],
        "costs": {
            "commission_bps": costs.commission_bps,
            "slippage_bps": costs.slippage_bps,
        },
        "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
        "formula_fingerprint": hashlib.sha256(
            best_factor.encode("utf-8")
        ).hexdigest(),
        "holdout_fingerprint": holdout_fingerprint_value,
        "source": "compatibility_default",
        "created_at": _utc_now(),
    }
    target = finalization_plan_path(str(session["id"]), plan_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(plan))
    session.setdefault("finalization_plans", []).append(_plan_summary(plan))
    for round_summary in session.get("rounds", []):
        if int(round_summary.get("index", -1)) == round_index:
            round_summary["latest_strategy_plan_id"] = plan_id
    return plan


def _legacy_finalization_id(round_index: int) -> str:
    return f"legacy-round-{round_index}"


def _finalization_summary(
    artifact: dict[str, Any],
    *,
    legacy: bool = False,
) -> dict[str, Any]:
    return {
        "evaluation_id": str(artifact["evaluation_id"]),
        "round_index": int(artifact["round_index"]),
        "status": str(artifact.get("status", "done")),
        "evidence_status": artifact.get("evidence_status"),
        "same_holdout_read_index": artifact.get("same_holdout_read_index"),
        "session_holdout_reads": int(artifact.get("session_holdout_reads", 0)),
        "holdout_fingerprint": artifact.get("holdout_fingerprint"),
        "strategy_plan_id": artifact.get("strategy_plan_id"),
        "comparison_id": artifact.get("comparison_id"),
        "primary_strategy_id": artifact.get("primary_strategy_id"),
        "started_at": artifact.get("started_at"),
        "completed_at": artifact.get("completed_at"),
        "report_available": artifact.get("report") is not None,
        "inherited_evidence_sources": list(
            artifact.get("inherited_evidence_sources") or []
        ),
        "inherited_test_reads": int(artifact.get("inherited_test_reads", 0)),
        "test_reads": int(artifact.get("test_reads", 0)),
        "legacy": legacy,
    }


def _legacy_finalizations(session_id: str) -> list[dict[str, Any]]:
    """Adapt embedded pre-v2 reports into read-only finalization summaries."""
    session = load_session(session_id)
    inherited = max(
        0,
        _int_or(session.get("inherited_test_reads"), 0),
    )
    summaries: list[dict[str, Any]] = []
    session_reads = 0
    for summary in list_rounds(session_id):
        payload = load_round(session_id, int(summary["index"]))
        if payload is None or payload.get("report") is None:
            continue
        session_reads += 1
        read_index = _int_or(
            (payload.get("round_metadata") or {}).get("test_read_index"),
            session_reads,
        )
        summaries.append(
            {
                "evaluation_id": _legacy_finalization_id(int(summary["index"])),
                "round_index": int(summary["index"]),
                "status": "done",
                "evidence_status": (
                    "locked_first_read"
                    if read_index == 1
                    else "repeated_same_holdout"
                ),
                "same_holdout_read_index": read_index,
                "session_holdout_reads": session_reads,
                "holdout_fingerprint": (
                    payload.get("holdout_fingerprint")
                    or (payload.get("round_metadata") or {}).get(
                        "holdout_fingerprint"
                    )
                    or "legacy-unknown"
                ),
                "started_at": (payload.get("round_metadata") or {}).get("started_at"),
                "completed_at": (payload.get("round_metadata") or {}).get(
                    "completed_at"
                ),
                "report_available": True,
                "inherited_evidence_sources": list(
                    session.get("inherited_evidence_sources") or []
                ),
                "inherited_test_reads": inherited,
                "test_reads": inherited + session_reads,
                "legacy": True,
            }
        )
    return summaries


def list_finalizations(session_id: str) -> list[dict[str, Any]]:
    session = load_session(session_id)
    stored = session.get("finalizations")
    if isinstance(stored, list) and stored:
        return [dict(item) for item in stored if isinstance(item, dict)]
    return _legacy_finalizations(session_id)


def _adapt_legacy_no_exposure(report: Any) -> Any:
    """Expose pre-health all-cash reports as invalid without rewriting artifacts."""
    if not isinstance(report, dict) or report.get("portfolio_health") is not None:
        return report
    metrics = dict(report.get("metrics") or {})
    avg_gross = metrics.get("avg_gross")
    avg_positions = metrics.get("avg_positions")
    try:
        no_exposure = float(avg_gross or 0.0) <= 1e-12 and float(
            avg_positions or 0.0
        ) <= 1e-12
    except (TypeError, ValueError):
        no_exposure = False
    if not no_exposure:
        return report
    calendar = _int_or(report.get("calendar_observations", report.get("observations")), 0)
    # Old artifacts encoded a cash-only path as valid-looking zero performance.
    # Adapt those values at read time so clients cannot mistake absence of a
    # portfolio for measured Sharpe or drawdown evidence.  The persisted
    # artifact remains untouched.
    metrics.update(
        {
            "gross_sharpe": None,
            "net_sharpe": None,
            "max_drawdown": None,
            "usable": False,
        }
    )
    report["metrics"] = metrics
    report["calendar_observations"] = calendar
    report["active_observations"] = 0
    report["observations"] = 0
    report["portfolio_health"] = {
        "valid": False,
        "reason": "no_exposure",
        "eligible_dates": calendar,
        "calendar_observations": calendar,
        "eligible_observations": calendar,
        "active_observations": 0,
        "two_sided_observations": 0,
        "exposure_coverage": 0.0,
        "two_sided_coverage": 0.0,
        "flat_factor_dates": calendar,
        "minimum_coverage": 0.60,
        "issues": ["legacy report contains no portfolio exposure"],
        "legacy_adapter": True,
    }
    integrity = dict(report.get("integrity") or {})
    issues = list(integrity.get("issues") or [])
    if "legacy report contains no portfolio exposure" not in issues:
        issues.append("legacy report contains no portfolio exposure")
    integrity.update({"valid": False, "issues": issues})
    report["integrity"] = integrity
    equity = list(report.get("normalized_equity") or [])
    if equity:
        report["normalized_equity"] = [equity[0]]
    return report


def _adapt_finalization_payload(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("report")
    source_oos = payload.get("oos_backtest")
    if source_oos is None and isinstance(report, dict):
        source_oos = report.get("oos_backtest")
    oos = _adapt_legacy_no_exposure(source_oos)
    payload["oos_backtest"] = oos
    if isinstance(report, dict):
        report["oos_backtest"] = oos
    for item in payload.get("strategy_results", []):
        if isinstance(item, dict):
            item["oos_backtest"] = _adapt_legacy_no_exposure(
                item.get("oos_backtest")
            )
    return payload


def load_finalization(
    session_id: str,
    evaluation_id: str,
) -> dict[str, Any] | None:
    payload = _read_result(finalization_path(session_id, evaluation_id))
    if payload is not None:
        return _adapt_finalization_payload(payload)
    match = re.fullmatch(r"legacy-round-(\d+)", evaluation_id)
    if match is None:
        return None
    round_index = int(match.group(1))
    round_payload = load_round(session_id, round_index)
    if round_payload is None or round_payload.get("report") is None:
        return None
    summary = next(
        (
            item
            for item in _legacy_finalizations(session_id)
            if item["evaluation_id"] == evaluation_id
        ),
        None,
    )
    if summary is None:
        return None
    return _adapt_finalization_payload({
        "schema_version": 0,
        **summary,
        "best_factor": round_payload.get("best_factor"),
        "report": round_payload.get("report"),
        "oos_backtest": round_payload.get("oos_backtest")
        or (round_payload.get("report") or {}).get("oos_backtest"),
        "context": round_payload.get("context"),
        "selection": round_payload.get("selection"),
    })


def _migrate_legacy_result(session: dict[str, Any]) -> None:
    """Pin a pre-schema latest result before a continuation can replace its alias."""
    if session.get("rounds"):
        return
    directory = session_dir(session["id"])
    legacy = _read_result(directory / "result.json")
    if legacy is None or legacy.get("report") is None:
        session.setdefault("rounds", [])
        return
    index = 0
    legacy = dict(legacy)
    legacy.setdefault("round_index", index)
    summary = _legacy_round_summary(session, legacy, index=index)
    legacy.setdefault("round_metadata", summary)
    target = round_path(session["id"], index)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        atomic_write_text(target, json.dumps(legacy))
    session["rounds"] = [summary]


def _adjustment_version() -> int:
    # Data normalization owns this version.  The fallback keeps old installations
    # readable while allowing the adjustment implementation to introduce its constant.
    from alphalineage.data import adjust

    return int(getattr(adjust, "ADJUSTMENT_SCHEMA_VERSION", 1))


def panel_fingerprint(panel: Panel) -> str:
    """Public stable fingerprint used to invalidate scores when cached data changes."""
    return _panel_fingerprint(panel)


def continuation_restart_reason(session_id: str) -> str | None:
    """Return why a checkpoint cannot safely continue under current semantics."""
    checkpoint = _read_result(session_dir(session_id) / "checkpoint.json")
    if checkpoint is None:
        return None
    saved_evolution = _int_or(checkpoint.get("evolution_version"), 1)
    if saved_evolution != EVOLUTION_VERSION:
        return (
            "This checkpoint uses an older evolution algorithm and cannot be continued "
            "without mixing incompatible populations."
        )
    rounds = list_rounds(session_id)
    if rounds and rounds[-1].get("restart_required"):
        return str(rounds[-1].get("invalid_reason") or "This session requires a clean restart.")
    return None


def _selected_lineage_node_id(store: LineageStore, tree: Node, generation: int) -> int | None:
    key = to_json(tree)
    same_generation = [
        node.id
        for node in store.nodes
        if node.generation == generation and to_json(node.tree) == key
    ]
    if same_generation:
        return same_generation[-1]
    matches = [node.id for node in store.nodes if to_json(node.tree) == key]
    return matches[-1] if matches else None


def lineage_for_round(session_id: str, index: int) -> dict[str, Any] | None:
    """Return the replayable cumulative lineage as it stood at a completed round."""
    summary = next((item for item in list_rounds(session_id) if item["index"] == index), None)
    if summary is None:
        return None
    lineage = _read_result(session_dir(session_id) / "lineage.json")
    if lineage is None:
        return None
    generation_boundary = summary.get("gen_end")
    if generation_boundary is not None:
        lineage["nodes"] = [
            node
            for node in lineage.get("nodes", [])
            if int(node.get("generation", 0)) <= int(generation_boundary)
        ]
    lineage["metadata"] = {
        **dict(lineage.get("metadata") or {}),
        "round_index": index,
        "generation_boundary": generation_boundary,
        "selected_lineage_node_id": summary.get("selected_lineage_node_id"),
    }
    return lineage


def _ensure_evidence_fields(session: dict[str, Any]) -> None:
    """Populate separated inherited/session evidence counters for old sessions."""
    legacy_reports = sum(
        bool(item.get("report_available")) for item in session.get("rounds", [])
    )
    if legacy_reports == 0:
        latest = _read_result(session_dir(session["id"]) / "result.json")
        legacy_reports = int(bool(latest is not None and latest.get("report") is not None))
    existing_total = max(0, _int_or(session.get("test_reads"), 0))
    session.setdefault("session_holdout_reads", legacy_reports)
    session.setdefault(
        "inherited_test_reads",
        max(0, existing_total - int(session["session_holdout_reads"])),
    )
    session.setdefault(
        "inherited_evidence_sources",
        [{"factor_id": factor_id} for factor_id in session.get("seed_factor_ids", [])],
    )
    session.setdefault("holdout_read_counts", {})
    session.setdefault("finalizations", [])
    session.setdefault("strategy_comparisons", [])
    session.setdefault("finalization_plans", [])
    session["test_reads"] = int(session["inherited_test_reads"]) + int(
        session["session_holdout_reads"]
    )


def holdout_fingerprint(
    panel: Panel,
    boundaries: Boundaries,
    config: GPConfig,
) -> str:
    """Fingerprint the exact frozen holdout data and reporting horizon."""
    test_dates = panel.dates[panel.dates >= pd.Timestamp(boundaries.test_start)]
    payload = {
        "panel": _panel_fingerprint(panel),
        "test_start": boundaries.test_start,
        "test_dates": [stamp.isoformat() for stamp in test_dates],
        "horizon": int(config.horizon),
        # Legacy (same-close) sessions keep their existing fingerprint; any other execution
        # timing is a different holdout measurement.
        **({} if config.execution == "close" else {"execution": config.execution}),
        "adjustment_version": _adjustment_version(),
        "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
        "validation_selection_version": VALIDATION_SELECTION_VERSION,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def claim_finalization(
    session_id: str,
    *,
    job_id: str,
    evaluation_id: str,
    round_index: int,
    holdout_fingerprint_value: str,
    confirm_repeat: bool,
    strategy_plan_id: str | None = None,
) -> tuple[bool, str | None]:
    """Reserve one atomic finalization and enforce explicit repeat consent."""
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        _ensure_evidence_fields(session)
        if session.get("active_finalization_job_id"):
            return False, "a holdout finalization is already running"
        round_payload = load_round(session_id, round_index)
        if round_payload is None:
            return False, "unknown round"
        plan: dict[str, Any] | None = None
        if strategy_plan_id is not None:
            plan = load_finalization_plan(session_id, strategy_plan_id)
            if plan is None or int(plan.get("round_index", -1)) != round_index:
                return False, "unknown strategy finalization plan for this round"
            pinned_holdout = plan.get("holdout_fingerprint")
            if pinned_holdout and pinned_holdout != holdout_fingerprint_value:
                return (
                    False,
                    "the pinned strategy plan belongs to a different holdout/data revision",
                )
        read_counts = dict(session.get("holdout_read_counts") or {})
        prior_same = max(0, _int_or(read_counts.get(holdout_fingerprint_value), 0))
        if prior_same == 0 and int(session.get("session_holdout_reads", 0)) > 0 and not read_counts:
            # Pre-fingerprint reports belong to this session's same frozen boundary.
            prior_same = int(session["session_holdout_reads"])
        if prior_same > 0 and not confirm_repeat:
            return (
                False,
                "this frozen holdout has already been read; set confirm_repeat=true "
                "to create explicitly exploratory repeated evidence",
            )
        if strategy_plan_id is None:
            plan = _ensure_compatibility_finalization_plan(
                session,
                round_payload,
                round_index,
                holdout_fingerprint_value,
            )
            strategy_plan_id = str(plan["strategy_plan_id"])
        session["active_finalization_job_id"] = job_id
        session["active_finalization"] = {
            "job_id": job_id,
            "evaluation_id": evaluation_id,
            "round_index": round_index,
            "holdout_fingerprint": holdout_fingerprint_value,
            "strategy_plan_id": strategy_plan_id,
            "comparison_id": plan.get("comparison_id") if plan is not None else None,
            "primary_strategy_id": (
                plan.get("primary_strategy_id") if plan is not None else None
            ),
            "status": "queued",
            "started_at": _utc_now(),
        }
        save_session(session)
        return True, None


def update_finalization_status(session_id: str, job_id: str, status: str) -> None:
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        active = dict(session.get("active_finalization") or {})
        if active.get("job_id") != job_id:
            return
        active["status"] = status
        session["active_finalization"] = active
        save_session(session)


def finish_finalization_job(
    session_id: str,
    job_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """Clear a finalization reservation without publishing a partial artifact."""
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        active = dict(session.get("active_finalization") or {})
        if active.get("job_id") != job_id:
            return
        session["active_finalization_job_id"] = None
        session["active_finalization"] = None
        session["last_finalization_job"] = {
            **active,
            "status": status,
            "completed_at": _utc_now(),
            "error": error,
        }
        save_session(session)


def claim_job(
    session_id: str,
    job_id: str,
    *,
    requested_generations: int | None = None,
    config: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
) -> bool:
    """Atomically reserve a session for one queued/running segment."""
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        if session.get("active_job_id"):
            return False
        now = _utc_now()
        session["active_job_id"] = job_id
        session["last_job_id"] = job_id
        session["last_job"] = {
            "id": job_id,
            "status": "queued",
            "queued_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "requested_generations": requested_generations,
            "config": dict(config or {}),
            "resources": dict(resources or {}),
            "error": None,
        }
        if requested_generations is not None:
            session["last_requested_generations"] = int(requested_generations)
        save_session(session)
        return True


def update_job_status(session_id: str, job_id: str, status: str) -> None:
    """Persist the active segment status without letting an old worker clobber a new job."""
    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        if session.get("active_job_id") != job_id:
            return
        last_job = dict(session.get("last_job") or {"id": job_id})
        now = _utc_now()
        last_job.update({"status": status, "updated_at": now})
        if status == "running" and not last_job.get("started_at"):
            last_job["started_at"] = now
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
            now = _utc_now()
            last_job.update(
                {
                    "status": status,
                    "updated_at": now,
                    "completed_at": now,
                    "error": error,
                }
            )
            session["last_job"] = last_job
            committed = any(
                item.get("job_id") == job_id for item in session.get("segments", [])
            )
            if not committed and status in {"stopped", "failed", "interrupted"}:
                session.setdefault("segments", []).append(
                    {
                        "index": len(session.get("segments", [])),
                        "job_id": job_id,
                        "universe": session.get("universe"),
                        "config": last_job.get("config") or session.get("config", {}),
                        "requested_generations": last_job.get("requested_generations"),
                        "gen_start": None,
                        "gen_end": None,
                        "new_trials": 0,
                        "status": status,
                        "resources": last_job.get("resources") or None,
                        "termination_reason": (
                            "user_stopped" if status == "stopped" else status
                        ),
                        "report_cancelled": True,
                        "report_available": False,
                        "started_at": last_job.get("started_at"),
                        "completed_at": now,
                    }
                )
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
    inherited_evidence_sources: Sequence[dict[str, Any]] = (),
    boundaries: Boundaries,
    trial_baseline: int,
    test_reads_baseline: int,
    created_at: str,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and persist a new session's ``session.json`` (no segment run yet)."""
    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "id": _slug(name),
        "name": name,
        "created_at": created_at,
        "updated_at": created_at,
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
        "inherited_test_reads": int(test_reads_baseline),
        "inherited_evidence_sources": (
            [dict(item) for item in inherited_evidence_sources]
            if inherited_evidence_sources
            else [{"factor_id": factor_id} for factor_id in seed_factor_ids]
        ),
        "session_holdout_reads": 0,
        "holdout_read_counts": {},
        "test_reads": int(test_reads_baseline),
        "segments": [],
        "rounds": [],
        "finalizations": [],
        "strategy_comparisons": [],
        "finalization_plans": [],
        "last_requested_generations": int(config.generations),
        "last_job_id": None,
        "active_job_id": None,
        "last_job": None,
        "active_finalization_job_id": None,
        "active_finalization": None,
        "last_finalization_job": None,
        "active_strategy_comparison_job_id": None,
        "active_strategy_comparison": None,
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
    _migrate_legacy_result(session)
    session.setdefault("rounds", [])
    # Persist a migrated legacy result before this segment can replace result.json.
    save_session(session)
    boundaries = Boundaries.from_dict(session["boundaries"])
    panel_fingerprint = _panel_fingerprint(panel)
    wall_started_at = (
        (session.get("last_job") or {}).get("started_at")
        or (session.get("last_job") or {}).get("queued_at")
        or _utc_now()
    )

    split = split_from_boundaries(panel.dates, boundaries)  # ValueError if locked segs empty (P1)
    train_panel = Panel({f: df.loc[df.index.isin(split.train)] for f, df in panel.fields.items()})
    test_start = pd.Timestamp(boundaries.test_start)
    if len(train_panel.dates) and train_panel.dates.max() >= test_start:
        raise RuntimeError("train panel leaked a locked test date (invariant 1)")

    checkpoint = directory / "checkpoint.json"
    lineage_path = directory / "lineage.json"
    warm = checkpoint.exists()
    if warm:
        restart_reason = continuation_restart_reason(session_id)
        if restart_reason:
            raise RuntimeError(f"Restart with the same setup: {restart_reason}")
        completed_rounds = list_rounds(session_id)
        prior_fingerprint = (
            completed_rounds[-1].get("panel_fingerprint")
            if completed_rounds
            else (session.get("segments") or [{}])[-1].get("panel_fingerprint")
        )
        if prior_fingerprint is not None and prior_fingerprint != panel_fingerprint:
            # The population structure is compatible, but every cached fitness belongs
            # to a different data revision and must be recomputed before evolution.
            rescore = True

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
    ) -> tuple[
        GP,
        ValidationSelection,
        list[Node],
        int,
        int,
        dict[str, float],
    ]:
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
            gp.run(generations=target, stop=stop)
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
            gp.run(generations=target, seeds=list(seeds), stop=stop)
        training_seconds = time.monotonic() - training_started

        gp.save_checkpoint(checkpoint)
        store.save(lineage_path)
        if progress is not None and hasattr(progress, "set_phase"):
            progress.set_phase("validating")
        validation_started = time.monotonic()
        fwd = forward_returns(panel, gp.config.horizon, gp.config.execution)
        searched_candidates = gp.searched_individuals()
        selection_result = select_validation_candidate(
            searched_candidates,
            panel,
            fwd,
            split.valid,
            method=gp.config.ic_method,
            parsimony=gp.config.parsimony,
            complexity_penalty_mode=getattr(
                gp.config,
                "complexity_penalty_mode",
                "per_node",
            ),
            complexity_penalty_value=getattr(
                gp.config,
                "resolved_complexity_penalty_value",
                gp.config.parsimony,
            ),
            max_nodes=gp.config.max_nodes,
            validation_folds=getattr(gp.config, "validation_folds", 3),
            fold_embargo=boundaries.embargo,
            min_names=gp.config.min_names,
            workers=workers,
            memory_budget_bytes=memory_budget,
        )
        record_validation = getattr(gp, "record_formula_validation_scores", None)
        if callable(record_validation):
            record_validation(selection_result.candidate_validation_scores)
            gp.save_checkpoint(checkpoint)
        trials = orient_training_candidates(searched_candidates)
        return gp, selection_result, trials, gen_start, target, {
            "training_seconds": training_seconds,
            "validation_seconds": time.monotonic() - validation_started,
            "reporting_seconds": 0.0,
            "total_seconds": time.monotonic() - started,
        }

    if lease_context is None:
        gp, selection_result, report_trials, gen_start, target, timings = execute()
    else:
        with lease_context as lease:
            gp, selection_result, report_trials, gen_start, target, timings = execute(lease)

    session["cumulative_trials"] = gp.trial_count
    _ensure_evidence_fields(session)
    session["last_job_id"] = job_id
    termination_reason = gp.termination_reason
    stopped_early = termination_reason == "user_stopped"
    resolved_resources = resources.to_dict() if resources is not None else None
    completed_at = _utc_now()
    segment_index = len(session["segments"])
    segment_status = "stopped" if stopped_early else "done"
    selected_lineage_node_id = _selected_lineage_node_id(
        store,
        selection_result.source_tree,
        gp.generation,
    )
    selection = selection_result.metadata()
    coverage_source = getattr(gp, "formula_exploration", {})
    if callable(coverage_source):
        coverage_source = coverage_source()
    parameter_coverage = (
        dict(coverage_source) if isinstance(coverage_source, dict) else {}
    )
    evidence_status = (
        "post_holdout_adaptive"
        if int(session.get("session_holdout_reads", 0)) > 0
        else "validation_only"
    )
    frozen_holdout_fingerprint = holdout_fingerprint(panel, boundaries, config)
    segment = {
        "index": segment_index,
        "job_id": job_id,
        "universe": session["universe"],
        "universe_definition": session.get("universe_definition"),
        "config": config.to_dict(),
        "requested_generations": int(generations),
        "gen_start": gen_start,
        "gen_end": gp.generation,
        "new_trials": gp.trial_count - prev_cumulative,
        "status": segment_status,
        "resources": resolved_resources,
        "termination_reason": termination_reason,
        "report_cancelled": False,
        "report_available": False,
        "started_at": wall_started_at,
        "completed_at": completed_at,
        "timings": timings,
        "panel_fingerprint": panel_fingerprint,
        "scorer_version": SCORER_VERSION,
        "evolution_version": EVOLUTION_VERSION,
        "adjustment_version": _adjustment_version(),
        "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
        "validation_selection_version": VALIDATION_SELECTION_VERSION,
        "selected_lineage_node_id": selected_lineage_node_id,
        "selection": selection,
        "parameter_coverage": parameter_coverage,
        "evidence_status": evidence_status,
    }
    session["segments"].append(segment)
    session["last_requested_generations"] = int(generations)

    result = {
        "best_factor": to_json(selection_result.oriented_tree),
        "report": None,
        "oos_backtest": None,
        "validation_only": True,
        "generations": gp.generation,
        "history": gp.history,
        "lineage": store.to_dict(),
        "session_id": session_id,
        "segment": segment_index,
        "test_reads": session["test_reads"],
        "session_holdout_reads": int(session["session_holdout_reads"]),
        "inherited_test_reads": int(session["inherited_test_reads"]),
        "inherited_evidence_sources": list(
            session.get("inherited_evidence_sources") or []
        ),
        "holdout_fingerprint": frozen_holdout_fingerprint,
        "cumulative_trials": session["cumulative_trials"],
        "repeated_oos_warning": int(session["session_holdout_reads"]) > 1,
        "test_read_index": None,
        "evidence_status": evidence_status,
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
        "report_cancelled": False,
        "selection": selection,
        "parameter_coverage": parameter_coverage,
        "report_trials": [to_json(tree) for tree in report_trials],
        "searched_trials": int(session["trial_baseline"]) + gp.trial_count,
        "n_user_operators": user_operator_count(allowed_operators),
    }
    # A committed generation produces one immutable validation round. The locked
    # holdout remains untouched until the user explicitly finalizes a selected round.
    round_index = len(session["rounds"])
    while round_path(session_id, round_index).exists():
        round_index += 1
    round_metadata = {
        "schema_version": ROUND_SCHEMA_VERSION,
        "index": round_index,
        "segment_index": segment_index,
        "validation_only": True,
        "report_available": False,
        "finalization_available": False,
        "latest_finalization_id": None,
        "latest_strategy_comparison_id": None,
        "latest_strategy_plan_id": None,
        "test_read_index": None,
        "evidence_status": evidence_status,
        "status": segment_status,
        "started_at": wall_started_at,
        "completed_at": completed_at,
        "requested_generations": int(generations),
        "gen_start": gen_start,
        "gen_end": gp.generation,
        "termination_reason": termination_reason,
        "selected_lineage_node_id": selected_lineage_node_id,
        "resources": resolved_resources,
        "timings": timings,
        "panel_fingerprint": panel_fingerprint,
        "holdout_fingerprint": frozen_holdout_fingerprint,
        "scorer_version": SCORER_VERSION,
        "evolution_version": EVOLUTION_VERSION,
        "adjustment_version": _adjustment_version(),
        "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
        "validation_selection_version": VALIDATION_SELECTION_VERSION,
        "selection": selection,
        "parameter_coverage": parameter_coverage,
        "config": config.to_dict(),
        "searched_trials": result["searched_trials"],
        "n_user_operators": result["n_user_operators"],
    }
    result["round_index"] = round_index
    result["round_metadata"] = round_metadata
    artifact_path = round_path(session_id, round_index)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(artifact_path, json.dumps(result))
    atomic_write_text(directory / "result.json", json.dumps(result))
    summary = _round_summary_from_result(result, index=round_index)
    session["rounds"].append(summary)
    segment["round_index"] = round_index
    save_session(session)
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("done")
    return {key: value for key, value in result.items() if key != "report_trials"}


def finalize_round(
    session_id: str,
    round_index: int,
    *,
    job_id: str,
    evaluation_id: str,
    panel: Panel,
    strategy_plan_id: str | None = None,
    progress: Any = None,
    stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable validation round against the frozen holdout once.

    The artifact and counters are committed only after ``build_report`` has crossed
    its non-cancellable locked-test boundary and returned a complete report.
    """
    started_monotonic = time.monotonic()
    started_at = _utc_now()
    session = load_session(session_id)
    boundaries = Boundaries.from_dict(session["boundaries"])
    round_payload = load_round(session_id, round_index)
    if round_payload is None:
        raise ValueError("unknown round")
    metadata = dict(round_payload.get("round_metadata") or {})
    config = GPConfig.from_dict(metadata.get("config") or session["config"])
    split = split_from_boundaries(panel.dates, boundaries)
    fingerprint = holdout_fingerprint(panel, boundaries, config)
    active = dict(session.get("active_finalization") or {})
    if (
        active.get("job_id") != job_id
        or active.get("evaluation_id") != evaluation_id
        or active.get("holdout_fingerprint") != fingerprint
        or active.get("strategy_plan_id") != strategy_plan_id
    ):
        raise RuntimeError("finalization reservation no longer matches this holdout")

    best_factor = round_payload.get("best_factor")
    if not isinstance(best_factor, str):
        raise ValueError("round has no selected formula")
    best_tree = from_json(best_factor)
    raw_trials = round_payload.get("report_trials") or [best_factor]
    trials = [from_json(item) for item in raw_trials if isinstance(item, str)]
    if not trials:
        trials = [best_tree]
    searched_trials = _int_or(
        round_payload.get("searched_trials", metadata.get("searched_trials")),
        len(trials),
    )
    n_user_operators = _int_or(
        round_payload.get("n_user_operators", metadata.get("n_user_operators")),
        0,
    )
    if strategy_plan_id is None:
        strategy_specs = [default_strategy_spec()]
        primary_strategy_id = strategy_specs[0].id
        comparison_id = None
        comparison_fingerprint = None
        comparison_evidence_status = "compatibility_default"
        cost_model = _round_cost_model(round_payload)
    else:
        strategy_plan = load_finalization_plan(session_id, strategy_plan_id)
        if (
            strategy_plan is None
            or int(strategy_plan.get("round_index", -1)) != round_index
        ):
            raise ValueError("unknown strategy finalization plan for this round")
        if strategy_plan.get("holdout_fingerprint") not in {None, fingerprint}:
            raise RuntimeError("strategy plan holdout fingerprint no longer matches")
        strategy_specs = [
            PortfolioStrategySpec.from_dict(item)
            for item in strategy_plan.get("strategies", [])
            if isinstance(item, dict)
        ]
        primary_strategy_id = str(strategy_plan.get("primary_strategy_id") or "")
        comparison_id = str(strategy_plan.get("comparison_id") or "") or None
        comparison_fingerprint = strategy_plan.get("comparison_fingerprint")
        comparison_evidence_status = str(
            strategy_plan.get("source_comparison_evidence_status")
            or "validation_only"
        )
        pinned_costs = dict(strategy_plan.get("costs") or {})
        cost_model = TransactionCostModel(
            commission_bps=float(pinned_costs.get("commission_bps", 1.0)),
            slippage_bps=float(pinned_costs.get("slippage_bps", 5.0)),
        )

    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("reporting")
    report_cache_path = session_dir(session_id) / "report_stats.json"
    cache_context = _report_cache_context(
        session,
        panel,
        boundaries,
        config,
        panel_fingerprint=_panel_fingerprint(panel),
    )
    cache_context = hashlib.sha256(
        (
            cache_context
            + json.dumps(
                {
                    "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
                    "strategies": [item.to_dict() for item in strategy_specs],
                    "primary_strategy_id": primary_strategy_id,
                    "commission_bps": cost_model.commission_bps,
                    "slippage_bps": cost_model.slippage_bps,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ).encode("utf-8")
    ).hexdigest()
    report_summaries = _load_report_cache(report_cache_path, cache_context)
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
            best_tree,
            trials,
            split,
            panel,
            searched_trials=searched_trials,
            min_names=config.min_names,
            horizon=config.horizon,
            execution=config.execution,
            ic_method=config.ic_method,
            n_user_operators=n_user_operators,
            strategy_specs=strategy_specs,
            primary_strategy_id=primary_strategy_id,
            costs=cost_model,
            progress=report_progress,
            stop=stop,
            on_finalizing=on_finalizing,
            summary_cache=report_summaries,
            summary_key=_report_tree_key,
            fwd=forward_returns(panel, config.horizon, config.execution),
        )
    finally:
        _save_report_cache(report_cache_path, cache_context, report_summaries)

    selection = dict(round_payload.get("selection") or {})
    report["validation_passed"] = bool(selection.get("validated"))
    report["validation_reason"] = selection.get("reason")
    if not report["validation_passed"]:
        report["significant"] = False

    with _SESSION_IO_LOCK:
        session = load_session(session_id)
        _ensure_evidence_fields(session)
        active = dict(session.get("active_finalization") or {})
        if (
            active.get("job_id") != job_id
            or active.get("strategy_plan_id") != strategy_plan_id
        ):
            raise RuntimeError("finalization reservation was released before commit")
        read_counts = dict(session.get("holdout_read_counts") or {})
        same_read_index = _int_or(read_counts.get(fingerprint), 0) + 1
        session_read_index = int(session["session_holdout_reads"]) + 1
        evidence_status = (
            "locked_first_read"
            if session_read_index == 1
            else "repeated_same_holdout"
        )
        source_status = str(metadata.get("evidence_status") or "validation_only")
        if (
            evidence_status != "locked_first_read"
            or source_status == "post_holdout_adaptive"
            or comparison_evidence_status == "post_holdout_adaptive"
        ):
            report["significant"] = False
        completed_at = _utc_now()
        artifact = {
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "evaluation_id": evaluation_id,
            "round_index": round_index,
            "status": "done",
            "evidence_status": evidence_status,
            "source_round_evidence_status": source_status,
            "source_comparison_evidence_status": comparison_evidence_status,
            "same_holdout_read_index": same_read_index,
            "session_holdout_reads": session_read_index,
            "holdout_fingerprint": fingerprint,
            "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
            "strategy_plan_id": strategy_plan_id,
            "comparison_id": comparison_id,
            "comparison_fingerprint": comparison_fingerprint,
            "primary_strategy_id": report.get(
                "primary_strategy_id", primary_strategy_id
            ),
            "strategy_results": list(report.get("strategy_results") or []),
            "started_at": started_at,
            "completed_at": completed_at,
            "report_available": True,
            "inherited_evidence_sources": list(
                session.get("inherited_evidence_sources") or []
            ),
            "inherited_test_reads": int(session["inherited_test_reads"]),
            "test_reads": int(session["inherited_test_reads"]) + session_read_index,
            "best_factor": best_factor,
            "report": report,
            "oos_backtest": report.get("oos_backtest"),
            "context": {
                **report_context(split, config, embargo=boundaries.embargo),
                "weighting_scheme": next(
                    item.scheme
                    for item in strategy_specs
                    if item.id == primary_strategy_id
                ),
                "quantile": next(
                    item.quantile
                    for item in strategy_specs
                    if item.id == primary_strategy_id
                ),
                "commission_bps": cost_model.commission_bps,
                "slippage_bps": cost_model.slippage_bps,
                "strategies": [item.to_dict() for item in strategy_specs],
                "universe": session["universe"],
                "universe_revision": (
                    session.get("universe_definition") or {}
                ).get("fingerprint"),
                "universe_definition": session.get("universe_definition"),
                "as_of": session["as_of"],
            },
            "selection": selection,
            "timings": {
                "reporting_seconds": time.monotonic() - started_monotonic,
                "total_seconds": time.monotonic() - started_monotonic,
            },
        }
        target = finalization_path(session_id, evaluation_id)
        if target.exists():
            raise RuntimeError("finalization evaluation id already exists")

        # Adapt legacy summaries before the first native finalization is added.
        if not session.get("finalizations"):
            session["finalizations"] = _legacy_finalizations(session_id)
        atomic_write_text(target, json.dumps(artifact))
        read_counts[fingerprint] = same_read_index
        session["holdout_read_counts"] = read_counts
        session["session_holdout_reads"] = session_read_index
        session["test_reads"] = artifact["test_reads"]
        summary = _finalization_summary(artifact)
        session.setdefault("finalizations", []).append(summary)
        for round_summary in session.get("rounds", []):
            if int(round_summary.get("index", -1)) == round_index:
                round_summary["finalization_available"] = True
                round_summary["latest_finalization_id"] = evaluation_id
        session["active_finalization_job_id"] = None
        session["active_finalization"] = None
        session["last_finalization_job"] = {
            "job_id": job_id,
            "evaluation_id": evaluation_id,
            "round_index": round_index,
            "strategy_plan_id": strategy_plan_id,
            "comparison_id": comparison_id,
            "primary_strategy_id": primary_strategy_id,
            "status": "done",
            "completed_at": completed_at,
        }
        save_session(session)
    if progress is not None and hasattr(progress, "set_phase"):
        progress.set_phase("done")
    return artifact
