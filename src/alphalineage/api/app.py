"""P5-T1 / P7 - the FastAPI app: submit GP searches and extend the system safely.

Lightweight by design: an in-process job runner and the cached Parquet panel - no Redis, no
Postgres. User operators are registered as typed macros (data, not code - invariant 5); custom
universes are point-in-time. The `get_panel` dependency is overridable so tests inject a
synthetic panel and never touch the network or the cache.

Not investment advice. Research output only.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from alphalineage.api.jobs import JobStore
from alphalineage.api.service import run_search
from alphalineage.core.extensions import (
    USER_OPERATORS,
    InvalidOperator,
    register_operator,
    unregister_operator,
)
from alphalineage.core.gp import GPConfig
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import REGISTRY, Primitive
from alphalineage.core.types import DType
from alphalineage.data import paths
from alphalineage.data.universe import Membership, Universe, sample_universe

app = FastAPI(title="AlphaLineage", version="0.1.0")
# Allow the browser `app` build (Vite dev server) to call the local backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_jobs = JobStore()
_universes: dict[str, Universe] = {}

_DEFAULT_UNIVERSE = "sp500-lite"
_DEFAULT_AS_OF = "2026-06-01"
_WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


# --- models ----------------------------------------------------------------------
class OperatorSpec(BaseModel):
    name: str
    arg_types: list[str]
    out_type: str
    body: dict[str, Any]


class MembershipSpec(BaseModel):
    symbol: str
    entry: str
    exit: str | None = None


class UniverseSpec(BaseModel):
    name: str
    memberships: list[MembershipSpec]


class RunRequest(BaseModel):
    universe: str = _DEFAULT_UNIVERSE
    config: dict[str, Any] = Field(default_factory=dict)
    operators: list[OperatorSpec] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    status: str


class WorkspaceSnapshot(BaseModel):
    id: str | None = None
    name: str = "Workspace"
    version: int = Field(ge=1)
    savedAt: str
    run: dict[str, Any] | None = None
    universes: list[UniverseSpec] = Field(default_factory=list)
    operators: list[OperatorSpec] = Field(default_factory=list)
    universeDraft: dict[str, Any] | None = None
    operatorDraft: dict[str, Any] | None = None
    ui: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSummary(BaseModel):
    id: str
    name: str
    savedAt: str
    hasRun: bool


def _primitive_info(prim: Primitive) -> dict[str, Any]:
    return {
        "name": prim.name,
        "kind": prim.kind.value,
        "arg_types": [t.value for t in prim.arg_types],
        "out_type": prim.out_type.value,
        "user": prim.macro_body is not None,
    }


def _register(spec: OperatorSpec) -> Primitive:
    try:
        return register_operator(
            spec.name, [DType(t) for t in spec.arg_types], DType(spec.out_type), spec.body
        )
    except (InvalidOperator, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _model_dump(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if dump is not None:
        return dump(mode="json")
    return model.dict()


def _new_workspace_id(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip().lower()).strip("-._")
    return f"{slug or 'workspace'}-{uuid.uuid4().hex[:8]}"


def _workspace_path(workspace_id: str) -> Path:
    if not _WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise HTTPException(status_code=400, detail="invalid workspace id")
    return paths.workspaces_dir() / f"{workspace_id}.json"


def _workspace_snapshot_from_file(path: Path) -> WorkspaceSnapshot:
    try:
        return WorkspaceSnapshot(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"invalid workspace file: {path.name}") from exc


def _save_workspace_snapshot(snapshot: WorkspaceSnapshot) -> WorkspaceSnapshot:
    workspace_id = snapshot.id or _new_workspace_id(snapshot.name)
    if not _WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise HTTPException(status_code=400, detail="invalid workspace id")
    payload = _model_dump(snapshot)
    payload["id"] = workspace_id
    target = _workspace_path(workspace_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return WorkspaceSnapshot(**payload)


def _universe_from_spec(spec: UniverseSpec) -> Universe:
    members = [
        Membership(
            m.symbol,
            pd.Timestamp(m.entry),
            pd.Timestamp(m.exit) if m.exit else None,
        )
        for m in spec.memberships
    ]
    return Universe(spec.name, members)


def _universe_to_spec(universe: Universe) -> dict[str, Any]:
    return {
        "name": universe.name,
        "memberships": [
            {
                "symbol": m.symbol,
                "entry": m.entry.date().isoformat(),
                "exit": m.exit.date().isoformat() if m.exit is not None else None,
            }
            for m in universe.memberships
        ],
    }


def _persist_universe(spec: UniverseSpec) -> Universe:
    universe = _universe_from_spec(spec)
    _universes[spec.name] = universe
    universe.save()
    return universe


def _load_persisted_universes() -> None:
    universe_dir = paths.universe_dir()
    if not universe_dir.exists():
        return
    for source in universe_dir.glob("*.parquet"):
        if source.stem in _universes:
            continue
        try:
            _universes[source.stem] = Universe.load(source.stem, source)
        except Exception:
            continue


def _save_run_workspace(job_id: str, result: Any) -> None:
    if not isinstance(result, dict):
        return
    _save_workspace_snapshot(
        WorkspaceSnapshot(
            id=f"run-{job_id}",
            name=f"Run {job_id[:8]}",
            version=1,
            savedAt=_now_iso(),
            run=result,
            ui={"selectedTab": "dashboard", "source": "backend-run"},
        )
    )


def get_panel() -> Panel:
    """Build the working panel from the cached universe (overridden in tests)."""
    symbols = sample_universe(_DEFAULT_UNIVERSE).members_asof(_DEFAULT_AS_OF)
    return Panel.from_cache(symbols)


# --- endpoints -------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/primitives")
def list_primitives() -> list[dict[str, Any]]:
    """The primitive palette (built-ins + user operators) for the operator composer."""
    return [_primitive_info(p) for p in REGISTRY.values()]


@app.post("/operators")
def add_operator(spec: OperatorSpec) -> dict[str, Any]:
    return _primitive_info(_register(spec))


@app.get("/operators")
def list_operators() -> list[dict[str, Any]]:
    return [_primitive_info(p) for p in USER_OPERATORS.values()]


@app.delete("/operators/{name}")
def remove_operator(name: str) -> dict[str, str]:
    unregister_operator(name)
    return {"removed": name}


@app.get("/universes")
def list_universes() -> list[dict[str, Any]]:
    _load_persisted_universes()
    sample = sample_universe(_DEFAULT_UNIVERSE)
    custom = [
        {**_universe_to_spec(universe), "symbols": universe.all_symbols(), "source": "custom"}
        for universe in sorted(_universes.values(), key=lambda u: u.name)
    ]
    return [
        {**_universe_to_spec(sample), "symbols": sample.all_symbols(), "source": "sample"},
        *custom,
    ]


@app.post("/universes")
def add_universe(spec: UniverseSpec) -> dict[str, Any]:
    universe = _persist_universe(spec)
    return {"name": spec.name, "symbols": universe.all_symbols()}


@app.get("/workspaces", response_model=list[WorkspaceSummary])
def list_workspaces() -> list[WorkspaceSummary]:
    workspace_dir = paths.workspaces_dir()
    if not workspace_dir.exists():
        return []
    workspace_paths = list(workspace_dir.glob("*.json"))
    summaries = [
        WorkspaceSummary(
            id=snapshot.id or path.stem,
            name=snapshot.name,
            savedAt=snapshot.savedAt,
            hasRun=snapshot.run is not None,
        )
        for path, snapshot in (
            (path, _workspace_snapshot_from_file(path)) for path in workspace_paths
        )
    ]
    return sorted(summaries, key=lambda item: item.savedAt, reverse=True)


@app.post("/workspaces", response_model=WorkspaceSnapshot)
def save_workspace(snapshot: WorkspaceSnapshot) -> WorkspaceSnapshot:
    for spec in snapshot.universes:
        _persist_universe(spec)
    return _save_workspace_snapshot(snapshot)


@app.get("/workspaces/{workspace_id}", response_model=WorkspaceSnapshot)
def get_workspace(workspace_id: str) -> WorkspaceSnapshot:
    target = _workspace_path(workspace_id)
    if not target.exists():
        raise HTTPException(status_code=404, detail="unknown workspace")
    return _workspace_snapshot_from_file(target)


@app.delete("/workspaces/{workspace_id}")
def delete_workspace(workspace_id: str) -> dict[str, str]:
    target = _workspace_path(workspace_id)
    if not target.exists():
        raise HTTPException(status_code=404, detail="unknown workspace")
    target.unlink()
    return {"removed": workspace_id}


@app.post("/runs", response_model=JobResponse)
def submit_run(req: RunRequest, panel: Panel = Depends(get_panel)) -> JobResponse:  # noqa: B008
    for spec in req.operators:  # register user operators before the search sees them
        _register(spec)
    _load_persisted_universes()
    if req.universe in _universes:  # a custom point-in-time universe overrides the default panel
        symbols = _universes[req.universe].members_asof(_DEFAULT_AS_OF)
        panel = Panel.from_cache(symbols)
    config = GPConfig.from_dict(req.config)
    job_id = _jobs.submit(run_search, config, panel, on_success=_save_run_workspace)
    return JobResponse(job_id=job_id, status="queued")


@app.get("/runs/{job_id}")
def get_run(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {"job_id": job.id, "status": job.status, "result": job.result, "error": job.error}


@app.get("/runs/{job_id}/lineage")
def get_lineage(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None or not isinstance(job.result, dict):
        raise HTTPException(status_code=404, detail="no lineage yet")
    lineage: dict[str, Any] = job.result.get("lineage", {})
    return lineage
