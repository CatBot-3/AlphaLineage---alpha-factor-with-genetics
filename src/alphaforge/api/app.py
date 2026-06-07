"""P5-T1 / P7 — the FastAPI app: submit GP searches and extend the system safely.

Lightweight by design: an in-process job runner and the cached Parquet panel — no Redis, no
Postgres. User operators are registered as typed macros (data, not code — invariant 5); custom
universes are point-in-time. The `get_panel` dependency is overridable so tests inject a
synthetic panel and never touch the network or the cache.

Not investment advice. Research output only.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from alphaforge.api.jobs import JobStore
from alphaforge.api.service import run_search
from alphaforge.core.extensions import (
    USER_OPERATORS,
    InvalidOperator,
    register_operator,
    unregister_operator,
)
from alphaforge.core.gp import GPConfig
from alphaforge.core.panel import Panel
from alphaforge.core.primitives import REGISTRY, Primitive
from alphaforge.core.types import DType
from alphaforge.data.universe import Membership, Universe, sample_universe

app = FastAPI(title="AlphaForge", version="0.1.0")
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


@app.post("/universes")
def add_universe(spec: UniverseSpec) -> dict[str, Any]:
    members = [
        Membership(
            m.symbol,
            pd.Timestamp(m.entry),
            pd.Timestamp(m.exit) if m.exit else None,
        )
        for m in spec.memberships
    ]
    universe = Universe(spec.name, members)
    _universes[spec.name] = universe
    return {"name": spec.name, "symbols": universe.all_symbols()}


@app.post("/runs", response_model=JobResponse)
def submit_run(req: RunRequest, panel: Panel = Depends(get_panel)) -> JobResponse:  # noqa: B008
    for spec in req.operators:  # register user operators before the search sees them
        _register(spec)
    if req.universe in _universes:  # a custom point-in-time universe overrides the default panel
        symbols = _universes[req.universe].members_asof(_DEFAULT_AS_OF)
        panel = Panel.from_cache(symbols)
    config = GPConfig.from_dict(req.config)
    job_id = _jobs.submit(run_search, config, panel)
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
