"""P5-T1 — the FastAPI app: submit a GP search as a background job, fetch its result.

Lightweight by design (invariant-free packaging choice): an in-process job runner and the
cached Parquet panel — no Redis, no Postgres. The `get_panel` dependency is overridable so
tests inject a synthetic panel and never touch the network or the cache.

Not investment advice. Research output only.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from alphaforge.api.jobs import JobStore
from alphaforge.api.service import run_search
from alphaforge.core.gp import GPConfig
from alphaforge.core.panel import Panel
from alphaforge.data.universe import sample_universe

app = FastAPI(title="AlphaForge", version="0.1.0")
_jobs = JobStore()

_DEFAULT_UNIVERSE = "sp500-lite"
_DEFAULT_AS_OF = "2026-06-01"


class RunRequest(BaseModel):
    universe: str = _DEFAULT_UNIVERSE
    config: dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    job_id: str
    status: str


def get_panel() -> Panel:
    """Build the working panel from the cached universe (overridden in tests)."""
    symbols = sample_universe(_DEFAULT_UNIVERSE).members_asof(_DEFAULT_AS_OF)
    return Panel.from_cache(symbols)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=JobResponse)
def submit_run(req: RunRequest, panel: Panel = Depends(get_panel)) -> JobResponse:  # noqa: B008
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
