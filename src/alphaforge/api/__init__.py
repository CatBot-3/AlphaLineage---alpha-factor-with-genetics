"""Phase 5 — the HTTP API: submit GP searches as background jobs and fetch results.

Lightweight (FastAPI + in-process jobs); start with ``uvicorn alphaforge.api.app:app``.
"""

from alphaforge.api.jobs import Job, JobStore

__all__ = ["Job", "JobStore"]
