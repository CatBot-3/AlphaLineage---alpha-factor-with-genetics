"""In-process background job runner (P5-T1).

A dependency-free stand-in for RQ/Redis: ``submit`` runs a function on a daemon thread and
tracks its status/result. The interface (submit -> id, get(id) -> Job) mirrors a real queue, so
swapping in RQ + Redis for production is a localized change.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | failed
    result: Any = None
    error: str | None = None
    # Live snapshot object (e.g. RunProgress) read by GET /runs/{id} while the job runs.
    progress: Any = None
    # Cooperative-cancellation flag the work function polls; set by JobStore.cancel.
    cancel: threading.Event = field(default_factory=threading.Event)


class JobStore:
    """Thread-safe registry of background jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_success: Callable[[str, Any], None] | None = None,
        progress: Any = None,
        cancel: threading.Event | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        job = Job(
            id=job_id if job_id is not None else uuid.uuid4().hex,
            progress=progress,
            cancel=cancel if cancel is not None else threading.Event(),
        )
        with self._lock:
            self._jobs[job.id] = job

        def _run() -> None:
            with self._lock:
                job.status = "running"
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - record any failure for the caller
                with self._lock:
                    job.error = repr(exc)
                    job.status = "failed"
            else:
                if on_success is not None:
                    try:
                        on_success(job.id, result)
                    except Exception:
                        # Persistence callbacks are best-effort; the run itself succeeded.
                        pass
                with self._lock:
                    job.result = result
                    job.status = "done"

        threading.Thread(target=_run, daemon=True).start()
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Signal a job to stop; the work function decides when to honor it. Returns found."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel.set()
        return True
