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

from alphalineage.api.resources import TrainingLeaseCancelled
from alphalineage.core.gp import TrainingCancelled


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | stopped | failed
    result: Any = None
    error: str | None = None
    termination_reason: str | None = None
    # Live snapshot object (e.g. RunProgress) read by GET /runs/{id} while the job runs.
    progress: Any = None
    # Immutable-at-submission context used by lifecycle guards (for example pinned formula
    # revisions).  Work functions should receive execution inputs directly rather than reading it.
    metadata: dict[str, Any] = field(default_factory=dict)
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
        metadata: dict[str, Any] | None = None,
        cancel: threading.Event | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        job = Job(
            id=job_id if job_id is not None else uuid.uuid4().hex,
            progress=progress,
            metadata=dict(metadata or {}),
            cancel=cancel if cancel is not None else threading.Event(),
        )
        with self._lock:
            self._jobs[job.id] = job

        def _run() -> None:
            with self._lock:
                job.status = "running"
            try:
                result = fn(*args, **kwargs)
            except (TrainingCancelled, TrainingLeaseCancelled):
                with self._lock:
                    job.termination_reason = "user_stopped"
                    job.status = "stopped"
                if progress is not None and hasattr(progress, "finish"):
                    progress.finish("user_stopped")
            except Exception as exc:  # noqa: BLE001 - record any failure for the caller
                with self._lock:
                    job.error = repr(exc)
                    job.status = "failed"
            else:
                termination_reason = (
                    str(result.get("termination_reason", "completed"))
                    if isinstance(result, dict)
                    else "completed"
                )
                if on_success is not None:
                    try:
                        on_success(job.id, result)
                    except Exception:
                        # Persistence callbacks are best-effort; the run itself succeeded.
                        pass
                with self._lock:
                    job.result = result
                    job.termination_reason = termination_reason
                    job.status = "stopped" if termination_reason == "user_stopped" else "done"
                if progress is not None and hasattr(progress, "finish"):
                    progress.finish(termination_reason)

        threading.Thread(target=_run, daemon=True).start()
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Return insertion-ordered job snapshots for small in-process job collections."""
        with self._lock:
            return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        """Forget a completed job; active jobs must first be cooperatively stopped."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {"queued", "running"}:
                return False
            del self._jobs[job_id]
            return True

    def cancel(self, job_id: str) -> bool:
        """Signal cancellable work to stop; locked-test finalization is non-interruptible."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in {"queued", "running"}:
                return False
            if job.progress is not None and hasattr(job.progress, "snapshot"):
                snapshot = job.progress.snapshot()
                phase = snapshot.get("phase") if isinstance(snapshot, dict) else None
                if phase in {"finalizing", "done", "stopped", "failed"}:
                    return False
            job.cancel.set()
            if job.progress is not None and hasattr(job.progress, "set_phase"):
                job.progress.set_phase("stopping")
        return True
