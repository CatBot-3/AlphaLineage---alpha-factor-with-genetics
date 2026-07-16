"""Portable, bounded CPU policy for local training jobs.

Resource choices are deliberately separate from :class:`GPConfig`: changing how many
workers evaluate a population must never change the statistical search or its checkpoint.
This module owns machine detection, the user-facing policy resolver, and one lazily-created
process-wide executor that prevents independent jobs from creating unbounded thread pools.
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any, Literal, TypeVar

ResourceProfile = Literal["light", "auto", "maximum", "custom"]

MAX_TRAINING_WORKERS = 32
MIN_CUSTOM_PERCENT = 10
MAX_CUSTOM_PERCENT = 100
MEMORY_FRACTION = 0.25
MIN_MEMORY_BUDGET = 256 * 1024 * 1024
MAX_MEMORY_BUDGET = 4 * 1024 * 1024 * 1024
_FALLBACK_AVAILABLE_MEMORY = 4 * 1024 * 1024 * 1024

_T = TypeVar("_T")


def detected_cpu_count() -> int:
    """CPUs available to this process, respecting affinity where Python exposes it."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
            if count > 0:
                return count
        except (OSError, NotImplementedError):
            pass
    return max(1, os.cpu_count() or 1)


def _windows_available_memory() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    return int(status.available_physical) if ok else None


def available_memory_bytes() -> int:
    """Best-effort physical memory currently available, with a safe portable fallback."""
    windows = _windows_available_memory()
    if windows is not None and windows > 0:
        return windows
    sysconf = getattr(os, "sysconf", None)
    if sysconf is not None:
        try:
            pages = int(sysconf("SC_AVPHYS_PAGES"))
            page_size = int(sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except (OSError, TypeError, ValueError):
            pass
    return _FALLBACK_AVAILABLE_MEMORY


def worker_capacity(cpus: int | None = None) -> int:
    """Bounded worker capacity, reserving one CPU for the app on machines with >2 CPUs."""
    detected = max(1, cpus if cpus is not None else detected_cpu_count())
    usable = detected - 1 if detected > 2 else detected
    return max(1, min(MAX_TRAINING_WORKERS, usable))


def memory_budget_bytes(available: int | None = None) -> int:
    """Twenty-five percent of available memory, bounded to a desktop-safe budget."""
    detected = max(1, available if available is not None else available_memory_bytes())
    return max(
        MIN_MEMORY_BUDGET,
        min(MAX_MEMORY_BUDGET, int(detected * MEMORY_FRACTION)),
    )


@dataclass(frozen=True)
class ResourcePolicy:
    profile: ResourceProfile = "auto"
    custom_percent: int | None = None

    def __post_init__(self) -> None:
        if self.profile not in {"light", "auto", "maximum", "custom"}:
            raise ValueError(f"unknown training resource profile {self.profile!r}")
        if self.profile == "custom":
            if (
                isinstance(self.custom_percent, bool)
                or not isinstance(self.custom_percent, int)
                or not MIN_CUSTOM_PERCENT <= self.custom_percent <= MAX_CUSTOM_PERCENT
            ):
                raise ValueError(
                    f"cpu_budget_percent must be an integer from {MIN_CUSTOM_PERCENT} to "
                    f"{MAX_CUSTOM_PERCENT} for the custom profile"
                )
        elif self.custom_percent is not None:
            raise ValueError("cpu_budget_percent is only valid for the custom profile")

    @property
    def percent(self) -> int:
        return {
            "light": 25,
            "auto": 50,
            "maximum": 100,
            "custom": int(self.custom_percent or 50),
        }[self.profile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "cpu_budget_percent": self.custom_percent,
        }


@dataclass(frozen=True)
class ResolvedResources:
    profile: ResourceProfile
    cpu_budget_percent: int | None
    percent: int
    detected_cpus: int
    worker_capacity: int
    requested_workers: int
    effective_workers: int
    workers: int
    available_memory_bytes: int
    memory_budget_bytes: int
    memory_per_worker_bytes: int
    run_memory_budget_bytes: int
    accelerated: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_resources(
    policy: ResourcePolicy | None = None,
    *,
    cpus: int | None = None,
    available_memory: int | None = None,
    accelerated: bool = True,
    fallback_reason: str | None = None,
) -> ResolvedResources:
    """Resolve a visible percentage policy into a deterministic bounded worker count."""
    selected = policy or ResourcePolicy()
    detected = max(1, cpus if cpus is not None else detected_cpu_count())
    capacity = worker_capacity(detected)
    requested_workers = max(1, min(capacity, math.ceil(capacity * selected.percent / 100)))
    effective_workers = requested_workers if accelerated else 1
    available = max(
        1, available_memory if available_memory is not None else available_memory_bytes()
    )
    budget = memory_budget_bytes(available)
    memory_per_worker = max(1, budget // capacity)
    return ResolvedResources(
        profile=selected.profile,
        cpu_budget_percent=selected.custom_percent,
        percent=selected.percent,
        detected_cpus=detected,
        worker_capacity=capacity,
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        # Backward-compatible alias used by existing API/UI clients.
        workers=effective_workers,
        available_memory_bytes=available,
        memory_budget_bytes=budget,
        memory_per_worker_bytes=memory_per_worker,
        run_memory_budget_bytes=min(
            budget,
            max(MIN_MEMORY_BUDGET, memory_per_worker * effective_workers),
        ),
        accelerated=accelerated,
        fallback_reason=None if accelerated else (fallback_reason or "acceleration unavailable"),
    )


class TrainingLeaseCancelled(RuntimeError):
    """Raised when a queued resource lease is cancelled before it acquires capacity."""


def _cancelled(cancel: threading.Event | Callable[[], bool] | None) -> bool:
    if cancel is None:
        return False
    return cancel.is_set() if isinstance(cancel, threading.Event) else bool(cancel())


class TrainingLease:
    """Weighted reservation for one run on the shared training scheduler."""

    def __init__(
        self,
        scheduler: TrainingScheduler,
        resources: ResolvedResources,
        cancel: threading.Event | Callable[[], bool] | None,
    ) -> None:
        self._scheduler = scheduler
        self.resources = resources
        self.max_workers = resources.workers
        self.requested_memory_budget_bytes = resources.run_memory_budget_bytes
        # A scheduler may have been created while less memory was available than when this
        # request was resolved. Cap one oversized request so it remains runnable without ever
        # exceeding the process-wide guard.
        self.memory_budget_bytes = min(
            self.requested_memory_budget_bytes,
            scheduler.max_memory_bytes,
        )
        self.memory_per_worker_bytes = max(1, self.memory_budget_bytes // self.max_workers)
        self._cancel = cancel
        self._acquired = False
        self._slots = threading.BoundedSemaphore(self.max_workers)

    def __enter__(self) -> TrainingLease:
        self._scheduler._reserve(
            self.max_workers,
            self.memory_budget_bytes,
            self._cancel,
        )
        self._acquired = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._acquired:
            self._scheduler._release(self.max_workers, self.memory_budget_bytes)
            self._acquired = False

    def submit(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> Future[_T]:
        """Submit Python fallback work without exceeding this run's reserved workers."""
        if not self._acquired:
            raise RuntimeError("training lease must be entered before submitting work")
        while not self._slots.acquire(timeout=0.1):
            if _cancelled(self._cancel):
                raise TrainingLeaseCancelled("training stopped while waiting for a worker")
        try:
            future = self._scheduler._submit(fn, *args, **kwargs)
        except Exception:
            self._slots.release()
            raise
        future.add_done_callback(lambda _future: self._slots.release())
        return future


class TrainingScheduler:
    """Process-wide CPU and memory reservations shared by all training jobs.

    A run still limits its own number of in-flight futures to ``ResolvedResources.workers``.
    The process-wide executor is one safety rail; an independent weighted memory reservation is
    the other. Simultaneous sessions therefore cannot exceed either machine budget in aggregate,
    including profiles whose per-run minimum memory is larger than their CPU-proportional share.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        max_memory_bytes: int | None = None,
    ) -> None:
        if max_workers is not None and (
            isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer or None")
        if max_memory_bytes is not None and (
            isinstance(max_memory_bytes, bool)
            or not isinstance(max_memory_bytes, int)
            or max_memory_bytes <= 0
        ):
            raise ValueError("max_memory_bytes must be a positive integer or None")
        self.max_workers = max_workers if max_workers is not None else worker_capacity()
        self.max_memory_bytes = (
            max_memory_bytes if max_memory_bytes is not None else memory_budget_bytes()
        )
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition()
        self._available_workers = self.max_workers
        self._available_memory_bytes = self.max_memory_bytes

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="alphalineage-score",
                )
            return self._executor

    def _submit(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> Future[_T]:
        return self._get_executor().submit(fn, *args, **kwargs)

    def acquire(
        self,
        resources: ResolvedResources,
        *,
        cancel: threading.Event | Callable[[], bool] | None = None,
    ) -> TrainingLease:
        """Return a context manager that atomically reserves this run's CPU and RAM weights."""
        if resources.workers > self.max_workers:
            raise ValueError(
                f"run requests {resources.workers} workers, scheduler capacity is "
                f"{self.max_workers}"
            )
        return TrainingLease(self, resources, cancel)

    def _reserve(
        self,
        workers: int,
        memory: int,
        cancel: threading.Event | Callable[[], bool] | None,
    ) -> None:
        with self._condition:
            while self._available_workers < workers or self._available_memory_bytes < memory:
                if _cancelled(cancel):
                    raise TrainingLeaseCancelled("training stopped while waiting for resources")
                self._condition.wait(timeout=0.1)
            if _cancelled(cancel):
                raise TrainingLeaseCancelled("training stopped while waiting for resources")
            self._available_workers -= workers
            self._available_memory_bytes -= memory

    def _release(self, workers: int, memory: int) -> None:
        with self._condition:
            self._available_workers = min(
                self.max_workers,
                self._available_workers + workers,
            )
            self._available_memory_bytes = min(
                self.max_memory_bytes,
                self._available_memory_bytes + memory,
            )
            self._condition.notify_all()

    @property
    def available_workers(self) -> int:
        with self._condition:
            return self._available_workers

    @property
    def available_memory_bytes(self) -> int:
        with self._condition:
            return self._available_memory_bytes

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)


TRAINING_SCHEDULER = TrainingScheduler()


def training_capabilities(
    *,
    accelerated: bool = True,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Machine-relative options shown before a user launches a search."""
    detected = detected_cpu_count()
    available = available_memory_bytes()
    profiles = {
        name: resolve_resources(
            ResourcePolicy(name),  # type: ignore[arg-type]
            cpus=detected,
            available_memory=available,
            accelerated=accelerated,
            fallback_reason=fallback_reason,
        ).to_dict()
        for name in ("light", "auto", "maximum")
    }
    return {
        "default_profile": "auto",
        "detected_cpus": detected,
        "worker_capacity": worker_capacity(detected),
        "available_memory_bytes": available,
        "memory_budget_bytes": memory_budget_bytes(available),
        "cpu_budget_percent_min": MIN_CUSTOM_PERCENT,
        "cpu_budget_percent_max": MAX_CUSTOM_PERCENT,
        "profiles": profiles,
    }
