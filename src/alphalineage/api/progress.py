"""Live run progress (A2) - a thread-safe recorder the GP drives while a job runs.

A ``RunProgress`` is handed to the GP as its lineage recorder. It forwards every
``on_init`` / ``on_generation`` call to an inner recorder (the ``LineageStore`` that
persists the run) and, under a lock, maintains a small snapshot - current generation,
target, a per-generation best/mean-fitness history, and the best tree seen so far - that
the ``GET /runs/{id}`` endpoint reads while the background thread is still working.

Drives the frontend progress view; the inner store is untouched so lineage is unaffected.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from alphalineage.core.tree import Node, to_json


class RunProgress:
    """A recorder wrapper that exposes a thread-safe snapshot of a running search."""

    def __init__(
        self,
        *,
        target_generations: int = 0,
        resources: dict[str, Any] | None = None,
    ) -> None:
        self._inner: Any = None
        self._lock = threading.Lock()
        self._phase = "queued"
        self._generation = 0
        self._target = target_generations
        self._history: list[dict[str, float]] = []
        self._best: tuple[Node, float] | None = None
        self._resources = dict(resources) if resources is not None else None
        self._candidate_done = 0
        self._candidate_total = 0
        self._report_done = 0
        self._report_total = 0
        self._factors_per_second: float | None = None
        self._termination_reason: str | None = None

    def attach(self, inner: Any) -> None:
        """Set the inner recorder (the persisting LineageStore) calls are forwarded to."""
        self._inner = inner

    def set_target(self, target_generations: int) -> None:
        with self._lock:
            self._target = target_generations

    def set_phase(self, phase: str) -> None:
        """Expose a coarse job phase without coupling the scorer to the API."""
        with self._lock:
            self._phase = phase

    def set_candidate_progress(
        self,
        done: int,
        total: int,
        *,
        factors_per_second: float | None = None,
    ) -> None:
        """Update the active population's completed-factor count."""
        with self._lock:
            self._candidate_done = max(0, int(done))
            self._candidate_total = max(0, int(total))
            self._factors_per_second = (
                max(0.0, float(factors_per_second)) if factors_per_second is not None else None
            )

    def set_report_progress(self, done: int, total: int) -> None:
        """Update final trial-report progress, which follows the generation loop."""
        with self._lock:
            self._phase = "reporting"
            self._report_done = max(0, int(done))
            self._report_total = max(0, int(total))

    def on_scoring(
        self,
        *,
        phase: str,
        generation: int,
        done: int,
        total: int,
        factors_per_second: float | None = None,
    ) -> None:
        """Recorder hook used by deterministic population batches for live progress."""
        with self._lock:
            self._phase = "initializing" if phase == "initializing" else "training"
            self._generation = max(self._generation, int(generation))
            self._candidate_done = max(0, int(done))
            self._candidate_total = max(0, int(total))
            self._factors_per_second = (
                max(0.0, float(factors_per_second)) if factors_per_second is not None else None
            )

    def finish(self, termination_reason: str = "completed") -> None:
        with self._lock:
            self._phase = "stopped" if termination_reason == "user_stopped" else "done"
            self._termination_reason = termination_reason

    # --- recorder protocol (called by GP) ----------------------------------------
    def on_init(
        self,
        trees: Sequence[Node],
        *,
        fitnesses: Sequence[float] | None = None,
        ops: Sequence[str] | None = None,
    ) -> None:
        if self._inner is not None:
            self._inner.on_init(trees, fitnesses=fitnesses, ops=ops)
        self._observe(0, trees, fitnesses)

    def on_generation(
        self, generation: int, entries: Sequence[tuple[Node, list[int], str, float]]
    ) -> None:
        if self._inner is not None:
            self._inner.on_generation(generation, entries)
        trees = [e[0] for e in entries]
        fits = [float(e[3]) for e in entries if len(e) > 3]
        self._observe(generation, trees, fits if fits else None)

    def _observe(
        self, generation: int, trees: Sequence[Node], fits: Sequence[float] | None
    ) -> None:
        with self._lock:
            self._phase = "training"
            self._generation = max(self._generation, generation)
            self._candidate_done = 0
            self._candidate_total = 0
            if fits:
                best_i = max(range(len(fits)), key=lambda i: fits[i])
                best_tree, best_fit = trees[best_i], float(fits[best_i])
                if self._best is None or best_fit >= self._best[1]:
                    self._best = (best_tree, best_fit)
                self._history.append(
                    {
                        "generation": int(generation),
                        "best_fitness": float(max(fits)),
                        "mean_fitness": float(sum(fits) / len(fits)),
                    }
                )

    # --- snapshot -----------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            best = (
                {"tree": to_json(self._best[0]), "fitness": self._best[1]}
                if self._best is not None
                else None
            )
            return {
                "phase": self._phase,
                "generation": self._generation,
                "target_generations": self._target,
                "history": list(self._history),
                "best": best,
                "resources": dict(self._resources) if self._resources is not None else None,
                "candidate_done": self._candidate_done,
                "candidate_total": self._candidate_total,
                "report_done": self._report_done,
                "report_total": self._report_total,
                "factors_per_second": self._factors_per_second,
                "termination_reason": self._termination_reason,
            }


class SyncProgress:
    """A thread-safe ``{done, total, current_symbol}`` counter for a data-pull job.

    Handed to ``JobStore.submit(..., progress=...)`` the same way ``RunProgress`` is for GP
    runs, so ``GET .../{job_id}`` can read a live snapshot while the job's background thread
    is still working through its symbol list.
    """

    def __init__(self, *, total: int = 0) -> None:
        self._lock = threading.Lock()
        self._done = 0
        self._total = total
        self._current_symbol: str | None = None

    def advance(self, symbol: str) -> None:
        with self._lock:
            self._done += 1
            self._current_symbol = symbol

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "done": self._done,
                "total": self._total,
                "current_symbol": self._current_symbol,
            }
