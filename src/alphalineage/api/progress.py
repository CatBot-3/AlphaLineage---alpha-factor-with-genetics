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

    def __init__(self, *, target_generations: int = 0) -> None:
        self._inner: Any = None
        self._lock = threading.Lock()
        self._phase = "queued"
        self._generation = 0
        self._target = target_generations
        self._history: list[dict[str, float]] = []
        self._best: tuple[Node, float] | None = None

    def attach(self, inner: Any) -> None:
        """Set the inner recorder (the persisting LineageStore) calls are forwarded to."""
        self._inner = inner

    def set_target(self, target_generations: int) -> None:
        with self._lock:
            self._target = target_generations

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
            self._phase = "running"
            self._generation = max(self._generation, generation)
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
            }
