"""Find the worker count that actually helps, by timing the run's own scoring batches.

A fixed CPU percentage is a guess about a machine this code has never seen. It is also a guess
about the *workload*: most of a search's wall clock is native scoring, which parallelises, but
the rest is single-threaded Python, and that ratio caps the achievable speedup no matter how
many cores are free. On a sixteen-core machine the usual result is that half the cores sit idle
while the other half deliver a fraction of the speedup their number suggests.

This tuner replaces the guess with a measurement. It hands the GP a worker count before each
scoring batch and is told what the batch cost; while calibrating it walks a ladder of candidate
counts, and once every rung has enough samples it settles on the *knee* - the smallest count
whose cost is within a tolerance of the best one seen. Spending twelve cores to beat eight by
three percent is not a win worth the heat.

The one property that makes this safe: **worker count cannot change what a search finds.**
Results are collected in input order and every tree's score is a pure function of the tree and
the panel, so a run at one worker and the same run at eight produce identical scores and an
identical final population (``tests/test_tuning.py`` asserts this). Probing therefore costs
only the timing difference of the probe itself, and the chosen count never enters a checkpoint,
a fingerprint, or a reported metric.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass
from typing import Any

#: A rung must collect this many samples before the ladder is considered walked.
SAMPLES_PER_RUNG = 2
#: Batches smaller than this teach nothing: fixed per-call overhead dominates the measurement.
MIN_BATCH_FOR_PROBE = 4
#: Accept a smaller worker count when it comes within this factor of the best cost seen.
KNEE_TOLERANCE = 1.10
#: After settling, re-walk the ladder if the settled rung's cost drifts past this factor of its
#: calibrated median for several batches running (another program took the cores, or gave them
#: back).
DRIFT_TOLERANCE = 1.50
DRIFT_PATIENCE = 4


def worker_ladder(max_workers: int) -> list[int]:
    """Candidate worker counts: powers of two, always including 1 and the ceiling.

    A ladder rather than a sweep because the cost curve is smooth and the interesting part is
    where it flattens; sampling every integer up to thirty-two would spend the whole run
    calibrating.
    """
    if max_workers <= 1:
        return [1]
    rungs = []
    rung = 1
    while rung < max_workers:
        rungs.append(rung)
        rung *= 2
    rungs.append(max_workers)
    return rungs


@dataclass(frozen=True)
class TunerSnapshot:
    """What the tuner has learned, for display. Never feeds a metric or a fingerprint."""

    state: str
    workers: int
    max_workers: int
    ladder: tuple[int, ...]
    samples: int
    #: Measured cost ratio of one worker to the chosen count, once both have been sampled.
    speedup: float | None
    #: Cost per unit of work at each sampled rung, for the "why not more cores" explanation.
    cost_by_workers: dict[int, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "workers": self.workers,
            "max_workers": self.max_workers,
            "ladder": list(self.ladder),
            "samples": self.samples,
            "speedup": self.speedup,
            "cost_by_workers": {str(k): v for k, v in sorted(self.cost_by_workers.items())},
        }


class WorkerTuner:
    """Chooses a worker count per scoring batch and learns from what each batch cost.

    Thread-safe because the GP may notify from a scoring callback while the API thread reads a
    snapshot for the progress endpoint.
    """

    def __init__(self, max_workers: int, *, samples_per_rung: int = SAMPLES_PER_RUNG) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if samples_per_rung < 1:
            raise ValueError("samples_per_rung must be at least 1")
        self.max_workers = max_workers
        self._ladder = worker_ladder(max_workers)
        # Probe from the ceiling downwards. A run that ends before calibration finishes has
        # then spent its batches on the *fast* rungs rather than the slow ones, so an
        # unfinished calibration costs little; a long run walks the whole ladder either way.
        self._probe_order = list(reversed(self._ladder))
        self._samples_per_rung = samples_per_rung
        self._costs: dict[int, list[float]] = {rung: [] for rung in self._ladder}
        # The serial rung is the most expensive one to sample and the least likely to win, so
        # it is confirmed once rather than repeatedly. It still has to be sampled at all: a
        # workload that does not parallelise must be able to say so, and it is the denominator
        # of the speedup the run reports.
        self._budget = {
            rung: (1 if rung == 1 and max_workers > 1 else samples_per_rung)
            for rung in self._ladder
        }
        self._lock = threading.Lock()
        self._cursor = 0
        self._settled: int | None = None
        self._drifting = 0
        # Until the first batch is measured, behave like the profile asked: use everything.
        # ``None`` marks a batch that was too small to teach the ladder anything.
        self._probing: int | None = max_workers

    # --- the GP's two calls -------------------------------------------------------
    def choose(self, batch_size: int) -> int:
        """Worker count for the next batch, capped by the work actually available."""
        with self._lock:
            if self._settled is not None:
                chosen = self._settled
            elif batch_size < MIN_BATCH_FOR_PROBE:
                # Too small to measure; run it on the current best guess and do not record it.
                chosen = self._settled or self.max_workers
                self._probing = None
            else:
                chosen = self._next_rung()
                self._probing = chosen
            return max(1, min(chosen, batch_size, self.max_workers))

    def observe(self, workers: int, batch_size: int, units: int, elapsed: float) -> None:
        """Record what a batch cost. ``units`` is work volume (total nodes), not tree count.

        Trees differ in size by an order of magnitude, so seconds-per-batch is far too noisy to
        compare rungs; seconds per node is stable enough to rank them after a few samples.
        """
        if batch_size < MIN_BATCH_FOR_PROBE or units <= 0 or elapsed <= 0.0:
            return
        with self._lock:
            if self._settled is not None:
                self._observe_settled(workers, elapsed / units)
                return
            if self._probing is None or workers != min(self._probing, batch_size, self.max_workers):
                # The batch did not run at the count we asked for (it was capped by batch size),
                # so it says nothing about that rung.
                return
            rung = self._probing
            self._costs[rung].append(elapsed / units)
            self._cursor += 1
            if all(len(samples) >= self._budget[key] for key, samples in self._costs.items()):
                self._settled = self._knee()
                self._drifting = 0

    def _next_rung(self) -> int:
        """The next rung still short of its sample budget, walking from the ceiling down."""
        for offset in range(len(self._probe_order)):
            rung = self._probe_order[(self._cursor + offset) % len(self._probe_order)]
            if len(self._costs[rung]) < self._budget[rung]:
                return rung
        return self.max_workers

    # --- internals ----------------------------------------------------------------
    def _knee(self) -> int:
        """Smallest rung whose median cost is within the tolerance of the best rung's."""
        medians = {
            rung: statistics.median(samples) for rung, samples in self._costs.items() if samples
        }
        if not medians:
            return self.max_workers
        best = min(medians.values())
        for rung in sorted(medians):
            if medians[rung] <= best * KNEE_TOLERANCE:
                return rung
        return max(medians, key=lambda rung: -medians[rung])

    def _observe_settled(self, workers: int, cost: float) -> None:
        """Watch for the machine changing under us and recalibrate if it clearly has."""
        if workers != self._settled:
            return
        samples = self._costs.get(self._settled) or []
        if not samples:
            return
        if cost > statistics.median(samples) * DRIFT_TOLERANCE:
            self._drifting += 1
            if self._drifting >= DRIFT_PATIENCE:
                self._costs = {rung: [] for rung in self._ladder}
                self._cursor = 0
                self._settled = None
                self._drifting = 0
        else:
            self._drifting = 0

    def restore(self, state: dict[str, Any] | None) -> None:
        """Adopt timings measured earlier so a continued run does not recalibrate.

        Timings describe the machine, not the search, so carrying them across segments is free.
        State from a different machine, a different ceiling, or a build before this existed is
        ignored and calibration simply starts again; nothing downstream depends on the value.
        """
        if not isinstance(state, dict) or int(state.get("max_workers", 0) or 0) != self.max_workers:
            return
        stored = state.get("cost_by_workers")
        if not isinstance(stored, dict):
            return
        restored: dict[int, list[float]] = {}
        for key, value in stored.items():
            try:
                rung, cost = int(key), float(value)
            except (TypeError, ValueError):
                return
            if rung not in self._costs or cost <= 0.0:
                return
            restored[rung] = [cost] * self._budget[rung]
        if len(restored) != len(self._costs):
            return
        with self._lock:
            self._costs = restored
            self._settled = self._knee()
            self._drifting = 0

    def snapshot(self) -> TunerSnapshot:
        with self._lock:
            medians = {
                rung: statistics.median(samples) for rung, samples in self._costs.items() if samples
            }
            chosen = self._settled if self._settled is not None else self.max_workers
            serial = medians.get(1)
            picked = medians.get(chosen)
            speedup = serial / picked if serial and picked else None
            return TunerSnapshot(
                state="settled" if self._settled is not None else "calibrating",
                workers=chosen,
                max_workers=self.max_workers,
                ladder=tuple(self._ladder),
                samples=sum(len(samples) for samples in self._costs.values()),
                speedup=speedup,
                cost_by_workers=medians,
            )
