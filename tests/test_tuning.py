"""Adaptive worker count: measure the machine instead of guessing a percentage of it."""

from __future__ import annotations

import pandas as pd
import pytest

from alphalineage.core.gp import GP, GPConfig
from alphalineage.core.tuning import (
    DRIFT_PATIENCE,
    KNEE_TOLERANCE,
    MIN_BATCH_FOR_PROBE,
    SAMPLES_PER_RUNG,
    WorkerTuner,
    worker_ladder,
)

_SMALL = dict(population_size=24, generations=3, max_depth=4, max_nodes=20, seed=0)


def _walk(tuner: WorkerTuner, cost_of, *, batches: int = 60, batch: int = 32, units: int = 500):
    """Feed the tuner synthetic batches whose cost follows ``cost_of(workers)``."""
    chosen: list[int] = []
    for _ in range(batches):
        workers = tuner.choose(batch)
        chosen.append(workers)
        tuner.observe(workers, batch, units, cost_of(workers) * units)
    return chosen


def test_ladder_always_spans_one_to_the_ceiling() -> None:
    assert worker_ladder(1) == [1]
    assert worker_ladder(2) == [1, 2]
    assert worker_ladder(8) == [1, 2, 4, 8]
    # A non-power-of-two ceiling (cores - 1) still gets sampled.
    assert worker_ladder(15) == [1, 2, 4, 8, 15]
    assert worker_ladder(3) == [1, 2, 3]


def test_settles_on_the_knee_not_on_the_largest_core_count() -> None:
    """Amdahl: past the knee more workers buy almost nothing, so the tuner should stop."""
    # Half the work is serial, which is roughly what this project's own profile shows.
    serial_share = 0.5

    def cost(workers: int) -> float:
        return serial_share + (1.0 - serial_share) / workers

    tuner = WorkerTuner(16)
    _walk(tuner, cost)
    snapshot = tuner.snapshot()

    assert snapshot.state == "settled"
    # 16 workers is the fastest rung, but 8 comes within the tolerance of it and 4 does not,
    # so 8 is the knee: the last count whose cores are still earning their place.
    assert cost(8) <= cost(16) * KNEE_TOLERANCE
    assert cost(4) > cost(16) * KNEE_TOLERANCE
    assert snapshot.workers == 8
    assert snapshot.max_workers == 16
    assert snapshot.speedup == pytest.approx(cost(1) / cost(8), rel=1e-6)


def test_a_workload_that_does_not_parallelize_settles_on_one_worker() -> None:
    tuner = WorkerTuner(16)
    _walk(tuner, lambda workers: 1.0)
    assert tuner.snapshot().workers == 1


def test_a_workload_that_parallelizes_perfectly_uses_the_whole_ceiling() -> None:
    tuner = WorkerTuner(8)
    _walk(tuner, lambda workers: 1.0 / workers)
    assert tuner.snapshot().workers == 8


def test_every_rung_is_sampled_before_settling() -> None:
    tuner = WorkerTuner(8)
    chosen = _walk(tuner, lambda workers: 0.25 + 0.75 / workers, batches=40)
    calibration = chosen[: len(worker_ladder(8)) * SAMPLES_PER_RUNG]
    assert set(calibration) == set(worker_ladder(8))
    # Once settled it stops probing and stays on one rung.
    assert len(set(chosen[-10:])) == 1


def test_a_batch_too_small_to_measure_is_neither_probed_nor_recorded() -> None:
    tuner = WorkerTuner(8)
    tiny = MIN_BATCH_FOR_PROBE - 1
    workers = tuner.choose(tiny)
    assert workers == tiny  # capped by the work available, never more workers than trees
    tuner.observe(workers, tiny, 100, 5.0)
    assert tuner.snapshot().samples == 0
    assert tuner.snapshot().state == "calibrating"


def test_recalibrates_when_the_machine_changes_under_a_settled_run() -> None:
    tuner = WorkerTuner(8)
    _walk(tuner, lambda workers: 0.1 + 0.9 / workers)
    settled = tuner.snapshot()
    assert settled.state == "settled"

    # Another program takes the cores: the settled rung suddenly costs far more.
    for _ in range(DRIFT_PATIENCE):
        workers = tuner.choose(32)
        tuner.observe(workers, 32, 500, 500 * 10.0)
    assert tuner.snapshot().state == "calibrating"


def test_restoring_timings_skips_calibration_but_only_for_the_same_ceiling() -> None:
    tuner = WorkerTuner(8)
    _walk(tuner, lambda workers: 0.2 + 0.8 / workers)
    state = tuner.snapshot().to_dict()
    assert state["state"] == "settled"

    warm = WorkerTuner(8)
    warm.restore(state)
    assert warm.snapshot().state == "settled"
    assert warm.snapshot().workers == state["workers"]

    # A checkpoint from a machine with a different core count teaches nothing here.
    other = WorkerTuner(4)
    other.restore(state)
    assert other.snapshot().state == "calibrating"
    # Neither does a state written before this existed, or a corrupt one.
    blank = WorkerTuner(8)
    blank.restore(None)
    blank.restore({"max_workers": 8, "cost_by_workers": {"1": "not a number"}})
    assert blank.snapshot().state == "calibrating"


def test_rejects_an_impossible_ceiling() -> None:
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="max_workers"):
            WorkerTuner(bad)  # type: ignore[arg-type]


# --- the property the whole design rests on ---------------------------------------------------
def test_worker_count_cannot_change_what_a_search_finds(signal_panel) -> None:
    """Probing mid-run is only safe because the count is invisible to the result."""
    panel, _ = signal_panel

    def fingerprint(workers: int, tuner: WorkerTuner | None = None):
        gp = GP(GPConfig(**_SMALL), panel, workers=workers, tuner=tuner)
        gp.run()
        return (
            sorted((key, value[0]) for key, value in gp._cache.items()),
            [(str(ind.tree), ind.fitness) for ind in gp.population],
        )

    serial = fingerprint(1)
    assert fingerprint(2) == serial
    # And a run whose worker count *changes between batches* lands in the same place.
    assert fingerprint(2, WorkerTuner(2)) == serial


def test_a_tuned_run_records_its_measurements_and_carries_them_to_the_next_segment(
    signal_panel, tmp_path
) -> None:
    panel, _ = signal_panel
    tuner = WorkerTuner(2)
    gp = GP(GPConfig(**_SMALL), panel, workers=2, tuner=tuner)
    gp.run()
    checkpoint = tmp_path / "checkpoint.json"
    gp.save_checkpoint(checkpoint)

    import json

    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert stored["worker_tuning"]["max_workers"] == 2
    assert stored["worker_tuning"]["cost_by_workers"]

    warm_tuner = WorkerTuner(2)
    warm = GP.from_checkpoint(checkpoint, panel, workers=2, tuner=warm_tuner)
    assert warm.tuner is warm_tuner
    assert warm_tuner.snapshot().cost_by_workers == tuner.snapshot().cost_by_workers


def test_chunks_stay_large_enough_to_be_worth_a_wave(signal_panel) -> None:
    """Small chunks spend a many-worker run's time on pool spin-up and load imbalance."""
    panel, _ = signal_panel
    gp = GP(GPConfig(**_SMALL), panel, workers=8, tuner=WorkerTuner(8))
    assert gp._scoring_chunk_size(200) >= 16
    # Still several waves, so a stop request lands and progress moves.
    assert gp._scoring_chunk_size(200) <= 200
    assert gp._scoring_chunk_size(4) <= 16


def test_progress_publishes_the_tuning_snapshot() -> None:
    from alphalineage.api.progress import RunProgress

    progress = RunProgress(target_generations=1)
    assert progress.snapshot()["tuning"] is None
    tuner = WorkerTuner(4)
    _walk(tuner, lambda workers: 0.3 + 0.7 / workers)
    progress.set_tuning(tuner.snapshot().to_dict())
    published = progress.snapshot()["tuning"]
    assert published["state"] == "settled"
    assert published["workers"] in worker_ladder(4)
    assert pd.notna(published["speedup"])
