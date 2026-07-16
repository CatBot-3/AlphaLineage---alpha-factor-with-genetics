from __future__ import annotations

import threading

import pytest

from alphalineage.api.resources import (
    MAX_MEMORY_BUDGET,
    MIN_MEMORY_BUDGET,
    ResourcePolicy,
    TrainingLeaseCancelled,
    TrainingScheduler,
    memory_budget_bytes,
    resolve_resources,
    worker_capacity,
)


def test_device_relative_profiles_are_visible_and_bounded():
    assert worker_capacity(1) == 1
    assert worker_capacity(2) == 2
    assert worker_capacity(20) == 19
    assert worker_capacity(64) == 32

    light = resolve_resources(ResourcePolicy("light"), cpus=20, available_memory=16 << 30)
    auto = resolve_resources(ResourcePolicy("auto"), cpus=20, available_memory=16 << 30)
    maximum = resolve_resources(ResourcePolicy("maximum"), cpus=20, available_memory=16 << 30)
    custom = resolve_resources(ResourcePolicy("custom", 70), cpus=20, available_memory=16 << 30)
    assert (light.percent, light.workers) == (25, 5)
    assert (auto.percent, auto.workers) == (50, 10)
    assert (maximum.percent, maximum.workers) == (100, 19)
    assert (custom.percent, custom.workers) == (70, 14)
    assert (auto.requested_workers, auto.effective_workers, auto.workers) == (10, 10, 10)
    assert auto.memory_budget_bytes == 4 << 30
    assert auto.run_memory_budget_bytes == auto.memory_per_worker_bytes * auto.workers


def test_memory_guard_is_twenty_five_percent_with_bounds():
    assert memory_budget_bytes(1 << 30) == MIN_MEMORY_BUDGET
    assert memory_budget_bytes(8 << 30) == 2 << 30
    assert memory_budget_bytes(64 << 30) == MAX_MEMORY_BUDGET


def test_python_fallback_downshifts_every_profile_to_one_worker():
    resolved = resolve_resources(
        ResourcePolicy("maximum"),
        cpus=20,
        available_memory=16 << 30,
        accelerated=False,
        fallback_reason="Python evaluator selected",
    )
    assert resolved.percent == 100
    assert resolved.requested_workers == 19
    assert resolved.effective_workers == 1
    assert resolved.workers == 1
    assert resolved.accelerated is False
    assert resolved.fallback_reason == "Python evaluator selected"


@pytest.mark.parametrize(
    "policy",
    [
        lambda: ResourcePolicy("custom"),
        lambda: ResourcePolicy("custom", 9),
        lambda: ResourcePolicy("auto", 50),
    ],
)
def test_resource_policy_rejects_ambiguous_custom_values(policy):
    with pytest.raises(ValueError):
        policy()


def test_scheduler_reserves_weighted_capacity_until_lease_exits():
    scheduler = TrainingScheduler(max_workers=2, max_memory_bytes=1 << 30)
    maximum = resolve_resources(ResourcePolicy("maximum"), cpus=3, available_memory=4 << 30)
    light = resolve_resources(ResourcePolicy("light"), cpus=3, available_memory=4 << 30)
    entered = threading.Event()
    attempting = threading.Event()

    def wait_for_one() -> None:
        attempting.set()
        with scheduler.acquire(light):
            entered.set()

    with scheduler.acquire(maximum):
        thread = threading.Thread(target=wait_for_one)
        thread.start()
        assert attempting.wait(1)
        assert not entered.is_set()
        assert scheduler.available_workers == 0
    assert entered.wait(1)
    thread.join(1)
    assert scheduler.available_workers == 2


def test_scheduler_wait_is_cooperatively_cancellable():
    scheduler = TrainingScheduler(max_workers=1, max_memory_bytes=1 << 30)
    one = resolve_resources(ResourcePolicy("maximum"), cpus=1, available_memory=4 << 30)
    cancel = threading.Event()
    stopped = threading.Event()

    def wait_and_cancel() -> None:
        try:
            with scheduler.acquire(one, cancel=cancel):
                pass
        except TrainingLeaseCancelled:
            stopped.set()

    with scheduler.acquire(one):
        thread = threading.Thread(target=wait_and_cancel)
        thread.start()
        cancel.set()
        assert stopped.wait(1)
    thread.join(1)


def test_scheduler_queues_when_ram_is_full_even_with_cpu_capacity():
    scheduler = TrainingScheduler(
        max_workers=4,
        max_memory_bytes=2 * MIN_MEMORY_BUDGET,
    )
    light = resolve_resources(ResourcePolicy("light"), cpus=5, available_memory=2 << 30)
    assert light.workers == 1
    assert light.run_memory_budget_bytes == MIN_MEMORY_BUDGET

    attempting = threading.Event()
    entered = threading.Event()
    release = threading.Event()

    def acquire_third() -> None:
        attempting.set()
        with scheduler.acquire(light):
            entered.set()
            assert release.wait(2)

    with scheduler.acquire(light):
        with scheduler.acquire(light):
            thread = threading.Thread(target=acquire_third)
            thread.start()
            assert attempting.wait(1)
            assert not entered.wait(0.1)
            # Two CPU slots are idle, proving RAM alone is gating the third lease.
            assert scheduler.available_workers == 2
            assert scheduler.available_memory_bytes == 0
        assert entered.wait(1)
        release.set()
        thread.join(1)

    assert scheduler.available_workers == 4
    assert scheduler.available_memory_bytes == 2 * MIN_MEMORY_BUDGET


def test_scheduler_memory_wait_is_cooperatively_cancellable():
    scheduler = TrainingScheduler(max_workers=2, max_memory_bytes=MIN_MEMORY_BUDGET)
    light = resolve_resources(ResourcePolicy("light"), cpus=3, available_memory=1 << 30)
    cancel = threading.Event()
    stopped = threading.Event()

    def wait_for_memory() -> None:
        try:
            with scheduler.acquire(light, cancel=cancel):
                pass
        except TrainingLeaseCancelled:
            stopped.set()

    with scheduler.acquire(light):
        assert scheduler.available_workers == 1
        assert scheduler.available_memory_bytes == 0
        thread = threading.Thread(target=wait_for_memory)
        thread.start()
        cancel.set()
        assert stopped.wait(1)
    thread.join(1)


def test_scheduler_downshifts_oversized_memory_request_without_changing_cpu_weight():
    scheduler = TrainingScheduler(max_workers=4, max_memory_bytes=MIN_MEMORY_BUDGET)
    maximum = resolve_resources(ResourcePolicy("maximum"), cpus=5, available_memory=16 << 30)
    assert maximum.workers == 4
    assert maximum.run_memory_budget_bytes == MAX_MEMORY_BUDGET

    with scheduler.acquire(maximum) as lease:
        assert lease.max_workers == 4
        assert lease.requested_memory_budget_bytes == MAX_MEMORY_BUDGET
        assert lease.memory_budget_bytes == MIN_MEMORY_BUDGET
        assert scheduler.available_workers == 0
        assert scheduler.available_memory_bytes == 0

    assert scheduler.available_workers == 4
    assert scheduler.available_memory_bytes == MIN_MEMORY_BUDGET
