"""Unit tests for retry/backoff and the rate limiter (no real sleeping)."""

from __future__ import annotations

import pytest

from alphaforge.data.retry import RateLimiter, with_retry
from alphaforge.data.tiingo_client import RateLimitError


def test_retry_succeeds_after_failures_and_honors_retry_after():
    calls = {"n": 0}
    slept: list[float] = []

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError(retry_after=2.0)
        return "ok"

    out = with_retry(
        fn,
        max_attempts=5,
        base_delay=0.1,
        retry_on=(RateLimitError,),
        get_retry_after=lambda e: getattr(e, "retry_after", None),
        sleep=slept.append,
        jitter=0.0,
    )

    assert out == "ok"
    assert calls["n"] == 3
    assert slept == [2.0, 2.0]  # Retry-After overrode exponential backoff


def test_retry_exponential_backoff_schedule():
    slept: list[float] = []

    def always_fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        with_retry(
            always_fail,
            max_attempts=4,
            base_delay=1.0,
            retry_on=(ValueError,),
            sleep=slept.append,
            jitter=0.0,
        )

    assert slept == [1.0, 2.0, 4.0]  # 3 sleeps before the 4th (final) attempt raises


def test_rate_limiter_blocks_when_window_full():
    now = {"t": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    rl = RateLimiter(max_calls=2, period_s=100.0, clock=lambda: now["t"], sleep=sleep)
    rl.acquire()
    rl.acquire()
    rl.acquire()  # third within the window must block

    assert slept == [100.0]
