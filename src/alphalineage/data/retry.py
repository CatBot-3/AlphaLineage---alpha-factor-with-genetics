"""Retry/backoff and rate-limit helpers for network providers.

Both helpers take injectable ``sleep`` / ``clock`` callables so tests exercise the
backoff schedule and rate-limit gating without real wall-clock delays.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    get_retry_after: Callable[[BaseException], float | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    jitter: float = 0.1,
    rng: random.Random | None = None,
) -> T:
    """Call ``fn`` with exponential backoff; re-raise the last error if it never succeeds.

    On a retryable exception the delay is ``base_delay * 2**(attempt-1)`` (capped at
    ``max_delay``), unless ``get_retry_after`` returns an explicit wait (e.g. an HTTP
    ``Retry-After`` from a 429), which overrides it. A random jitter of up to
    ``jitter * delay`` is added. The final attempt does not sleep before raising.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    generator = rng or random.Random()
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if get_retry_after is not None:
                retry_after = get_retry_after(exc)
                if retry_after is not None:
                    delay = retry_after
            if jitter:
                delay += generator.uniform(0.0, jitter * delay)
            sleep(delay)


@dataclass
class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` within ``period_s`` seconds.

    Tiingo's free tier is roughly 50 symbols/hour; this gates the provider so a bulk
    pull blocks rather than getting 429-throttled. ``clock`` / ``sleep`` are injectable.
    """

    max_calls: int
    period_s: float = 3600.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _calls: list[float] = field(default_factory=list)

    def _prune(self, now: float) -> None:
        self._calls = [t for t in self._calls if now - t < self.period_s]

    def acquire(self) -> None:
        """Block (via ``sleep``) until a call slot is free, then record the call."""
        now = self.clock()
        self._prune(now)
        if len(self._calls) >= self.max_calls:
            wait = self.period_s - (now - self._calls[0])
            if wait > 0:
                self.sleep(wait)
            now = self.clock()
            self._prune(now)
        self._calls.append(self.clock())
