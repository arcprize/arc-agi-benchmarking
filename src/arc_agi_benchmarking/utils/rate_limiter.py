"""Thread-safe request rate limiting for synchronous provider SDK calls."""

from __future__ import annotations

import threading
import time
from typing import Callable


class RequestRateLimiter:
    """A token bucket consumed immediately before an outbound API request.

    Provider adapters run in executor threads, so this limiter deliberately uses
    a regular threading lock and blocking sleep instead of asyncio primitives.
    """

    def __init__(
        self,
        rate: float,
        capacity: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(rate, (int, float)) or not rate > 0:
            raise ValueError("Rate must be a positive number")
        if not isinstance(capacity, (int, float)) or capacity < 1:
            raise ValueError("Capacity must be at least 1")

        self._rate = float(rate)
        self._capacity = float(capacity)
        self._available_requests = self._capacity
        self._clock = clock
        self._sleeper = sleeper
        self._last_refill_time = self._clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill the bucket. The caller must hold ``self._lock``."""
        now = self._clock()
        elapsed = now - self._last_refill_time
        if elapsed > 0:
            self._available_requests = min(
                self._capacity,
                self._available_requests + elapsed * self._rate,
            )
            self._last_refill_time = now

    def acquire(self, requests_needed: int = 1) -> float:
        """Consume request allowance, blocking as needed.

        Returns the total number of seconds spent waiting.
        """
        if not isinstance(requests_needed, int) or requests_needed <= 0:
            raise ValueError("requests_needed must be a positive integer")
        if requests_needed > self._capacity:
            raise ValueError(
                f"Requested requests ({requests_needed}) exceeds bucket capacity "
                f"({self._capacity}) - acquisition impossible."
            )

        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._available_requests >= requests_needed:
                    self._available_requests -= requests_needed
                    return waited
                wait_time = (requests_needed - self._available_requests) / self._rate

            self._sleeper(wait_time)
            waited += wait_time

    def get_available_requests(self) -> float:
        """Return the approximate current request allowance."""
        with self._lock:
            self._refill()
            return self._available_requests
