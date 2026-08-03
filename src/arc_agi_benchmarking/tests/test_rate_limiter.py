import threading

import pytest

from arc_agi_benchmarking.utils.rate_limiter import RequestRateLimiter


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_limiter(rate=10, capacity=1):
    fake_time = FakeTime()
    limiter = RequestRateLimiter(
        rate=rate,
        capacity=capacity,
        clock=fake_time.monotonic,
        sleeper=fake_time.sleep,
    )
    return limiter, fake_time


def test_limiter_initialization_and_validation():
    limiter, _ = make_limiter(rate=10, capacity=20)
    assert limiter._rate == 10.0
    assert limiter._capacity == 20.0

    with pytest.raises(ValueError, match="positive"):
        RequestRateLimiter(rate=0, capacity=1)
    with pytest.raises(ValueError, match="at least 1"):
        RequestRateLimiter(rate=1, capacity=0)


def test_acquire_consumes_allowance_and_waits_for_refill():
    limiter, fake_time = make_limiter(rate=5, capacity=1)

    assert limiter.acquire() == 0
    waited = limiter.acquire()

    assert waited == pytest.approx(0.2)
    assert fake_time.sleeps == [pytest.approx(0.2)]
    assert limiter.get_available_requests() == pytest.approx(0)


def test_limiter_supports_configured_burst_capacity():
    limiter, fake_time = make_limiter(rate=2, capacity=3)

    assert limiter.acquire(3) == 0
    assert limiter.acquire() == pytest.approx(0.5)
    assert fake_time.sleeps == [pytest.approx(0.5)]


def test_invalid_acquisition_is_rejected():
    limiter, _ = make_limiter(capacity=2)
    with pytest.raises(ValueError, match="positive integer"):
        limiter.acquire(0)
    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        limiter.acquire(3)


def test_acquire_is_thread_safe():
    limiter = RequestRateLimiter(rate=10_000, capacity=1)
    acquired = []

    def worker():
        limiter.acquire()
        acquired.append(True)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert acquired == [True] * 5
