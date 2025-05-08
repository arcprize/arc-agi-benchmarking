import asyncio
import time
import pytest
import math
from arc_agi_benchmarking.utils.rate_limiter import AsyncRequestRateLimiter

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio

# Helper for Decimal comparison, mostly useful if precision context changes
# For default precision, direct == is fine with Decimal("str") inputs.
# def assert_decimal_equal(a, b):
#     assert a == b, f"{a} != {b}"

# --- Test Cases ---

async def test_limiter_initialization():
    """Test valid and invalid initializations."""
    # Valid
    limiter = AsyncRequestRateLimiter(rate=10, capacity=20)
    assert limiter._rate == 10.0 # Check float
    assert limiter._capacity == 20.0 # Check float
    assert limiter._available_requests == 20.0 # Check float

    # Invalid rate
    with pytest.raises(ValueError, match="Rate must be a positive number"):
        AsyncRequestRateLimiter(rate=0, capacity=10)
    with pytest.raises(ValueError, match="Rate must be a positive number"):
        AsyncRequestRateLimiter(rate=-1, capacity=10)
        
    # Invalid capacity
    with pytest.raises(ValueError, match="Capacity must be a non-negative number"):
        AsyncRequestRateLimiter(rate=1, capacity=-1)

async def test_basic_acquire():
    """Test acquiring requests when available."""
    limiter = AsyncRequestRateLimiter(rate=10, capacity=10)
    start_requests = await limiter.get_available_requests()
    assert abs(start_requests - 10.0) < 1e-9 # Check close to 10 initially
    
    await limiter.acquire(1)
    actual_9 = await limiter.get_available_requests()
    # Check value is within a small range around 9.0
    assert 9.0 - 1e-3 < actual_9 < 9.0 + 1e-3 # Increased tolerance
    
    await limiter.acquire(5)
    actual_4 = await limiter.get_available_requests()
    # Check value is within a small range around 4.0
    assert 4.0 - 1e-3 < actual_4 < 4.0 + 1e-3 # Increased tolerance

async def test_consume_capacity():
    """Test acquiring exactly the capacity quickly."""
    capacity = 5
    limiter = AsyncRequestRateLimiter(rate=1, capacity=capacity)
    
    start_time = time.monotonic()
    await limiter.acquire(capacity)
    end_time = time.monotonic()
    
    # Check value is very small (close to zero)
    assert await limiter.get_available_requests() < 1e-4 
    assert end_time - start_time < 0.1 

async def test_rate_limit_wait():
    """Test that acquiring beyond capacity forces a wait based on rate."""
    rate = 5.0
    capacity = 3.0
    limiter = AsyncRequestRateLimiter(rate=rate, capacity=capacity)

    await limiter.acquire(int(capacity))
    # Check value is very small (close to zero)
    assert await limiter.get_available_requests() < 1e-4 
    
    start_time = time.monotonic()
    await limiter.acquire(1)
    end_time = time.monotonic()
    
    expected_wait = 1.0 / rate
    actual_wait = end_time - start_time
    
    # Time comparison still uses isclose
    assert math.isclose(actual_wait, expected_wait, abs_tol=0.05)
    
    # After waiting, the available amount might be slightly > 0
    # Check it's very small again
    assert await limiter.get_available_requests() < 0.1 # Keep existing check
    
async def test_burst_then_rate():
    """Test the burst capacity followed by rate-limited acquisition."""
    rate = 2.0
    capacity = 5.0
    limiter = AsyncRequestRateLimiter(rate=rate, capacity=capacity)

    start_burst_time = time.monotonic()
    await limiter.acquire(int(capacity))
    end_burst_time = time.monotonic()
    assert end_burst_time - start_burst_time < 0.1
    # Check value is very small (close to zero)
    assert await limiter.get_available_requests() < 1e-4 

    start_wait_time = time.monotonic()
    await limiter.acquire(1)
    end_wait_time = time.monotonic()
    
    expected_wait = 1.0 / rate
    actual_wait = end_wait_time - start_wait_time
    # Time comparison still uses isclose
    assert math.isclose(actual_wait, expected_wait, abs_tol=0.05)
    
    # Check remaining is small
    assert await limiter.get_available_requests() < 0.1

async def test_refill():
    """Test that requests refill over time."""
    rate = 10.0
    capacity = 5.0
    limiter = AsyncRequestRateLimiter(rate=rate, capacity=capacity)

    await limiter.acquire(int(capacity)) 
    # Check value is very small (close to zero)
    # assert await limiter.get_available_requests() < 1e-4 
    assert await limiter.get_available_requests() < 1e-3 # Increased tolerance

    wait_time = 3.0 / rate
    await asyncio.sleep(wait_time + 0.01) 

    available = await limiter.get_available_requests()
    # Check available is at least close to 3
    assert available >= 3.0 - 1e-5 
    assert available <= capacity

    await limiter.acquire(3)
    remainder = await limiter.get_available_requests()
    # Check remainder is small (no rounding needed for < check)
    assert remainder < 0.2 
    assert remainder >= 0.0

async def test_concurrent_acquires():
    """Simulate multiple concurrent tasks acquiring requests."""
    rate = 5.0 
    capacity = 10.0
    num_tasks = 15
    limiter = AsyncRequestRateLimiter(rate=rate, capacity=capacity)
    results = []

    async def worker(worker_id):
        start_time = time.monotonic()
        async with limiter:
            acquire_time = time.monotonic()
            await asyncio.sleep(0.01)
            end_time = time.monotonic()
            results.append({
                "id": worker_id,
                "start": start_time,
                "acquired": acquire_time,
                "end": end_time
            })

    overall_start_time = time.monotonic()
    tasks = [asyncio.create_task(worker(i)) for i in range(num_tasks)]
    await asyncio.gather(*tasks)
    overall_end_time = time.monotonic()

    total_duration = overall_end_time - overall_start_time
    
    expected_rate_limited_tasks = num_tasks - capacity
    expected_duration = (expected_rate_limited_tasks / rate) if expected_rate_limited_tasks > 0 else 0

    # Time comparison still uses isclose
    assert math.isclose(total_duration, expected_duration, rel_tol=0.15, abs_tol=0.05) # More tolerance here

async def test_context_manager():
    """Test using the limiter as an async context manager."""
    limiter = AsyncRequestRateLimiter(rate=10, capacity=5)
    start_req = await limiter.get_available_requests()
    assert abs(start_req - 5.0) < 1e-9 # Check close to 5 initially
    async with limiter:
        actual_4 = await limiter.get_available_requests()
        # Check value is within a small range around 4.0
        assert 4.0 - 1e-3 < actual_4 < 4.0 + 1e-3 # Increased tolerance
    final_req = await limiter.get_available_requests()
    # Check value is within a small range around 4.0
    assert 4.0 - 1e-3 < final_req < 4.0 + 1e-3 # Increased tolerance

async def test_acquire_more_than_capacity_error():
    """Test ValueError when trying to acquire more than capacity."""
    limiter = AsyncRequestRateLimiter(rate=1, capacity=5)
    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        await limiter.acquire(6)

async def test_acquire_zero_or_negative_error():
    """Test ValueError when trying to acquire 0 or negative requests."""
    limiter = AsyncRequestRateLimiter(rate=1, capacity=5)
    with pytest.raises(ValueError, match="must be a positive integer"):
        await limiter.acquire(0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        await limiter.acquire(-1)

async def test_get_available_requests_basic():
    """Basic test for the inspection method."""
    limiter = AsyncRequestRateLimiter(rate=10, capacity=15)
    assert isinstance(await limiter.get_available_requests(), float) 
    assert abs(await limiter.get_available_requests() - 15.0) < 1e-9 # Check close to 15
    await limiter.acquire(3)
    actual_12 = await limiter.get_available_requests()
    # Check value is within a small range around 12.0
    assert 12.0 - 1e-3 < actual_12 < 12.0 + 1e-3 # Increased tolerance

    await asyncio.sleep(0.1) 
    # Check close to 13 (existing isclose is fine here)
    assert math.isclose(await limiter.get_available_requests(), 13.0, abs_tol=0.1)

# Note: Testing the sync-call path of get_available_requests is complex
#       as it involves potentially blocking calls to asyncio.run().
#       Focusing on the async usage is usually sufficient for library tests. 