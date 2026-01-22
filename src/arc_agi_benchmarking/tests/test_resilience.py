"""
Tests for the resilience module (timeout and circuit breaker functionality).
"""

import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock

from arc_agi_benchmarking.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    TaskTimeoutError,
    request_timeout,
    task_timeout,
)
from arc_agi_benchmarking.resilience.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    get_circuit_breaker,
    get_circuit_breaker_registry,
)
from arc_agi_benchmarking.resilience.timeout import (
    with_timeout,
    sync_timeout,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_REASONING_TIMEOUT,
    DEFAULT_TASK_TIMEOUT,
)


# =============================================================================
# Timeout Tests
# =============================================================================


@pytest.mark.asyncio
class TestRequestTimeout:
    """Tests for the request_timeout async context manager."""

    async def test_successful_operation_within_timeout(self):
        """Test that operations completing within timeout succeed."""
        async with request_timeout(1.0, "test operation"):
            await asyncio.sleep(0.1)
        # Should complete without raising

    async def test_timeout_raises_task_timeout_error(self):
        """Test that operations exceeding timeout raise TaskTimeoutError."""
        with pytest.raises(TaskTimeoutError) as exc_info:
            async with request_timeout(0.1, "slow operation"):
                await asyncio.sleep(1.0)

        assert "slow operation" in str(exc_info.value)
        assert exc_info.value.timeout == 0.1
        assert exc_info.value.elapsed is not None
        assert exc_info.value.elapsed >= 0.1

    async def test_invalid_timeout_value_logs_warning(self):
        """Test that invalid (zero or negative) timeout values are handled."""
        # With zero timeout, should execute without timeout enforcement
        async with request_timeout(0, "zero timeout"):
            await asyncio.sleep(0.05)

        # With negative timeout, should execute without timeout enforcement
        async with request_timeout(-1, "negative timeout"):
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
class TestTaskTimeout:
    """Tests for the task_timeout function."""

    async def test_async_function_within_timeout(self):
        """Test async function completing within timeout."""
        async def async_func(value: int) -> int:
            await asyncio.sleep(0.05)
            return value * 2

        result = await task_timeout(async_func, 1.0, "async test", 5)
        assert result == 10

    async def test_sync_function_within_timeout(self):
        """Test sync function completing within timeout."""
        def sync_func(value: int) -> int:
            time.sleep(0.05)
            return value * 3

        result = await task_timeout(sync_func, 1.0, "sync test", 7)
        assert result == 21

    async def test_async_function_timeout(self):
        """Test async function exceeding timeout."""
        async def slow_async():
            await asyncio.sleep(2.0)
            return "never reached"

        with pytest.raises(TaskTimeoutError) as exc_info:
            await task_timeout(slow_async, 0.1, "slow async")

        assert "slow async" in str(exc_info.value)
        assert exc_info.value.timeout == 0.1

    async def test_sync_function_timeout(self):
        """Test sync function exceeding timeout."""
        def slow_sync():
            time.sleep(2.0)
            return "never reached"

        with pytest.raises(TaskTimeoutError) as exc_info:
            await task_timeout(slow_sync, 0.1, "slow sync")

        assert "slow sync" in str(exc_info.value)

    async def test_kwargs_passed_correctly(self):
        """Test that kwargs are passed to the function."""
        async def func_with_kwargs(a: int, b: int = 0) -> int:
            return a + b

        result = await task_timeout(func_with_kwargs, 1.0, "kwargs test", 5, b=10)
        assert result == 15


@pytest.mark.asyncio
class TestWithTimeoutDecorator:
    """Tests for the with_timeout decorator."""

    async def test_decorator_on_async_function(self):
        """Test decorator on async function."""
        @with_timeout(1.0, "decorated function")
        async def decorated_func():
            await asyncio.sleep(0.05)
            return "success"

        result = await decorated_func()
        assert result == "success"

    async def test_decorator_timeout(self):
        """Test decorator raises timeout."""
        @with_timeout(0.1, "slow decorated")
        async def slow_decorated():
            await asyncio.sleep(1.0)

        with pytest.raises(TaskTimeoutError):
            await slow_decorated()

    async def test_decorator_uses_function_name_as_default(self):
        """Test decorator uses function name when operation not specified."""
        @with_timeout(0.1)
        async def my_slow_function():
            await asyncio.sleep(1.0)

        with pytest.raises(TaskTimeoutError) as exc_info:
            await my_slow_function()

        assert "my_slow_function" in str(exc_info.value)


class TestSyncTimeout:
    """Tests for the sync_timeout context manager."""

    def test_logs_warning_when_exceeded(self):
        """Test that sync_timeout logs a warning when exceeded."""
        # sync_timeout doesn't interrupt, just logs warnings
        with sync_timeout(0.01, "quick operation"):
            time.sleep(0.05)  # Will exceed but won't raise

    def test_no_warning_when_within_limit(self):
        """Test that sync_timeout doesn't warn when within limit."""
        with sync_timeout(1.0, "within limit"):
            time.sleep(0.01)


class TestDefaultTimeoutValues:
    """Tests for default timeout constants."""

    def test_default_values_are_reasonable(self):
        """Verify default timeout values are sensible."""
        assert DEFAULT_REQUEST_TIMEOUT == 300  # 5 minutes
        assert DEFAULT_REASONING_TIMEOUT == 900  # 15 minutes
        assert DEFAULT_TASK_TIMEOUT == 1800  # 30 minutes
        assert DEFAULT_REQUEST_TIMEOUT < DEFAULT_REASONING_TIMEOUT < DEFAULT_TASK_TIMEOUT


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreakerBasics:
    """Tests for basic CircuitBreaker functionality."""

    def test_initial_state_is_closed(self):
        """Test that a new circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() is True

    def test_record_success(self):
        """Test recording a successful request."""
        cb = CircuitBreaker("test")
        cb.record_success()

        stats = cb.get_stats()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["failed_requests"] == 0
        assert stats["consecutive_successes"] == 1

    def test_record_failure(self):
        """Test recording a failed request."""
        cb = CircuitBreaker("test", failure_threshold=5)
        cb.record_failure()

        stats = cb.get_stats()
        assert stats["total_requests"] == 1
        assert stats["failed_requests"] == 1
        assert stats["consecutive_failures"] == 1
        assert cb.state == CircuitBreakerState.CLOSED  # Still closed

    def test_opens_after_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        cb = CircuitBreaker("test", failure_threshold=3)

        for i in range(3):
            cb.record_failure()

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False

    def test_raise_if_open(self):
        """Test that raise_if_open raises when circuit is open."""
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            cb.raise_if_open()

        assert exc_info.value.provider == "test"
        assert exc_info.value.failure_count == 1

    def test_success_resets_consecutive_failures(self):
        """Test that success resets consecutive failure count."""
        cb = CircuitBreaker("test", failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.stats.current_consecutive_failures == 2

        cb.record_success()
        assert cb.stats.current_consecutive_failures == 0


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_closed_to_open_transition(self):
        """Test transition from CLOSED to OPEN."""
        cb = CircuitBreaker("test", failure_threshold=2)

        assert cb.state == CircuitBreakerState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_open_to_half_open_after_recovery(self):
        """Test transition from OPEN to HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Accessing state triggers the transition
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        """Test transition from HALF_OPEN to CLOSED on success."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1, success_threshold=2)
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN  # Need 2 successes

        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Test transition from HALF_OPEN to OPEN on failure."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset functionality."""

    def test_reset_returns_to_closed(self):
        """Test that reset returns circuit to CLOSED state."""
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        cb.reset()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.stats.total_requests == 0
        assert cb.stats.failed_requests == 0


class TestCircuitBreakerExceptionFiltering:
    """Tests for circuit breaker exception filtering."""

    def test_excluded_exceptions_not_counted(self):
        """Test that excluded exceptions are not counted as failures."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=2,
            excluded_exceptions={ValueError},
        )

        cb.record_failure(ValueError("excluded"))
        assert cb.stats.failed_requests == 0
        assert cb.state == CircuitBreakerState.CLOSED

        cb.record_failure(RuntimeError("counted"))
        cb.record_failure(RuntimeError("counted"))
        assert cb.state == CircuitBreakerState.OPEN

    def test_only_specified_exceptions_counted(self):
        """Test that only specified failure_exceptions are counted."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=2,
            failure_exceptions={TimeoutError},
        )

        # This shouldn't count
        cb.record_failure(ValueError("not counted"))
        assert cb.stats.failed_requests == 0

        # These should count
        cb.record_failure(TimeoutError("counted"))
        cb.record_failure(TimeoutError("counted"))
        assert cb.state == CircuitBreakerState.OPEN


class TestCircuitBreakerRegistry:
    """Tests for the CircuitBreakerRegistry."""

    def test_get_or_create(self):
        """Test get_or_create returns same instance for same name."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("provider1", failure_threshold=5)
        cb2 = registry.get_or_create("provider1", failure_threshold=10)  # Different threshold

        assert cb1 is cb2  # Same instance
        assert cb1.config.failure_threshold == 5  # Original threshold kept

    def test_different_names_different_instances(self):
        """Test different names create different instances."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("provider1")
        cb2 = registry.get_or_create("provider2")

        assert cb1 is not cb2

    def test_get_returns_none_if_not_exists(self):
        """Test get returns None if circuit breaker doesn't exist."""
        registry = CircuitBreakerRegistry()
        assert registry.get("nonexistent") is None

    def test_get_all_stats(self):
        """Test getting stats for all circuit breakers."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("provider1")
        registry.get_or_create("provider2")

        stats = registry.get_all_stats()
        assert "provider1" in stats
        assert "provider2" in stats

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create("provider1", failure_threshold=1)
        cb2 = registry.get_or_create("provider2", failure_threshold=1)

        cb1.record_failure()
        cb2.record_failure()

        registry.reset_all()

        assert cb1.state == CircuitBreakerState.CLOSED
        assert cb2.state == CircuitBreakerState.CLOSED

    def test_remove(self):
        """Test removing a circuit breaker from registry."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("provider1")

        assert registry.remove("provider1") is True
        assert registry.get("provider1") is None
        assert registry.remove("nonexistent") is False


class TestGlobalCircuitBreakerRegistry:
    """Tests for the global circuit breaker registry."""

    def test_get_circuit_breaker_registry_returns_same_instance(self):
        """Test global registry returns same instance."""
        registry1 = get_circuit_breaker_registry()
        registry2 = get_circuit_breaker_registry()
        assert registry1 is registry2

    def test_get_circuit_breaker_convenience_function(self):
        """Test convenience function for getting circuit breakers."""
        cb = get_circuit_breaker("test_provider", failure_threshold=3)
        assert cb.name == "test_provider"
        assert cb.config.failure_threshold == 3


class TestCircuitBreakerStats:
    """Tests for circuit breaker statistics."""

    def test_stats_track_all_metrics(self):
        """Test that stats track all expected metrics."""
        cb = CircuitBreaker("test", failure_threshold=5)

        cb.record_success()
        cb.record_failure()
        cb.record_success()

        stats = cb.get_stats()

        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["total_requests"] == 3
        assert stats["successful_requests"] == 2
        assert stats["failed_requests"] == 1
        assert stats["consecutive_successes"] == 1
        assert stats["consecutive_failures"] == 0

    def test_rejected_requests_counted(self):
        """Test that rejected requests are counted when circuit is open."""
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.stats.rejected_requests == 0

        # raise_if_open should increment rejected_requests
        with pytest.raises(CircuitBreakerOpenError):
            cb.raise_if_open()
        assert cb.stats.rejected_requests == 1

        with pytest.raises(CircuitBreakerOpenError):
            cb.raise_if_open()
        assert cb.stats.rejected_requests == 2


class TestCircuitBreakerThreadSafety:
    """Tests for circuit breaker thread safety."""

    def test_concurrent_record_operations(self):
        """Test that concurrent operations are handled safely."""
        import threading

        cb = CircuitBreaker("test", failure_threshold=100)
        errors = []

        def record_ops():
            try:
                for _ in range(50):
                    cb.record_failure()
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_ops) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cb.stats.total_requests == 1000  # 10 threads * 100 ops each


class TestCircuitBreakerRepr:
    """Tests for circuit breaker string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        cb = CircuitBreaker("test", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()

        repr_str = repr(cb)
        assert "test" in repr_str
        assert "closed" in repr_str
        assert "2/5" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
class TestTimeoutAndCircuitBreakerIntegration:
    """Integration tests for timeout and circuit breaker working together."""

    async def test_timeout_feeds_circuit_breaker(self):
        """Test that timeout errors can be recorded by circuit breaker."""
        cb = CircuitBreaker("test", failure_threshold=2)

        async def slow_operation():
            await asyncio.sleep(1.0)

        for _ in range(2):
            try:
                await task_timeout(slow_operation, 0.05, "slow op")
            except TaskTimeoutError as e:
                cb.record_failure(e)

        assert cb.state == CircuitBreakerState.OPEN

    async def test_circuit_breaker_prevents_timeout_attempts(self):
        """Test that circuit breaker prevents further attempts after opening."""
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()  # Open the circuit

        # Circuit is open, should not attempt
        with pytest.raises(CircuitBreakerOpenError):
            cb.raise_if_open()

        # The slow operation is never called
        call_count = 0

        async def tracked_slow_operation():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(1.0)

        # Check circuit before attempting
        try:
            cb.raise_if_open()
            await task_timeout(tracked_slow_operation, 0.1, "tracked")
        except CircuitBreakerOpenError:
            pass  # Expected

        assert call_count == 0  # Operation was never called
