"""Circuit breaker implementation for preventing cascading failures."""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Set

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        recovery_time: Optional[float] = None,
        failure_count: Optional[int] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.recovery_time = recovery_time
        self.failure_count = failure_count


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    failure_window: float = 0.0
    failure_exceptions: Optional[Set[type]] = None
    excluded_exceptions: Optional[Set[type]] = None


@dataclass
class CircuitBreakerStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_transitions: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change_time: Optional[float] = None
    current_consecutive_failures: int = 0
    current_consecutive_successes: int = 0


class CircuitBreaker:
    """Circuit breaker with configurable thresholds."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        failure_window: float = 0.0,
        failure_exceptions: Optional[Set[type]] = None,
        excluded_exceptions: Optional[Set[type]] = None,
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            failure_window=failure_window,
            failure_exceptions=failure_exceptions,
            excluded_exceptions=excluded_exceptions,
        )
        self.stats = CircuitBreakerStats()
        self._state = CircuitBreakerState.CLOSED
        self._failure_times: list[float] = []
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to(CircuitBreakerState.HALF_OPEN)
            return self._state

    def _should_attempt_recovery(self) -> bool:
        if self.stats.last_failure_time is None:
            return True
        elapsed = time.time() - self.stats.last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        old_state = self._state
        self._state = new_state
        self.stats.state_transitions += 1
        self.stats.last_state_change_time = time.time()
        logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")

    def _count_recent_failures(self) -> int:
        if self.config.failure_window <= 0:
            return self.stats.current_consecutive_failures
        cutoff = time.time() - self.config.failure_window
        self._failure_times = [t for t in self._failure_times if t > cutoff]
        return len(self._failure_times)

    def _should_count_as_failure(self, exception: Exception) -> bool:
        exc_type = type(exception)
        if self.config.excluded_exceptions and exc_type in self.config.excluded_exceptions:
            return False
        if self.config.failure_exceptions:
            return exc_type in self.config.failure_exceptions
        return True

    def can_execute(self) -> bool:
        return self.state != CircuitBreakerState.OPEN

    def raise_if_open(self) -> None:
        if not self.can_execute():
            with self._lock:
                self.stats.rejected_requests += 1
            recovery_time = None
            if self.stats.last_failure_time:
                recovery_time = max(
                    0, self.config.recovery_timeout - (time.time() - self.stats.last_failure_time)
                )
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. Retry after {recovery_time:.1f}s" if recovery_time else "",
                provider=self.name,
                recovery_time=recovery_time,
                failure_count=self.stats.current_consecutive_failures,
            )

    def record_success(self) -> None:
        with self._lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.time()
            self.stats.current_consecutive_failures = 0
            self.stats.current_consecutive_successes += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self.stats.current_consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitBreakerState.CLOSED)
                    self.stats.current_consecutive_successes = 0

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        with self._lock:
            if exception and not self._should_count_as_failure(exception):
                return

            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.last_failure_time = time.time()
            self.stats.current_consecutive_failures += 1
            self.stats.current_consecutive_successes = 0
            self._failure_times.append(time.time())

            failure_count = self._count_recent_failures()
            logger.warning(
                f"Circuit breaker '{self.name}' failure ({failure_count}/{self.config.failure_threshold})"
            )

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._transition_to(CircuitBreakerState.OPEN)
            elif self._state == CircuitBreakerState.CLOSED:
                if failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitBreakerState.OPEN)

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self.stats = CircuitBreakerStats()
            self._failure_times = []

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "rejected_requests": self.stats.rejected_requests,
                "state_transitions": self.stats.state_transitions,
                "consecutive_failures": self.stats.current_consecutive_failures,
                "consecutive_successes": self.stats.current_consecutive_successes,
                "last_failure_time": self.stats.last_failure_time,
                "last_success_time": self.stats.last_success_time,
            }

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name='{self.name}', state={self._state.value}, "
            f"failures={self.stats.current_consecutive_failures}/{self.config.failure_threshold})"
        )


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers by name."""

    def __init__(self, default_config: Optional[CircuitBreakerConfig] = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self._default_config = default_config or CircuitBreakerConfig()

    def get(self, name: str) -> Optional[CircuitBreaker]:
        with self._lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold or self._default_config.failure_threshold,
                    recovery_timeout=recovery_timeout or self._default_config.recovery_timeout,
                    success_threshold=kwargs.get("success_threshold", self._default_config.success_threshold),
                    failure_window=kwargs.get("failure_window", self._default_config.failure_window),
                    failure_exceptions=kwargs.get("failure_exceptions", self._default_config.failure_exceptions),
                    excluded_exceptions=kwargs.get("excluded_exceptions", self._default_config.excluded_exceptions),
                )
            return self._breakers[name]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: cb.get_stats() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False


_global_registry: Optional[CircuitBreakerRegistry] = None
_global_registry_lock = threading.Lock()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = CircuitBreakerRegistry()
    return _global_registry


def get_circuit_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    return get_circuit_breaker_registry().get_or_create(name, **kwargs)
