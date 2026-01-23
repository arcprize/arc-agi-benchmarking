"""
Resilience module for timeout and circuit breaker functionality.

This module provides resilience patterns to handle:
- Request timeouts to prevent indefinite hangs
- Circuit breakers to prevent cascading failures

Usage:
    from arc_agi_benchmarking.resilience import (
        CircuitBreaker,
        CircuitBreakerOpenError,
        CircuitBreakerState,
        TaskTimeoutError,
        request_timeout,
        task_timeout,
    )
"""

from arc_agi_benchmarking.resilience.timeout import (
    TaskTimeoutError,
    request_timeout,
    task_timeout,
)
from arc_agi_benchmarking.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    CircuitBreakerRegistry,
    get_circuit_breaker,
    get_circuit_breaker_registry,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitBreakerRegistry",
    "CircuitBreakerState",
    "TaskTimeoutError",
    "get_circuit_breaker",
    "get_circuit_breaker_registry",
    "request_timeout",
    "task_timeout",
]
