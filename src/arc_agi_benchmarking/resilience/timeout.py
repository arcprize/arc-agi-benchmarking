"""Timeout utilities for API calls and task execution."""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Python 3.11+ has asyncio.timeout, older versions need a polyfill
_HAS_ASYNCIO_TIMEOUT = sys.version_info >= (3, 11)


class TaskTimeoutError(Exception):
    """Raised when a task or request times out."""

    def __init__(self, message: str, elapsed: Optional[float] = None, timeout: Optional[float] = None):
        super().__init__(message)
        self.elapsed = elapsed
        self.timeout = timeout


if _HAS_ASYNCIO_TIMEOUT:
    @asynccontextmanager
    async def request_timeout(seconds: float, operation: str = "request"):
        """Async context manager for request timeouts."""
        if seconds <= 0:
            yield
            return

        start_time = time.monotonic()
        try:
            async with asyncio.timeout(seconds):
                yield
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            raise TaskTimeoutError(
                f"{operation} timed out after {elapsed:.2f}s (limit: {seconds}s)",
                elapsed=elapsed,
                timeout=seconds,
            )
else:
    @asynccontextmanager
    async def request_timeout(seconds: float, operation: str = "request"):
        """Async context manager for request timeouts (Python 3.10 compatible)."""
        if seconds <= 0:
            yield
            return

        start_time = time.monotonic()
        task = asyncio.current_task()
        loop = asyncio.get_event_loop()

        timeout_handle = loop.call_later(seconds, task.cancel)
        try:
            yield
        except asyncio.CancelledError:
            elapsed = time.monotonic() - start_time
            if elapsed >= seconds:
                raise TaskTimeoutError(
                    f"{operation} timed out after {elapsed:.2f}s (limit: {seconds}s)",
                    elapsed=elapsed,
                    timeout=seconds,
                )
            raise
        finally:
            timeout_handle.cancel()


@contextmanager
def sync_timeout(seconds: float, operation: str = "operation"):
    """Tracks elapsed time and logs warning if exceeded. Does NOT interrupt."""
    start_time = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start_time
        if elapsed > seconds:
            logger.warning(f"{operation} exceeded duration: {elapsed:.2f}s > {seconds}s")


async def task_timeout(
    coro_or_func: Callable[..., T],
    timeout_seconds: float,
    operation: str = "task",
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a coroutine or sync function with a timeout.

    Note: For sync functions, the timeout only cancels the await, not the thread.
    The underlying thread continues running until completion.
    """
    if timeout_seconds <= 0:
        if asyncio.iscoroutinefunction(coro_or_func):
            return await coro_or_func(*args, **kwargs)
        else:
            return await asyncio.to_thread(coro_or_func, *args, **kwargs)

    start_time = time.monotonic()
    try:
        if asyncio.iscoroutinefunction(coro_or_func):
            coro = coro_or_func(*args, **kwargs)
        else:
            coro = asyncio.to_thread(coro_or_func, *args, **kwargs)
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start_time
        raise TaskTimeoutError(
            f"{operation} timed out after {elapsed:.2f}s (limit: {timeout_seconds}s)",
            elapsed=elapsed,
            timeout=timeout_seconds,
        )


def with_timeout(timeout_seconds: float, operation: Optional[str] = None):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation or func.__name__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await task_timeout(func, timeout_seconds, op_name, *args, **kwargs)

        return wrapper

    return decorator


DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_REASONING_TIMEOUT = 900
DEFAULT_TASK_TIMEOUT = 1800
