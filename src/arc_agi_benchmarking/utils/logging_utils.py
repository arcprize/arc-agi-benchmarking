"""Centralized structured logging configuration.

Provides JSON-formatted logging with consistent fields across the application.
Supports both console and file output with configurable formats.
"""

import json
import logging
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Standard fields included in all log records
STANDARD_FIELDS = [
    "timestamp",
    "level",
    "logger",
    "message",
]

# Optional context fields that may be present
CONTEXT_FIELDS = [
    "task_id",
    "config",
    "provider",
    "attempt",
    "duration_ms",
    "tokens_used",
    "cost_usd",
    "error_type",
]

_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "env",
    "environment",
    "password",
    "proxy_authorization",
    "secret",
    "token",
}


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or any(
        normalized.endswith(f"_{suffix}") for suffix in _SENSITIVE_KEYS
    )


def serialize_for_raw_log(value: Any, *, _seen: Optional[set[int]] = None) -> Any:
    """Convert SDK values to JSON-safe data while recursively redacting secrets."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return serialize_for_raw_log(value.value, _seen=_seen)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}

    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return {"type": type(value).__name__, "value": "[CIRCULAR]"}

    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            return {
                str(key): (
                    _REDACTED
                    if _is_sensitive_key(key)
                    else serialize_for_raw_log(item, _seen=seen)
                )
                for key, item in value.items()
            }
        finally:
            seen.remove(value_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seen.add(value_id)
        try:
            return [serialize_for_raw_log(item, _seen=seen) for item in value]
        finally:
            seen.remove(value_id)

    converted: Any = None
    if hasattr(value, "model_dump"):
        with suppress(Exception):
            converted = value.model_dump(mode="json")
    if converted is None and hasattr(value, "to_dict"):
        with suppress(Exception):
            converted = value.to_dict()
    if converted is None and is_dataclass(value) and not isinstance(value, type):
        with suppress(Exception):
            converted = asdict(value)
    if converted is None and hasattr(value, "dict"):
        with suppress(Exception):
            converted = value.dict()
    if converted is None and hasattr(value, "__dict__"):
        with suppress(Exception):
            converted = {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_") and not callable(item)
            }
    if converted is not None:
        seen.add(value_id)
        try:
            return serialize_for_raw_log(converted, _seen=seen)
        finally:
            seen.remove(value_id)

    with suppress(Exception):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    return {"type": type(value).__name__, "repr": "[UNAVAILABLE]"}


@dataclass(frozen=True)
class RawAPIRequestHandle:
    started_monotonic: float
    context: dict[str, Any]
    operation: str


class RawAPILogger:
    """Emit correlated API lifecycle events through the existing logger."""

    schema_version = 1

    def __init__(
        self,
        run_id: Optional[str] = None,
        event_logger: Optional[logging.Logger] = None,
    ):
        self.run_id = run_id or str(uuid.uuid4())
        self.logger = event_logger or logging.getLogger(
            "arc_agi_benchmarking.raw_api"
        )

    def start_request(
        self,
        operation: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        context: Mapping[str, Any],
        request_payload: Any = None,
    ) -> RawAPIRequestHandle:
        timestamp = datetime.now(timezone.utc)
        handle = RawAPIRequestHandle(
            started_monotonic=time.monotonic(),
            context=serialize_for_raw_log(dict(context)),
            operation=operation,
        )
        payload = (
            request_payload
            if request_payload is not None
            else {"args": list(args), "kwargs": dict(kwargs)}
        )
        self._emit(
            {
                **self._base_event(handle, "request_started", timestamp),
                "request": serialize_for_raw_log(payload),
            }
        )
        return handle

    def record_success(self, handle: RawAPIRequestHandle, response: Any) -> None:
        self._emit(
            {
                **self._base_event(
                    handle,
                    "request_succeeded",
                    datetime.now(timezone.utc),
                ),
                "duration_ms": self._duration_ms(handle),
                "response": serialize_for_raw_log(response),
            }
        )

    def record_failure(self, handle: RawAPIRequestHandle, error: BaseException) -> None:
        error_data: dict[str, Any] = {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }
        for attr in ("status_code", "request_id", "body", "response"):
            value = getattr(error, attr, None)
            if value is not None:
                error_data[attr] = serialize_for_raw_log(value)
        self._emit(
            {
                **self._base_event(
                    handle,
                    "request_failed",
                    datetime.now(timezone.utc),
                ),
                "duration_ms": self._duration_ms(handle),
                "error": serialize_for_raw_log(error_data),
            }
        )

    def record_task_timeout(
        self,
        *,
        task_id: str,
        config: str,
        elapsed: Optional[float],
        timeout: Optional[float],
    ) -> None:
        self._emit(
            {
                "schema_version": self.schema_version,
                "event": "task_timed_out",
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": {"task_id": task_id, "config": config},
                "elapsed_seconds": elapsed,
                "timeout_seconds": timeout,
            }
        )

    def _base_event(
        self,
        handle: RawAPIRequestHandle,
        event: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event": event,
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "operation": handle.operation,
            "context": handle.context,
        }

    def _duration_ms(self, handle: RawAPIRequestHandle) -> float:
        return round((time.monotonic() - handle.started_monotonic) * 1000, 3)

    def _emit(self, event: Mapping[str, Any]) -> None:
        self.logger.info(event["event"], extra={"raw_api_event": dict(event)})


class _ExcludeRawAPIEvents(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not hasattr(record, "raw_api_event")


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs logs as single-line JSON objects with consistent fields.
    Additional context can be added via the `extra` parameter in log calls.
    """

    def __init__(self, include_extra: bool = True):
        """Initialize the formatter.

        Args:
            include_extra: Whether to include extra fields from log records.
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        raw_api_event = getattr(record, "raw_api_event", None)
        if raw_api_event is not None:
            return json.dumps(raw_api_event, default=str)

        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add context fields if present in the record
        if self.include_extra:
            for field in CONTEXT_FIELDS:
                if hasattr(record, field):
                    value = getattr(record, field)
                    if value is not None:
                        log_data[field] = value

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for console output.

    Formats logs with timestamp, level, and message in a readable format.
    Includes context fields inline when present.
    """

    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record for human reading."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build context string from extra fields
        context_parts = []
        for field in CONTEXT_FIELDS:
            if hasattr(record, field):
                value = getattr(record, field)
                if value is not None:
                    # Format field name nicely
                    display_name = field.replace("_", " ").title()
                    if field == "duration_ms":
                        context_parts.append(f"{value}ms")
                    elif field == "cost_usd":
                        context_parts.append(f"${value:.4f}")
                    elif field == "tokens_used":
                        context_parts.append(f"{value} tokens")
                    else:
                        context_parts.append(f"{field}={value}")

        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        base_msg = f"{timestamp} | {record.levelname:<8} | {record.name} | {record.getMessage()}{context_str}"

        # Add exception info if present
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"

        return base_msg


def setup_logging(
    level: str = "INFO",
    json_console: bool = False,
    log_file: Optional[Path] = None,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE).
        json_console: If True, output JSON to console instead of human-readable.
        log_file: Optional path to write JSON logs to a file.
        quiet_libraries: If True, suppress verbose logging from third-party libraries.

    Returns:
        The root logger for the application.
    """
    # Handle special "NONE" level
    if level.upper() == "NONE":
        logging.disable(logging.CRITICAL)
        return logging.getLogger("arc_agi_benchmarking")

    # Enable logging if previously disabled
    logging.disable(logging.NOTSET)

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure the root logger so all loggers inherit the config
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.addFilter(_ExcludeRawAPIEvents())

    if json_console:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(HumanReadableFormatter())

    root_logger.addHandler(console_handler)

    # File handler (always JSON)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Quiet noisy libraries
    if quiet_libraries:
        noisy_loggers = [
            "httpx",
            "httpcore",
            "urllib3",
            "requests",
            "anthropic",
            "google",
            "openai",
            "pydantic",
            "transformers",
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Return the app logger for convenience
    return logging.getLogger("arc_agi_benchmarking")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the application prefix.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A logger instance.
    """
    # Ensure it's under the app namespace
    if not name.startswith("arc_agi_benchmarking"):
        name = f"arc_agi_benchmarking.{name}"
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding fields to log records.

    Usage:
        with LogContext(task_id="abc123", config="gpt-4o"):
            logger.info("Processing task")  # Will include task_id and config
    """

    def __init__(self, **kwargs: Any):
        """Initialize with context fields."""
        self.context = kwargs
        self._old_factory = None

    def __enter__(self):
        """Add context to log records."""
        self._old_factory = logging.getLogRecordFactory()
        context = self.context

        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log record factory."""
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
        return False


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any,
) -> None:
    """Log a message with additional context fields.

    Args:
        logger: The logger to use.
        level: Log level (e.g., logging.INFO).
        message: Log message.
        **context: Additional fields to include in the log record.

    Example:
        log_with_context(logger, logging.INFO, "Task completed",
                        task_id="abc123", duration_ms=1500, tokens_used=2000)
    """
    # Filter out keys that might already be set by the global record factory
    # to avoid KeyError from logging's extra dict handling
    safe_context = {}
    for key, value in context.items():
        if key not in ["message", "asctime"]:
            safe_context[key] = value

    # Create a temporary record factory that adds our context
    old_factory = logging.getLogRecordFactory()

    def factory_with_context(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        for key, value in safe_context.items():
            # Only set if not already set or if we're overriding with new value
            setattr(record, key, value)
        return record

    try:
        logging.setLogRecordFactory(factory_with_context)
        logger.log(level, message)
    finally:
        logging.setLogRecordFactory(old_factory)
