"""Append-only logging for provider API requests and responses."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import json
import logging
from pathlib import Path
import re
import threading
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence
import uuid


logger = logging.getLogger(__name__)

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


def _safe_task_label(task_id: Optional[str]) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id or "unknown_task"))
    return label if label not in {"", ".", ".."} else "unknown_task"


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or any(
        normalized.endswith(f"_{suffix}")
        for suffix in (
            "api_key",
            "authorization",
            "cookie",
            "password",
            "secret",
            "token",
        )
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

    try:
        representation = repr(value)
    except Exception:
        representation = "[UNAVAILABLE]"
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": representation,
    }


@dataclass(frozen=True)
class RawAPIRequestHandle:
    request_id: str
    task_id: Optional[str]
    started_at: datetime
    started_monotonic: float
    context: dict[str, Any]
    operation: str


class RawAPIRecorder:
    """Write correlated request lifecycle events to per-task JSONL files."""

    schema_version = 1

    def __init__(
        self,
        log_dir: Path | str | None = None,
        run_id: Optional[str] = None,
        *,
        log_file: Path | str | None = None,
    ):
        if (log_dir is None) == (log_file is None):
            raise ValueError("Specify exactly one of log_dir or log_file")

        self.log_file = (
            Path(log_file).expanduser().resolve() if log_file is not None else None
        )
        self.log_dir = (
            self.log_file.parent
            if self.log_file is not None
            else Path(log_dir).expanduser().resolve()
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_dir.is_dir():
            raise ValueError(f"Raw API log path is not a directory: {self.log_dir}")
        try:
            with tempfile.TemporaryFile(dir=self.log_dir):
                pass
        except OSError as error:
            raise ValueError(
                f"Raw API log directory is not writable: {self.log_dir}"
            ) from error
        self.run_id = run_id or str(uuid.uuid4())
        self._write_lock = threading.Lock()

    def start_request(
        self,
        operation: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        context: Mapping[str, Any],
        request_payload: Any = None,
    ) -> RawAPIRequestHandle:
        started_at = datetime.now(timezone.utc)
        safe_context = serialize_for_raw_log(dict(context))
        handle = RawAPIRequestHandle(
            request_id=str(uuid.uuid4()),
            task_id=context.get("task_id"),
            started_at=started_at,
            started_monotonic=time.monotonic(),
            context=safe_context,
            operation=operation,
        )
        payload = request_payload
        if payload is None:
            payload = {"args": list(args), "kwargs": dict(kwargs)}
        self._append(
            handle.task_id,
            {
                **self._base_event(handle, "request_started", started_at),
                "request": serialize_for_raw_log(payload),
            },
        )
        return handle

    def record_success(self, handle: RawAPIRequestHandle, response: Any) -> None:
        now = datetime.now(timezone.utc)
        self._append(
            handle.task_id,
            {
                **self._base_event(handle, "request_succeeded", now),
                "duration_ms": round((time.monotonic() - handle.started_monotonic) * 1000, 3),
                "response": serialize_for_raw_log(response),
            },
        )

    def record_failure(self, handle: RawAPIRequestHandle, error: BaseException) -> None:
        now = datetime.now(timezone.utc)
        error_data: dict[str, Any] = {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }
        for attr in ("status_code", "request_id", "body", "response"):
            value = getattr(error, attr, None)
            if value is not None:
                error_data[attr] = serialize_for_raw_log(value)
        self._append(
            handle.task_id,
            {
                **self._base_event(handle, "request_failed", now),
                "duration_ms": round((time.monotonic() - handle.started_monotonic) * 1000, 3),
                "error": serialize_for_raw_log(error_data),
            },
        )

    def record_task_timeout(
        self,
        *,
        task_id: str,
        config: str,
        elapsed: Optional[float],
        timeout: Optional[float],
    ) -> None:
        self._append(
            task_id,
            {
                "schema_version": self.schema_version,
                "event": "task_timed_out",
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": {"task_id": task_id, "config": config},
                "elapsed_seconds": elapsed,
                "timeout_seconds": timeout,
            },
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
            "request_id": handle.request_id,
            "timestamp": timestamp.isoformat(),
            "operation": handle.operation,
            "context": handle.context,
        }

    def _append(self, task_id: Optional[str], event: Mapping[str, Any]) -> None:
        task_label = _safe_task_label(task_id)
        path = self.log_file or self.log_dir / f"{task_label}.jsonl"
        line = json.dumps(event, ensure_ascii=True, separators=(",", ":"), default=str)
        try:
            with self._write_lock:
                with path.open("a", encoding="utf-8") as output:
                    output.write(f"{line}\n")
                    output.flush()
        except Exception:
            logger.exception("Failed to write raw API log event to %s", path)
