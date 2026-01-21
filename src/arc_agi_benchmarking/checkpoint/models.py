"""Data models for checkpointing."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    """Status of a task in the batch."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AttemptResult:
    """Result of a single attempt within a task."""

    attempt_index: int
    test_pair_index: int
    response: Any
    cost_usd: Decimal = Decimal("0")
    tokens_input: int = 0
    tokens_output: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    timestamp: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return {
            "attempt_index": self.attempt_index,
            "test_pair_index": self.test_pair_index,
            "response": self.response,
            "cost_usd": str(self.cost_usd),
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttemptResult":
        return cls(
            attempt_index=data["attempt_index"],
            test_pair_index=data["test_pair_index"],
            response=data["response"],
            cost_usd=Decimal(data.get("cost_usd", "0")),
            tokens_input=data.get("tokens_input", 0),
            tokens_output=data.get("tokens_output", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            error=data.get("error"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class TaskCheckpoint:
    """Checkpoint for within-task progress."""

    schema_version: int = 1
    task_id: str = ""
    completed_attempts: list[AttemptResult] = field(default_factory=list)
    total_cost_usd: Decimal = Decimal("0")
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    started_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "completed_attempts": [a.to_dict() for a in self.completed_attempts],
            "total_cost_usd": str(self.total_cost_usd),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskCheckpoint":
        version = data.get("schema_version", 1)
        if version != 1:
            raise ValueError(f"Unsupported checkpoint schema version: {version}")
        return cls(
            schema_version=version,
            task_id=data["task_id"],
            completed_attempts=[
                AttemptResult.from_dict(a) for a in data.get("completed_attempts", [])
            ],
            total_cost_usd=Decimal(data.get("total_cost_usd", "0")),
            total_tokens_input=data.get("total_tokens_input", 0),
            total_tokens_output=data.get("total_tokens_output", 0),
            started_at=datetime.fromisoformat(data["started_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class TaskProgress:
    """Progress of a single task within a batch."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    attempts_completed: int = 0
    attempts_total: int = 0
    cost_usd: Decimal = Decimal("0")
    error: str | None = None
    worker_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "attempts_completed": self.attempts_completed,
            "attempts_total": self.attempts_total,
            "cost_usd": str(self.cost_usd),
            "error": self.error,
            "worker_id": self.worker_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskProgress":
        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            attempts_completed=data.get("attempts_completed", 0),
            attempts_total=data.get("attempts_total", 0),
            cost_usd=Decimal(data.get("cost_usd", "0")),
            error=data.get("error"),
            worker_id=data.get("worker_id"),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
        )


@dataclass
class BatchProgress:
    """Overall progress of a benchmark batch."""

    schema_version: int = 1
    run_id: str = ""
    tasks: dict[str, TaskProgress] = field(default_factory=dict)
    total_cost_usd: Decimal = Decimal("0")
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    started_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)

    @property
    def in_progress_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

    @property
    def total_count(self) -> int:
        return len(self.tasks)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "total_cost_usd": str(self.total_cost_usd),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchProgress":
        version = data.get("schema_version", 1)
        if version != 1:
            raise ValueError(f"Unsupported batch progress schema version: {version}")
        return cls(
            schema_version=version,
            run_id=data["run_id"],
            tasks={
                tid: TaskProgress.from_dict(t) for tid, t in data.get("tasks", {}).items()
            },
            total_cost_usd=Decimal(data.get("total_cost_usd", "0")),
            total_tokens_input=data.get("total_tokens_input", 0),
            total_tokens_output=data.get("total_tokens_output", 0),
            started_at=datetime.fromisoformat(data["started_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
