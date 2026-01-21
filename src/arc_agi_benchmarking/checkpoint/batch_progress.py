"""Batch progress manager for tracking overall run progress."""

import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal

from arc_agi_benchmarking.checkpoint.models import (
    BatchProgress,
    TaskProgress,
    TaskStatus,
)
from arc_agi_benchmarking.storage import StorageBackend

logger = logging.getLogger(__name__)


class BatchProgressManager:
    """Manages batch-level progress tracking.

    Tracks which tasks are pending, in-progress, completed, or failed.
    Persists progress to storage for resume capability.
    """

    def __init__(
        self,
        storage: StorageBackend,
        run_id: str,
        progress_key: str = "progress.json",
    ):
        self.storage = storage
        self.run_id = run_id
        self.progress_key = progress_key
        self._progress: BatchProgress | None = None
        self._worker_id = f"{os.getpid()}@{os.uname().nodename}"

    @property
    def progress(self) -> BatchProgress:
        if self._progress is None:
            self._progress = self._load_or_create()
        return self._progress

    def _load_or_create(self) -> BatchProgress:
        data = self.storage.read_text(self.progress_key)
        if data:
            try:
                progress = BatchProgress.from_dict(json.loads(data))
                if progress.run_id != self.run_id:
                    logger.warning(
                        f"Run ID mismatch: expected {self.run_id}, "
                        f"found {progress.run_id}. Starting fresh."
                    )
                    return BatchProgress(run_id=self.run_id)
                return progress
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load progress, starting fresh: {e}")
        return BatchProgress(run_id=self.run_id)

    def _save(self) -> None:
        self.progress.updated_at = datetime.now(timezone.utc)
        self.storage.write_text(
            self.progress_key,
            json.dumps(self.progress.to_dict(), indent=2),
        )

    def initialize_tasks(self, task_ids: list[str], attempts_per_task: int = 1) -> None:
        """Initialize progress tracking for a list of tasks.

        Only adds tasks that don't already exist (preserves resumed state).
        """
        for task_id in task_ids:
            if task_id not in self.progress.tasks:
                self.progress.tasks[task_id] = TaskProgress(
                    task_id=task_id,
                    attempts_total=attempts_per_task,
                )
        self._save()

    def claim_task(self, task_id: str) -> bool:
        """Attempt to claim a task for processing.

        Returns True if the task was successfully claimed, False if it's
        already being processed or completed.
        """
        task = self.progress.tasks.get(task_id)
        if not task:
            return False

        if task.status != TaskStatus.PENDING:
            return False

        task.status = TaskStatus.IN_PROGRESS
        task.worker_id = self._worker_id
        task.started_at = datetime.now(timezone.utc)
        self._save()
        return True

    def get_next_pending_task(self) -> str | None:
        """Get the next pending task ID, or None if all tasks are done."""
        for task_id, task in self.progress.tasks.items():
            if task.status == TaskStatus.PENDING:
                return task_id
        return None

    def claim_next_task(self) -> str | None:
        """Claim the next available pending task.

        Returns the task ID if successful, None if no tasks available.
        Uses a retry loop to handle race conditions where another worker
        claims a task between get_next_pending_task() and claim_task().
        """
        while True:
            task_id = self.get_next_pending_task()
            if task_id is None:
                return None
            if self.claim_task(task_id):
                return task_id

    def mark_completed(
        self,
        task_id: str,
        cost_usd: Decimal = Decimal("0"),
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> None:
        """Mark a task as completed."""
        task = self.progress.tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        task.cost_usd = cost_usd

        self.progress.total_cost_usd += cost_usd
        self.progress.total_tokens_input += tokens_input
        self.progress.total_tokens_output += tokens_output
        self._save()

    def mark_failed(
        self,
        task_id: str,
        error: str,
        cost_usd: Decimal = Decimal("0"),
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> None:
        """Mark a task as failed, accumulating any costs incurred."""
        task = self.progress.tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = datetime.now(timezone.utc)
        task.cost_usd = cost_usd

        self.progress.total_cost_usd += cost_usd
        self.progress.total_tokens_input += tokens_input
        self.progress.total_tokens_output += tokens_output
        self._save()

    def update_task_progress(
        self,
        task_id: str,
        attempts_completed: int,
        cost_usd: Decimal = Decimal("0"),
    ) -> None:
        """Update progress within a task (e.g., after each attempt)."""
        task = self.progress.tasks.get(task_id)
        if not task:
            return

        task.attempts_completed = attempts_completed
        task.cost_usd = cost_usd
        self._save()

    def reset_stale_tasks(self, max_age_seconds: int = 3600) -> int:
        """Reset tasks that have been in-progress too long (stale workers).

        Returns the number of tasks reset.
        """
        now = datetime.now(timezone.utc)
        reset_count = 0

        for task in self.progress.tasks.values():
            if task.status != TaskStatus.IN_PROGRESS:
                continue
            if task.started_at is None:
                continue

            age = (now - task.started_at).total_seconds()
            if age > max_age_seconds:
                logger.info(
                    f"Resetting stale task {task.task_id} "
                    f"(age: {age:.0f}s, worker: {task.worker_id})"
                )
                task.status = TaskStatus.PENDING
                task.worker_id = None
                task.started_at = None
                reset_count += 1

        if reset_count > 0:
            self._save()

        return reset_count

    def is_complete(self) -> bool:
        """Check if all tasks are completed or failed."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self.progress.tasks.values()
        )

    def retry_failed_tasks(self) -> int:
        """Reset failed tasks back to pending for retry.

        Returns the number of tasks reset.
        """
        reset_count = 0
        for task in self.progress.tasks.values():
            if task.status == TaskStatus.FAILED:
                task.status = TaskStatus.PENDING
                task.error = None
                task.worker_id = None
                task.started_at = None
                task.completed_at = None
                reset_count += 1

        if reset_count > 0:
            self._save()

        return reset_count

    def get_summary(self) -> dict:
        """Get a summary of the current progress."""
        return {
            "run_id": self.run_id,
            "total": self.progress.total_count,
            "pending": self.progress.pending_count,
            "in_progress": self.progress.in_progress_count,
            "completed": self.progress.completed_count,
            "failed": self.progress.failed_count,
            "total_cost_usd": str(self.progress.total_cost_usd),
        }
