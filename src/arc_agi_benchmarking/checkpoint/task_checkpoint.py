"""Task checkpoint manager for within-task progress tracking."""

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arc_agi_benchmarking.checkpoint.models import (
    AttemptResult,
    TaskCheckpoint,
)
from arc_agi_benchmarking.storage import StorageBackend

logger = logging.getLogger(__name__)


class TaskCheckpointManager:
    """Manages within-task checkpointing.

    Tracks completed attempts within a task and enables resuming
    from the last successful attempt after interruption.
    """

    def __init__(
        self,
        storage: StorageBackend,
        task_id: str,
        checkpoint_dir: str = "checkpoints",
    ):
        self.storage = storage
        self.task_id = task_id
        self.checkpoint_dir = checkpoint_dir
        self._checkpoint: TaskCheckpoint | None = None

    @property
    def checkpoint_key(self) -> str:
        return f"{self.checkpoint_dir}/{self.task_id}.json"

    @property
    def checkpoint(self) -> TaskCheckpoint:
        if self._checkpoint is None:
            self._checkpoint = self._load_or_create()
        return self._checkpoint

    def _load_or_create(self) -> TaskCheckpoint:
        data = self.storage.read_text(self.checkpoint_key)
        if data:
            try:
                checkpoint = TaskCheckpoint.from_dict(json.loads(data))
                logger.info(
                    f"Resumed checkpoint for {self.task_id} "
                    f"with {len(checkpoint.completed_attempts)} completed attempts"
                )
                return checkpoint
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load checkpoint, starting fresh: {e}")
        return TaskCheckpoint(task_id=self.task_id)

    def _save(self) -> None:
        self.checkpoint.updated_at = datetime.now(timezone.utc)
        data = json.dumps(self.checkpoint.to_dict(), indent=2)
        self.storage.write_text(self.checkpoint_key, data)

    def get_completed_attempts(self) -> list[AttemptResult]:
        """Get list of completed attempts."""
        return list(self.checkpoint.completed_attempts)

    def get_next_attempt_index(self, test_pair_index: int, max_attempts: int) -> int | None:
        """Get the next attempt index to run for a test pair.

        Returns None if all attempts for this test pair are complete.
        """
        completed = {
            a.attempt_index
            for a in self.checkpoint.completed_attempts
            if a.test_pair_index == test_pair_index
        }

        for i in range(max_attempts):
            if i not in completed:
                return i
        return None

    def is_test_pair_complete(self, test_pair_index: int, max_attempts: int) -> bool:
        """Check if all attempts for a test pair are complete."""
        return self.get_next_attempt_index(test_pair_index, max_attempts) is None

    def record_attempt(
        self,
        test_pair_index: int,
        attempt_index: int,
        response: Any,
        cost_usd: Decimal = Decimal("0"),
        tokens_input: int = 0,
        tokens_output: int = 0,
        duration_seconds: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Record a completed attempt and save checkpoint."""
        result = AttemptResult(
            attempt_index=attempt_index,
            test_pair_index=test_pair_index,
            response=response,
            cost_usd=cost_usd,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            duration_seconds=duration_seconds,
            error=error,
        )

        self.checkpoint.completed_attempts.append(result)
        self.checkpoint.total_cost_usd += cost_usd
        self.checkpoint.total_tokens_input += tokens_input
        self.checkpoint.total_tokens_output += tokens_output
        self._save()

        logger.debug(
            f"Checkpointed attempt {attempt_index} for test pair {test_pair_index} "
            f"of task {self.task_id}"
        )

    def get_results_for_test_pair(self, test_pair_index: int) -> list[AttemptResult]:
        """Get all completed attempts for a specific test pair."""
        return [
            a
            for a in self.checkpoint.completed_attempts
            if a.test_pair_index == test_pair_index
        ]

    def delete_checkpoint(self) -> None:
        """Delete the checkpoint file after successful task completion."""
        self.storage.delete(self.checkpoint_key)
        self._checkpoint = None
        logger.debug(f"Deleted checkpoint for task {self.task_id}")

    def get_summary(self) -> dict:
        """Get a summary of the checkpoint state."""
        return {
            "task_id": self.task_id,
            "completed_attempts": len(self.checkpoint.completed_attempts),
            "total_cost_usd": str(self.checkpoint.total_cost_usd),
            "total_tokens_input": self.checkpoint.total_tokens_input,
            "total_tokens_output": self.checkpoint.total_tokens_output,
        }
