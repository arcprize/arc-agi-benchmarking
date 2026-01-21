"""Tests for checkpointing and progress tracking."""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from arc_agi_benchmarking.checkpoint import (
    AttemptResult,
    BatchProgress,
    BatchProgressManager,
    TaskCheckpoint,
    TaskCheckpointManager,
    TaskProgress,
    TaskStatus,
)
from arc_agi_benchmarking.storage import LocalStorageBackend


class TestTaskStatus:
    def test_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


class TestAttemptResult:
    def test_to_dict_and_from_dict(self):
        result = AttemptResult(
            attempt_index=0,
            test_pair_index=1,
            response={"answer": [[1, 2], [3, 4]]},
            cost_usd=Decimal("0.005"),
            tokens_input=100,
            tokens_output=50,
            duration_seconds=1.5,
        )

        data = result.to_dict()
        restored = AttemptResult.from_dict(data)

        assert restored.attempt_index == 0
        assert restored.test_pair_index == 1
        assert restored.response == {"answer": [[1, 2], [3, 4]]}
        assert restored.cost_usd == Decimal("0.005")
        assert restored.tokens_input == 100
        assert restored.tokens_output == 50
        assert restored.duration_seconds == 1.5

    def test_with_error(self):
        result = AttemptResult(
            attempt_index=0,
            test_pair_index=0,
            response=None,
            error="API timeout",
        )

        data = result.to_dict()
        restored = AttemptResult.from_dict(data)

        assert restored.error == "API timeout"
        assert restored.response is None


class TestTaskCheckpoint:
    def test_to_dict_and_from_dict(self):
        checkpoint = TaskCheckpoint(
            task_id="task_001",
            total_cost_usd=Decimal("0.01"),
        )
        checkpoint.completed_attempts.append(
            AttemptResult(
                attempt_index=0,
                test_pair_index=0,
                response={"answer": [[1]]},
            )
        )

        data = checkpoint.to_dict()
        restored = TaskCheckpoint.from_dict(data)

        assert restored.task_id == "task_001"
        assert restored.total_cost_usd == Decimal("0.01")
        assert len(restored.completed_attempts) == 1

    def test_unsupported_version_raises(self):
        data = {"schema_version": 99, "task_id": "test"}
        with pytest.raises(ValueError, match="Unsupported checkpoint schema version"):
            TaskCheckpoint.from_dict(data)


class TestTaskProgress:
    def test_to_dict_and_from_dict(self):
        progress = TaskProgress(
            task_id="task_001",
            status=TaskStatus.IN_PROGRESS,
            attempts_completed=2,
            attempts_total=3,
            cost_usd=Decimal("0.05"),
            worker_id="worker_1",
            started_at=datetime.utcnow(),
        )

        data = progress.to_dict()
        restored = TaskProgress.from_dict(data)

        assert restored.task_id == "task_001"
        assert restored.status == TaskStatus.IN_PROGRESS
        assert restored.attempts_completed == 2
        assert restored.attempts_total == 3
        assert restored.cost_usd == Decimal("0.05")


class TestBatchProgress:
    def test_to_dict_and_from_dict(self):
        batch = BatchProgress(run_id="run_123")
        batch.tasks["task_1"] = TaskProgress(task_id="task_1", status=TaskStatus.COMPLETED)
        batch.tasks["task_2"] = TaskProgress(task_id="task_2", status=TaskStatus.PENDING)

        data = batch.to_dict()
        restored = BatchProgress.from_dict(data)

        assert restored.run_id == "run_123"
        assert len(restored.tasks) == 2
        assert restored.tasks["task_1"].status == TaskStatus.COMPLETED

    def test_count_properties(self):
        batch = BatchProgress(run_id="test")
        batch.tasks["t1"] = TaskProgress(task_id="t1", status=TaskStatus.PENDING)
        batch.tasks["t2"] = TaskProgress(task_id="t2", status=TaskStatus.IN_PROGRESS)
        batch.tasks["t3"] = TaskProgress(task_id="t3", status=TaskStatus.COMPLETED)
        batch.tasks["t4"] = TaskProgress(task_id="t4", status=TaskStatus.FAILED)

        assert batch.pending_count == 1
        assert batch.in_progress_count == 1
        assert batch.completed_count == 1
        assert batch.failed_count == 1
        assert batch.total_count == 4


class TestBatchProgressManager:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorageBackend:
        return LocalStorageBackend(tmp_path)

    @pytest.fixture
    def manager(self, storage: LocalStorageBackend) -> BatchProgressManager:
        return BatchProgressManager(storage, run_id="test_run")

    def test_initialize_empty_task_list(self, manager: BatchProgressManager):
        manager.initialize_tasks([])
        assert manager.progress.total_count == 0

    def test_claim_nonexistent_task(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1"])
        assert not manager.claim_task("nonexistent")

    def test_initialize_preserves_existing(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1"], attempts_per_task=2)
        manager.claim_task("task_1")
        manager.mark_completed("task_1")

        manager.initialize_tasks(["task_1", "task_2"], attempts_per_task=3)

        assert manager.progress.tasks["task_1"].status == TaskStatus.COMPLETED
        assert manager.progress.tasks["task_1"].attempts_total == 2
        assert manager.progress.tasks["task_2"].attempts_total == 3

    def test_corrupted_json_recovery(self, storage: LocalStorageBackend):
        storage.write_text("progress.json", "not valid json{{{")
        manager = BatchProgressManager(storage, run_id="test_run")
        assert manager.progress.total_count == 0

    def test_initialize_tasks(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1", "task_2", "task_3"], attempts_per_task=2)

        assert manager.progress.total_count == 3
        assert all(
            t.status == TaskStatus.PENDING for t in manager.progress.tasks.values()
        )
        assert all(t.attempts_total == 2 for t in manager.progress.tasks.values())

    def test_claim_task(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1"])

        assert manager.claim_task("task_1")
        assert manager.progress.tasks["task_1"].status == TaskStatus.IN_PROGRESS
        assert manager.progress.tasks["task_1"].worker_id is not None

    def test_claim_task_already_claimed(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1"])
        manager.claim_task("task_1")

        assert not manager.claim_task("task_1")

    def test_claim_next_task(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1", "task_2"])

        task_id = manager.claim_next_task()
        assert task_id == "task_1"

        task_id = manager.claim_next_task()
        assert task_id == "task_2"

        task_id = manager.claim_next_task()
        assert task_id is None

    def test_mark_completed(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1"])
        manager.claim_task("task_1")
        manager.mark_completed(
            "task_1",
            cost_usd=Decimal("0.05"),
            tokens_input=100,
            tokens_output=50,
        )

        task = manager.progress.tasks["task_1"]
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert manager.progress.total_cost_usd == Decimal("0.05")

    def test_mark_failed(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1"])
        manager.claim_task("task_1")
        manager.mark_failed("task_1", error="API error")

        task = manager.progress.tasks["task_1"]
        assert task.status == TaskStatus.FAILED
        assert task.error == "API error"

    def test_is_complete(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1", "task_2"])

        assert not manager.is_complete()

        manager.claim_task("task_1")
        manager.mark_completed("task_1")
        assert not manager.is_complete()

        manager.claim_task("task_2")
        manager.mark_failed("task_2", "error")
        assert manager.is_complete()

    def test_persistence(self, storage: LocalStorageBackend):
        manager1 = BatchProgressManager(storage, run_id="test_run")
        manager1.initialize_tasks(["task_1", "task_2"])
        manager1.claim_task("task_1")
        manager1.mark_completed("task_1", cost_usd=Decimal("0.10"))

        manager2 = BatchProgressManager(storage, run_id="test_run")

        assert manager2.progress.total_count == 2
        assert manager2.progress.tasks["task_1"].status == TaskStatus.COMPLETED
        assert manager2.progress.tasks["task_2"].status == TaskStatus.PENDING
        assert manager2.progress.total_cost_usd == Decimal("0.10")

    def test_reset_stale_tasks(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1"])
        manager.claim_task("task_1")

        manager.progress.tasks["task_1"].started_at = datetime.utcnow() - timedelta(
            hours=2
        )
        manager._save()

        reset_count = manager.reset_stale_tasks(max_age_seconds=3600)

        assert reset_count == 1
        assert manager.progress.tasks["task_1"].status == TaskStatus.PENDING

    def test_get_summary(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1", "task_2"])
        manager.claim_task("task_1")
        manager.mark_completed("task_1", cost_usd=Decimal("0.05"))

        summary = manager.get_summary()

        assert summary["total"] == 2
        assert summary["completed"] == 1
        assert summary["pending"] == 1
        assert summary["total_cost_usd"] == "0.05"

    def test_retry_failed_tasks(self, manager: BatchProgressManager):
        manager.initialize_tasks(["task_1", "task_2", "task_3"])
        manager.claim_task("task_1")
        manager.mark_completed("task_1")
        manager.claim_task("task_2")
        manager.mark_failed("task_2", "error1")
        manager.claim_task("task_3")
        manager.mark_failed("task_3", "error2")

        assert manager.progress.completed_count == 1
        assert manager.progress.failed_count == 2

        reset_count = manager.retry_failed_tasks()

        assert reset_count == 2
        assert manager.progress.completed_count == 1
        assert manager.progress.failed_count == 0
        assert manager.progress.pending_count == 2
        assert manager.progress.tasks["task_2"].status == TaskStatus.PENDING
        assert manager.progress.tasks["task_2"].error is None
        assert manager.progress.tasks["task_3"].status == TaskStatus.PENDING


class TestTaskCheckpointManager:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorageBackend:
        return LocalStorageBackend(tmp_path)

    @pytest.fixture
    def manager(self, storage: LocalStorageBackend) -> TaskCheckpointManager:
        return TaskCheckpointManager(storage, task_id="task_001")

    def test_corrupted_json_recovery(self, storage: LocalStorageBackend):
        storage.write_text("checkpoints/task_001.json", "{invalid")
        manager = TaskCheckpointManager(storage, task_id="task_001")
        assert len(manager.get_completed_attempts()) == 0

    def test_multiple_test_pairs(self, manager: TaskCheckpointManager):
        manager.record_attempt(test_pair_index=0, attempt_index=0, response="a")
        manager.record_attempt(test_pair_index=1, attempt_index=0, response="b")
        manager.record_attempt(test_pair_index=2, attempt_index=0, response="c")

        assert len(manager.get_results_for_test_pair(0)) == 1
        assert len(manager.get_results_for_test_pair(1)) == 1
        assert len(manager.get_results_for_test_pair(2)) == 1
        assert manager.checkpoint.total_cost_usd == Decimal("0")

    def test_record_attempt(self, manager: TaskCheckpointManager):
        manager.record_attempt(
            test_pair_index=0,
            attempt_index=0,
            response={"answer": [[1, 2]]},
            cost_usd=Decimal("0.01"),
            tokens_input=50,
            tokens_output=25,
        )

        attempts = manager.get_completed_attempts()
        assert len(attempts) == 1
        assert attempts[0].test_pair_index == 0
        assert attempts[0].attempt_index == 0

    def test_get_next_attempt_index(self, manager: TaskCheckpointManager):
        assert manager.get_next_attempt_index(test_pair_index=0, max_attempts=2) == 0

        manager.record_attempt(
            test_pair_index=0,
            attempt_index=0,
            response=None,
        )
        assert manager.get_next_attempt_index(test_pair_index=0, max_attempts=2) == 1

        manager.record_attempt(
            test_pair_index=0,
            attempt_index=1,
            response=None,
        )
        assert manager.get_next_attempt_index(test_pair_index=0, max_attempts=2) is None

    def test_is_test_pair_complete(self, manager: TaskCheckpointManager):
        assert not manager.is_test_pair_complete(test_pair_index=0, max_attempts=2)

        manager.record_attempt(test_pair_index=0, attempt_index=0, response=None)
        assert not manager.is_test_pair_complete(test_pair_index=0, max_attempts=2)

        manager.record_attempt(test_pair_index=0, attempt_index=1, response=None)
        assert manager.is_test_pair_complete(test_pair_index=0, max_attempts=2)

    def test_get_results_for_test_pair(self, manager: TaskCheckpointManager):
        manager.record_attempt(test_pair_index=0, attempt_index=0, response="a")
        manager.record_attempt(test_pair_index=1, attempt_index=0, response="b")
        manager.record_attempt(test_pair_index=0, attempt_index=1, response="c")

        results = manager.get_results_for_test_pair(0)
        assert len(results) == 2
        assert results[0].response == "a"
        assert results[1].response == "c"

    def test_persistence(self, storage: LocalStorageBackend):
        manager1 = TaskCheckpointManager(storage, task_id="task_001")
        manager1.record_attempt(
            test_pair_index=0,
            attempt_index=0,
            response={"answer": [[1]]},
            cost_usd=Decimal("0.02"),
        )

        manager2 = TaskCheckpointManager(storage, task_id="task_001")

        attempts = manager2.get_completed_attempts()
        assert len(attempts) == 1
        assert manager2.checkpoint.total_cost_usd == Decimal("0.02")

    def test_delete_checkpoint(self, manager: TaskCheckpointManager, storage: LocalStorageBackend):
        manager.record_attempt(test_pair_index=0, attempt_index=0, response=None)
        assert storage.exists(manager.checkpoint_key)

        manager.delete_checkpoint()
        assert not storage.exists(manager.checkpoint_key)

    def test_get_summary(self, manager: TaskCheckpointManager):
        manager.record_attempt(
            test_pair_index=0,
            attempt_index=0,
            response=None,
            cost_usd=Decimal("0.01"),
            tokens_input=100,
            tokens_output=50,
        )

        summary = manager.get_summary()

        assert summary["task_id"] == "task_001"
        assert summary["completed_attempts"] == 1
        assert summary["total_cost_usd"] == "0.01"
        assert summary["total_tokens_input"] == 100
