#!/usr/bin/env python3
"""Demo script to test checkpointing with fake tasks."""

import random
import time
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from arc_agi_benchmarking.checkpoint import (
    BatchProgressManager,
    TaskCheckpointManager,
)
from arc_agi_benchmarking.storage import LocalStorageBackend


def simulate_task(task_checkpoint: TaskCheckpointManager, num_test_pairs: int = 3, max_attempts: int = 2, fail_task: bool = False):
    """Simulate running a task with checkpointing.

    Returns True if task succeeded, False if it failed.
    """
    all_succeeded = True

    for test_pair_idx in range(num_test_pairs):
        pair_succeeded = False
        while True:
            attempt_idx = task_checkpoint.get_next_attempt_index(test_pair_idx, max_attempts)
            if attempt_idx is None:
                break  # All attempts done for this test pair

            # Simulate work
            print(f"    Running test pair {test_pair_idx}, attempt {attempt_idx}...")
            time.sleep(0.1)  # Simulate API call

            # Simulate occasional failure (or forced failure)
            if fail_task or random.random() < 0.3:
                print(f"    ❌ Simulated failure!")
                task_checkpoint.record_attempt(
                    test_pair_index=test_pair_idx,
                    attempt_index=attempt_idx,
                    response=None,
                    error="Simulated API error",
                    cost_usd=Decimal("0.001"),
                )
            else:
                print(f"    ✓ Success!")
                task_checkpoint.record_attempt(
                    test_pair_index=test_pair_idx,
                    attempt_index=attempt_idx,
                    response={"answer": [[1, 2], [3, 4]]},
                    cost_usd=Decimal("0.005"),
                    tokens_input=100,
                    tokens_output=50,
                )
                pair_succeeded = True
                break  # Success, move to next test pair

        if not pair_succeeded:
            all_succeeded = False

    return all_succeeded


def run_batch(storage: LocalStorageBackend, task_ids: list[str], interrupt_after: int | None = None, force_fail_task: str | None = None):
    """Run a batch of tasks with checkpointing."""
    batch_manager = BatchProgressManager(storage, run_id="demo_run")
    batch_manager.initialize_tasks(task_ids)

    print(f"\n{'='*50}")
    print(f"Starting batch: {batch_manager.get_summary()}")
    print(f"{'='*50}\n")

    tasks_processed = 0
    while True:
        task_id = batch_manager.claim_next_task()
        if not task_id:
            break

        print(f"\n▶ Processing task: {task_id}")

        task_checkpoint = TaskCheckpointManager(storage, task_id)

        # Check if resuming
        existing = len(task_checkpoint.get_completed_attempts())
        if existing > 0:
            print(f"  Resuming from checkpoint ({existing} attempts already done)")

        try:
            fail_this_task = (task_id == force_fail_task)
            success = simulate_task(task_checkpoint, fail_task=fail_this_task)

            if success:
                batch_manager.mark_completed(
                    task_id,
                    cost_usd=task_checkpoint.checkpoint.total_cost_usd,
                    tokens_input=task_checkpoint.checkpoint.total_tokens_input,
                    tokens_output=task_checkpoint.checkpoint.total_tokens_output,
                )
                task_checkpoint.delete_checkpoint()
                print(f"  ✓ Task {task_id} completed")
            else:
                batch_manager.mark_failed(
                    task_id,
                    "All attempts exhausted",
                    cost_usd=task_checkpoint.checkpoint.total_cost_usd,
                    tokens_input=task_checkpoint.checkpoint.total_tokens_input,
                    tokens_output=task_checkpoint.checkpoint.total_tokens_output,
                )
                task_checkpoint.delete_checkpoint()
                print(f"  ✗ Task {task_id} failed (all attempts exhausted)")

        except Exception as e:
            batch_manager.mark_failed(task_id, str(e))
            print(f"  ✗ Task {task_id} failed: {e}")

        tasks_processed += 1
        if interrupt_after and tasks_processed >= interrupt_after:
            print(f"\n⚠️  Simulating interruption after {tasks_processed} tasks!")
            return False

    print(f"\n{'='*50}")
    print(f"Batch complete: {batch_manager.get_summary()}")
    print(f"{'='*50}\n")
    return batch_manager


def main():
    task_ids = [f"task_{i:03d}" for i in range(5)]

    with TemporaryDirectory() as tmpdir:
        storage = LocalStorageBackend(Path(tmpdir))
        print(f"Using temp storage: {tmpdir}")

        # First run - force task_002 to fail
        print("\n" + "="*60)
        print("RUN 1: Processing tasks (task_002 will fail)")
        print("="*60)
        batch_manager = run_batch(storage, task_ids, force_fail_task="task_002")

        # Show failed task
        print(f"\nFailed tasks: {batch_manager.progress.failed_count}")

        # Retry failed tasks
        print("\n" + "="*60)
        print("RUN 2: Retrying failed tasks")
        print("="*60)
        reset_count = batch_manager.retry_failed_tasks()
        print(f"Reset {reset_count} failed task(s) back to pending")

        # Run again - should only process the previously failed task
        run_batch(storage, task_ids)

        # Third run - should be a no-op
        print("\n" + "="*60)
        print("RUN 3: Running again (should be no-op)")
        print("="*60)
        run_batch(storage, task_ids)


if __name__ == "__main__":
    main()
