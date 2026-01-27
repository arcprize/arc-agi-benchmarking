"""Checkpointing and progress tracking for benchmark runs."""

from arc_agi_benchmarking.checkpoint.models import (
    TaskStatus,
    TaskProgress,
    BatchProgress,
    TaskCheckpoint,
    AttemptResult,
)
from arc_agi_benchmarking.checkpoint.batch_progress import BatchProgressManager
from arc_agi_benchmarking.checkpoint.task_checkpoint import TaskCheckpointManager

__all__ = [
    "TaskStatus",
    "TaskProgress",
    "BatchProgress",
    "TaskCheckpoint",
    "AttemptResult",
    "BatchProgressManager",
    "TaskCheckpointManager",
]

# Optional DynamoDB support
try:
    from arc_agi_benchmarking.checkpoint.dynamodb_progress import DynamoDBProgressManager

    __all__.append("DynamoDBProgressManager")
except ImportError:
    pass  # boto3 not installed
