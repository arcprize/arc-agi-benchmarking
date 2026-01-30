"""DynamoDB-based progress manager for distributed benchmark runs."""

import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


# DynamoDB status constants (UPPERCASE to match PRD and AWS orchestration)
class DynamoDBTaskStatus:
    """Task status values for DynamoDB storage.

    FAILED: Task failed but can be retried (retry_count < max_retries)
    FAILED_PERMANENT: Task failed and retries exhausted (will not be re-queued)
    """

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"  # Retryable failure
    FAILED_PERMANENT = "FAILED_PERMANENT"  # Retries exhausted, won't be re-queued


class DynamoDBProgressManager:
    """Manages distributed progress tracking using DynamoDB.

    Uses conditional updates to prevent race conditions when multiple
    workers claim tasks concurrently.
    """

    def __init__(
        self,
        runs_table_name: str,
        tasks_table_name: str,
        region_name: Optional[str] = None,
    ):
        """Initialize the DynamoDB progress manager.

        Args:
            runs_table_name: Name of the benchmark runs DynamoDB table.
            tasks_table_name: Name of the task progress DynamoDB table.
            region_name: AWS region. Defaults to AWS_REGION env var or us-west-2.
        """
        import boto3

        self._region = region_name or os.environ.get("AWS_REGION", "us-west-2")
        self._dynamodb = boto3.resource("dynamodb", region_name=self._region)
        self._runs_table = self._dynamodb.Table(runs_table_name)
        self._tasks_table = self._dynamodb.Table(tasks_table_name)
        self._worker_id = f"{os.getpid()}@{os.uname().nodename}"

    def create_run(
        self,
        config_name: str,
        task_ids: list[str],
        data_source: str,
        triggered_by: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> str:
        """Create a new benchmark run.

        Args:
            config_name: Name of the model configuration.
            task_ids: List of task IDs to process.
            data_source: Source of task data (e.g., "evaluation", "training").
            triggered_by: Who triggered the run (optional).
            commit_sha: Git commit SHA (optional).

        Returns:
            The generated run_id.
        """
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self._runs_table.put_item(
            Item={
                "run_id": run_id,
                "config_name": config_name,
                "data_source": data_source,
                "total_tasks": len(task_ids),
                "completed_tasks": 0,
                "failed_tasks": 0,
                "total_cost_usd": Decimal("0"),
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "status": "IN_PROGRESS",
                "triggered_by": triggered_by or "unknown",
                "commit_sha": commit_sha or "unknown",
                "created_at": now,
                "updated_at": now,
            }
        )

        # Batch write task entries
        with self._tasks_table.batch_writer() as batch:
            for task_id in task_ids:
                batch.put_item(
                    Item={
                        "run_id": run_id,
                        "task_id": task_id,
                        "status": DynamoDBTaskStatus.PENDING,
                        "retry_count": 0,
                        "created_at": now,
                        "updated_at": now,
                    }
                )

        logger.info(f"Created run {run_id} with {len(task_ids)} tasks")
        return run_id

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get run details."""
        response = self._runs_table.get_item(Key={"run_id": run_id})
        return response.get("Item")

    def get_pending_tasks(self, run_id: str) -> list[str]:
        """Get list of tasks that are pending or need retry.

        Handles pagination for large task lists (>1MB response).
        """
        task_ids = []
        query_kwargs = {
            "KeyConditionExpression": "run_id = :rid",
            "FilterExpression": "#status IN (:pending, :failed)",
            "ExpressionAttributeNames": {"#status": "status"},
            "ExpressionAttributeValues": {
                ":rid": run_id,
                ":pending": DynamoDBTaskStatus.PENDING,
                ":failed": DynamoDBTaskStatus.FAILED,
            },
        }

        while True:
            response = self._tasks_table.query(**query_kwargs)
            task_ids.extend(item["task_id"] for item in response.get("Items", []))

            # Check for more pages
            if "LastEvaluatedKey" not in response:
                break
            query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        return task_ids

    def claim_task(self, run_id: str, task_id: str) -> bool:
        """Attempt to claim a task for processing.

        Uses conditional update to prevent race conditions.

        Returns:
            True if the task was claimed, False otherwise.
        """
        now = datetime.now(timezone.utc).isoformat()

        try:
            self._tasks_table.update_item(
                Key={"run_id": run_id, "task_id": task_id},
                UpdateExpression=(
                    "SET #status = :in_progress, "
                    "worker_id = :worker, "
                    "started_at = :now, "
                    "updated_at = :now"
                ),
                ConditionExpression="#status = :pending OR #status = :failed",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":in_progress": DynamoDBTaskStatus.IN_PROGRESS,
                    ":pending": DynamoDBTaskStatus.PENDING,
                    ":failed": DynamoDBTaskStatus.FAILED,
                    ":worker": self._worker_id,
                    ":now": now,
                },
            )
            logger.info(f"Claimed task {task_id} in run {run_id}")
            return True
        except self._dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            logger.debug(f"Failed to claim task {task_id} (already claimed)")
            return False

    def mark_completed(
        self,
        run_id: str,
        task_id: str,
        result_s3_key: Optional[str] = None,
        cost_usd: Decimal = Decimal("0"),
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> None:
        """Mark a task as completed."""
        now = datetime.now(timezone.utc).isoformat()

        # Update task
        update_expr = (
            "SET #status = :completed, "
            "completed_at = :now, "
            "updated_at = :now, "
            "cost_usd = :cost, "
            "tokens_input = :tokens_in, "
            "tokens_output = :tokens_out"
        )
        expr_values = {
            ":completed": DynamoDBTaskStatus.COMPLETED,
            ":now": now,
            ":cost": cost_usd,
            ":tokens_in": tokens_input,
            ":tokens_out": tokens_output,
        }

        if result_s3_key:
            update_expr += ", result_s3_key = :s3_key"
            expr_values[":s3_key"] = result_s3_key

        self._tasks_table.update_item(
            Key={"run_id": run_id, "task_id": task_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues=expr_values,
        )

        # Update run aggregates (including cost and token rollups)
        self._runs_table.update_item(
            Key={"run_id": run_id},
            UpdateExpression=(
                "SET completed_tasks = completed_tasks + :one, "
                "total_cost_usd = total_cost_usd + :cost, "
                "total_input_tokens = total_input_tokens + :tokens_in, "
                "total_output_tokens = total_output_tokens + :tokens_out, "
                "updated_at = :now"
            ),
            ExpressionAttributeValues={
                ":one": 1,
                ":cost": cost_usd,
                ":tokens_in": tokens_input,
                ":tokens_out": tokens_output,
                ":now": now,
            },
        )

        logger.info(f"Marked task {task_id} as completed (cost=${cost_usd})")

    def mark_failed(
        self,
        run_id: str,
        task_id: str,
        error: str,
        max_retries: int = 3,
    ) -> bool:
        """Mark a task as failed.

        Atomically updates retry_count, error, and status in a single DynamoDB call.
        If retries are exhausted, marks the task as FAILED_PERMANENT so it won't
        be re-queued by get_pending_tasks.

        Args:
            run_id: The run ID.
            task_id: The task ID.
            error: Error message.
            max_retries: Maximum retry attempts before marking as permanently failed.

        Returns:
            True if retries exhausted, False if can retry.
        """
        now = datetime.now(timezone.utc).isoformat()

        # First, get current retry_count to determine the target status
        response = self._tasks_table.get_item(
            Key={"run_id": run_id, "task_id": task_id}
        )
        item = response.get("Item", {})
        current_retry_count = int(item.get("retry_count", 0))
        new_retry_count = current_retry_count + 1
        retries_exhausted = new_retry_count >= max_retries

        # Determine target status based on whether retries will be exhausted
        target_status = (
            DynamoDBTaskStatus.FAILED_PERMANENT
            if retries_exhausted
            else DynamoDBTaskStatus.FAILED
        )

        # Atomic update: retry_count, error, status, and updated_at in one call
        # Use conditional expression to ensure retry_count hasn't changed
        # Note: 'error' and 'status' are DynamoDB reserved keywords
        try:
            self._tasks_table.update_item(
                Key={"run_id": run_id, "task_id": task_id},
                UpdateExpression=(
                    "SET retry_count = :new_count, "
                    "#err = :error, "
                    "#status = :status, "
                    "updated_at = :now"
                ),
                ConditionExpression=(
                    "attribute_not_exists(retry_count) OR retry_count = :old_count"
                ),
                ExpressionAttributeNames={
                    "#err": "error",
                    "#status": "status",
                },
                ExpressionAttributeValues={
                    ":new_count": new_retry_count,
                    ":old_count": current_retry_count,
                    ":error": error,
                    ":status": target_status,
                    ":now": now,
                },
            )
        except self._dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            # Another worker updated retry_count - re-read and retry
            logger.debug(f"Contention on mark_failed for {task_id}, retrying")
            return self.mark_failed(run_id, task_id, error, max_retries)

        # Note: We don't increment failed_tasks here - that's done by the
        # handle_error Lambda to avoid double-counting when Step Functions
        # also catches the failure.
        if retries_exhausted:
            logger.warning(
                f"Task {task_id} failed permanently after {new_retry_count} retries"
            )
        else:
            logger.info(
                f"Task {task_id} failed (retry {new_retry_count}/{max_retries})"
            )

        return retries_exhausted

    def mark_run_completed(self, run_id: str) -> None:
        """Mark the entire run as completed."""
        now = datetime.now(timezone.utc).isoformat()

        self._runs_table.update_item(
            Key={"run_id": run_id},
            UpdateExpression=(
                "SET #status = :completed, completed_at = :now, updated_at = :now"
            ),
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":completed": "COMPLETED",
                ":now": now,
            },
        )
        logger.info(f"Marked run {run_id} as completed")

    def get_run_summary(self, run_id: str) -> dict:
        """Get a summary of the run progress including cost and token totals."""
        run = self.get_run(run_id)
        if not run:
            return {}

        return {
            "run_id": run_id,
            "status": run.get("status"),
            "total": run.get("total_tasks", 0),
            "completed": run.get("completed_tasks", 0),
            "failed": run.get("failed_tasks", 0),
            "pending": run.get("total_tasks", 0)
            - run.get("completed_tasks", 0)
            - run.get("failed_tasks", 0),
            "total_cost_usd": float(run.get("total_cost_usd", 0)),
            "total_input_tokens": run.get("total_input_tokens", 0),
            "total_output_tokens": run.get("total_output_tokens", 0),
        }
