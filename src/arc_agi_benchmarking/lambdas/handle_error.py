"""
Handle Error Lambda handler for Step Functions orchestration.

Handles task failures by updating DynamoDB with error information
and incrementing the failed task count.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variables
RUNS_TABLE = os.environ.get("RUNS_TABLE", "arc_benchmark_runs")
TASKS_TABLE = os.environ.get("TASKS_TABLE", "arc_task_progress")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))


def get_dynamodb_client():
    """Get DynamoDB client, supporting LocalStack for testing."""
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client("dynamodb", endpoint_url=endpoint_url)
    return boto3.client("dynamodb")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Handle a task error from Step Functions.

    Expected event format:
    {
        "run_id": "run-2024-01-15-abc123",
        "task_id": "00576224",
        "error": {
            "Error": "States.TaskFailed",
            "Cause": "Batch job failed..."
        }
    }

    Returns:
    {
        "run_id": "run-2024-01-15-abc123",
        "task_id": "00576224",
        "status": "FAILED" | "RETRY_SCHEDULED",
        "retry_count": 1,
        "error_message": "..."
    }
    """
    logger.info(f"Handling task error: {json.dumps(event)}")

    run_id = event["run_id"]
    task_id = event["task_id"]
    error_info = event.get("error", {})

    # Extract error details
    error_type = error_info.get("Error", "Unknown")
    error_cause = error_info.get("Cause", "No details available")

    # Truncate error cause if too long (DynamoDB attribute limit)
    if len(error_cause) > 4000:
        error_cause = error_cause[:4000] + "... (truncated)"

    dynamodb = get_dynamodb_client()
    updated_at = datetime.now(timezone.utc).isoformat()

    try:
        # Get current task state to check retry count
        task_response = dynamodb.get_item(
            TableName=TASKS_TABLE,
            Key={"run_id": {"S": run_id}, "task_id": {"S": task_id}},
        )

        current_retry_count = 0
        if "Item" in task_response:
            current_retry_count = int(
                task_response["Item"].get("retry_count", {"N": "0"})["N"]
            )

        new_retry_count = current_retry_count + 1
        should_retry = new_retry_count < MAX_RETRIES

        if should_retry:
            # Mark as failed but retryable. Note: Step Functions Map state doesn't
            # automatically re-queue - a separate sweep/retry mechanism is needed.
            new_status = "FAILED"
            logger.info(
                f"Task {task_id} failed (attempt {new_retry_count}/{MAX_RETRIES}), eligible for retry"
            )
        else:
            # Mark as permanently failed - retries exhausted
            new_status = "FAILED_PERMANENT"
            logger.warning(
                f"Task {task_id} has exhausted retries ({MAX_RETRIES}), marking as FAILED_PERMANENT"
            )

        # Update task record
        dynamodb.update_item(
            TableName=TASKS_TABLE,
            Key={"run_id": {"S": run_id}, "task_id": {"S": task_id}},
            UpdateExpression="""
                SET #status = :status,
                    updated_at = :updated_at,
                    retry_count = :retry_count,
                    last_error_type = :error_type,
                    last_error_cause = :error_cause,
                    last_error_at = :error_at
            """,
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": {"S": new_status},
                ":updated_at": {"S": updated_at},
                ":retry_count": {"N": str(new_retry_count)},
                ":error_type": {"S": error_type},
                ":error_cause": {"S": error_cause},
                ":error_at": {"S": updated_at},
            },
        )

        # Only increment failed_tasks counter when permanently failed (retries exhausted)
        if new_status == "FAILED_PERMANENT":
            dynamodb.update_item(
                TableName=RUNS_TABLE,
                Key={"run_id": {"S": run_id}},
                UpdateExpression="SET failed_tasks = failed_tasks + :inc, updated_at = :updated_at",
                ExpressionAttributeValues={
                    ":inc": {"N": "1"},
                    ":updated_at": {"S": updated_at},
                },
            )
            logger.info(f"Incremented failed_tasks counter for run {run_id}")

        return {
            "run_id": run_id,
            "task_id": task_id,
            "status": "RETRY_SCHEDULED" if should_retry else "FAILED_PERMANENT",
            "retry_count": new_retry_count,
            "error_message": f"{error_type}: {error_cause[:200]}",
        }

    except ClientError as e:
        logger.error(f"DynamoDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
