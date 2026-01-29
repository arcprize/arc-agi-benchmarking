"""
Initialize Lambda handler for Step Functions orchestration.

Creates a new benchmark run record in DynamoDB and initializes task records
for all tasks to be processed.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variables
RUNS_TABLE = os.environ.get("RUNS_TABLE", "arc_benchmark_runs")
TASKS_TABLE = os.environ.get("TASKS_TABLE", "arc_task_progress")


def get_dynamodb_client():
    """Get DynamoDB client, supporting LocalStack for testing."""
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client("dynamodb", endpoint_url=endpoint_url)
    return boto3.client("dynamodb")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Initialize a benchmark run.

    Expected event format:
    {
        "config_name": "gpt-4o",
        "data_source": "evaluation",
        "task_ids": ["00576224", "66e6c45b", ...],
        "triggered_by": "github-actions",
        "commit_sha": "abc123..."
    }

    Returns:
    {
        "run_id": "run-2024-01-15-abc123",
        "task_ids": ["00576224", "66e6c45b", ...],
        "config_name": "gpt-4o",
        "total_tasks": 100
    }
    """
    logger.info(f"Initializing benchmark run: {json.dumps(event)}")

    # Extract parameters
    config_name = event["config_name"]
    data_source = event.get("data_source", "evaluation")
    task_ids = event["task_ids"]
    triggered_by = event.get("triggered_by", "unknown")
    commit_sha = event.get("commit_sha", "")

    # Generate run ID
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    run_id = f"run-{config_name}-{timestamp}-{short_uuid}"

    dynamodb = get_dynamodb_client()
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        # Create run record
        run_item = {
            "run_id": {"S": run_id},
            "config_name": {"S": config_name},
            "data_source": {"S": data_source},
            "status": {"S": "RUNNING"},
            "created_at": {"S": created_at},
            "triggered_by": {"S": triggered_by},
            "commit_sha": {"S": commit_sha},
            "total_tasks": {"N": str(len(task_ids))},
            "completed_tasks": {"N": "0"},
            "failed_tasks": {"N": "0"},
            "total_cost_usd": {"N": "0"},
        }

        dynamodb.put_item(TableName=RUNS_TABLE, Item=run_item)
        logger.info(f"Created run record: {run_id}")

        # Create task records in batches
        batch_size = 25  # DynamoDB batch write limit
        for i in range(0, len(task_ids), batch_size):
            batch = task_ids[i : i + batch_size]
            request_items = []

            for task_id in batch:
                task_item = {
                    "PutRequest": {
                        "Item": {
                            "run_id": {"S": run_id},
                            "task_id": {"S": task_id},
                            "status": {"S": "PENDING"},
                            "created_at": {"S": created_at},
                            "retry_count": {"N": "0"},
                        }
                    }
                }
                request_items.append(task_item)

            # Batch write with retry for unprocessed items
            unprocessed = {TASKS_TABLE: request_items}
            while unprocessed.get(TASKS_TABLE):
                response = dynamodb.batch_write_item(RequestItems=unprocessed)
                unprocessed = response.get("UnprocessedItems", {})
                if unprocessed.get(TASKS_TABLE):
                    logger.warning(
                        f"Retrying {len(unprocessed[TASKS_TABLE])} unprocessed items"
                    )

        logger.info(f"Created {len(task_ids)} task records for run {run_id}")

        return {
            "run_id": run_id,
            "task_ids": task_ids,
            "config_name": config_name,
            "total_tasks": len(task_ids),
        }

    except ClientError as e:
        logger.error(f"DynamoDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
