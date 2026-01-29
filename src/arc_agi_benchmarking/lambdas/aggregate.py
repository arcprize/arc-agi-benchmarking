"""
Aggregate Lambda handler for Step Functions orchestration.

Aggregates results from all completed tasks, calculates metrics,
and stores the aggregated results in S3.
"""

import json
import logging
import os
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
S3_BUCKET = os.environ.get("S3_BUCKET", "")


def get_dynamodb_client():
    """Get DynamoDB client, supporting LocalStack for testing."""
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client("dynamodb", endpoint_url=endpoint_url)
    return boto3.client("dynamodb")


def get_s3_client():
    """Get S3 client, supporting LocalStack for testing."""
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client("s3", endpoint_url=endpoint_url)
    return boto3.client("s3")


def decimal_to_float(obj: Any) -> Any:
    """Convert Decimal to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decimal_to_float(v) for v in obj]
    return obj


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Aggregate results from all completed tasks.

    Expected event format:
    {
        "run_id": "run-2024-01-15-abc123",
        "task_results": [...]  # Results from Map state (optional)
    }

    Returns:
    {
        "run_id": "run-2024-01-15-abc123",
        "total_tasks": 100,
        "completed_tasks": 95,
        "failed_tasks": 5,
        "total_cost_usd": 12.34,
        "accuracy": 0.85,
        "results_s3_key": "runs/run-2024-01-15-abc123/results/aggregated.json"
    }
    """
    logger.info(f"Aggregating results: {json.dumps(event)}")

    run_id = event["run_id"]

    dynamodb = get_dynamodb_client()

    try:
        # Get run record
        run_response = dynamodb.get_item(
            TableName=RUNS_TABLE, Key={"run_id": {"S": run_id}}
        )

        if "Item" not in run_response:
            raise ValueError(f"Run not found: {run_id}")

        run_item = run_response["Item"]
        config_name = run_item.get("config_name", {"S": "unknown"})["S"]
        total_tasks = int(run_item.get("total_tasks", {"N": "0"})["N"])

        # Query all task records for this run
        tasks = []
        last_evaluated_key = None

        while True:
            query_params = {
                "TableName": TASKS_TABLE,
                "KeyConditionExpression": "run_id = :run_id",
                "ExpressionAttributeValues": {":run_id": {"S": run_id}},
            }
            if last_evaluated_key:
                query_params["ExclusiveStartKey"] = last_evaluated_key

            response = dynamodb.query(**query_params)
            tasks.extend(response.get("Items", []))

            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break

        # Calculate aggregated metrics
        completed_count = 0
        failed_count = 0
        total_cost = Decimal("0")
        correct_count = 0
        task_results = []

        for task in tasks:
            task_id = task.get("task_id", {"S": "unknown"})["S"]
            status = task.get("status", {"S": "UNKNOWN"})["S"]
            cost = Decimal(task.get("cost_usd", {"N": "0"})["N"])
            # Note: is_correct is populated by a separate scoring step after the run.
            # During initial aggregation, it will be False/missing for all tasks.
            is_correct = task.get("is_correct", {"BOOL": False}).get("BOOL", False)
            result_s3_key = task.get("result_s3_key", {"S": ""})["S"]

            total_cost += cost

            if status == "COMPLETED":
                completed_count += 1
                if is_correct:
                    correct_count += 1
            elif status in ("FAILED", "FAILED_PERMANENT"):
                # Count both retryable failures and permanent failures
                failed_count += 1

            task_results.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "cost_usd": float(cost),
                    "is_correct": is_correct,
                    "result_s3_key": result_s3_key,
                }
            )

        # Calculate accuracy (only for completed tasks)
        accuracy = correct_count / completed_count if completed_count > 0 else 0.0

        # Build aggregated results
        aggregated = {
            "run_id": run_id,
            "config_name": config_name,
            "aggregated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_count,
                "failed_tasks": failed_count,
                "pending_tasks": total_tasks - completed_count - failed_count,
                "correct_tasks": correct_count,
                "accuracy": accuracy,
                "total_cost_usd": float(total_cost),
            },
            "task_results": task_results,
        }

        # Store aggregated results in S3
        results_s3_key = ""
        if S3_BUCKET:
            s3 = get_s3_client()
            results_s3_key = f"runs/{run_id}/results/aggregated.json"

            s3.put_object(
                Bucket=S3_BUCKET,
                Key=results_s3_key,
                Body=json.dumps(aggregated, indent=2, default=str),
                ContentType="application/json",
            )
            logger.info(f"Stored aggregated results: s3://{S3_BUCKET}/{results_s3_key}")

        # Update run record with aggregated metrics
        dynamodb.update_item(
            TableName=RUNS_TABLE,
            Key={"run_id": {"S": run_id}},
            UpdateExpression="""
                SET completed_tasks = :completed,
                    failed_tasks = :failed,
                    total_cost_usd = :cost,
                    accuracy = :accuracy,
                    updated_at = :updated_at
            """,
            ExpressionAttributeValues={
                ":completed": {"N": str(completed_count)},
                ":failed": {"N": str(failed_count)},
                ":cost": {"N": str(total_cost)},
                ":accuracy": {"N": str(accuracy)},
                ":updated_at": {"S": datetime.now(timezone.utc).isoformat()},
            },
        )

        logger.info(
            f"Aggregation complete for {run_id}: "
            f"{completed_count}/{total_tasks} completed, "
            f"{failed_count} failed, "
            f"accuracy={accuracy:.2%}, "
            f"cost=${total_cost:.2f}"
        )

        return {
            "run_id": run_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "total_cost_usd": float(total_cost),
            "accuracy": accuracy,
            "results_s3_key": results_s3_key,
        }

    except ClientError as e:
        logger.error(f"AWS error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
