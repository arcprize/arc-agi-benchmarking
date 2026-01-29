"""
Complete Lambda handler for Step Functions orchestration.

Marks the benchmark run as complete and publishes final metrics
to CloudWatch.
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
CLOUDWATCH_NAMESPACE = os.environ.get("CLOUDWATCH_NAMESPACE", "ArcBenchmark")


def get_dynamodb_client():
    """Get DynamoDB client, supporting LocalStack for testing."""
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client("dynamodb", endpoint_url=endpoint_url)
    return boto3.client("dynamodb")


def get_cloudwatch_client():
    """Get CloudWatch client, supporting LocalStack for testing."""
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client("cloudwatch", endpoint_url=endpoint_url)
    return boto3.client("cloudwatch")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Mark a benchmark run as complete.

    Expected event format:
    {
        "run_id": "run-2024-01-15-abc123",
        "aggregation": {
            "total_tasks": 100,
            "completed_tasks": 95,
            "failed_tasks": 5,
            "total_cost_usd": 12.34,
            "accuracy": 0.85
        }
    }

    Returns:
    {
        "run_id": "run-2024-01-15-abc123",
        "status": "COMPLETED",
        "completed_at": "2024-01-15T12:00:00Z",
        "final_metrics": {...}
    }
    """
    logger.info(f"Completing benchmark run: {json.dumps(event)}")

    run_id = event["run_id"]
    aggregation = event.get("aggregation", {})

    dynamodb = get_dynamodb_client()
    completed_at = datetime.now(timezone.utc).isoformat()

    try:
        # Get run record for config name
        run_response = dynamodb.get_item(
            TableName=RUNS_TABLE, Key={"run_id": {"S": run_id}}
        )

        if "Item" not in run_response:
            raise ValueError(f"Run not found: {run_id}")

        run_item = run_response["Item"]
        config_name = run_item.get("config_name", {"S": "unknown"})["S"]
        created_at = run_item.get("created_at", {"S": completed_at})["S"]

        # Calculate duration
        try:
            start_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            duration_seconds = (end_time - start_time).total_seconds()
        except (ValueError, TypeError):
            duration_seconds = 0

        # Determine final status
        failed_tasks = aggregation.get("failed_tasks", 0)
        total_tasks = aggregation.get("total_tasks", 0)

        if failed_tasks == total_tasks and total_tasks > 0:
            final_status = "FAILED"
        elif failed_tasks > 0:
            final_status = "COMPLETED_WITH_ERRORS"
        else:
            final_status = "COMPLETED"

        # Update run record
        dynamodb.update_item(
            TableName=RUNS_TABLE,
            Key={"run_id": {"S": run_id}},
            UpdateExpression="""
                SET #status = :status,
                    completed_at = :completed_at,
                    duration_seconds = :duration,
                    updated_at = :updated_at
            """,
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": {"S": final_status},
                ":completed_at": {"S": completed_at},
                ":duration": {"N": str(int(duration_seconds))},
                ":updated_at": {"S": completed_at},
            },
        )

        logger.info(f"Run {run_id} marked as {final_status}")

        # Publish metrics to CloudWatch
        try:
            cloudwatch = get_cloudwatch_client()
            metrics = [
                {
                    "MetricName": "RunCompleted",
                    "Dimensions": [{"Name": "ConfigName", "Value": config_name}],
                    "Value": 1,
                    "Unit": "Count",
                },
                {
                    "MetricName": "TasksCompleted",
                    "Dimensions": [{"Name": "ConfigName", "Value": config_name}],
                    "Value": aggregation.get("completed_tasks", 0),
                    "Unit": "Count",
                },
                {
                    "MetricName": "TasksFailed",
                    "Dimensions": [{"Name": "ConfigName", "Value": config_name}],
                    "Value": aggregation.get("failed_tasks", 0),
                    "Unit": "Count",
                },
                {
                    "MetricName": "Accuracy",
                    "Dimensions": [{"Name": "ConfigName", "Value": config_name}],
                    "Value": aggregation.get("accuracy", 0) * 100,  # Percentage
                    "Unit": "Percent",
                },
                {
                    "MetricName": "TotalCostUSD",
                    "Dimensions": [{"Name": "ConfigName", "Value": config_name}],
                    "Value": aggregation.get("total_cost_usd", 0),
                    "Unit": "None",
                },
                {
                    "MetricName": "DurationSeconds",
                    "Dimensions": [{"Name": "ConfigName", "Value": config_name}],
                    "Value": duration_seconds,
                    "Unit": "Seconds",
                },
            ]

            cloudwatch.put_metric_data(Namespace=CLOUDWATCH_NAMESPACE, MetricData=metrics)
            logger.info(f"Published {len(metrics)} metrics to CloudWatch")

        except ClientError as e:
            # Log but don't fail if CloudWatch publish fails
            logger.warning(f"Failed to publish CloudWatch metrics: {e}")

        final_metrics = {
            "total_tasks": aggregation.get("total_tasks", 0),
            "completed_tasks": aggregation.get("completed_tasks", 0),
            "failed_tasks": aggregation.get("failed_tasks", 0),
            "accuracy": aggregation.get("accuracy", 0),
            "total_cost_usd": aggregation.get("total_cost_usd", 0),
            "duration_seconds": int(duration_seconds),
        }

        return {
            "run_id": run_id,
            "status": final_status,
            "completed_at": completed_at,
            "final_metrics": final_metrics,
        }

    except ClientError as e:
        logger.error(f"DynamoDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
