"""
Tests for Step Functions Lambda handlers.
"""

import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# Check if moto is available for AWS mocking
try:
    from moto import mock_aws

    HAS_MOTO = True
except ImportError:
    HAS_MOTO = False

    def mock_aws(func):
        """Stub decorator when moto is not installed."""
        return pytest.mark.skip(reason="moto not installed")(func)


# Skip all tests if boto3 is not installed
pytest.importorskip("boto3")

import boto3


@pytest.fixture
def aws_credentials():
    """Mock AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"


@pytest.fixture
def dynamodb_tables(aws_credentials):
    """Create DynamoDB tables for testing."""
    if not HAS_MOTO:
        pytest.skip("moto not installed")

    with mock_aws():
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")

        # Create runs table
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        # Create tasks table
        dynamodb.create_table(
            TableName="arc_task_progress",
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "task_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "task_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        yield dynamodb


@pytest.fixture
def s3_bucket(aws_credentials):
    """Create S3 bucket for testing."""
    if not HAS_MOTO:
        pytest.skip("moto not installed")

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket="arc-benchmark-test",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        yield s3


class TestInitializeHandler:
    """Tests for the initialize Lambda handler."""

    @pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
    @mock_aws
    def test_initialize_creates_run_and_tasks(self, aws_credentials):
        """Test that initialize creates run and task records."""
        # Setup
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName="arc_task_progress",
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "task_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "task_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        from arc_agi_benchmarking.lambdas.initialize import handler

        event = {
            "config_name": "gpt-4o",
            "data_source": "sample",
            "task_ids": ["task1", "task2", "task3"],
            "triggered_by": "test",
            "commit_sha": "abc123",
        }

        # Execute
        result = handler(event, None)

        # Verify
        assert "run_id" in result
        assert result["config_name"] == "gpt-4o"
        assert result["total_tasks"] == 3
        assert result["task_ids"] == ["task1", "task2", "task3"]

        # Verify run record created
        run_response = dynamodb.get_item(
            TableName="arc_benchmark_runs",
            Key={"run_id": {"S": result["run_id"]}},
        )
        assert "Item" in run_response
        assert run_response["Item"]["status"]["S"] == "RUNNING"

        # Verify task records created
        tasks_response = dynamodb.query(
            TableName="arc_task_progress",
            KeyConditionExpression="run_id = :run_id",
            ExpressionAttributeValues={":run_id": {"S": result["run_id"]}},
        )
        assert len(tasks_response["Items"]) == 3

    @pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
    @mock_aws
    def test_initialize_handles_large_task_list(self, aws_credentials):
        """Test batch writing for large task lists."""
        # Setup
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName="arc_task_progress",
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "task_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "task_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        from arc_agi_benchmarking.lambdas.initialize import handler

        # Create 50 tasks (requires 2 batches)
        task_ids = [f"task{i:04d}" for i in range(50)]
        event = {
            "config_name": "test-config",
            "task_ids": task_ids,
        }

        # Execute
        result = handler(event, None)

        # Verify all tasks created
        assert result["total_tasks"] == 50
        tasks_response = dynamodb.query(
            TableName="arc_task_progress",
            KeyConditionExpression="run_id = :run_id",
            ExpressionAttributeValues={":run_id": {"S": result["run_id"]}},
        )
        assert len(tasks_response["Items"]) == 50


class TestHandleErrorHandler:
    """Tests for the handle_error Lambda handler."""

    @pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
    @mock_aws
    def test_handle_error_increments_retry(self, aws_credentials):
        """Test that handle_error increments retry count."""
        # Setup
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName="arc_task_progress",
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "task_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "task_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Create run and task
        dynamodb.put_item(
            TableName="arc_benchmark_runs",
            Item={
                "run_id": {"S": "test-run"},
                "status": {"S": "RUNNING"},
                "failed_tasks": {"N": "0"},
            },
        )
        dynamodb.put_item(
            TableName="arc_task_progress",
            Item={
                "run_id": {"S": "test-run"},
                "task_id": {"S": "task1"},
                "status": {"S": "RUNNING"},
                "retry_count": {"N": "0"},
            },
        )

        from arc_agi_benchmarking.lambdas.handle_error import handler

        event = {
            "run_id": "test-run",
            "task_id": "task1",
            "error": {
                "Error": "States.TaskFailed",
                "Cause": "Container exited with code 1",
            },
        }

        # Execute
        result = handler(event, None)

        # Verify retry scheduled (not failed yet)
        assert result["status"] == "RETRY_SCHEDULED"
        assert result["retry_count"] == 1

        # Verify task updated - status is FAILED (retryable) not PENDING
        # Note: Step Functions Map doesn't re-queue, so we mark as FAILED not PENDING
        task_response = dynamodb.get_item(
            TableName="arc_task_progress",
            Key={"run_id": {"S": "test-run"}, "task_id": {"S": "task1"}},
        )
        assert task_response["Item"]["retry_count"]["N"] == "1"
        assert task_response["Item"]["status"]["S"] == "FAILED"

    @pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
    @mock_aws
    def test_handle_error_marks_failed_after_max_retries(self, aws_credentials):
        """Test that task is marked FAILED after max retries."""
        # Setup
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName="arc_task_progress",
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "task_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "task_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Create run and task with 2 retries already
        dynamodb.put_item(
            TableName="arc_benchmark_runs",
            Item={
                "run_id": {"S": "test-run"},
                "status": {"S": "RUNNING"},
                "failed_tasks": {"N": "0"},
            },
        )
        dynamodb.put_item(
            TableName="arc_task_progress",
            Item={
                "run_id": {"S": "test-run"},
                "task_id": {"S": "task1"},
                "status": {"S": "RUNNING"},
                "retry_count": {"N": "2"},  # Already at max-1
            },
        )

        from arc_agi_benchmarking.lambdas.handle_error import handler

        event = {
            "run_id": "test-run",
            "task_id": "task1",
            "error": {"Error": "Timeout", "Cause": "Task timed out"},
        }

        # Execute
        result = handler(event, None)

        # Verify marked as permanently failed
        assert result["status"] == "FAILED_PERMANENT"
        assert result["retry_count"] == 3

        # Verify task status
        task_response = dynamodb.get_item(
            TableName="arc_task_progress",
            Key={"run_id": {"S": "test-run"}, "task_id": {"S": "task1"}},
        )
        assert task_response["Item"]["status"]["S"] == "FAILED_PERMANENT"

        # Verify run's failed_tasks incremented
        run_response = dynamodb.get_item(
            TableName="arc_benchmark_runs",
            Key={"run_id": {"S": "test-run"}},
        )
        assert run_response["Item"]["failed_tasks"]["N"] == "1"


class TestAggregateHandler:
    """Tests for the aggregate Lambda handler."""

    @pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
    @mock_aws
    def test_aggregate_calculates_metrics(self, aws_credentials):
        """Test that aggregate calculates correct metrics."""
        # Setup
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName="arc_task_progress",
            KeySchema=[
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "task_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "task_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Create run
        dynamodb.put_item(
            TableName="arc_benchmark_runs",
            Item={
                "run_id": {"S": "test-run"},
                "config_name": {"S": "gpt-4o"},
                "status": {"S": "RUNNING"},
                "total_tasks": {"N": "4"},
            },
        )

        # Create completed tasks
        tasks = [
            {"task_id": "t1", "status": "COMPLETED", "is_correct": True, "cost": "0.10"},
            {"task_id": "t2", "status": "COMPLETED", "is_correct": True, "cost": "0.15"},
            {"task_id": "t3", "status": "COMPLETED", "is_correct": False, "cost": "0.12"},
            {"task_id": "t4", "status": "FAILED", "is_correct": False, "cost": "0.05"},
        ]
        for task in tasks:
            dynamodb.put_item(
                TableName="arc_task_progress",
                Item={
                    "run_id": {"S": "test-run"},
                    "task_id": {"S": task["task_id"]},
                    "status": {"S": task["status"]},
                    "is_correct": {"BOOL": task["is_correct"]},
                    "cost_usd": {"N": task["cost"]},
                },
            )

        from arc_agi_benchmarking.lambdas.aggregate import handler

        event = {"run_id": "test-run"}

        # Execute
        result = handler(event, None)

        # Verify
        assert result["run_id"] == "test-run"
        assert result["total_tasks"] == 4
        assert result["completed_tasks"] == 3
        assert result["failed_tasks"] == 1
        assert abs(result["total_cost_usd"] - 0.42) < 0.01
        assert abs(result["accuracy"] - (2 / 3)) < 0.01  # 2 correct out of 3 completed


class TestCompleteHandler:
    """Tests for the complete Lambda handler."""

    @pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
    @mock_aws
    def test_complete_marks_run_completed(self, aws_credentials):
        """Test that complete marks run as COMPLETED."""
        # Setup
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        # Create run
        dynamodb.put_item(
            TableName="arc_benchmark_runs",
            Item={
                "run_id": {"S": "test-run"},
                "config_name": {"S": "gpt-4o"},
                "status": {"S": "RUNNING"},
                "created_at": {"S": datetime.now(timezone.utc).isoformat()},
            },
        )

        from arc_agi_benchmarking.lambdas.complete import handler

        event = {
            "run_id": "test-run",
            "aggregation": {
                "total_tasks": 100,
                "completed_tasks": 100,
                "failed_tasks": 0,
                "accuracy": 0.85,
                "total_cost_usd": 12.34,
            },
        }

        # Execute
        result = handler(event, None)

        # Verify
        assert result["run_id"] == "test-run"
        assert result["status"] == "COMPLETED"
        assert "completed_at" in result
        assert result["final_metrics"]["total_tasks"] == 100

        # Verify run record updated
        run_response = dynamodb.get_item(
            TableName="arc_benchmark_runs",
            Key={"run_id": {"S": "test-run"}},
        )
        assert run_response["Item"]["status"]["S"] == "COMPLETED"

    @pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
    @mock_aws
    def test_complete_marks_completed_with_errors(self, aws_credentials):
        """Test that run with failures is marked COMPLETED_WITH_ERRORS."""
        # Setup
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        dynamodb.put_item(
            TableName="arc_benchmark_runs",
            Item={
                "run_id": {"S": "test-run"},
                "config_name": {"S": "gpt-4o"},
                "status": {"S": "RUNNING"},
                "created_at": {"S": datetime.now(timezone.utc).isoformat()},
            },
        )

        from arc_agi_benchmarking.lambdas.complete import handler

        event = {
            "run_id": "test-run",
            "aggregation": {
                "total_tasks": 100,
                "completed_tasks": 95,
                "failed_tasks": 5,  # Some failures
                "accuracy": 0.80,
                "total_cost_usd": 10.00,
            },
        }

        # Execute
        result = handler(event, None)

        # Verify status
        assert result["status"] == "COMPLETED_WITH_ERRORS"

    @pytest.mark.skipif(not HAS_MOTO, reason="moto not installed")
    @mock_aws
    def test_complete_marks_failed_if_all_tasks_failed(self, aws_credentials):
        """Test that run is marked FAILED if all tasks failed."""
        # Setup
        dynamodb = boto3.client("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="arc_benchmark_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        dynamodb.put_item(
            TableName="arc_benchmark_runs",
            Item={
                "run_id": {"S": "test-run"},
                "config_name": {"S": "gpt-4o"},
                "status": {"S": "RUNNING"},
                "created_at": {"S": datetime.now(timezone.utc).isoformat()},
            },
        )

        from arc_agi_benchmarking.lambdas.complete import handler

        event = {
            "run_id": "test-run",
            "aggregation": {
                "total_tasks": 10,
                "completed_tasks": 0,
                "failed_tasks": 10,  # All failed
                "accuracy": 0,
                "total_cost_usd": 0.50,
            },
        }

        # Execute
        result = handler(event, None)

        # Verify status
        assert result["status"] == "FAILED"
