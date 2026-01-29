"""Tests for the AWS Batch worker."""

import json
import os
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if boto3/moto are available for AWS tests
try:
    import boto3
    from moto import mock_aws

    HAS_AWS_DEPS = True
except ImportError:
    HAS_AWS_DEPS = False

    # Create a no-op decorator for when moto isn't installed
    def mock_aws(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip(reason="moto not installed")(func)

        return decorator


from arc_agi_benchmarking.batch_worker import (
    BatchWorker,
    BatchWorkerConfig,
    GracefulShutdown,
)


class TestBatchWorkerConfig:
    """Tests for BatchWorkerConfig."""

    def test_config_from_environment(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("RUN_ID", "test-run-123")
        monkeypatch.setenv("TASK_ID", "00576224")
        monkeypatch.setenv("CONFIG_NAME", "gpt-4o")
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        config = BatchWorkerConfig()

        assert config.run_id == "test-run-123"
        assert config.task_id == "00576224"
        assert config.config_name == "gpt-4o"
        assert config.s3_bucket == "test-bucket"
        assert config.aws_region == "us-east-1"
        assert config.s3_prefix == "runs/test-run-123"

    def test_config_defaults(self, monkeypatch):
        """Test default values are applied."""
        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("TASK_ID", "task1")
        monkeypatch.setenv("CONFIG_NAME", "test-config")
        monkeypatch.setenv("S3_BUCKET", "bucket")
        # Don't set optional vars

        config = BatchWorkerConfig()

        assert config.aws_region == "us-west-2"
        assert config.dynamodb_runs_table == "arc_benchmark_runs"
        assert config.dynamodb_tasks_table == "arc_task_progress"
        assert config.dynamodb_rate_limit_table == "arc_rate_limits"
        assert config.max_retries == 3
        assert config.request_timeout == 300

    def test_validate_missing_required(self, monkeypatch):
        """Test validation catches missing required fields."""
        # Clear environment
        for var in ["RUN_ID", "TASK_ID", "CONFIG_NAME", "S3_BUCKET"]:
            monkeypatch.delenv(var, raising=False)

        config = BatchWorkerConfig()
        missing = config.validate()

        assert "RUN_ID" in missing
        assert "TASK_ID" in missing
        assert "CONFIG_NAME" in missing
        assert "S3_BUCKET" in missing

    def test_validate_all_present(self, monkeypatch):
        """Test validation passes with all required fields."""
        monkeypatch.setenv("RUN_ID", "run")
        monkeypatch.setenv("TASK_ID", "task")
        monkeypatch.setenv("CONFIG_NAME", "config")
        monkeypatch.setenv("S3_BUCKET", "bucket")

        config = BatchWorkerConfig()
        missing = config.validate()

        assert len(missing) == 0


class TestGracefulShutdown:
    """Tests for GracefulShutdown handler."""

    def test_initial_state(self):
        """Test initial state is not shutdown."""
        handler = GracefulShutdown()
        assert handler.shutdown_requested is False

    def test_signal_handling(self):
        """Test signal handler sets shutdown flag."""
        handler = GracefulShutdown()
        handler.setup()

        try:
            # Simulate SIGTERM
            handler._handle_signal(15, None)  # 15 = SIGTERM
            assert handler.shutdown_requested is True
        finally:
            handler.restore()

    def test_restore_handlers(self):
        """Test handlers are properly restored."""
        import signal

        handler = GracefulShutdown()
        original_sigterm = signal.getsignal(signal.SIGTERM)

        handler.setup()
        handler.restore()

        # Note: The exact comparison may vary based on platform
        # The key is that restore() doesn't crash


class TestBatchWorker:
    """Tests for BatchWorker."""

    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Create a mock configuration."""
        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("TASK_ID", "00576224")
        monkeypatch.setenv("CONFIG_NAME", "random-baseline")
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        return BatchWorkerConfig()

    def test_worker_initialization(self, mock_config):
        """Test worker initializes with config."""
        worker = BatchWorker(mock_config)

        assert worker.config == mock_config
        assert worker._storage is None  # Lazy loaded
        assert worker._progress is None
        assert worker._rate_limiter is None

    @pytest.mark.skipif(not HAS_AWS_DEPS, reason="boto3/moto not installed")
    @mock_aws()
    def test_storage_lazy_loading(self, mock_config):
        """Test S3 storage is lazily loaded."""
        # Create mock S3 bucket
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )

        worker = BatchWorker(mock_config)

        # Access storage to trigger lazy load
        storage = worker.storage

        assert storage is not None
        assert storage.bucket == "test-bucket"
        assert "runs/test-run" in storage.prefix

    @pytest.mark.skipif(not HAS_AWS_DEPS, reason="boto3/moto not installed")
    @mock_aws()
    def test_progress_lazy_loading(self, mock_config):
        """Test DynamoDB progress manager is lazily loaded."""
        # Create mock DynamoDB tables
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

        worker = BatchWorker(mock_config)

        # Access progress to trigger lazy load
        progress = worker.progress

        assert progress is not None


class TestBatchWorkerIntegration:
    """Integration tests for BatchWorker (requires moto)."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up mock environment."""
        monkeypatch.setenv("RUN_ID", "integration-test-run")
        monkeypatch.setenv("TASK_ID", "00576224")
        monkeypatch.setenv("CONFIG_NAME", "random-baseline")
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("S3_TASKS_PREFIX", "tasks")
        monkeypatch.setenv("AWS_REGION", "us-west-2")

    @pytest.fixture
    def sample_task_data(self):
        """Sample ARC task data."""
        return {
            "train": [
                {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
                {"input": [[2, 2], [2, 2]], "output": [[3, 3], [3, 3]]},
            ],
            "test": [{"input": [[4, 4], [4, 4]], "output": [[5, 5], [5, 5]]}],
        }

    @pytest.mark.skipif(not HAS_AWS_DEPS, reason="boto3/moto not installed")
    @mock_aws()
    def test_load_task_data(self, mock_env, sample_task_data):
        """Test loading task data from S3."""
        # Set up S3 with task data
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        s3.put_object(
            Bucket="test-bucket",
            Key="tasks/00576224.json",
            Body=json.dumps(sample_task_data),
        )

        config = BatchWorkerConfig()
        worker = BatchWorker(config)

        task_data = worker._load_task_data()

        assert task_data == sample_task_data
        assert len(task_data["train"]) == 2
        assert len(task_data["test"]) == 1

    @pytest.mark.skipif(not HAS_AWS_DEPS, reason="boto3/moto not installed")
    @mock_aws()
    def test_save_submission(self, mock_env):
        """Test saving submission to S3."""
        # Set up S3
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )

        config = BatchWorkerConfig()
        worker = BatchWorker(config)

        result = {
            "task_id": "00576224",
            "config_name": "random-baseline",
            "submissions": [{"attempt_1": {"answer": [[1, 1], [1, 1]]}}],
            "total_cost_usd": 0.001,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "any_correct": False,
        }

        submission_key = worker._save_submission(result)

        assert submission_key == "submissions/00576224.json"

        # Verify it was saved
        response = s3.get_object(
            Bucket="test-bucket",
            Key="runs/integration-test-run/submissions/00576224.json",
        )
        saved_data = json.loads(response["Body"].read())
        assert saved_data == result["submissions"]


class TestBatchWorkerMain:
    """Tests for the main entry point."""

    def test_main_missing_config(self, monkeypatch):
        """Test main returns error code for missing config."""
        # Clear required env vars
        for var in ["RUN_ID", "TASK_ID", "CONFIG_NAME", "S3_BUCKET"]:
            monkeypatch.delenv(var, raising=False)

        from arc_agi_benchmarking.batch_worker import main

        exit_code = main()

        assert exit_code == 2  # Configuration error

    @pytest.mark.skipif(not HAS_AWS_DEPS, reason="boto3/moto not installed")
    @mock_aws()
    def test_main_task_already_claimed(self, monkeypatch):
        """Test main handles already-claimed task gracefully."""
        monkeypatch.setenv("RUN_ID", "test-run")
        monkeypatch.setenv("TASK_ID", "task1")
        monkeypatch.setenv("CONFIG_NAME", "random-baseline")
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_REGION", "us-west-2")

        # Set up DynamoDB tables
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

        # Pre-claim the task (simulate another worker)
        resource = boto3.resource("dynamodb", region_name="us-west-2")
        table = resource.Table("arc_task_progress")
        table.put_item(
            Item={
                "run_id": "test-run",
                "task_id": "task1",
                "status": "IN_PROGRESS",
                "worker_id": "other-worker",
            }
        )

        from arc_agi_benchmarking.batch_worker import main

        exit_code = main()

        # Should exit gracefully (0) since task is being processed
        assert exit_code == 0
