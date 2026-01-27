"""Tests for AWS integration components."""

import os
import urllib.error
from decimal import Decimal
from unittest.mock import patch

import pytest

from arc_agi_benchmarking.utils.execution_context import (
    ExecutionEnvironment,
    _can_reach_ec2_metadata,
    detect_environment,
    is_aws_environment,
    is_local_environment,
)


class TestExecutionContext:
    """Tests for execution environment detection."""

    def test_local_environment_default(self):
        """Without AWS env vars, should detect local."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "arc_agi_benchmarking.utils.execution_context._can_reach_ec2_metadata",
                return_value=False,
            ):
                assert detect_environment() == ExecutionEnvironment.LOCAL
                assert is_local_environment()
                assert not is_aws_environment()

    def test_batch_fargate_environment(self):
        """With Batch + Fargate env vars, should detect Fargate."""
        env = {
            "AWS_BATCH_JOB_ID": "job-123",
            "AWS_EXECUTION_ENV": "AWS_ECS_FARGATE",
        }
        with patch.dict(os.environ, env, clear=True):
            assert detect_environment() == ExecutionEnvironment.FARGATE
            assert is_aws_environment()
            assert not is_local_environment()

    def test_batch_ec2_environment(self):
        """With Batch on EC2, should detect EC2."""
        env = {"AWS_BATCH_JOB_ID": "job-123"}
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "arc_agi_benchmarking.utils.execution_context._can_reach_ec2_metadata",
                return_value=True,
            ):
                assert detect_environment() == ExecutionEnvironment.EC2
                assert is_aws_environment()

    def test_ecs_fargate_environment(self):
        """ECS Fargate without Batch should detect Fargate."""
        env = {"AWS_EXECUTION_ENV": "AWS_ECS_FARGATE"}
        with patch.dict(os.environ, env, clear=True):
            assert detect_environment() == ExecutionEnvironment.FARGATE

    def test_ec2_environment(self):
        """EC2 instance should be detected via metadata endpoint."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "arc_agi_benchmarking.utils.execution_context._can_reach_ec2_metadata",
                return_value=True,
            ):
                assert detect_environment() == ExecutionEnvironment.EC2

    def test_imdsv2_401_response_detected_as_ec2(self):
        """IMDSv2 returns 401 without token, but still indicates EC2."""
        mock_error = urllib.error.HTTPError(
            url="http://169.254.169.254/latest/meta-data/",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=mock_error):
            assert _can_reach_ec2_metadata() is True

    def test_imdsv2_403_response_detected_as_ec2(self):
        """IMDSv2 may return 403 for some paths, still indicates EC2."""
        mock_error = urllib.error.HTTPError(
            url="http://169.254.169.254/latest/meta-data/",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=mock_error):
            assert _can_reach_ec2_metadata() is True

    def test_404_response_not_ec2(self):
        """A 404 response indicates this is not an EC2 metadata endpoint."""
        mock_error = urllib.error.HTTPError(
            url="http://169.254.169.254/latest/meta-data/",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=mock_error):
            assert _can_reach_ec2_metadata() is False

    def test_connection_refused_not_ec2(self):
        """Connection refused indicates not on EC2."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            assert _can_reach_ec2_metadata() is False


# Tests for DynamoDB components require moto or real AWS credentials
# Skip if boto3/moto not available

try:
    import boto3
    from moto import mock_aws

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore

    # No-op decorator when moto isn't installed
    def mock_aws():  # type: ignore
        def decorator(func):
            return func
        return decorator


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3/moto not installed")
class TestDynamoDBProgressManager:
    """Tests for DynamoDB progress manager."""

    def _create_tables(self):
        """Helper to create mock DynamoDB tables."""
        dynamodb = boto3.resource("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="test_runs",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName="test_tasks",
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

    @mock_aws
    def test_create_run(self):
        """Test creating a new benchmark run."""
        from arc_agi_benchmarking.checkpoint.dynamodb_progress import (
            DynamoDBProgressManager,
        )

        self._create_tables()

        manager = DynamoDBProgressManager(
            runs_table_name="test_runs",
            tasks_table_name="test_tasks",
            region_name="us-west-2",
        )

        run_id = manager.create_run(
            config_name="test-config",
            task_ids=["task_1", "task_2", "task_3"],
            data_source="evaluation",
        )

        assert run_id is not None

        run = manager.get_run(run_id)
        assert run["config_name"] == "test-config"
        assert run["total_tasks"] == 3
        assert run["status"] == "IN_PROGRESS"  # UPPERCASE per PRD
        # Verify aggregate fields are initialized
        assert run["total_cost_usd"] == Decimal("0")
        assert run["total_input_tokens"] == 0
        assert run["total_output_tokens"] == 0

    @mock_aws
    def test_claim_task(self):
        """Test claiming a task."""
        from arc_agi_benchmarking.checkpoint.dynamodb_progress import (
            DynamoDBProgressManager,
        )

        self._create_tables()

        manager = DynamoDBProgressManager(
            runs_table_name="test_runs",
            tasks_table_name="test_tasks",
            region_name="us-west-2",
        )

        run_id = manager.create_run(
            config_name="test-config",
            task_ids=["task_1"],
            data_source="evaluation",
        )

        # First claim should succeed
        assert manager.claim_task(run_id, "task_1") is True

        # Second claim should fail (already claimed)
        assert manager.claim_task(run_id, "task_1") is False

    @mock_aws
    def test_mark_completed_rolls_up_aggregates(self):
        """Test that mark_completed rolls up cost and token totals to run."""
        from arc_agi_benchmarking.checkpoint.dynamodb_progress import (
            DynamoDBProgressManager,
        )

        self._create_tables()

        manager = DynamoDBProgressManager(
            runs_table_name="test_runs",
            tasks_table_name="test_tasks",
            region_name="us-west-2",
        )

        run_id = manager.create_run(
            config_name="test-config",
            task_ids=["task_1", "task_2"],
            data_source="evaluation",
        )

        # Complete first task with cost/tokens
        manager.claim_task(run_id, "task_1")
        manager.mark_completed(
            run_id,
            "task_1",
            cost_usd=Decimal("0.50"),
            tokens_input=1000,
            tokens_output=500,
        )

        # Verify run aggregates updated
        run = manager.get_run(run_id)
        assert run["completed_tasks"] == 1
        assert run["total_cost_usd"] == Decimal("0.50")
        assert run["total_input_tokens"] == 1000
        assert run["total_output_tokens"] == 500

        # Complete second task
        manager.claim_task(run_id, "task_2")
        manager.mark_completed(
            run_id,
            "task_2",
            cost_usd=Decimal("0.75"),
            tokens_input=2000,
            tokens_output=1000,
        )

        # Verify aggregates are cumulative
        run = manager.get_run(run_id)
        assert run["completed_tasks"] == 2
        assert run["total_cost_usd"] == Decimal("1.25")
        assert run["total_input_tokens"] == 3000
        assert run["total_output_tokens"] == 1500

        # Verify summary includes aggregates
        summary = manager.get_run_summary(run_id)
        assert summary["total_cost_usd"] == 1.25
        assert summary["total_input_tokens"] == 3000
        assert summary["total_output_tokens"] == 1500

    @mock_aws
    def test_task_status_uses_uppercase(self):
        """Test that task statuses use UPPERCASE per PRD DynamoDB examples."""
        from arc_agi_benchmarking.checkpoint.dynamodb_progress import (
            DynamoDBProgressManager,
            DynamoDBTaskStatus,
        )

        self._create_tables()

        manager = DynamoDBProgressManager(
            runs_table_name="test_runs",
            tasks_table_name="test_tasks",
            region_name="us-west-2",
        )

        run_id = manager.create_run(
            config_name="test-config",
            task_ids=["task_1"],
            data_source="evaluation",
        )

        # Verify status constants are uppercase
        assert DynamoDBTaskStatus.PENDING == "PENDING"
        assert DynamoDBTaskStatus.IN_PROGRESS == "IN_PROGRESS"
        assert DynamoDBTaskStatus.COMPLETED == "COMPLETED"
        assert DynamoDBTaskStatus.FAILED == "FAILED"

        # Verify task starts with PENDING
        tasks_table = boto3.resource("dynamodb", region_name="us-west-2").Table(
            "test_tasks"
        )
        task = tasks_table.get_item(Key={"run_id": run_id, "task_id": "task_1"})["Item"]
        assert task["status"] == "PENDING"

    @mock_aws
    def test_exhausted_retries_not_requeued(self):
        """Test that tasks with exhausted retries are NOT returned by get_pending_tasks."""
        from arc_agi_benchmarking.checkpoint.dynamodb_progress import (
            DynamoDBProgressManager,
            DynamoDBTaskStatus,
        )

        self._create_tables()

        manager = DynamoDBProgressManager(
            runs_table_name="test_runs",
            tasks_table_name="test_tasks",
            region_name="us-west-2",
        )

        run_id = manager.create_run(
            config_name="test-config",
            task_ids=["task_1", "task_2"],
            data_source="evaluation",
        )

        # Both tasks should be pending initially
        pending = manager.get_pending_tasks(run_id)
        assert set(pending) == {"task_1", "task_2"}

        # Fail task_1 with max_retries=2, fail it twice
        manager.claim_task(run_id, "task_1")
        assert manager.mark_failed(run_id, "task_1", "error1", max_retries=2) is False

        # task_1 should still be retryable (1 retry < 2 max)
        pending = manager.get_pending_tasks(run_id)
        assert "task_1" in pending

        # Fail again - now retries should be exhausted
        manager.claim_task(run_id, "task_1")
        assert manager.mark_failed(run_id, "task_1", "error2", max_retries=2) is True

        # task_1 should NOT be in pending anymore (exhausted)
        pending = manager.get_pending_tasks(run_id)
        assert "task_1" not in pending
        assert "task_2" in pending  # task_2 still pending

        # Verify task_1 has FAILED_PERMANENT status
        tasks_table = boto3.resource("dynamodb", region_name="us-west-2").Table(
            "test_tasks"
        )
        task = tasks_table.get_item(Key={"run_id": run_id, "task_id": "task_1"})["Item"]
        assert task["status"] == DynamoDBTaskStatus.FAILED_PERMANENT

        # Verify run's failed_tasks count was incremented
        run = manager.get_run(run_id)
        assert run["failed_tasks"] == 1

    @mock_aws
    def test_failed_permanent_status_constant(self):
        """Test that FAILED_PERMANENT status constant exists."""
        from arc_agi_benchmarking.checkpoint.dynamodb_progress import DynamoDBTaskStatus

        assert DynamoDBTaskStatus.FAILED_PERMANENT == "FAILED_PERMANENT"


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3/moto not installed")
class TestDistributedRateLimiter:
    """Tests for distributed rate limiter."""

    def _create_rate_limit_table(self):
        """Helper to create mock rate limits table."""
        dynamodb = boto3.resource("dynamodb", region_name="us-west-2")
        dynamodb.create_table(
            TableName="test_rate_limits",
            KeySchema=[{"AttributeName": "provider", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "provider", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

    @pytest.mark.asyncio
    async def test_rate_period_interface(self):
        """Test that limiter uses rate/period interface matching provider_config.yml."""
        from arc_agi_benchmarking.resilience.distributed_rate_limiter import (
            DistributedRateLimiter,
        )

        with mock_aws():
            self._create_rate_limit_table()

            # Matches provider_config.yml style: 20 requests per 60 seconds
            limiter = DistributedRateLimiter(
                provider="test-provider",
                rate=20.0,
                period=60.0,
                table_name="test_rate_limits",
                region_name="us-west-2",
            )

            # Verify properties
            assert limiter.rate == 20.0
            assert limiter.period == 60.0

            # Should succeed (bucket starts with rate tokens)
            result = await limiter.acquire(5, timeout=1.0)
            assert result is True

            # Check available tokens
            available = await limiter.get_available_tokens()
            assert available < 20.0

    @pytest.mark.asyncio
    async def test_acquire_tokens(self):
        """Test acquiring tokens from distributed limiter."""
        from arc_agi_benchmarking.resilience.distributed_rate_limiter import (
            DistributedRateLimiter,
        )

        with mock_aws():
            self._create_rate_limit_table()

            # 10 requests per 10 seconds = 1 request per second refill
            limiter = DistributedRateLimiter(
                provider="test-provider",
                rate=10.0,
                period=10.0,
                table_name="test_rate_limits",
                region_name="us-west-2",
            )

            # Should succeed (bucket starts full with rate tokens)
            result = await limiter.acquire(5, timeout=1.0)
            assert result is True

            # Check available tokens (should be ~5 left)
            available = await limiter.get_available_tokens()
            assert available < 10.0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using limiter as context manager."""
        from arc_agi_benchmarking.resilience.distributed_rate_limiter import (
            DistributedRateLimiter,
        )

        with mock_aws():
            self._create_rate_limit_table()

            limiter = DistributedRateLimiter(
                provider="test-provider",
                rate=10.0,
                period=10.0,
                table_name="test_rate_limits",
                region_name="us-west-2",
            )

            async with limiter:
                pass  # Should acquire 1 token

            available = await limiter.get_available_tokens()
            assert available < 10.0

    def test_invalid_rate_raises_error(self):
        """Test that invalid rate raises ValueError."""
        from arc_agi_benchmarking.resilience.distributed_rate_limiter import (
            DistributedRateLimiter,
        )

        with mock_aws():
            self._create_rate_limit_table()

            with pytest.raises(ValueError, match="Rate must be positive"):
                DistributedRateLimiter(
                    provider="test",
                    rate=0,
                    period=60.0,
                    table_name="test_rate_limits",
                    region_name="us-west-2",
                )

    def test_invalid_period_raises_error(self):
        """Test that invalid period raises ValueError."""
        from arc_agi_benchmarking.resilience.distributed_rate_limiter import (
            DistributedRateLimiter,
        )

        with mock_aws():
            self._create_rate_limit_table()

            with pytest.raises(ValueError, match="Period must be positive"):
                DistributedRateLimiter(
                    provider="test",
                    rate=10.0,
                    period=0,
                    table_name="test_rate_limits",
                    region_name="us-west-2",
                )
