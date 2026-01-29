#!/usr/bin/env python3
"""
AWS Batch worker for executing individual benchmark tasks.

This module provides the entry point for AWS Batch jobs that execute
ARC benchmark tasks. It integrates with:
- S3 for task data and submission storage
- DynamoDB for distributed progress tracking
- Distributed rate limiting across workers

Supports both public and private datasets:
- Public ARC-AGI-2 data: DATA_SOURCE=evaluation or DATA_SOURCE=training
- Private test data: DATA_SOURCE=private

Task data is loaded from s3://{S3_BUCKET}/tasks/{DATA_SOURCE}/{TASK_ID}.json

Environment variables:
    Required:
        RUN_ID: The benchmark run ID (from DynamoDB)
        TASK_ID: The task to execute
        CONFIG_NAME: The model configuration name (from models.yml)
        S3_BUCKET: S3 bucket for data and results

    Optional:
        DATA_SOURCE: Dataset to use - evaluation, training, or private (default: evaluation)
        S3_PREFIX: S3 prefix for this run (default: runs/{RUN_ID})
        S3_TASKS_PREFIX: S3 prefix for task data (default: tasks/{DATA_SOURCE})
        AWS_REGION: AWS region (default: us-west-2)
        DYNAMODB_RUNS_TABLE: DynamoDB runs table (default: arc_benchmark_runs)
        DYNAMODB_TASKS_TABLE: DynamoDB tasks table (default: arc_task_progress)
        DYNAMODB_RATE_LIMIT_TABLE: DynamoDB rate limit table (default: arc_rate_limits)
        MAX_RETRIES: Maximum retries for failed tasks (default: 3)
        REQUEST_TIMEOUT: Timeout for API requests in seconds (default: 300)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BatchWorkerConfig:
    """Configuration loaded from environment variables."""

    def __init__(self):
        # Required
        self.run_id = os.environ.get("RUN_ID")
        self.task_id = os.environ.get("TASK_ID")
        self.config_name = os.environ.get("CONFIG_NAME")
        self.s3_bucket = os.environ.get("S3_BUCKET")

        # Optional with defaults
        self.s3_prefix = os.environ.get("S3_PREFIX", f"runs/{self.run_id}")
        self.data_source = os.environ.get("DATA_SOURCE", "evaluation")
        self.s3_tasks_prefix = os.environ.get(
            "S3_TASKS_PREFIX", f"tasks/{self.data_source}"
        )
        self.aws_region = os.environ.get("AWS_REGION", "us-west-2")

        # DynamoDB tables
        self.dynamodb_runs_table = os.environ.get(
            "DYNAMODB_RUNS_TABLE", "arc_benchmark_runs"
        )
        self.dynamodb_tasks_table = os.environ.get(
            "DYNAMODB_TASKS_TABLE", "arc_task_progress"
        )
        self.dynamodb_rate_limit_table = os.environ.get(
            "DYNAMODB_RATE_LIMIT_TABLE", "arc_rate_limits"
        )

        # Execution settings
        self.max_retries = int(os.environ.get("MAX_RETRIES", "3"))
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "300"))

        # Worker ID (for tracking which worker processed the task)
        self.worker_id = os.environ.get(
            "AWS_BATCH_JOB_ID", f"worker-{os.getpid()}"
        )

    def validate(self) -> list[str]:
        """Validate required configuration. Returns list of missing fields."""
        missing = []
        if not self.run_id:
            missing.append("RUN_ID")
        if not self.task_id:
            missing.append("TASK_ID")
        if not self.config_name:
            missing.append("CONFIG_NAME")
        if not self.s3_bucket:
            missing.append("S3_BUCKET")
        return missing


class GracefulShutdown:
    """Handles graceful shutdown on SIGTERM/SIGINT."""

    def __init__(self):
        self.shutdown_requested = False
        self._original_sigterm = None
        self._original_sigint = None

    def setup(self):
        """Install signal handlers."""
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def restore(self):
        """Restore original signal handlers."""
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)


class BatchWorker:
    """AWS Batch worker for executing benchmark tasks."""

    def __init__(self, config: BatchWorkerConfig):
        self.config = config
        self.shutdown_handler = GracefulShutdown()

        # Lazy-loaded components
        self._storage: Optional[Any] = None
        self._progress: Optional[Any] = None
        self._rate_limiter: Optional[Any] = None
        self._model_config: Optional[Any] = None
        self._provider_config: Optional[dict] = None

    @property
    def storage(self):
        """Lazy-load S3 storage backend."""
        if self._storage is None:
            from arc_agi_benchmarking.storage import S3StorageBackend

            self._storage = S3StorageBackend(
                bucket=self.config.s3_bucket,
                prefix=self.config.s3_prefix,
                region_name=self.config.aws_region,
            )
            logger.info(f"Initialized S3 storage: {self._storage}")
        return self._storage

    @property
    def progress(self):
        """Lazy-load DynamoDB progress manager."""
        if self._progress is None:
            from arc_agi_benchmarking.checkpoint import DynamoDBProgressManager

            self._progress = DynamoDBProgressManager(
                runs_table_name=self.config.dynamodb_runs_table,
                tasks_table_name=self.config.dynamodb_tasks_table,
                region_name=self.config.aws_region,
            )
            logger.info("Initialized DynamoDB progress manager")
        return self._progress

    @property
    def rate_limiter(self):
        """Lazy-load distributed rate limiter."""
        if self._rate_limiter is None:
            from arc_agi_benchmarking.resilience import DistributedRateLimiter

            provider = self.model_config.provider
            provider_config = self._get_provider_config()

            self._rate_limiter = DistributedRateLimiter(
                provider=provider,
                rate=provider_config.get("rate", 20),
                period=provider_config.get("period", 60),
                table_name=self.config.dynamodb_rate_limit_table,
                region_name=self.config.aws_region,
            )
            logger.info(
                f"Initialized distributed rate limiter for {provider} "
                f"({provider_config.get('rate', 20)} req/{provider_config.get('period', 60)}s)"
            )
        return self._rate_limiter

    @property
    def model_config(self):
        """Lazy-load model configuration."""
        if self._model_config is None:
            from arc_agi_benchmarking.utils import read_models_config

            self._model_config = read_models_config(self.config.config_name)
            logger.info(
                f"Loaded model config: {self._model_config.name} "
                f"(provider: {self._model_config.provider})"
            )
        return self._model_config

    def _get_provider_config(self) -> dict:
        """Load provider rate limit configuration."""
        if self._provider_config is None:
            from arc_agi_benchmarking.utils import read_provider_rate_limits

            try:
                all_limits = read_provider_rate_limits()
                provider = self.model_config.provider
                self._provider_config = all_limits.get(
                    provider, {"rate": 20, "period": 60}
                )
            except FileNotFoundError:
                logger.warning(
                    "provider_config.yml not found, using default rate limits"
                )
                self._provider_config = {"rate": 20, "period": 60}
        return self._provider_config

    def _load_task_data(self) -> dict:
        """Load task data from S3."""
        # Task data is stored without run prefix (shared across runs)
        from arc_agi_benchmarking.storage import S3StorageBackend

        tasks_storage = S3StorageBackend(
            bucket=self.config.s3_bucket,
            prefix=self.config.s3_tasks_prefix,
            region_name=self.config.aws_region,
        )

        task_key = f"{self.config.task_id}.json"
        data = tasks_storage.read(task_key)
        if data is None:
            raise FileNotFoundError(
                f"Task data not found: s3://{self.config.s3_bucket}/"
                f"{self.config.s3_tasks_prefix}/{task_key}"
            )
        return json.loads(data.decode("utf-8"))

    async def _execute_task(self, task_data: dict) -> dict:
        """Execute the benchmark task.

        This method mirrors the logic in main.py's ARCTester but is adapted
        for the batch worker context with distributed rate limiting.

        Args:
            task_data: The task data loaded from S3.

        Returns:
            Dictionary with results including submissions and metrics.
        """
        from arc_agi_benchmarking.adapters import (
            AnthropicAdapter,
            OpenAIAdapter,
            DeepseekAdapter,
            GeminiAdapter,
            FireworksAdapter,
            GrokAdapter,
            OpenRouterAdapter,
            XAIAdapter,
        )
        from arc_agi_benchmarking.prompts.prompt_manager import (
            convert_task_pairs_to_prompt,
        )
        from arc_agi_benchmarking.schemas import ARCPair

        PROVIDER_ADAPTERS = {
            "anthropic": AnthropicAdapter,
            "openai": OpenAIAdapter,
            "deepseek": DeepseekAdapter,
            "gemini": GeminiAdapter,
            "fireworks": FireworksAdapter,
            "grok": GrokAdapter,
            "openrouter": OpenRouterAdapter,
            "xai": XAIAdapter,
        }

        # Initialize provider adapter
        adapter_cls = PROVIDER_ADAPTERS.get(self.model_config.provider)
        if adapter_cls is None:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")

        provider = adapter_cls(self.config.config_name)

        # Extract training and test pairs from task data
        train_pairs = [
            ARCPair(input=pair["input"], output=pair["output"])
            for pair in task_data.get("train", [])
        ]
        test_pairs = task_data.get("test", [])

        # Configuration
        num_attempts = self.model_config.kwargs.get("num_attempts", 2)
        retry_attempts = self.model_config.kwargs.get("retry_attempts", 2)

        # Track results
        task_attempts = []
        total_cost = Decimal("0")
        total_input_tokens = 0
        total_output_tokens = 0

        for pair_idx, test_pair in enumerate(test_pairs):
            if self.shutdown_handler.shutdown_requested:
                logger.warning("Shutdown requested, stopping task execution")
                break

            test_input = ARCPair(input=test_pair["input"], output=None)
            expected_output = test_pair.get("output")

            pair_attempts = {}

            for attempt_num in range(1, num_attempts + 1):
                attempt_key = f"attempt_{attempt_num}"
                pair_attempts[attempt_key] = None

                for retry in range(retry_attempts):
                    if self.shutdown_handler.shutdown_requested:
                        break

                    try:
                        # Acquire rate limit token
                        async with self.rate_limiter:
                            logger.debug(
                                f"Executing attempt {attempt_num}, retry {retry + 1} "
                                f"for pair {pair_idx + 1}"
                            )

                            # Generate prompt and get prediction
                            prompt = convert_task_pairs_to_prompt(
                                train_pairs, test_input
                            )

                            # Run prediction in thread pool (providers are sync)
                            attempt_obj = await asyncio.to_thread(
                                provider.make_prediction,
                                prompt,
                                task_id=self.config.task_id,
                                test_id=self.config.config_name,
                                pair_index=pair_idx,
                            )

                            # Parse response if needed
                            if isinstance(attempt_obj.answer, str):
                                parsed = provider.extract_json_from_response(
                                    attempt_obj.answer
                                )
                                if parsed is not None:
                                    attempt_obj.answer = parsed

                            # Check correctness if expected output available
                            if expected_output is not None:
                                attempt_obj.correct = attempt_obj.answer == expected_output

                            # Track costs
                            if attempt_obj.metadata and attempt_obj.metadata.cost:
                                total_cost += Decimal(
                                    str(attempt_obj.metadata.cost.total_cost)
                                )
                            if attempt_obj.metadata and attempt_obj.metadata.usage:
                                total_input_tokens += (
                                    attempt_obj.metadata.usage.prompt_tokens or 0
                                )
                                total_output_tokens += (
                                    attempt_obj.metadata.usage.completion_tokens or 0
                                )

                            pair_attempts[attempt_key] = attempt_obj.model_dump(
                                mode="json"
                            )
                            logger.info(
                                f"Pair {pair_idx + 1}, attempt {attempt_num} succeeded"
                            )
                            break  # Success, exit retry loop

                    except Exception as e:
                        logger.warning(
                            f"Pair {pair_idx + 1}, attempt {attempt_num}, "
                            f"retry {retry + 1} failed: {e}"
                        )
                        if retry == retry_attempts - 1:
                            logger.error(
                                f"All retries exhausted for pair {pair_idx + 1}, "
                                f"attempt {attempt_num}"
                            )

            if any(v is not None for v in pair_attempts.values()):
                task_attempts.append(pair_attempts)

        return {
            "task_id": self.config.task_id,
            "config_name": self.config.config_name,
            "submissions": task_attempts,
            "total_cost_usd": float(total_cost),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "any_correct": any(
                attempt.get("correct", False)
                for pair in task_attempts
                for attempt in pair.values()
                if attempt is not None
            ),
        }

    def _save_submission(self, result: dict) -> str:
        """Save submission to S3.

        Args:
            result: The task result dictionary.

        Returns:
            The S3 key where the submission was saved.
        """
        submission_key = f"submissions/{self.config.task_id}.json"
        submission_data = json.dumps(result["submissions"], indent=2).encode("utf-8")
        self.storage.write(submission_key, submission_data)
        logger.info(f"Saved submission to s3://{self.config.s3_bucket}/{submission_key}")
        return submission_key

    async def run(self) -> int:
        """Execute the batch worker.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        self.shutdown_handler.setup()
        start_time = time.monotonic()

        try:
            logger.info(
                f"Starting batch worker: run={self.config.run_id}, "
                f"task={self.config.task_id}, config={self.config.config_name}"
            )

            # Claim task atomically
            if not self.progress.claim_task(self.config.run_id, self.config.task_id):
                logger.warning(
                    f"Task {self.config.task_id} already claimed by another worker"
                )
                return 0  # Not an error - task is being processed

            # Load task data
            logger.info(f"Loading task data for {self.config.task_id}")
            task_data = self._load_task_data()

            # Execute task
            logger.info(f"Executing task {self.config.task_id}")
            result = await self._execute_task(task_data)

            # Save submission
            submission_key = self._save_submission(result)

            # Mark completed
            self.progress.mark_completed(
                run_id=self.config.run_id,
                task_id=self.config.task_id,
                result_s3_key=submission_key,
                cost_usd=Decimal(str(result["total_cost_usd"])),
                tokens_input=result["total_input_tokens"],
                tokens_output=result["total_output_tokens"],
            )

            elapsed = time.monotonic() - start_time
            logger.info(
                f"Task {self.config.task_id} completed successfully in {elapsed:.2f}s "
                f"(cost=${result['total_cost_usd']:.6f})"
            )
            return 0

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.exception(f"Task {self.config.task_id} failed after {elapsed:.2f}s: {e}")

            # Mark failed (may allow retry)
            retries_exhausted = self.progress.mark_failed(
                run_id=self.config.run_id,
                task_id=self.config.task_id,
                error=str(e),
                max_retries=self.config.max_retries,
            )

            if retries_exhausted:
                logger.error(f"Task {self.config.task_id} failed permanently")
            else:
                logger.warning(f"Task {self.config.task_id} failed, will be retried")

            return 1  # Non-zero exit triggers Batch retry

        finally:
            self.shutdown_handler.restore()


def main() -> int:
    """Main entry point for the batch worker."""
    # Load and validate configuration
    config = BatchWorkerConfig()
    missing = config.validate()

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return 2  # Configuration error

    # Create and run worker
    worker = BatchWorker(config)

    try:
        return asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130  # Standard exit code for SIGINT


if __name__ == "__main__":
    sys.exit(main())
