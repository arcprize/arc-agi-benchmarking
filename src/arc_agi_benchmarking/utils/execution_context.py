"""Execution context detection for local vs AWS environments."""

import os
import urllib.request
from enum import Enum


class ExecutionEnvironment(Enum):
    """Detected execution environment."""

    LOCAL = "local"
    EC2 = "ec2"
    FARGATE = "fargate"
    BATCH = "batch"


def detect_environment() -> ExecutionEnvironment:
    """Detect the current execution environment.

    Returns:
        ExecutionEnvironment indicating where the code is running.
    """
    # Check for AWS Batch
    if os.environ.get("AWS_BATCH_JOB_ID"):
        # Batch on Fargate vs EC2
        if os.environ.get("AWS_EXECUTION_ENV") == "AWS_ECS_FARGATE":
            return ExecutionEnvironment.FARGATE
        # Try EC2 metadata endpoint
        if _can_reach_ec2_metadata():
            return ExecutionEnvironment.EC2
        return ExecutionEnvironment.FARGATE

    # Check for ECS Fargate (non-Batch)
    if os.environ.get("AWS_EXECUTION_ENV") == "AWS_ECS_FARGATE":
        return ExecutionEnvironment.FARGATE

    # Check for EC2 (non-Batch)
    if _can_reach_ec2_metadata():
        return ExecutionEnvironment.EC2

    return ExecutionEnvironment.LOCAL


def _can_reach_ec2_metadata() -> bool:
    """Check if EC2 instance metadata endpoint is reachable.

    Handles both IMDSv1 (direct access) and IMDSv2 (token required).
    For IMDSv2, the endpoint returns 401 Unauthorized without a token,
    but a 401 response still indicates we're running on EC2.
    """
    try:
        urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/",
            timeout=0.5,
        )
        return True
    except urllib.error.HTTPError as e:
        # 401 means IMDSv2 is required but we're still on EC2
        # Any response from the metadata endpoint means we're on EC2
        return e.code in (401, 403)
    except (urllib.error.URLError, OSError):
        return False


def is_aws_environment() -> bool:
    """Check if running in any AWS environment."""
    return detect_environment() != ExecutionEnvironment.LOCAL


def is_local_environment() -> bool:
    """Check if running locally."""
    return detect_environment() == ExecutionEnvironment.LOCAL
