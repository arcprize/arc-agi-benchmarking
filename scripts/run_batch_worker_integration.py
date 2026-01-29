#!/usr/bin/env python3
"""
Local integration test for batch worker using LocalStack.

Prerequisites:
    pip install localstack awscli-local boto3
    localstack start

Usage:
    python scripts/test_batch_worker_local.py
"""

import json
import os
import subprocess
import sys
import time

import boto3
from botocore.config import Config

# LocalStack endpoint
LOCALSTACK_ENDPOINT = "http://localhost:4566"

# Test configuration
TEST_RUN_ID = "test-run-001"
TEST_TASK_ID = "66e6c45b"
TEST_CONFIG = "random-baseline"
BUCKET_NAME = "arc-benchmark-test"
REGION = "us-west-2"

# Table names
RUNS_TABLE = "arc_benchmark_runs"
TASKS_TABLE = "arc_benchmark_tasks"
RATE_LIMIT_TABLE = "arc_rate_limits"


def get_localstack_client(service: str):
    """Create a boto3 client pointing to LocalStack."""
    return boto3.client(
        service,
        endpoint_url=LOCALSTACK_ENDPOINT,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(retries={"max_attempts": 0}),
    )


def setup_dynamodb():
    """Create DynamoDB tables in LocalStack."""
    dynamodb = get_localstack_client("dynamodb")

    tables = [
        {
            "TableName": RUNS_TABLE,
            "KeySchema": [{"AttributeName": "run_id", "KeyType": "HASH"}],
            "AttributeDefinitions": [{"AttributeName": "run_id", "AttributeType": "S"}],
        },
        {
            "TableName": TASKS_TABLE,
            "KeySchema": [
                {"AttributeName": "run_id", "KeyType": "HASH"},
                {"AttributeName": "task_id", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "run_id", "AttributeType": "S"},
                {"AttributeName": "task_id", "AttributeType": "S"},
            ],
        },
        {
            "TableName": RATE_LIMIT_TABLE,
            "KeySchema": [{"AttributeName": "bucket_key", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "bucket_key", "AttributeType": "S"}
            ],
        },
    ]

    for table_config in tables:
        try:
            dynamodb.create_table(
                **table_config,
                BillingMode="PAY_PER_REQUEST",
            )
            print(f"Created table: {table_config['TableName']}")
        except dynamodb.exceptions.ResourceInUseException:
            print(f"Table already exists: {table_config['TableName']}")

    # Wait for tables to be active
    for table_config in tables:
        waiter = dynamodb.get_waiter("table_exists")
        waiter.wait(TableName=table_config["TableName"])


def setup_s3():
    """Create S3 bucket and upload test task."""
    s3 = get_localstack_client("s3")

    # Create bucket
    try:
        s3.create_bucket(
            Bucket=BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        print(f"Created bucket: {BUCKET_NAME}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"Bucket already exists: {BUCKET_NAME}")

    # Upload test task
    task_path = f"data/sample/tasks/{TEST_TASK_ID}.json"
    if os.path.exists(task_path):
        with open(task_path) as f:
            task_data = f.read()
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"tasks/{TEST_TASK_ID}.json",
            Body=task_data,
        )
        print(f"Uploaded task: tasks/{TEST_TASK_ID}.json")
    else:
        print(f"Warning: Task file not found: {task_path}")


def setup_task_record():
    """Create a task record in DynamoDB for the worker to claim."""
    dynamodb = get_localstack_client("dynamodb")

    # Create run record
    dynamodb.put_item(
        TableName=RUNS_TABLE,
        Item={
            "run_id": {"S": TEST_RUN_ID},
            "config_name": {"S": TEST_CONFIG},
            "status": {"S": "running"},
            "created_at": {"S": "2024-01-01T00:00:00Z"},
            "total_tasks": {"N": "1"},
            "completed_tasks": {"N": "0"},
        },
    )
    print(f"Created run record: {TEST_RUN_ID}")

    # Create task record (pending status so worker can claim it)
    dynamodb.put_item(
        TableName=TASKS_TABLE,
        Item={
            "run_id": {"S": TEST_RUN_ID},
            "task_id": {"S": TEST_TASK_ID},
            "status": {"S": "pending"},
            "created_at": {"S": "2024-01-01T00:00:00Z"},
        },
    )
    print(f"Created task record: {TEST_TASK_ID}")


def run_batch_worker():
    """Run the batch worker with LocalStack configuration."""
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": TEST_RUN_ID,
            "TASK_ID": TEST_TASK_ID,
            "CONFIG_NAME": TEST_CONFIG,
            "S3_BUCKET": BUCKET_NAME,
            "AWS_REGION": REGION,
            "DYNAMODB_RUNS_TABLE": RUNS_TABLE,
            "DYNAMODB_TASKS_TABLE": TASKS_TABLE,
            "DYNAMODB_RATE_LIMIT_TABLE": RATE_LIMIT_TABLE,
            # Point to LocalStack
            "AWS_ENDPOINT_URL": LOCALSTACK_ENDPOINT,
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
        }
    )

    print("\n" + "=" * 60)
    print("Running batch worker...")
    print("=" * 60 + "\n")

    result = subprocess.run(
        [sys.executable, "-m", "arc_agi_benchmarking.batch_worker"],
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    return result.returncode


def check_results():
    """Check the results in DynamoDB and S3."""
    dynamodb = get_localstack_client("dynamodb")
    s3 = get_localstack_client("s3")

    print("\n" + "=" * 60)
    print("Checking results...")
    print("=" * 60 + "\n")

    # Check task status
    response = dynamodb.get_item(
        TableName=TASKS_TABLE,
        Key={
            "run_id": {"S": TEST_RUN_ID},
            "task_id": {"S": TEST_TASK_ID},
        },
    )

    if "Item" in response:
        item = response["Item"]
        print(f"Task status: {item.get('status', {}).get('S', 'unknown')}")
        if "result_s3_key" in item:
            print(f"Result S3 key: {item['result_s3_key']['S']}")
        if "error" in item:
            print(f"Error: {item['error']['S']}")

    # List submissions in S3
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix="submissions/",
        )
        if "Contents" in response:
            print("\nSubmissions in S3:")
            for obj in response["Contents"]:
                print(f"  - {obj['Key']}")
    except Exception as e:
        print(f"Could not list S3 objects: {e}")


def main():
    print("=" * 60)
    print("Batch Worker Local Integration Test")
    print("=" * 60)
    print(f"\nLocalStack endpoint: {LOCALSTACK_ENDPOINT}")
    print(f"Run ID: {TEST_RUN_ID}")
    print(f"Task ID: {TEST_TASK_ID}")
    print(f"Config: {TEST_CONFIG}")
    print()

    # Check if LocalStack is running
    try:
        s3 = get_localstack_client("s3")
        s3.list_buckets()
    except Exception as e:
        print(f"Error: Cannot connect to LocalStack at {LOCALSTACK_ENDPOINT}")
        print("Please start LocalStack first: localstack start")
        print(f"Details: {e}")
        sys.exit(1)

    # Setup
    setup_dynamodb()
    setup_s3()
    setup_task_record()

    # Run worker
    exit_code = run_batch_worker()

    # Check results
    check_results()

    print(f"\nBatch worker exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
