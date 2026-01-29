# AWS Infrastructure for ARC-AGI Benchmarking

This directory contains AWS infrastructure configuration for running ARC benchmarks at scale using AWS Batch.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Batch                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Job Queue: arc-benchmark-queue                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Job Definition: arc-benchmark-task                 │  │  │
│  │  │  - Container: arc-agi-benchmarking:latest           │  │  │
│  │  │  - Entry: python -m arc_agi_benchmarking.batch_worker│ │  │
│  │  │  - Env: RUN_ID, TASK_ID, CONFIG_NAME, S3_BUCKET     │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    DynamoDB     │  │       S3        │  │ Secrets Manager │
│                 │  │                 │  │                 │
│ arc_benchmark_  │  │ arc-benchmarks/ │  │ arc-benchmark/  │
│   runs          │  │ ├── tasks/      │  │   api-keys      │
│                 │  │ ├── runs/       │  │                 │
│ arc_task_       │  │ │   ├── sub/    │  │ - OPENAI_API_KEY│
│   progress      │  │ │   └── check/  │  │ - ANTHROPIC_... │
│                 │  │ └── results/    │  │ - GOOGLE_...    │
│ arc_rate_limits │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Files

- **batch-job-definition.json**: AWS Batch job definition for benchmark tasks
- **iam-policies.json**: IAM roles and policies for Batch jobs
- **dynamodb-tables.json**: DynamoDB table schemas

## Setup Instructions

### 1. Create DynamoDB Tables

```bash
# Create benchmark runs table
aws dynamodb create-table \
  --table-name arc_benchmark_runs \
  --attribute-definitions \
    AttributeName=run_id,AttributeType=S \
    AttributeName=config_name,AttributeType=S \
    AttributeName=created_at,AttributeType=S \
  --key-schema AttributeName=run_id,KeyType=HASH \
  --global-secondary-indexes \
    '[{"IndexName":"config-created-index","KeySchema":[{"AttributeName":"config_name","KeyType":"HASH"},{"AttributeName":"created_at","KeyType":"RANGE"}],"Projection":{"ProjectionType":"ALL"}}]' \
  --billing-mode PAY_PER_REQUEST

# Create task progress table
aws dynamodb create-table \
  --table-name arc_task_progress \
  --attribute-definitions \
    AttributeName=run_id,AttributeType=S \
    AttributeName=task_id,AttributeType=S \
  --key-schema \
    AttributeName=run_id,KeyType=HASH \
    AttributeName=task_id,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST

# Create rate limits table
aws dynamodb create-table \
  --table-name arc_rate_limits \
  --attribute-definitions AttributeName=provider,AttributeType=S \
  --key-schema AttributeName=provider,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### 2. Create S3 Bucket

```bash
aws s3 mb s3://arc-benchmarks --region us-west-2

# Upload task data
aws s3 sync data/v2/tasks/ s3://arc-benchmarks/tasks/
```

### 3. Create Secrets

```bash
aws secretsmanager create-secret \
  --name arc-benchmark/api-keys \
  --secret-string '{
    "OPENAI_API_KEY": "sk-...",
    "ANTHROPIC_API_KEY": "sk-ant-...",
    "GOOGLE_API_KEY": "...",
    "DEEPSEEK_API_KEY": "...",
    "XAI_API_KEY": "...",
    "FIREWORKS_API_KEY": "...",
    "OPENROUTER_API_KEY": "..."
  }'
```

### 4. Create IAM Roles

See `iam-policies.json` for the role definitions. Create roles using the AWS Console or CLI.

### 5. Create ECR Repository and Push Image

```bash
# Create repository
aws ecr create-repository --repository-name arc-agi-benchmarking

# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com

# Build and push image
docker build -t arc-agi-benchmarking:latest .
docker tag arc-agi-benchmarking:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/arc-agi-benchmarking:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/arc-agi-benchmarking:latest
```

### 6. Create Batch Resources

```bash
# Create compute environment (Fargate)
aws batch create-compute-environment \
  --compute-environment-name arc-benchmark-fargate \
  --type MANAGED \
  --state ENABLED \
  --compute-resources type=FARGATE,maxvCpus=256,subnets=subnet-xxx,securityGroupIds=sg-xxx

# Create job queue
aws batch create-job-queue \
  --job-queue-name arc-benchmark-queue \
  --state ENABLED \
  --priority 1 \
  --compute-environment-order order=1,computeEnvironment=arc-benchmark-fargate

# Register job definition
aws batch register-job-definition --cli-input-json file://batch-job-definition.json
```

## Local Testing with LocalStack

You can test the batch worker locally without deploying to AWS using [LocalStack](https://localstack.cloud/).

### Prerequisites

```bash
pip install localstack awscli-local boto3
```

### Start LocalStack

```bash
# In a separate terminal
localstack start
```

### Run the Integration Test

```bash
python scripts/run_batch_worker_integration.py
```

This script will:
1. Create DynamoDB tables in LocalStack
2. Create an S3 bucket and upload a sample task
3. Create a task record for the worker to claim
4. Run the batch worker against LocalStack
5. Verify the results in DynamoDB and S3

### Manual Testing with LocalStack

```bash
# Set environment variables
export RUN_ID="test-run-001"
export TASK_ID="66e6c45b"
export CONFIG_NAME="random-baseline"
export S3_BUCKET="arc-benchmark-test"
export AWS_REGION="us-west-2"
export DYNAMODB_RUNS_TABLE="arc_benchmark_runs"
export DYNAMODB_TASKS_TABLE="arc_benchmark_tasks"
export DYNAMODB_RATE_LIMIT_TABLE="arc_rate_limits"
export AWS_ENDPOINT_URL="http://localhost:4566"
export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"

# Run the batch worker
python -m arc_agi_benchmarking.batch_worker
```

## Running a Benchmark

### Submit Individual Task

```bash
aws batch submit-job \
  --job-name "arc-gpt4o-00576224" \
  --job-queue arc-benchmark-queue \
  --job-definition arc-benchmark-task \
  --container-overrides '{
    "environment": [
      {"name": "RUN_ID", "value": "run-2024-01-15-001"},
      {"name": "TASK_ID", "value": "00576224"},
      {"name": "CONFIG_NAME", "value": "gpt-4o"}
    ]
  }'
```

### Monitor Job

```bash
# Check job status
aws batch describe-jobs --jobs <job-id>

# View logs
aws logs get-log-events \
  --log-group-name /aws/batch/arc-benchmark \
  --log-stream-name task/<job-id>
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUN_ID` | Yes | - | Benchmark run identifier |
| `TASK_ID` | Yes | - | Task to execute |
| `CONFIG_NAME` | Yes | - | Model config from models.yml |
| `S3_BUCKET` | Yes | - | S3 bucket for data/results |
| `S3_PREFIX` | No | `runs/{RUN_ID}` | S3 prefix for run data |
| `S3_TASKS_PREFIX` | No | `tasks` | S3 prefix for task data |
| `AWS_REGION` | No | `us-west-2` | AWS region |
| `DYNAMODB_RUNS_TABLE` | No | `arc_benchmark_runs` | DynamoDB runs table |
| `DYNAMODB_TASKS_TABLE` | No | `arc_task_progress` | DynamoDB tasks table |
| `DYNAMODB_RATE_LIMIT_TABLE` | No | `arc_rate_limits` | DynamoDB rate limit table |
| `MAX_RETRIES` | No | `3` | Max retries for failed tasks |
| `REQUEST_TIMEOUT` | No | `300` | API request timeout (seconds) |

## Cost Estimation

Approximate costs per 1000 tasks (us-west-2):

| Resource | Usage | Cost |
|----------|-------|------|
| Batch (Fargate) | 1 vCPU, 2GB, ~5min/task | ~$4.00 |
| DynamoDB | ~3000 writes | ~$0.004 |
| S3 | ~1000 PUTs, ~2000 GETs | ~$0.01 |
| CloudWatch | ~1GB logs | ~$0.50 |
| **Total (infra)** | | **~$4.50** |

Note: API costs (OpenAI, Anthropic, etc.) are additional and depend on the model used.
