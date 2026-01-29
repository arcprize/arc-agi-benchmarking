# Step Functions Lambda Handlers

This directory contains the Lambda handlers used by AWS Step Functions to orchestrate benchmark runs.

## Handlers

| Handler | Purpose | Input | Output |
|---------|---------|-------|--------|
| `initialize` | Create run record, initialize task records | config_name, task_ids, triggered_by | run_id, task_ids |
| `handle_error` | Handle task failures, manage retries | run_id, task_id, error | status, retry_count |
| `aggregate` | Aggregate results from all tasks | run_id | metrics, accuracy, cost |
| `complete` | Mark run complete, publish CloudWatch metrics | run_id, aggregation | final_status |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUNS_TABLE` | No | `arc_benchmark_runs` | DynamoDB runs table name |
| `TASKS_TABLE` | No | `arc_benchmark_tasks` | DynamoDB tasks table name |
| `S3_BUCKET` | No | - | S3 bucket for storing aggregated results |
| `MAX_RETRIES` | No | `3` | Maximum retry attempts before marking failed |
| `CLOUDWATCH_NAMESPACE` | No | `ArcBenchmark` | CloudWatch metrics namespace |
| `AWS_ENDPOINT_URL` | No | - | Custom endpoint (for LocalStack testing) |

## Deployment

The Terraform module in `infra-live-sandbox-teal` creates Lambda functions with stub code.
To deploy the actual handlers:

### Option 1: Direct Upload

```bash
# Package the handlers
cd src/arc_agi_benchmarking/lambdas
zip -r lambda_package.zip *.py

# Update each Lambda function
aws lambda update-function-code \
  --function-name arc-benchmark-sandbox-teal-initialize \
  --zip-file fileb://lambda_package.zip

aws lambda update-function-code \
  --function-name arc-benchmark-sandbox-teal-handle-error \
  --zip-file fileb://lambda_package.zip

aws lambda update-function-code \
  --function-name arc-benchmark-sandbox-teal-aggregate \
  --zip-file fileb://lambda_package.zip

aws lambda update-function-code \
  --function-name arc-benchmark-sandbox-teal-complete \
  --zip-file fileb://lambda_package.zip
```

### Option 2: Lambda Layer (Recommended)

Build a Lambda layer with the full package:

```bash
# Create layer directory structure
mkdir -p lambda_layer/python
pip install -t lambda_layer/python boto3

# Copy package code
cp -r src/arc_agi_benchmarking lambda_layer/python/

# Create layer zip
cd lambda_layer
zip -r ../lambda_layer.zip python

# Publish layer
aws lambda publish-layer-version \
  --layer-name arc-benchmark-layer \
  --zip-file fileb://lambda_layer.zip \
  --compatible-runtimes python3.11
```

Then update the Terraform module to use the layer.

## Testing

### Unit Tests

```bash
pytest src/arc_agi_benchmarking/tests/test_lambdas.py -v
```

### Local Testing with LocalStack

```bash
# Start LocalStack
localstack start

# Set environment for local testing
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export RUNS_TABLE=arc_benchmark_runs
export TASKS_TABLE=arc_benchmark_tasks

# Test handler directly
python -c "
from arc_agi_benchmarking.lambdas.initialize import handler
result = handler({
    'config_name': 'test',
    'task_ids': ['task1', 'task2'],
}, None)
print(result)
"
```

## State Machine Flow

```
Step Functions Execution
        │
        ▼
┌───────────────┐
│  Initialize   │  ← Creates run record, returns task_ids
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  ProcessTasks │  ← Map state (parallel)
│    (Map)      │
│       │       │
│   ┌───┴───┐   │
│   │ Batch │   │  ← Submit AWS Batch job
│   │  Job  │   │
│   └───┬───┘   │
│       │       │
│   ┌───┴───┐   │
│   │Handle │   │  ← On error, update retry count
│   │ Error │   │
│   └───────┘   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Aggregate   │  ← Calculate metrics, store results
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Complete    │  ← Mark run complete, publish metrics
└───────────────┘
```
