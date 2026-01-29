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
| `TASKS_TABLE` | No | `arc_task_progress` | DynamoDB tasks table name |
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
export TASKS_TABLE=arc_task_progress

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

## Retry Semantics

**Important**: The Step Functions Map state does not automatically re-queue failed tasks. The retry flow works as follows:

1. **Batch Job Fails** → Step Functions catches the error
2. **`handle_error` Lambda** → Increments retry count, marks task as:
   - `FAILED` (if retries remaining) — eligible for retry
   - `FAILED_PERMANENT` (if max retries exhausted) — no more retries
3. **Map continues** → Other tasks proceed; failed task is NOT re-queued

### Operational Requirement: Retry Sweeper

To actually retry failed tasks, you need an external mechanism:

**Option A: Scheduled Lambda Sweeper**
```python
# Scan for FAILED tasks, resubmit to Step Functions
tasks = dynamodb.query(
    TableName=TASKS_TABLE,
    IndexName='status-index',
    KeyConditionExpression='status = :s',
    ExpressionAttributeValues={':s': {'S': 'FAILED'}}
)
# Resubmit each task to a new or existing run
```

**Option B: Manual Re-run**
```bash
# Find failed tasks
aws dynamodb query --table-name arc_task_progress \
  --index-name status-index \
  --key-condition-expression "status = :s" \
  --expression-attribute-values '{":s":{"S":"FAILED"}}'

# Trigger new run with failed task IDs
```

**Option C: Accept partial completion**
If retry isn't critical, the `aggregate` Lambda will correctly count `FAILED` tasks in the final metrics.

This design was chosen because:
- Step Functions Map state doesn't support dynamic re-queuing
- Separating retry logic allows flexible retry policies (immediate, exponential backoff, manual)
- Failed tasks are preserved for debugging before retry

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
