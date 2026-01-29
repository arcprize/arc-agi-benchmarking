# GitHub Actions Workflows

## Workflows

### `python-tests.yml`
Runs pytest on all branches and PRs. Tests against Python 3.10, 3.11, and 3.12.

### `benchmark.yml`
Triggers AWS Step Functions benchmark executions.

**Triggers:**
- **Automatic**: Push to `main` that modifies `models.yml`
- **Manual**: Via workflow_dispatch with config name and data source

Note: `provider_config.yml` changes (rate limits, timeouts) do not trigger reruns since they don't affect benchmark results.

## Config Change Detection

The workflow uses `scripts/detect_config_changes.py` to identify which model configs changed:

- Compares `models.yml` between `github.event.before` and `github.sha` (full push range)
- Handles multi-commit pushes correctly
- Fails fast on YAML parse errors (use `--lenient` to skip validation)
- Returns empty list for removed configs (no benchmark to run)

The script is thoroughly tested (`tests/test_detect_config_changes.py`) and can be run locally:

```bash
# Detect changes between commits
python scripts/detect_config_changes.py --base HEAD~1 --head HEAD

# Get all configs (fallback mode)
python scripts/detect_config_changes.py --fallback-all
```

## Required Secrets

The benchmark workflow requires the following repository secrets:

| Secret | Description | Example |
|--------|-------------|---------|
| `AWS_ACCOUNT_ID` | AWS account ID for the sandbox environment | `123456789012` |
| `AWS_BENCHMARK_ROLE_ARN` | ARN of IAM role for GitHub Actions OIDC | `arn:aws:iam::123456789012:role/github-actions-benchmark` |
| `S3_BUCKET` | S3 bucket containing task data | `arc-benchmark-sandbox-teal-task-data` |

## Setting Up AWS OIDC for GitHub Actions

To allow GitHub Actions to assume an AWS role without long-lived credentials:

### 1. Create OIDC Identity Provider in AWS

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### 2. Create IAM Role for GitHub Actions

Create a role with this trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:arcprize/arc-agi-benchmarking:*"
        }
      }
    }
  ]
}
```

### 3. Attach Permissions to the Role

The role needs permissions to:
- Start Step Functions executions
- List S3 objects (to get task list)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "states:StartExecution",
        "states:DescribeExecution"
      ],
      "Resource": "arn:aws:states:*:ACCOUNT_ID:stateMachine:arc-benchmark-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::arc-benchmark-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::arc-benchmark-*/tasks/*"
    }
  ]
}
```

## Manual Benchmark Trigger

To manually trigger a benchmark:

1. Go to **Actions** → **Run Benchmarks**
2. Click **Run workflow**
3. Fill in:
   - **config_name**: Model config from `models.yml` (e.g., `gpt-4o`)
   - **data_source**: `sample`, `evaluation`, or `training`
   - **task_list**: Optional comma-separated task IDs

## Step Functions Execution Naming

Execution names follow this pattern to ensure uniqueness within AWS's 90-day window:

```
bm-{run_id}-{attempt}-{config}
```

- `run_id`: GitHub Actions run ID (unique per workflow run)
- `attempt`: Run attempt number (for retries)
- `config`: Model config name (truncated if needed)

**Truncation strategy**: If the full name exceeds 80 characters (Step Functions limit), the config name is truncated and an 8-character hash is appended:

```
bm-12345678-1-very-long-config-name-that-w-a1b2c3d4
              ^--- truncated config ---^   ^-hash-^
```

This ensures collision-free names even with long config names across multiple runs.

## Monitoring Benchmark Execution

After triggering a benchmark:
1. Check the workflow run summary for the Step Functions execution link
2. Open the link to view progress in the AWS Console
3. Results are stored in DynamoDB and S3

## Cost Considerations

Each benchmark run incurs costs for:
- AWS Batch compute (Fargate vCPU/memory)
- DynamoDB read/write capacity
- S3 storage and requests
- API provider costs (OpenAI, Anthropic, etc.)

The workflow will run for each detected config change, so be mindful of:
- Limiting configs in `models.yml` changes
- Using `sample` data source for testing
- Setting cost limits in the benchmark configuration
