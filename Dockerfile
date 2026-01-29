# Dockerfile for ARC-AGI Benchmarking AWS Batch Worker
#
# This container is designed to run as an AWS Batch job, executing
# individual benchmark tasks with distributed coordination via
# DynamoDB and S3.
#
# Build:
#   docker build -t arc-agi-benchmarking:latest .
#
# Run locally (for testing):
#   docker run --rm \
#     -e RUN_ID=test-run \
#     -e TASK_ID=00576224 \
#     -e CONFIG_NAME=gpt-4o \
#     -e S3_BUCKET=my-bucket \
#     -e AWS_ACCESS_KEY_ID=xxx \
#     -e AWS_SECRET_ACCESS_KEY=xxx \
#     -e AWS_REGION=us-west-2 \
#     arc-agi-benchmarking:latest

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Note: gcc not typically needed for this project (boto3/pydantic are pure Python)
# but included for potential native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for better layer caching)
COPY pyproject.toml ./

# Copy source code
COPY src/ src/

# Install the package with AWS extras
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -e ".[aws]" || pip install --no-cache-dir -e . boto3

# Set Python to run unbuffered (important for logging in containers)
ENV PYTHONUNBUFFERED=1

# Set default AWS region
ENV AWS_REGION=us-west-2

# Health check - verify the module can be imported
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from arc_agi_benchmarking import batch_worker; print('OK')" || exit 1

# Run the batch worker
ENTRYPOINT ["python", "-m", "arc_agi_benchmarking.batch_worker"]
