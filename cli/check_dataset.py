#!/usr/bin/env python3
"""
Check if a model configuration is allowed to run on a specific dataset.

Usage:
    python cli/check_dataset.py <config_name> <data_source>

Exit codes:
    0 - Dataset is allowed
    1 - Dataset is NOT allowed
    2 - Configuration not found or other error

Examples:
    python cli/check_dataset.py claude-3-5-sonnet-20241022 public-v2/evaluation
    python cli/check_dataset.py qwen3-max private-v1/evaluation  # Will exit 1
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from arc_agi_benchmarking.utils.task_utils import (
    get_allowed_datasets,
    is_dataset_allowed,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: check_dataset.py <config_name> <data_source>", file=sys.stderr)
        print("Example: check_dataset.py claude-3-5-sonnet-20241022 public-v2/evaluation", file=sys.stderr)
        sys.exit(2)

    config_name = sys.argv[1]
    data_source = sys.argv[2]

    try:
        allowed_datasets = get_allowed_datasets(config_name)
        is_allowed = is_dataset_allowed(config_name, data_source)

        # Extract dataset type from data_source
        dataset_type = data_source.split('/')[0]

        if is_allowed:
            print(f"OK: {config_name} can run on {data_source}")
            print(f"Allowed datasets: {', '.join(allowed_datasets)}")
            sys.exit(0)
        else:
            print(f"BLOCKED: {config_name} cannot run on {data_source}", file=sys.stderr)
            print(f"Allowed datasets: {', '.join(allowed_datasets)}", file=sys.stderr)
            print(f"Requested dataset type: {dataset_type}", file=sys.stderr)
            sys.exit(1)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
