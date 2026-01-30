#!/usr/bin/env python3
"""
Detect which model configurations changed between git commits.

This script compares models.yml between two git refs and outputs
the list of config names that were added or modified.

Usage:
    python scripts/detect_config_changes.py [--base BASE_REF] [--head HEAD_REF]

Examples:
    # Compare HEAD with HEAD~1 (default)
    python scripts/detect_config_changes.py

    # Compare specific refs
    python scripts/detect_config_changes.py --base main --head feature-branch

    # Compare with a specific commit
    python scripts/detect_config_changes.py --base abc123 --head HEAD

    # For GitHub Actions push events (covers multi-commit pushes)
    python scripts/detect_config_changes.py --base ${{ github.event.before }} --head ${{ github.sha }}

Output:
    JSON array of changed config names, e.g.: ["gpt-4o", "claude-3-opus"]

Exit codes:
    0: Success
    1: YAML parse error (head ref has invalid YAML)
    2: Other error
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


MODELS_FILE = "src/arc_agi_benchmarking/models.yml"

# Git uses all-zeros SHA to indicate "no previous commit" (first push)
NULL_SHA = "0000000000000000000000000000000000000000"


class YamlParseError(Exception):
    """Raised when YAML content cannot be parsed."""

    pass


def get_file_at_ref(filepath: str, ref: str) -> Optional[str]:
    """Get file contents at a specific git ref.

    Args:
        filepath: Path to the file relative to repo root
        ref: Git ref (commit SHA, branch name, HEAD~1, etc.)

    Returns:
        File contents as string, or None if file doesn't exist at that ref
    """
    # Handle null SHA (first push to repo/branch)
    # Git uses exactly 40 zeros for null SHA - don't use prefix match
    # as legitimate commits can start with zeros
    if ref == NULL_SHA:
        return None

    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{filepath}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        # File doesn't exist at this ref
        return None


def parse_models_yaml(content: Optional[str], strict: bool = False) -> dict:
    """Parse models.yml content into a dictionary keyed by config name.

    Handles both formats:
    - List format: models: [{name: "foo", ...}, {name: "bar", ...}]
    - Dict format: {foo: {...}, bar: {...}}

    Args:
        content: YAML content as string, or None
        strict: If True, raise YamlParseError on invalid YAML

    Returns:
        Dictionary of model configs keyed by name, or empty dict if content is None

    Raises:
        YamlParseError: If strict=True and YAML is invalid
    """
    if content is None:
        return {}
    try:
        data = yaml.safe_load(content) or {}

        # Handle list format: models: [{name: "foo", ...}, ...]
        if isinstance(data, dict) and "models" in data:
            models_list = data.get("models", [])
            if isinstance(models_list, list):
                return {
                    m["name"]: m
                    for m in models_list
                    if isinstance(m, dict) and "name" in m
                }

        # Handle dict format: {foo: {...}, bar: {...}}
        if isinstance(data, dict):
            return data

        return {}
    except yaml.YAMLError as e:
        if strict:
            raise YamlParseError(f"Invalid YAML: {e}") from e
        return {}


def configs_equal(config1: dict, config2: dict) -> bool:
    """Compare two config dictionaries for equality.

    Uses direct dictionary comparison which handles nested structures.

    Args:
        config1: First config dictionary
        config2: Second config dictionary

    Returns:
        True if configs are equal, False otherwise
    """
    return config1 == config2


def detect_changed_configs(
    base_ref: str, head_ref: str, strict: bool = True
) -> list[str]:
    """Detect which model configs changed between two git refs.

    A config is considered "changed" if:
    - It was added (exists in head but not base)
    - It was modified (exists in both but content differs)

    Removed configs are not included since there's no benchmark to run.

    Args:
        base_ref: Base git ref to compare from
        head_ref: Head git ref to compare to
        strict: If True, raise YamlParseError if head YAML is invalid

    Returns:
        Sorted list of config names that were added or modified

    Raises:
        YamlParseError: If strict=True and head YAML is invalid
    """
    # Get file contents at each ref
    base_content = get_file_at_ref(MODELS_FILE, base_ref)
    head_content = get_file_at_ref(MODELS_FILE, head_ref)

    # Parse YAML into dictionaries
    # Base can be lenient (old commits may have had issues)
    # Head must be strict - we need valid YAML to run benchmarks
    base_configs = parse_models_yaml(base_content, strict=False)
    head_configs = parse_models_yaml(head_content, strict=strict)

    changed = []

    # Find added or modified configs
    for config_name, config_value in head_configs.items():
        if config_name not in base_configs:
            # Added
            changed.append(config_name)
        elif not configs_equal(config_value, base_configs[config_name]):
            # Modified
            changed.append(config_name)

    # Note: Removed configs are intentionally not included
    # (no benchmark to run for a deleted config)

    return sorted(changed)


def compare_configs_direct(base_configs: dict, head_configs: dict) -> list[str]:
    """Compare two config dictionaries directly.

    This is useful for testing or when you already have parsed configs.

    Args:
        base_configs: Base config dictionary
        head_configs: Head config dictionary

    Returns:
        Sorted list of config names that were added or modified
    """
    changed = []

    for config_name, config_value in head_configs.items():
        if config_name not in base_configs:
            changed.append(config_name)
        elif not configs_equal(config_value, base_configs[config_name]):
            changed.append(config_name)

    return sorted(changed)


def main():
    parser = argparse.ArgumentParser(
        description="Detect which model configurations changed between git commits."
    )
    parser.add_argument(
        "--base",
        default="HEAD~1",
        help="Base git ref to compare from (default: HEAD~1)",
    )
    parser.add_argument(
        "--head", default="HEAD", help="Head git ref to compare to (default: HEAD)"
    )
    parser.add_argument(
        "--fallback-all",
        action="store_true",
        help="If no changes detected, output all configs",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configs to output (for safety)",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Don't fail on YAML parse errors (treat as empty)",
    )

    args = parser.parse_args()

    try:
        changed_configs = detect_changed_configs(
            args.base, args.head, strict=not args.lenient
        )
    except YamlParseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Hint: Fix the YAML syntax error or use --lenient to skip validation",
            file=sys.stderr,
        )
        return 1

    # Fallback to all configs if requested and none detected
    if not changed_configs and args.fallback_all:
        head_content = get_file_at_ref(MODELS_FILE, args.head)
        try:
            all_configs = parse_models_yaml(head_content, strict=not args.lenient)
        except YamlParseError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        changed_configs = sorted(all_configs.keys())

    # Apply max limit if specified
    if args.max_configs and len(changed_configs) > args.max_configs:
        changed_configs = changed_configs[: args.max_configs]

    # Output as JSON for easy parsing in CI
    print(json.dumps(changed_configs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
