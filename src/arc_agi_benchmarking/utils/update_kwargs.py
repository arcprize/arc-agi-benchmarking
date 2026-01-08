#!/usr/bin/env python3
"""
Script to update the metadata.kwargs field in ARC task attempt JSON files.

Point this script at a directory and it will recursively find *.json files,
replace the kwargs dict for every attempt_* entry, and save the updated files.
"""

import argparse
import json
from typing import Dict, Any
from pathlib import Path


def parse_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load the replacement kwargs from a JSON string or file.
    """
    if args.kwargs_json:
        try:
            return json.loads(args.kwargs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON passed to --kwargs_json: {exc}")

    if args.kwargs_file:
        kwargs_path = Path(args.kwargs_file)
        if not kwargs_path.exists():
            raise ValueError(f"--kwargs_file not found: {kwargs_path}")
        with open(kwargs_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in file {kwargs_path}: {exc}")

    raise ValueError("One of --kwargs_json or --kwargs_file is required")


def update_json_file_kwargs(filepath: Path, new_kwargs: Dict[str, Any]) -> None:
    """
    Replace the kwargs dict for each attempt in a single JSON file.
    """
    print(f"Processing {filepath}...")

    with open(filepath, "r") as f:
        data = json.load(f)

    updated = False
    for test_pair_idx, test_pair in enumerate(data):
        for attempt_key, attempt in test_pair.items():
            if not attempt_key.startswith("attempt_"):
                continue
            if not attempt:
                continue
            metadata = attempt.get("metadata")
            if not metadata:
                continue

            old_kwargs = metadata.get("kwargs")
            metadata["kwargs"] = new_kwargs

            if old_kwargs != new_kwargs:
                updated = True
                print(
                    f"  Updated {filepath.name} - test pair {test_pair_idx} - "
                    f"{attempt_key} kwargs"
                )

    if updated:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved updated kwargs to {filepath}")
    else:
        print(f"  No updates needed for {filepath}")


def main() -> None:
    """Parse arguments and process files recursively or a single file."""
    parser = argparse.ArgumentParser(
        description="Update metadata.kwargs in ARC task attempt JSON files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing JSON files (processed recursively)",
    )
    parser.add_argument(
        "--single_file",
        type=str,
        help="Single JSON file to update (for testing/validation)",
    )
    parser.add_argument(
        "--kwargs_json",
        type=str,
        help='Replacement kwargs as a JSON string, e.g. \'{"reasoning": {"effort": "high"}}\'',
    )
    parser.add_argument(
        "--kwargs_file",
        type=str,
        help="Path to a JSON file containing the replacement kwargs",
    )

    args = parser.parse_args()

    try:
        new_kwargs = parse_kwargs(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    # Validate target selection
    if not args.input_dir and not args.single_file:
        print("Error: Either --input_dir or --single_file must be provided")
        return
    if args.input_dir and args.single_file:
        print("Error: Only one of --input_dir or --single_file should be provided")
        return

    # Single file path
    if args.single_file:
        file_path = Path(args.single_file)
        if not file_path.exists():
            print(f"Error: File '{file_path}' does not exist")
            return
        if file_path.suffix.lower() != ".json":
            print(f"Error: '{file_path}' is not a JSON file")
            return

        print(f"Replacing kwargs with: {json.dumps(new_kwargs, indent=2)}")
        update_json_file_kwargs(file_path, new_kwargs)
        return

    # Directory path
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return

    json_files = [
        path for path in input_dir.rglob("*.json") if path.name != "results.json"
    ]
    if not json_files:
        print(f"No JSON files found in '{input_dir}' or its subdirectories")
        return

    print(f"Found {len(json_files)} JSON files to process (searching recursively)")
    print(f"Replacing kwargs with: {json.dumps(new_kwargs, indent=2)}")

    files_by_dir = {}
    for json_file in json_files:
        dir_path = json_file.parent
        files_by_dir.setdefault(dir_path, []).append(json_file)

    for dir_path in sorted(files_by_dir.keys()):
        rel_dir = dir_path.relative_to(input_dir) if dir_path != input_dir else "."
        print(f"\nDirectory: {rel_dir}")
        for json_file in sorted(files_by_dir[dir_path]):
            update_json_file_kwargs(json_file, new_kwargs)

    print("\nKwargs update complete!")


if __name__ == "__main__":
    main()
