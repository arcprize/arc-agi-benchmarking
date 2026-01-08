#!/usr/bin/env python3
"""
Script to update model names and test IDs in ARC task attempt JSON files.

This script reads JSON files containing task attempts and updates the model name
and test_id based on a provided config mapping.
"""

import argparse
import json
import os
from typing import Dict, Any
from pathlib import Path

# Import the required functions
from arc_agi_benchmarking.utils.task_utils import read_models_config

# Guard to avoid spamming the same debug line for many files
_printed_config_lookup = False


def update_json_file_model_and_config(filepath: Path, config_name: str) -> None:
    """
    Update model name and test_id in a single JSON file.
    
    Args:
        filepath: Path to the JSON file
        config_name: New model configuration name to use
    """
    global _printed_config_lookup
    print(f"Processing {filepath}...")

    if not _printed_config_lookup:
        base_dir = Path(__file__).resolve().parent.parent
        models_path = base_dir / "models.yml"
        private_path = base_dir / "models_private.yml"
        print(f"[DEBUG] Looking for config '{config_name}' in:")
        print(f"        models.yml        -> {models_path} (exists={models_path.exists()})")
        print(f"        models_private.yml -> {private_path} (exists={private_path.exists()})")
        print(f"        ARC_MODELS_YML env -> {os.environ.get('ARC_MODELS_YML', 'not set')}")
        _printed_config_lookup = True
    
    # Read the JSON file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get the model config for the new model name
    try:
        model_config = read_models_config(config_name)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Get the new model name from the config
    new_model_name = model_config.model_name
    
    # Update model name and test_id for each test pair
    updated = False
    for test_pair_idx, test_pair in enumerate(data):
        # Iterate through all attempts (attempt_1, attempt_2, etc.)
        for attempt_key in test_pair:
            if attempt_key.startswith('attempt_'):
                attempt = test_pair[attempt_key]

                if not attempt:
                    continue
                
                # Check if this attempt has metadata
                if 'metadata' in attempt:
                    metadata = attempt['metadata']
                    
                    # Update model name
                    old_model = metadata.get('model', 'N/A')
                    metadata['model'] = new_model_name
                    
                    # Update test_id to config name
                    old_test_id = metadata.get('test_id', 'N/A')
                    metadata['test_id'] = config_name
                    
                    updated = True
                    
                    print(f"  Updated {filepath.name} - test pair {test_pair_idx} - {attempt_key}")
                    print(f"    Model: {old_model} -> {new_model_name}")
                    print(f"    Test ID: {old_test_id} -> {config_name}")
    
    # Save the updated data back to the file
    if updated:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved updated model name and test_id to {filepath}")
    else:
        print(f"  No updates needed for {filepath}")


def update_single_file_test(filepath: str, config_name: str) -> None:
    """
    Test function to update a single file for validation.
    
    Args:
        filepath: Path to the single JSON file to test
        config_name: Model configuration name to use
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        print(f"Error: File '{filepath}' does not exist")
        return
    
    if not file_path.suffix.lower() == '.json':
        print(f"Error: '{filepath}' is not a JSON file")
        return
    
    print(f"Testing update on single file: {filepath}")
    print(f"Using config: {config_name}")
    print("-" * 50)
    
    update_json_file_model_and_config(file_path, config_name)
    
    print("-" * 50)
    print("Single file test complete!")


def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(
        description='Update model names and test IDs in ARC task attempt JSON files'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Directory containing JSON files with task attempts (searches recursively)'
    )
    parser.add_argument(
        '--single_file',
        type=str,
        help='Single JSON file to update (for testing purposes)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Model configuration name to use for updates'
    )
    
    args = parser.parse_args()
    
    # Check that either input_dir or single_file is provided
    if not args.input_dir and not args.single_file:
        print("Error: Either --input_dir or --single_file must be provided")
        return
    
    if args.input_dir and args.single_file:
        print("Error: Only one of --input_dir or --single_file should be provided")
        return
    
    # Handle single file testing
    if args.single_file:
        update_single_file_test(args.single_file, args.config)
        return
    
    # Handle directory processing
    input_dir = Path(args.input_dir)
    
    # Check if directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return
    
    # Find all JSON files recursively in the directory and all subdirectories
    json_files = list(input_dir.rglob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}' or its subdirectories")
        return
    
    print(f"Found {len(json_files)} JSON files to process (searching recursively)")
    print(f"Using config: {args.config}")
    
    # Group files by directory for better output organization
    files_by_dir = {}
    for json_file in json_files:
        dir_path = json_file.parent
        if dir_path not in files_by_dir:
            files_by_dir[dir_path] = []
        files_by_dir[dir_path].append(json_file)
    
    print(f"Files found in {len(files_by_dir)} directories")
    print()
    
    # Process each JSON file
    for dir_path in sorted(files_by_dir.keys()):
        print(f"\nDirectory: {dir_path.relative_to(input_dir) if dir_path != input_dir else '.'}")
        for json_file in sorted(files_by_dir[dir_path]):
            update_json_file_model_and_config(json_file, args.config)
    
    print("\nModel name and test_id update complete!")


if __name__ == '__main__':
    main()
