#!/usr/bin/env python3
"""
Script to update pricing information in ARC task attempt JSON files.

This script reads JSON files containing task attempts and updates the cost
information based on the current pricing in models.yml.
"""

import argparse
import json
import os
from typing import Dict, Any
from pathlib import Path

# Import the required functions
from arc_agi_benchmarking.utils.task_utils import read_models_config
from arc_agi_benchmarking.schemas import Usage, CompletionTokensDetails, Cost


def calculate_cost(usage: Dict[str, Any], pricing: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate usage costs based on token counts and pricing.
    
    This follows the logic from openai_base.py's _calculate_cost method.
    
    Args:
        usage: Dictionary containing token usage information
        pricing: Dictionary containing input/output pricing per 1M tokens
        
    Returns:
        Dictionary with prompt_cost, completion_cost, reasoning_cost, and total_cost
    """
    # Extract token counts
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    total_tokens = usage.get('total_tokens', 0) or 0
    
    # Get reasoning tokens if available
    completion_tokens_details = usage.get('completion_tokens_details', {})
    reasoning_tokens_explicit = completion_tokens_details.get('reasoning_tokens', 0) or 0
    
    # Determine effective token counts for cost calculation
    prompt_tokens_for_cost = prompt_tokens
    completion_tokens_for_cost = 0
    reasoning_tokens_for_cost = 0
    
    # Case A: Completion includes Reasoning (pt + ct == tt)
    if total_tokens == 0 or (prompt_tokens + completion_tokens == total_tokens):
        reasoning_tokens_for_cost = reasoning_tokens_explicit
        completion_tokens_for_cost = max(0, completion_tokens - reasoning_tokens_for_cost)
    # Case B: Reasoning is Separate or Inferred (pt + ct < tt)
    else:
        reasoning_tokens_for_cost = reasoning_tokens_explicit if reasoning_tokens_explicit else total_tokens - (prompt_tokens + completion_tokens)
        completion_tokens_for_cost = completion_tokens
    
    # Get pricing per token
    input_cost_per_token = pricing['input'] / 1_000_000
    output_cost_per_token = pricing['output'] / 1_000_000
    
    # Calculate costs
    prompt_cost = prompt_tokens_for_cost * input_cost_per_token
    completion_cost = completion_tokens_for_cost * output_cost_per_token
    reasoning_cost = reasoning_tokens_for_cost * output_cost_per_token
    total_cost = prompt_cost + completion_cost + reasoning_cost
    
    return {
        'prompt_cost': prompt_cost,
        'completion_cost': completion_cost,
        'reasoning_cost': reasoning_cost,
        'total_cost': total_cost
    }


def update_json_file_pricing(filepath: Path, config_name: str) -> None:
    """
    Update pricing in a single JSON file.
    
    Args:
        filepath: Path to the JSON file
        config_name: Model configuration name to get pricing from
    """
    print(f"Processing {filepath}...")
    
    # Read the JSON file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get the model config for pricing
    try:
        model_config = read_models_config(config_name)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Check if we have pricing information
    if not hasattr(model_config, 'pricing') or not model_config.pricing:
        print(f"Warning: No pricing information found for config '{config_name}'")
        return
    
    pricing = {
        'input': model_config.pricing.input,
        'output': model_config.pricing.output
    }
    
    # Update pricing for each test pair
    updated = False
    for test_pair_idx, test_pair in enumerate(data):
        # Iterate through all attempts (attempt_1, attempt_2, etc.)
        for attempt_key in test_pair:
            if attempt_key.startswith('attempt_'):
                attempt = test_pair[attempt_key]

                if not attempt:
                    continue
                
                # Check if this attempt has metadata and usage
                if 'metadata' in attempt and 'usage' in attempt['metadata']:
                    usage = attempt['metadata']['usage']
                    
                    # Calculate new costs
                    new_costs = calculate_cost(usage, pricing)
                    
                    # Update the cost field in metadata
                    if 'cost' not in attempt['metadata']:
                        attempt['metadata']['cost'] = {}
                    
                    attempt['metadata']['cost'].update(new_costs)
                    updated = True
                    
                    print(f"  Updated {filepath.name} - test pair {test_pair_idx} - {attempt_key} - {new_costs}")
    
    # Save the updated data back to the file
    if updated:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved updated pricing to {filepath}")
    else:
        print(f"  No updates needed for {filepath}")


def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(
        description='Update pricing information in ARC task attempt JSON files'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing JSON files with task attempts (searches recursively)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Model configuration name from models.yml to get pricing'
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
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
    print(f"Using pricing from config: {args.config}")
    
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
            update_json_file_pricing(json_file, args.config)
    
    print("\nPricing update complete!")


if __name__ == '__main__':
    main()
