import os
from arc_agi_benchmarking.schemas import ARCPair, ModelConfig
from typing import List
import json
import re
import yaml

def get_train_pairs_from_task(data_dir, task_id) -> List[ARCPair]:
    """
    Loads up task train pairs from task json file
    """

    task_file = os.path.join(data_dir, f"{task_id}.json")
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    pairs = []
    for pair in task_data['train']:
        pairs.append(ARCPair(input=pair['input'], output=pair['output']))

    return pairs

def get_test_input_from_task(data_dir, task_id) -> List[ARCPair]:
    task_file = os.path.join(data_dir, f"{task_id}.json")
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    pairs = []
    for pair in task_data['test']:
        pairs.append(ARCPair(input=pair['input']))

    return pairs

def get_test_pairs_from_task(data_dir, task_id) -> List[ARCPair]:
    """
    Loads up task test pairs from task json file with both input and output
    """
    task_file = os.path.join(data_dir, f"{task_id}.json")
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    pairs = []
    for pair in task_data['test']:
        pairs.append(ARCPair(input=pair['input'], output=pair.get('output')))

    return pairs


def save_submission(save_submission_dir: str, task_id: str, task_attempts) -> None:
    """
    Save the submission to a file with full attempt metadata.
    
    The save_submission_dir should be a directory path that includes the config name,
    e.g., 'submissions/o1_short_response' or 'submissions/gemini_pro'.
    """
    os.makedirs(save_submission_dir, exist_ok=True)
    submission_file = os.path.join(save_submission_dir, f"{task_id}.json")
    
    with open(submission_file, "w") as f:
        json.dump(task_attempts, f, indent=4)

    return submission_file

def normalize_model_name(name: str) -> str:
    """
    Normalize model name for comparison by:
    1. Converting dots to dashes
    2. Removing any date suffixes
    3. Removing 'latest' suffix
    4. Removing duplicate dashes
    
    Examples:
        claude-3.5-sonnet -> claude-3-5-sonnet
        claude-3-5-sonnet-20240315 -> claude-3-5-sonnet
        claude-3-5-sonnet-latest -> claude-3-5-sonnet
    """
    # Remove any date suffix (assuming YYYYMMDD format)
    name = re.sub(r'-\d{8}$', '', name)
    
    # Remove 'latest' suffix
    name = re.sub(r'-latest$', '', name)
    
    # Convert dots to dashes
    name = name.replace('.', '-')
    
    # Clean up multiple dashes
    name = re.sub(r'-+', '-', name)
    
    return name

# Default datasets if not specified - public v1 and v2 only (private requires explicit opt-in)
# This ensures private/sensitive data is never sent to a model without explicit configuration.
DEFAULT_ALLOWED_DATASETS = ["public-v1", "public-v2"]


def get_allowed_datasets(config: str) -> list[str]:
    """
    Get the list of allowed datasets for a model configuration.

    Args:
        config: The configuration name (from models.yml)

    Returns:
        List of allowed dataset identifiers (e.g., ["public-v1", "public-v2", "private-v1", "private-v2"])

    Note:
        If no 'datasets' field is specified in models.yml, defaults to ["public-v1", "public-v2"].
        To run on private datasets, the model config MUST explicitly include "private-v1"
        and/or "private-v2" in the datasets list.
    """
    model_config = read_models_config(config)
    return model_config.kwargs.get('datasets', DEFAULT_ALLOWED_DATASETS)


def is_dataset_allowed(config: str, data_source: str) -> bool:
    """
    Check if a data source is allowed for a model configuration.

    Args:
        config: The configuration name (from models.yml)
        data_source: The data source path (e.g., "public-v1/evaluation", "private-v2/training")

    Returns:
        True if the data source is allowed, False otherwise
    """
    allowed = get_allowed_datasets(config)
    # Extract the dataset identifier from data_source (first component of path)
    # e.g., "public-v1/evaluation" -> "public-v1"
    dataset_id = data_source.split('/')[0]
    return dataset_id in allowed


def read_models_config(config: str) -> ModelConfig:
    """
    Reads and parses both models.yml and models_private.yml configuration files 
    for a specific configuration.
    
    Args:
        config (str): The configuration name to look up (e.g., 'o1_high', 'gemini_short_response')
        
    Returns:
        ModelConfig: The configuration for the specified model
        
    Raises:
        ValueError: If no matching configuration is found
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(base_dir, "models.yml")
    models_private_file = os.path.join(base_dir, "models_private.yml")
    
    # Initialize with models from the main config file
    with open(models_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Add models from private config if it exists
    if os.path.exists(models_private_file):
        with open(models_private_file, 'r') as f:
            private_config_data = yaml.safe_load(f)
            # Merge the models lists
            if 'models' in private_config_data:
                config_data['models'].extend(private_config_data['models'])
    
    # Look for a model with the name matching the config parameter
    for model in config_data['models']:
        if model.get('name') == config:
            return ModelConfig(**model)
            
    raise ValueError(f"No matching configuration found for '{config}'")

def read_provider_rate_limits() -> dict:
    """
    Reads and parses the provider_config.yml file to get rate limit configurations.

    Assumes provider_config.yml is in the same base directory as models.yml.

    Returns:
        dict: A dictionary where keys are provider names and values are dicts
              containing 'rate' and 'period'.
              Example: {'openai': {'rate': 60, 'period': 60}}

    Raises:
        FileNotFoundError: If provider_config.yml is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    # Go up three levels from src/utils/task_utils.py to get to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    provider_config_file = os.path.join(base_dir, "provider_config.yml")

    if not os.path.exists(provider_config_file):
        raise FileNotFoundError(f"provider_config.yml not found at {provider_config_file}")

    with open(provider_config_file, 'r') as f:
        try:
            rate_limits_data = yaml.safe_load(f)
            if not isinstance(rate_limits_data, dict):
                raise yaml.YAMLError("provider_config.yml root should be a dictionary of providers.")
            # Basic validation for each provider's config (skip 'defaults' key)
            for provider, limits in rate_limits_data.items():
                if provider == 'defaults':
                    continue  # Skip defaults section
                if not isinstance(limits, dict) or 'rate' not in limits or 'period' not in limits:
                    raise yaml.YAMLError(
                        f"Provider '{provider}' in provider_config.yml must have 'rate' and 'period' keys."
                    )
                if not isinstance(limits['rate'], (int, float)) or not isinstance(limits['period'], (int, float)):
                    raise yaml.YAMLError(
                        f"'rate' and 'period' for provider '{provider}' must be numbers."
                    )
            return rate_limits_data
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing provider_config.yml: {e}")


from arc_agi_benchmarking.resilience.timeout import (
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_REASONING_TIMEOUT,
)

DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_RECOVERY = 60


def get_provider_timeout_config(provider_name: str, all_provider_limits: dict) -> dict:
    """Get timeout and circuit breaker configuration for a provider."""
    defaults = all_provider_limits.get('defaults', {})
    default_request_timeout = defaults.get('request_timeout', DEFAULT_REQUEST_TIMEOUT)
    default_reasoning_timeout = defaults.get('reasoning_timeout', DEFAULT_REASONING_TIMEOUT)
    default_cb_threshold = defaults.get('circuit_breaker_threshold', DEFAULT_CIRCUIT_BREAKER_THRESHOLD)
    default_cb_recovery = defaults.get('circuit_breaker_recovery', DEFAULT_CIRCUIT_BREAKER_RECOVERY)

    provider_config = all_provider_limits.get(provider_name, {})

    return {
        'request_timeout': provider_config.get('request_timeout', default_request_timeout),
        'reasoning_timeout': provider_config.get('reasoning_timeout', default_reasoning_timeout),
        'circuit_breaker_threshold': provider_config.get('circuit_breaker_threshold', default_cb_threshold),
        'circuit_breaker_recovery': provider_config.get('circuit_breaker_recovery', default_cb_recovery),
    }