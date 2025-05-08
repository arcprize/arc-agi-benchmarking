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