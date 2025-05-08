import os
import json

def validate_data(data_dir, task_id):
    """
    Validate the data directory and task id and checks to make sure the task json is valid
    """

    if not data_dir:
        raise ValueError("Data directory is required")

    if not task_id:
        raise ValueError("Task ID is required")

    # Convert data_dir to absolute path relative to the current working directory
    abs_data_dir = os.path.abspath(data_dir)

    if not os.path.isdir(abs_data_dir):
        raise ValueError(f"Invalid data directory: {abs_data_dir}")
    
    # Check if the task JSON file exists
    task_file = os.path.join(abs_data_dir, f"{task_id}.json")
    if not os.path.isfile(task_file):
        raise ValueError(f"Task file '{task_id}.json' not found in {abs_data_dir}")
    
    # Validate the JSON file
    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        # Check for required keys in the JSON
        required_keys = ['train', 'test']
        for key in required_keys:
            if key not in task_data:
                raise ValueError(f"Missing required key '{key}' in task file {task_file}")
        
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in task file {task_file}")
    
    return True