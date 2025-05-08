import os

def submission_exists(submission_dir, task_id):
    """
    Check if a submission already exists
    
    Args:
    submission_dir (str): Directory where submissions are stored (can be relative)
    task_id (str): ID of the task to check

    Returns:
    bool: True if the submission exists, False otherwise
    """
    # Convert submission_dir to absolute path
    abs_submission_dir = os.path.abspath(submission_dir)
    
    # Check if the submission file exists
    submission_file = os.path.join(abs_submission_dir, f"{task_id}.json")
    
    submission_exists = os.path.exists(submission_file)

    return submission_exists