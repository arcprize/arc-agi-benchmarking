import json
import os
from typing import Any, Dict, List, Optional


def _submission_file(submission_dir: str, task_id: str) -> str:
    return os.path.join(os.path.abspath(submission_dir), f"{task_id}.json")


def submission_exists(submission_dir: str, task_id: str) -> bool:
    """Return whether a submission file exists for ``task_id``."""
    return os.path.exists(_submission_file(submission_dir, task_id))


def load_submission(submission_dir: str, task_id: str) -> Optional[List[Any]]:
    """Load a submission, returning ``None`` when it is absent or malformed."""
    try:
        with open(_submission_file(submission_dir, task_id), "r") as f:
            submission = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    return submission if isinstance(submission, list) else None


def _metadata_pair_index(pair_attempts: Dict[str, Any]) -> Optional[int]:
    for attempt in pair_attempts.values():
        if not isinstance(attempt, dict):
            continue
        metadata = attempt.get("metadata")
        if not isinstance(metadata, dict):
            continue
        pair_index = metadata.get("pair_index")
        if isinstance(pair_index, int) and not isinstance(pair_index, bool):
            return pair_index
    return None


def normalize_submission_pairs(
    submission: Optional[List[Any]], expected_num_pairs: int
) -> List[Dict[str, Any]]:
    """Map saved pairs to task pair indexes, retaining valid prior attempts.

    Older submissions may omit a test pair when every attempt for that pair was
    null. Non-null attempts include their pair index in metadata, which lets us
    put later saved pairs back into their correct positions.
    """
    normalized: List[Dict[str, Any]] = [dict() for _ in range(expected_num_pairs)]
    if submission is None:
        return normalized

    for saved_position, pair_attempts in enumerate(submission):
        if not isinstance(pair_attempts, dict):
            continue

        metadata_index = _metadata_pair_index(pair_attempts)
        candidate_indexes = (metadata_index, saved_position)
        target_index = next(
            (
                index
                for index in candidate_indexes
                if index is not None
                and 0 <= index < expected_num_pairs
                and not normalized[index]
            ),
            None,
        )
        if target_index is not None:
            normalized[target_index] = dict(pair_attempts)

    return normalized


def submission_attempts_complete(
    task_attempts: List[Dict[str, Any]], num_attempts: int
) -> bool:
    """Return whether every expected pair has every expected non-null attempt."""
    return bool(task_attempts) and all(
        pair_attempts.get(f"attempt_{attempt_num}") is not None
        for pair_attempts in task_attempts
        for attempt_num in range(1, num_attempts + 1)
    )


def submission_is_complete(
    submission_dir: str,
    task_id: str,
    expected_num_pairs: int,
    num_attempts: int,
) -> bool:
    """Return whether a saved submission has all expected non-null attempts."""
    if expected_num_pairs < 1 or num_attempts < 1:
        return False
    submission = load_submission(submission_dir, task_id)
    if submission is None:
        return False
    normalized = normalize_submission_pairs(submission, expected_num_pairs)
    return submission_attempts_complete(normalized, num_attempts)
