"""
Test that the 'correct' field is properly set on Attempt objects during task execution.

This test suite verifies that when tasks are run through main.py's ARCTester,
the 'correct' field in each Attempt is set to True/False based on whether
the attempt's answer matches the ground truth output for that test pair.

The tests verify:
1. Correct attempts have correct=True
2. Incorrect attempts have correct=False
3. Multiple attempts per pair are all scored
4. Empty list answers have correct=False
5. The correct field is persisted to the saved JSON files
"""

import pytest
import json
import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from arc_agi_benchmarking.schemas import (
    Attempt, AttemptMetadata, Choice, Message, Usage,
    CompletionTokensDetails, Cost, ModelConfig, ModelPricing
)
from arc_agi_benchmarking.utils.task_utils import save_submission


# Test Data - A simple ARC task with known solutions
TEST_TASK_DATA = {
    "train": [
        {"input": [[1]], "output": [[1, 1]]},
        {"input": [[2]], "output": [[2, 2]]}
    ],
    "test": [
        {"input": [[3]], "output": [[3, 3]]},  # Test pair 0
        {"input": [[4]], "output": [[4, 4, 4]]}  # Test pair 1
    ]
}


def create_mock_attempt(answer, pair_index=0, task_id="test_task", test_id="test_config"):
    """Helper to create a mock Attempt object with all required fields."""
    metadata = AttemptMetadata(
        model="test-model",
        provider="test-provider",
        start_timestamp=datetime.now(timezone.utc),
        end_timestamp=datetime.now(timezone.utc),
        choices=[
            Choice(index=0, message=Message(role="user", content="test prompt")),
            Choice(index=1, message=Message(role="assistant", content=json.dumps(answer)))
        ],
        reasoning_summary=None,
        kwargs={},
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=0,
                accepted_prediction_tokens=10,
                rejected_prediction_tokens=0
            )
        ),
        cost=Cost(
            prompt_cost=0.001,
            completion_cost=0.002,
            reasoning_cost=0.0,
            total_cost=0.003
        ),
        task_id=task_id,
        pair_index=pair_index,
        test_id=test_id
    )

    return Attempt(answer=answer, metadata=metadata, correct=None)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with test task data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    task_file = data_dir / "test_task.json"
    with open(task_file, 'w') as f:
        json.dump(TEST_TASK_DATA, f)

    return str(data_dir)


@pytest.fixture
def temp_submission_dir(tmp_path):
    """Create a temporary directory for submissions."""
    submission_dir = tmp_path / "submissions"
    submission_dir.mkdir()
    return str(submission_dir)


@pytest.fixture
def mock_model_config():
    """Create a mock ModelConfig for testing."""
    return ModelConfig(
        name="test_config",
        model_name="test-model",
        provider="test-provider",
        api_key_env="TEST_API_KEY",
        pricing=ModelPricing(date="2024-01-01", input=1.0, output=2.0),
        kwargs={}
    )


class TestCorrectFieldSetDuringExecution:
    """Test that the correct field is set during task execution (Option 1)."""

    def test_correct_attempt_has_correct_true(self, temp_data_dir, temp_submission_dir, mock_model_config):
        """Test that a correct answer results in correct=True."""
        from main import ARCTester

        # Expected output for test pair 0
        correct_answer = [[3, 3]]
        mock_attempt = create_mock_attempt(correct_answer, pair_index=0)

        mock_provider = Mock()
        mock_provider.make_prediction = Mock(return_value=mock_attempt)

        with patch('main.utils.read_models_config', return_value=mock_model_config):
            with patch.object(ARCTester, 'init_provider', return_value=mock_provider):
                tester = ARCTester(
                    config="test_config",
                    save_submission_dir=temp_submission_dir,
                    overwrite_submission=True,
                    print_submission=False,
                    num_attempts=1,
                    retry_attempts=1
                )

                result = tester.generate_task_solution(temp_data_dir, "test_task")

                # Load the saved submission
                submission_file = os.path.join(temp_submission_dir, "test_task.json")
                assert os.path.exists(submission_file), "Submission file should be created"

                with open(submission_file, 'r') as f:
                    saved_submission = json.load(f)

                # Verify the correct field is True
                attempt_data = saved_submission[0]["attempt_1"]
                assert attempt_data["correct"] is True, "Correct answer should have correct=True"


    def test_incorrect_attempt_has_correct_false(self, temp_data_dir, temp_submission_dir, mock_model_config):
        """Test that an incorrect answer results in correct=False."""
        from main import ARCTester

        # Wrong answer for test pair 0 (expected [[3, 3]])
        incorrect_answer = [[9, 9]]
        mock_attempt = create_mock_attempt(incorrect_answer, pair_index=0)

        mock_provider = Mock()
        mock_provider.make_prediction = Mock(return_value=mock_attempt)

        with patch('main.utils.read_models_config', return_value=mock_model_config):
            with patch.object(ARCTester, 'init_provider', return_value=mock_provider):
                tester = ARCTester(
                    config="test_config",
                    save_submission_dir=temp_submission_dir,
                    overwrite_submission=True,
                    print_submission=False,
                    num_attempts=1,
                    retry_attempts=1
                )

                result = tester.generate_task_solution(temp_data_dir, "test_task")

                submission_file = os.path.join(temp_submission_dir, "test_task.json")
                with open(submission_file, 'r') as f:
                    saved_submission = json.load(f)

                attempt_data = saved_submission[0]["attempt_1"]
                assert attempt_data["correct"] is False, "Incorrect answer should have correct=False"


    def test_multiple_attempts_all_scored(self, temp_data_dir, temp_submission_dir, mock_model_config):
        """Test that all attempts are scored with correct field."""
        from main import ARCTester

        # First attempt wrong, second attempt correct
        wrong_answer = [[9, 9]]
        correct_answer = [[3, 3]]

        attempt_1 = create_mock_attempt(wrong_answer, pair_index=0)
        attempt_2 = create_mock_attempt(correct_answer, pair_index=0)

        call_count = [0]
        def mock_make_prediction(*args, **kwargs):
            call_count[0] += 1
            return attempt_1 if call_count[0] == 1 else attempt_2

        mock_provider = Mock()
        mock_provider.make_prediction = Mock(side_effect=mock_make_prediction)

        with patch('main.utils.read_models_config', return_value=mock_model_config):
            with patch.object(ARCTester, 'init_provider', return_value=mock_provider):
                tester = ARCTester(
                    config="test_config",
                    save_submission_dir=temp_submission_dir,
                    overwrite_submission=True,
                    print_submission=False,
                    num_attempts=2,
                    retry_attempts=1
                )

                result = tester.generate_task_solution(temp_data_dir, "test_task")

                submission_file = os.path.join(temp_submission_dir, "test_task.json")
                with open(submission_file, 'r') as f:
                    saved_submission = json.load(f)

                # Verify both attempts have correct field set
                pair_0_attempts = saved_submission[0]
                assert pair_0_attempts["attempt_1"]["correct"] is False, "First attempt should be incorrect"
                assert pair_0_attempts["attempt_2"]["correct"] is True, "Second attempt should be correct"


    def test_empty_list_answer_has_correct_false(self, temp_data_dir, temp_submission_dir, mock_model_config):
        """Test that an empty list answer has correct=False."""
        from main import ARCTester

        empty_answer = []
        mock_attempt = create_mock_attempt(empty_answer, pair_index=0)

        mock_provider = Mock()
        mock_provider.make_prediction = Mock(return_value=mock_attempt)

        with patch('main.utils.read_models_config', return_value=mock_model_config):
            with patch.object(ARCTester, 'init_provider', return_value=mock_provider):
                tester = ARCTester(
                    config="test_config",
                    save_submission_dir=temp_submission_dir,
                    overwrite_submission=True,
                    print_submission=False,
                    num_attempts=1,
                    retry_attempts=1
                )

                result = tester.generate_task_solution(temp_data_dir, "test_task")

                submission_file = os.path.join(temp_submission_dir, "test_task.json")
                with open(submission_file, 'r') as f:
                    saved_submission = json.load(f)

                attempt_data = saved_submission[0]["attempt_1"]
                assert attempt_data["correct"] is False, "Empty list should have correct=False"


    def test_multiple_pairs_each_scored_independently(self, temp_data_dir, temp_submission_dir, mock_model_config):
        """Test that each test pair is scored independently with its own ground truth."""
        from main import ARCTester

        # Pair 0: correct answer [[3, 3]]
        # Pair 1: wrong answer (expected [[4, 4, 4]])
        attempt_pair_0 = create_mock_attempt([[3, 3]], pair_index=0)
        attempt_pair_1 = create_mock_attempt([[9, 9, 9]], pair_index=1)

        call_count = [0]
        def mock_make_prediction(*args, **kwargs):
            call_count[0] += 1
            return attempt_pair_0 if call_count[0] == 1 else attempt_pair_1

        mock_provider = Mock()
        mock_provider.make_prediction = Mock(side_effect=mock_make_prediction)

        with patch('main.utils.read_models_config', return_value=mock_model_config):
            with patch.object(ARCTester, 'init_provider', return_value=mock_provider):
                tester = ARCTester(
                    config="test_config",
                    save_submission_dir=temp_submission_dir,
                    overwrite_submission=True,
                    print_submission=False,
                    num_attempts=1,
                    retry_attempts=1
                )

                result = tester.generate_task_solution(temp_data_dir, "test_task")

                submission_file = os.path.join(temp_submission_dir, "test_task.json")
                with open(submission_file, 'r') as f:
                    saved_submission = json.load(f)

                # Verify pair 0 is correct and pair 1 is incorrect
                assert saved_submission[0]["attempt_1"]["correct"] is True, "Pair 0 should be correct"
                assert saved_submission[1]["attempt_1"]["correct"] is False, "Pair 1 should be incorrect"


    def test_wrong_dimensions_has_correct_false(self, temp_data_dir, temp_submission_dir, mock_model_config):
        """Test that answers with wrong dimensions are marked as incorrect."""
        from main import ARCTester

        # Expected: [[3, 3]], Given: [[3, 3, 3]] (wrong dimensions)
        wrong_dims_answer = [[3, 3, 3]]
        mock_attempt = create_mock_attempt(wrong_dims_answer, pair_index=0)

        mock_provider = Mock()
        mock_provider.make_prediction = Mock(return_value=mock_attempt)

        with patch('main.utils.read_models_config', return_value=mock_model_config):
            with patch.object(ARCTester, 'init_provider', return_value=mock_provider):
                tester = ARCTester(
                    config="test_config",
                    save_submission_dir=temp_submission_dir,
                    overwrite_submission=True,
                    print_submission=False,
                    num_attempts=1,
                    retry_attempts=1
                )

                result = tester.generate_task_solution(temp_data_dir, "test_task")

                submission_file = os.path.join(temp_submission_dir, "test_task.json")
                with open(submission_file, 'r') as f:
                    saved_submission = json.load(f)

                attempt_data = saved_submission[0]["attempt_1"]
                assert attempt_data["correct"] is False, "Wrong dimensions should have correct=False"


    def test_correct_field_exists_in_attempt_schema(self):
        """Test that Attempt objects can hold the correct field."""
        attempt = create_mock_attempt([[1, 2]], pair_index=0)

        # Initially None
        assert attempt.correct is None, "correct field should initially be None"

        # Can be set to True
        attempt.correct = True
        assert attempt.correct is True

        # Can be set to False
        attempt.correct = False
        assert attempt.correct is False

        # Can be serialized/deserialized
        attempt_dict = attempt.model_dump()
        assert "correct" in attempt_dict
        assert attempt_dict["correct"] is False

        # Can be loaded from dict
        reconstructed = Attempt(**attempt_dict)
        assert reconstructed.correct is False


class TestCorrectFieldPersistence:
    """Test that the correct field is properly saved and loaded from JSON."""

    def test_correct_field_persists_in_json(self, tmp_path):
        """Test that the correct field is saved to and loaded from JSON files."""
        # Create an attempt with correct=True
        attempt_correct = create_mock_attempt([[1, 2]], pair_index=0)
        attempt_correct.correct = True

        # Create an attempt with correct=False
        attempt_incorrect = create_mock_attempt([[9, 9]], pair_index=0)
        attempt_incorrect.correct = False

        # Save to JSON like main.py does
        task_attempts = [
            {
                "attempt_1": attempt_correct.model_dump(mode='json'),
                "attempt_2": attempt_incorrect.model_dump(mode='json')
            }
        ]

        submission_dir = tmp_path / "submissions"
        submission_dir.mkdir()
        save_submission(str(submission_dir), "test_task", task_attempts)

        # Load back from JSON
        submission_file = submission_dir / "test_task.json"
        with open(submission_file, 'r') as f:
            loaded_submission = json.load(f)

        # Verify correct field persisted
        assert loaded_submission[0]["attempt_1"]["correct"] is True
        assert loaded_submission[0]["attempt_2"]["correct"] is False


    def test_correct_field_none_when_not_set(self, tmp_path):
        """Test that correct field is None when not explicitly set."""
        # Create an attempt without setting correct
        attempt = create_mock_attempt([[1, 2]], pair_index=0)
        assert attempt.correct is None

        # Save and reload
        task_attempts = [{"attempt_1": attempt.model_dump(mode='json')}]

        submission_dir = tmp_path / "submissions"
        submission_dir.mkdir()
        save_submission(str(submission_dir), "test_task", task_attempts)

        # Load back
        submission_file = submission_dir / "test_task.json"
        with open(submission_file, 'r') as f:
            loaded_submission = json.load(f)

        # Verify correct field is None
        assert loaded_submission[0]["attempt_1"]["correct"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
