import sys
import pytest
from unittest.mock import patch, MagicMock
import os
import json

from main import ARCTester 
from arc_agi_testing.adapters import ProviderAdapter # Import base class for type hinting/mocking structure
from arc_agi_testing.schemas import Attempt, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails # Import necessary schemas
from datetime import datetime, timezone # Import datetime

# Constants for the test
PROVIDER_CONFIG = "gpt-4o-2024-11-20"
TASK_ID = "f0afb749"
DATA_DIR = "data/arc-agi/data/evaluation"  # Assuming tests run from root
SAVE_DIR = "." # Save in the current directory for simplicity in test
NUM_ATTEMPTS = 1 # Reduce attempts for faster testing
RETRY_ATTEMPTS = 1

# --- Mock Attempt Object --- 
# Simulates the object returned by provider.make_prediction
mock_attempt = MagicMock(spec=Attempt)
# Mock metadata structure
mock_metadata = MagicMock(spec=AttemptMetadata)
mock_choice = MagicMock(spec=Choice)
mock_message = MagicMock(spec=Message)
mock_message.content = '[[0, 1], [1, 0]]' # The actual mock response content
mock_choice.message = mock_message
mock_metadata.choices = [mock_choice] # Needs to be a list
# Add dummy usage/cost/etc. to metadata if needed by later code (unlikely for this test path)
mock_metadata.usage = MagicMock(spec=Usage, prompt_tokens=10, completion_tokens=5, total_tokens=15, completion_tokens_details=MagicMock(spec=CompletionTokensDetails))
mock_metadata.cost = MagicMock(spec=Cost, total_cost=0.01)
mock_metadata.start_timestamp = datetime.now(timezone.utc)
mock_metadata.end_timestamp = datetime.now(timezone.utc)
# Add other necessary metadata fields used in serialization/saving if any
mock_metadata.model = PROVIDER_CONFIG
mock_metadata.provider = "openai" # Assuming based on config
mock_metadata.kwargs = {}
mock_metadata.task_id = TASK_ID
mock_metadata.pair_index = 0 # Assuming first pair for simplicity
mock_metadata.test_id = PROVIDER_CONFIG
mock_attempt.metadata = mock_metadata
# Configure model_dump to return a serializable dict
mock_attempt.answer = [[0, 1], [1, 0]] # Set the answer directly as it would be after parsing
mock_dump_dict = {
    "answer": [[0, 1], [1, 0]],
    "metadata": {
        # Include essential serializable metadata fields
        "model": mock_metadata.model,
        "provider": mock_metadata.provider,
        "start_timestamp": mock_metadata.start_timestamp.isoformat(),
        "end_timestamp": mock_metadata.end_timestamp.isoformat(),
        # Minimal choices representation
        "choices": [{"index": 0, "message": {"role": "assistant", "content": mock_message.content}}],
        "kwargs": mock_metadata.kwargs,
        # Simplified usage/cost for serialization
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "cost": {"total_cost": 0.01},
        "task_id": mock_metadata.task_id,
        "pair_index": mock_metadata.pair_index,
        "test_id": mock_metadata.test_id
    }
}
mock_attempt.model_dump.return_value = mock_dump_dict
# --- End Mock Attempt --- 

# --- Mock Provider Instance --- 
# Replaces the actual OpenAIAdapter instance via patching ARCTester.init_provider
mock_provider_instance = MagicMock(spec=ProviderAdapter)
# Configure its make_prediction method to return the mock Attempt
mock_provider_instance.make_prediction.return_value = mock_attempt
# REMOVE extract_json_from_response configuration entirely
# --- End Mock Provider ---

@patch('main.ARCTester.init_provider') 
def test_gpt_4o_provider_e2e_mocked(mock_init_provider): # REMOVE mock_backscan_parser arg
    """Runs test by calling ARCTester directly, mocking provider via init_provider."""
    # Configure the mock init_provider to return our mock provider instance
    mock_init_provider.return_value = mock_provider_instance


    # --- Delete existing submission file if it exists --- 
    submission_filename = f"{TASK_ID}.json" 
    try:
        os.remove(submission_filename)
        print(f"\n🧹 Removed existing submission file: {submission_filename}")
    except FileNotFoundError:
        print(f"\nℹ️ No existing submission file found: {submission_filename}")
    # --- End delete --- 
    
    # Instantiate ARCTester with test parameters
    arc_tester = ARCTester(
        config=PROVIDER_CONFIG,
        save_submission_dir=SAVE_DIR, 
        overwrite_submission=True, # Ensure we run even if file exists somehow
        print_submission=True, # Print submission for verification
        num_attempts=NUM_ATTEMPTS,
        retry_attempts=RETRY_ATTEMPTS,
        print_logs=True # Enable logs to see output
    )
   
    # Run the main logic
    try:
        result = arc_tester.generate_task_solution(
            data_dir=DATA_DIR,
            task_id=TASK_ID
        )
        print(f"\n✅ E2E Test completed successfully for {PROVIDER_CONFIG}")
            
        # Assert that the mock provider's make_prediction was called
        mock_provider_instance.make_prediction.assert_called()
        print(f"Mock make_prediction call count: {mock_provider_instance.make_prediction.call_count}")
        
        # Assert on the result or the saved file content
        assert result is not None, "generate_task_solution returned None"
        with open(submission_filename, 'r') as f:
            saved_data = json.load(f)
        expected_answer_from_mock = [[0, 1], [1, 0]] 
        assert saved_data[0]['attempt_1']['answer'] == expected_answer_from_mock, "Saved answer doesn't match expected mock output"
        print(f"Saved Answer: {saved_data[0]['attempt_1']['answer']}")

    except Exception as e:
        pytest.fail(f"ARCTester execution failed: {e}")
    finally:
        # Clean up the created submission file
        if os.path.exists(submission_filename):
            os.remove(submission_filename)
            print(f"\n🧹 Cleaned up submission file: {submission_filename}")