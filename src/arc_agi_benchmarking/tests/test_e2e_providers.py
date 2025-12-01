import sys
import pytest
from unittest.mock import patch, MagicMock
import os
import json
from pathlib import Path

from main import ARCTester 
from arc_agi_benchmarking.adapters import ProviderAdapter # Import base class for type hinting/mocking structure
from arc_agi_benchmarking.schemas import Attempt, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails # Import necessary schemas
from datetime import datetime, timezone # Import datetime

# Constants for the test
PROVIDER_CONFIG = "gpt-4o-2024-11-20"
TASK_ID = "66e6c45b"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "sample" / "tasks"
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
        retry_attempts=RETRY_ATTEMPTS
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


# --- Test for main.py CLI execution --- #

# Helper to setup sys.path for importing main.py from project root
# This assumes tests might be run from various locations (e.g., project root or src directory)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from main import main_cli # Import after sys.path modification

@patch('main.ARCTester.generate_task_solution')
@patch('main.ARCTester.init_provider')
@patch('main.utils.read_models_config') # Patched at the source where it's used by main.py
@patch('main.logging.basicConfig') # To suppress logging output during tests
@patch('main.set_metrics_enabled') # To avoid side effects with metrics
def test_main_cli_execution_and_arctester_mocking(mock_set_metrics_enabled,
                                           mock_logging_basic_config,
                                           mock_read_models_config,
                                           mock_init_provider,
                                           mock_generate_task_solution):
    """Tests that main.py's main_cli function can be called, initializes ARCTester,
    and calls its generate_task_solution method, all with mocks in place."""
    print("\n🧪 Running test_main_cli_execution_and_arctester_mocking")

    # 1. Configure the mock for read_models_config (used by ARCTester in main.py)
    mock_model_cfg = MagicMock()
    mock_model_cfg.provider = "cli_mock_provider"
    mock_model_cfg.model_name = "cli_mock_model"
    mock_model_cfg.kwargs = {}
    mock_model_cfg.pricing = MagicMock(input=0, output=0)
    mock_read_models_config.return_value = mock_model_cfg

    # 2. Configure the mock for init_provider (called by ARCTester in main.py)
    mock_cli_provider_instance = MagicMock(spec=ProviderAdapter)
    mock_init_provider.return_value = mock_cli_provider_instance

    # 3. Configure the mock for generate_task_solution (called by ARCTester in main.py)
    mock_generate_task_solution.return_value = None # Doesn't need to do anything complex

    # 4. Prepare CLI arguments for main_cli
    cli_test_args = [
        "--data_dir", "dummy/cli_data",
        "--task_id", "dummy_cli_task",
        "--config", "dummy_cli_config", # This will be passed to read_models_config
        "--log-level", "CRITICAL", # Keep test console clean
        "--save_submission_dir", "dummy/cli_submissions" # Example argument
    ]

    # 5. Call the main_cli function from main.py
    try:
        main_cli(cli_test_args)
        print("main_cli call completed.")
    except Exception as e:
        pytest.fail(f"main_cli raised an unexpected exception: {e}\nArgs: {cli_test_args}")

    # 6. Assertions
    mock_read_models_config.assert_called_once_with("dummy_cli_config")
    print(f"mock_read_models_config called with: {mock_read_models_config.call_args}")
    
    mock_init_provider.assert_called_once_with("cli_mock_provider")
    print(f"mock_init_provider called with: {mock_init_provider.call_args}")
    
    mock_generate_task_solution.assert_called_once()
    call_args_list = mock_generate_task_solution.call_args_list
    if call_args_list:
        args, kwargs = call_args_list[0]
        assert kwargs.get('data_dir') == "dummy/cli_data", f"Expected data_dir dummy/cli_data, got {kwargs.get('data_dir')}"
        assert kwargs.get('task_id') == "dummy_cli_task", f"Expected task_id dummy_cli_task, got {kwargs.get('task_id')}"
        print(f"mock_generate_task_solution called with kwargs: {kwargs}")
    else:
        pytest.fail("mock_generate_task_solution was not called with expected arguments")

    mock_set_metrics_enabled.assert_called_once_with(False) # Default behavior
    print(f"mock_set_metrics_enabled called with: {mock_set_metrics_enabled.call_args}")
    
    mock_logging_basic_config.assert_called_once()
    print("✅ test_main_cli_execution_and_arctester_mocking completed successfully.")
