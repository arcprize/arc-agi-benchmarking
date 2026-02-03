import pytest
import logging
import subprocess
import sys
import os
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path to allow importing main and cli.run_all
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports now possible after path adjustment
from main import ARCTester
from cli.run_all import main, run_single_test_wrapper, AsyncRequestRateLimiter, get_or_create_rate_limiter
from arc_agi_benchmarking.schemas import Attempt # Import Attempt schema

# --- Test ARCTester Logging (main.py) ---

@pytest.mark.parametrize(
    "log_level_arg, expected_debug_present, expected_info_present",
    [
        ("DEBUG", True, True),
        ("INFO", False, True),
        ("WARNING", False, False),
        ("ERROR", False, False),
    ]
)
@patch('main.utils')
@patch('main.ARCTester.init_provider')
def test_arctester_log_levels(
    mock_init_provider, mock_utils, caplog,
    log_level_arg, expected_debug_present, expected_info_present
):
    """
    Tests that ARCTester logs messages at the correct levels based on the configured log level.
    Uses caplog to set the level for the 'main' logger and checks record levels.
    """
    log_level_numeric = getattr(logging, log_level_arg.upper())
    caplog.set_level(log_level_numeric, logger='main')

    # Setup Mocks
    mock_provider_instance = MagicMock()
    
    # Create a mock Attempt object that can be dumped and has necessary attributes
    mock_attempt_obj = MagicMock(spec=Attempt)
    mock_attempt_obj.metadata = MagicMock()
    # Simulate a response that the extractor can handle
    mock_attempt_obj.metadata.choices = [MagicMock(message=MagicMock(content="Valid content for mock parser"))]
    # Define what model_dump should return - a serializable dict
    mock_attempt_obj.model_dump.return_value = {
        "task_id": "mock_task_001", 
        "test_id": "mock_config",
        "pair_index": 0,
        "answer": [[1]], # This should match what the extractor returns
        "metadata": {"usage": {"total_tokens": 10}} # Example metadata
    }
    # Mock make_prediction to return this mock Attempt
    mock_provider_instance.make_prediction.return_value = mock_attempt_obj
    
    mock_init_provider.return_value = mock_provider_instance

    # Mock provider extractor to return a valid answer structure when called
    mock_provider_instance.extract_json_from_response.return_value = [[1]]

    mock_utils.validate_data.return_value = True
    mock_utils.submission_exists.return_value = False
    mock_utils.get_train_pairs_from_task.return_value = [] # Empty training pairs
    mock_utils.get_test_input_from_task.return_value = [MagicMock(input=[[0]])] # Single test pair
    mock_utils.save_submission.return_value = "mock/path/saved.json"
    mock_utils.read_models_config.return_value = MagicMock(provider="mock_provider", model_name="mock_model")

    # Instantiate ARCTester
    arc_tester = ARCTester(
        config="mock_config",
        save_submission_dir="mock_submissions/mock_config",
        overwrite_submission=True,
        print_submission=True, # Keep this True to ensure submission log is attempted
        num_attempts=1,
        retry_attempts=1
    )

    # Call the method that logs
    arc_tester.generate_task_solution(data_dir="mock_data", task_id="mock_task_001")

    # Filter records from the 'main' logger
    main_records = [rec for rec in caplog.records if rec.name == 'main']

    # Assert based on expected presence of levels
    has_debug = any(rec.levelno == logging.DEBUG for rec in main_records)
    has_info = any(rec.levelno == logging.INFO for rec in main_records)
    # We could also check for WARNING/ERROR if we mock scenarios that trigger them

    assert has_debug == expected_debug_present, f"DEBUG logs presence mismatch. Expected={expected_debug_present}, Found={has_debug}. Level set={log_level_arg}"
    assert has_info == expected_info_present, f"INFO logs presence mismatch. Expected={expected_info_present}, Found={has_info}. Level set={log_level_arg}"

# --- Test Orchestrator Logging (cli/run_all.py) ---

def test_orchestrator_rate_limiter_log(caplog):
    """Specifically test if the rate limiter init log is captured."""
    caplog.set_level(logging.WARNING, logger='cli.run_all')
    
    # Ensure caches are clear for this specific test
    from cli import run_all
    run_all.PROVIDER_RATE_LIMITERS.clear()
    run_all.MODEL_CONFIG_CACHE.clear()
    
    # Call the function that logs - unknown provider triggers a WARNING
    _ = get_or_create_rate_limiter("test_provider", {})
    
    # Check if the warning log message exists (unknown providers log a warning about using defaults)
    found = any(rec.getMessage().startswith("No rate limit configuration found for provider 'test_provider'") 
                for rec in caplog.records if rec.name == 'cli.run_all' and rec.levelno == logging.WARNING)
    assert found, f"Rate limiter warning log not found. Logs: {caplog.text}"


@pytest.mark.parametrize(
    # Parameter is now just log_level_arg and expected_info_present
    "log_level_arg, expected_info_present",
    [
        ("DEBUG", True),  # DEBUG level should show INFO logs
        ("INFO", True),   # INFO level should show INFO logs
        ("WARNING", False), # WARNING level should filter out INFO logs
        ("ERROR", False),   # ERROR level should filter out INFO logs
    ]
)
@pytest.mark.asyncio
@patch('cli.run_all.run_single_test_wrapper')
@patch('cli.run_all.read_provider_rate_limits')
@patch('cli.run_all.get_model_config')
@patch('builtins.open')
@patch('cli.run_all.exit', side_effect=lambda code: None)
async def test_orchestrator_log_levels(
    mock_exit, mock_open, mock_get_model_config, mock_read_limits, mock_run_wrapper, caplog,
    # Removed expected_debug_present
    log_level_arg, expected_info_present
):
    """
    Tests that the cli/run_all.py orchestrator logs INFO messages correctly
    based on the configured log level. Uses caplog to set the level.
    (DEBUG messages aren't currently tested as none are directly logged by the main orchestrator path).
    """
    log_level_numeric = getattr(logging, log_level_arg.upper())
    caplog.set_level(log_level_numeric, logger='cli.run_all')

    # Mock setup
    mock_run_wrapper.return_value = True
    mock_read_limits.return_value = {"mock_provider": {"rate": 100, "period": 60}}
    mock_model_config = MagicMock(provider="mock_provider")
    mock_get_model_config.return_value = mock_model_config
    mock_file_handle = MagicMock()
    file_lines = ["task1\n", "task2\n"]
    mock_file_handle.__enter__.return_value.__iter__.return_value = iter(file_lines)
    mock_open.return_value = mock_file_handle
    # Call the main async function
    await main(
        task_list_file="dummy.txt",
        config_to_test="mock_config",
        data_dir="dummy_data",
        save_submission_dir="dummy_submissions",
        overwrite_submission=False,
        print_submission=False,
        num_attempts=1,
        retry_attempts=1,
        logs_base_dir=Path("test_logs")
    )

    # Filter records from the 'cli.run_all' logger
    cli_records = [rec for rec in caplog.records if rec.name == 'cli.run_all']

    # Assert based on expected presence of INFO level logs
    has_info = any(rec.levelno == logging.INFO for rec in cli_records)

    assert has_info == expected_info_present, f"INFO logs presence mismatch. Expected={expected_info_present}, Found={has_info}. Level set={log_level_arg}"

    # We could add an assertion here that DEBUG logs are *never* found in this test scenario
    has_debug = any(rec.levelno == logging.DEBUG for rec in cli_records)
    assert not has_debug, f"Unexpected DEBUG logs found for level {log_level_arg}"


@pytest.mark.asyncio
@patch('cli.run_all.run_single_test_wrapper', return_value=True)
@patch('cli.run_all.read_provider_rate_limits', return_value={})
@patch('cli.run_all.get_model_config')
@patch('builtins.open')
@patch('cli.run_all.exit', side_effect=lambda code: None) # Prevent SystemExit
async def test_orchestrator_log_level_none(mock_exit, mock_open, mock_get_model_config, mock_read_limits, mock_run_wrapper, caplog):
    """
    Tests that --log-level NONE effectively silences logging output.
    Uses caplog to check for absence of records.
    """
    # Set root logger high first, then specific logger if needed (though NONE handler should stop it)
    # logging.getLogger().setLevel(logging.CRITICAL + 1) # REMOVED - Rely on cli/run_all setup for NONE
    # logging.getLogger('cli.run_all').setLevel(logging.CRITICAL + 1) # REMOVED
    # Caplog captures everything regardless of handler levels, check records
    caplog.set_level(logging.DEBUG) 

    mock_model_config = MagicMock(provider="mock_provider")
    mock_get_model_config.return_value = mock_model_config
    
    # Make mock_open return an object that supports context management AND iteration
    mock_file_handle = MagicMock()
    file_lines = ["task1\n"]
    mock_file_handle.__enter__.return_value.__iter__.return_value = iter(file_lines)
    mock_open.return_value = mock_file_handle

    # Simulate the effect of --log-level NONE on the logger config
    # We already set the levels above, no need to simulate argparse/basicConfig here # REMOVED Comment
    # Setting levels directly as the test runs outside __main__ where basicConfig is called
    logging.getLogger('cli.run_all').setLevel(logging.CRITICAL + 1)
    # Also set root logger high in case of propagation or library logging
    logging.getLogger().setLevel(logging.CRITICAL + 1)
    
    await main(
        task_list_file="dummy.txt",
        config_to_test="mock_config",
        data_dir="dummy_data",
        save_submission_dir="dummy_submissions",
        overwrite_submission=False,
        print_submission=False,
        num_attempts=1,
        retry_attempts=1,
        logs_base_dir=Path("test_logs")
    )

    assert len(caplog.records) == 0, f"Expected no log records, but got {len(caplog.records)}: {caplog.text}"

    # Restore levels after test if needed, though pytest usually isolates
    # logging.getLogger('cli.run_all').setLevel(logging.NOTSET)
    # logging.getLogger().setLevel(logging.NOTSET)
