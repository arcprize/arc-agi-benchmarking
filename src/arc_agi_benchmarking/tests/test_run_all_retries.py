import pytest
import asyncio
import logging
from unittest.mock import patch, MagicMock

# Assuming 'cli.run_all' can be imported. This might require PYTHONPATH adjustments
# if 'src' isn't the root or if tests are run from a different context.
# If 'src' is the project root for modules:
# from cli.run_all import run_single_test_wrapper, AsyncRequestRateLimiter
# If 'src' is part of the import path (e.g. 'from src.cli.run_all ...'), adjust as needed.
# For now, sticking to the simpler form based on common pytest setups from project root.
from cli.run_all import run_single_test_wrapper, AsyncRequestRateLimiter

# This class was unused and causing a PytestCollectionWarning. Removing it.
# class TestRetryableException(Exception):
#     """A simple exception to simulate retryable errors for testing."""
#     pass

# Renamed to start with an underscore to avoid pytest collection warnings
class _TestRetryableExceptionInTestScope(Exception):
    """A simple exception defined in the test scope to simulate retryable errors."""
    pass

TEST_DATA_DIR = "data/sample/tasks"

class APICallSimulator:
    """Simulates the behavior of ARCTester.generate_task_solution for testing retries."""
    def __init__(self, fail_n_times: int, exception_to_raise: type[Exception]):
        self.call_count = 0
        self.fail_n_times = fail_n_times
        self.exception_to_raise = exception_to_raise

    def simulate_generate_task_solution(self, data_dir: str, task_id: str):
        self.call_count += 1
        logging.info(f"SIMULATOR: Call #{self.call_count} to generate_task_solution for {task_id}")
        if self.call_count <= self.fail_n_times:
            logging.info(f"SIMULATOR: Simulating failure with {self.exception_to_raise.__name__}")
            raise self.exception_to_raise(f"Simulated API error on attempt {self.call_count}")
        logging.info("SIMULATOR: Simulating success")
        # generate_task_solution doesn't return a value, it just completes or raises.
        return None


@pytest.mark.asyncio
# No @patch decorators on the function itself
async def test_retry_and_eventual_success(caplog): # Only pytest fixtures like caplog
    """
    Tests that tenacity retries on OUR TestRetryableExceptionInTestScope and eventually succeeds,
    using patch as a context manager.
    """
    # Ensure the relevant logger ('cli.run_all') captures WARNING messages
    caplog.set_level(logging.WARNING, logger='cli.run_all') 
    # Set overall capture level low if needed, but WARNING should be enough here.
    # caplog.set_level(logging.DEBUG) 

    config_name = "test_config_retry_success"
    task_id = "test_task_001"
    
    # Use patch as a context manager
    # The target string for EFFECTIVE_RETRYABLE_EXCEPTIONS should be where it's defined and used by tenacity.
    with patch('cli.run_all.EFFECTIVE_RETRYABLE_EXCEPTIONS', (_TestRetryableExceptionInTestScope,)) as _mocked_retry_config:
        # _mocked_retry_config is the new value tuple, not a MagicMock. We don't typically use it directly.
        with patch('cli.run_all.ARCTester') as MockARCTesterClass: # MockARCTesterClass is the mock of the ARCTester class
            mock_arc_instance = MockARCTesterClass.return_value # This is the mock for instances of ARCTester
            
            num_failures_to_simulate = 2
            simulator = APICallSimulator(
                fail_n_times=num_failures_to_simulate,
                exception_to_raise=_TestRetryableExceptionInTestScope
            )
            mock_arc_instance.generate_task_solution.side_effect = simulator.simulate_generate_task_solution

            limiter = AsyncRequestRateLimiter(rate=1000, capacity=1000)

            # Execute the function under test
            result = await run_single_test_wrapper(
                config_name, 
                task_id, 
                limiter,
                data_dir=TEST_DATA_DIR, # DEFAULT_DATA_DIR
                submissions_root="submissions_test_retries", # Changed from save_submission_dir_base
                overwrite_submission=True, # DEFAULT_OVERWRITE_SUBMISSION is False, but True for test clarity
                print_submission=False, # DEFAULT_PRINT_SUBMISSION
                num_attempts=1, # DEFAULT_NUM_ATTEMPTS is 2, using 1 for faster test
                retry_attempts=1  # DEFAULT_RETRY_ATTEMPTS is 2, using 1 for faster test
            )

            # Assertions
            assert result is True, "Wrapper should return True on eventual success."
            
            expected_calls = num_failures_to_simulate + 1
            assert mock_arc_instance.generate_task_solution.call_count == expected_calls, \
                f"generate_task_solution expected {expected_calls} calls, got {mock_arc_instance.generate_task_solution.call_count}"

            # Adjusted log check for tenacity's before_sleep_log output
            tenacity_retry_logs = [rec for rec in caplog.records if 
                                   rec.levelname == "WARNING" and 
                                   "Retrying" in rec.message and 
                                   rec.name == "cli.run_all"] # Check for the logger name used in cli/run_all.py
            assert len(tenacity_retry_logs) == num_failures_to_simulate, \
                f"Expected {num_failures_to_simulate} tenacity retry log(s) from 'cli.run_all' logger, found {len(tenacity_retry_logs)}. Logs: {caplog.text}"

            # Assert the ARCTester class was instantiated for each attempt (initial + retries)
            assert MockARCTesterClass.call_count == expected_calls, \
                f"Expected ARCTester class to be instantiated {expected_calls} times, but was {MockARCTesterClass.call_count}"

    # Optional: Print logs for debugging if the test fails
    # print("\nCaptured Logs for test_retry_and_eventual_success:")
    # for record in caplog.records:
    #     print(f"{record.levelname} - {record.name} - {record.message}")

# We can add more tests here for other scenarios:
# - Test for failure after all retries
# - Test for non-retryable exceptions 

# --- Add new test cases below --- #

# Renamed to start with an underscore
class _NonRetryableTestException(ValueError):
    """A simple non-retryable exception for testing."""
    pass


@pytest.mark.asyncio
async def test_failure_after_all_retries(caplog):
    """
    Tests that tenacity gives up after all configured attempts if the error persists.
    Assumes tenacity is configured for stop_after_attempt(4).
    """
    # Ensure the relevant logger ('cli.run_all') captures WARNING messages
    caplog.set_level(logging.WARNING, logger='cli.run_all')
    # caplog.set_level(logging.DEBUG) 

    config_name = "test_config_persistent_failure"
    task_id = "test_task_002"
    max_attempts_by_tenacity = 4 # From stop_after_attempt(4) in cli/run_all.py

    with patch('cli.run_all.EFFECTIVE_RETRYABLE_EXCEPTIONS', (_TestRetryableExceptionInTestScope,)):
        with patch('cli.run_all.ARCTester') as MockARCTesterClass:
            mock_arc_instance = MockARCTesterClass.return_value
            
            # Simulate failure for all attempts + 1 (to be sure it always fails)
            simulator = APICallSimulator(
                fail_n_times=max_attempts_by_tenacity + 1, 
                exception_to_raise=_TestRetryableExceptionInTestScope
            )
            mock_arc_instance.generate_task_solution.side_effect = simulator.simulate_generate_task_solution

            limiter = AsyncRequestRateLimiter(rate=1000, capacity=1000)
            result = await run_single_test_wrapper(
                config_name, 
                task_id, 
                limiter,
                data_dir=TEST_DATA_DIR,
                submissions_root="submissions_test_retries",
                overwrite_submission=True,
                print_submission=False,
                num_attempts=1,
                retry_attempts=1
            )

            assert result is False, "Wrapper should return False when all retries are exhausted."
            
            # generate_task_solution and ARCTester instantiation should be called max_attempts_by_tenacity times
            assert mock_arc_instance.generate_task_solution.call_count == max_attempts_by_tenacity, \
                f"generate_task_solution expected {max_attempts_by_tenacity} calls, got {mock_arc_instance.generate_task_solution.call_count}"
            assert MockARCTesterClass.call_count == max_attempts_by_tenacity, \
                f"Expected ARCTester class to be instantiated {max_attempts_by_tenacity} times, got {MockARCTesterClass.call_count}"

            # Tenacity logs max_attempts - 1 retries
            expected_retry_logs = max_attempts_by_tenacity - 1
            tenacity_retry_logs = [rec for rec in caplog.records if 
                                   rec.levelname == "WARNING" and "Retrying" in rec.message and rec.name == "cli.run_all"]
            assert len(tenacity_retry_logs) == expected_retry_logs, \
                f"Expected {expected_retry_logs} tenacity retry log(s), found {len(tenacity_retry_logs)}. Logs: {caplog.text}"


@pytest.mark.asyncio
async def test_non_retryable_exception(caplog):
    """
    Tests that tenacity does not retry for exceptions not in its configured list.
    """
    caplog.set_level(logging.INFO)
    config_name = "test_config_non_retryable"
    task_id = "test_task_003"

    # Patch EFFECTIVE_RETRYABLE_EXCEPTIONS to ONLY retry on TestRetryableExceptionInTestScope.
    # So, NonRetryableTestException should not be retried.
    with patch('cli.run_all.EFFECTIVE_RETRYABLE_EXCEPTIONS', (_TestRetryableExceptionInTestScope,)):
        with patch('cli.run_all.ARCTester') as MockARCTesterClass:
            mock_arc_instance = MockARCTesterClass.return_value
            
            simulator = APICallSimulator(
                fail_n_times=1, # Will fail on the first call
                exception_to_raise=_NonRetryableTestException # This exception is not in the patched config
            )
            mock_arc_instance.generate_task_solution.side_effect = simulator.simulate_generate_task_solution

            limiter = AsyncRequestRateLimiter(rate=1000, capacity=1000)
            result = await run_single_test_wrapper(
                config_name, 
                task_id, 
                limiter,
                data_dir=TEST_DATA_DIR,
                submissions_root="submissions_test_retries",
                overwrite_submission=True,
                print_submission=False,
                num_attempts=1,
                retry_attempts=1
            )

            assert result is False, "Wrapper should return False on non-retryable exception."
            
            # generate_task_solution and ARCTester instantiation should be called only once
            assert mock_arc_instance.generate_task_solution.call_count == 1, "generate_task_solution expected 1 call"
            assert MockARCTesterClass.call_count == 1, "Expected ARCTester class to be instantiated 1 time"

            # No tenacity retry logs expected
            tenacity_retry_logs = [rec for rec in caplog.records if 
                                   rec.levelname == "WARNING" and "Retrying" in rec.message and rec.name == "cli.run_all"]
            assert len(tenacity_retry_logs) == 0, f"Expected 0 tenacity retry log(s), found {len(tenacity_retry_logs)}. Logs: {caplog.text}" 
