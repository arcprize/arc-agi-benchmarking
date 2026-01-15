import asyncio
import os
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
import contextvars

import sys
import logging

from dotenv import load_dotenv
load_dotenv()

# Add the project root directory and the src directory to sys.path
# This allows cli/run_all.py to import 'main' and 'src' from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root) # For importing main.py

if src_dir not in sys.path:
    sys.path.insert(0, src_dir) # For importing arc_agi_benchmarking from src

from main import ARCTester
from arc_agi_benchmarking.utils.task_utils import read_models_config, read_provider_rate_limits
from arc_agi_benchmarking.utils.rate_limiter import AsyncRequestRateLimiter
from arc_agi_benchmarking.utils.metrics import set_metrics_enabled, set_metrics_filename_prefix
from arc_agi_benchmarking.utils.preflight import run_preflight
from arc_agi_benchmarking.utils.logging_utils import setup_logging, StructuredFormatter

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

# Context for per-task logging
LOG_CONFIG_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar("log_config", default=None)
LOG_TASK_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar("log_task", default=None)

# Extend log records with config/task context
_ORIGINAL_RECORD_FACTORY = logging.getLogRecordFactory()

def _record_factory(*args, **kwargs):
    record = _ORIGINAL_RECORD_FACTORY(*args, **kwargs)
    record.config_name = LOG_CONFIG_CTX.get()
    record.task_id = LOG_TASK_CTX.get()
    return record


# Apply the record factory globally (once)
logging.setLogRecordFactory(_record_factory)

# Attempt to import provider-specific exceptions for retrying
try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AnthropicRateLimitError = None
    logger.warning("Anthropic SDK not installed or RateLimitError not found. Retries for Anthropic rate limits will not be specific.")

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = None
    logger.warning("OpenAI SDK not installed or RateLimitError not found. Retries for OpenAI rate limits will not be specific.")

try:
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
except ImportError:
    GoogleResourceExhausted = None
    logger.warning("Google API Core SDK not installed or ResourceExhausted not found. Retries for Google rate limits will not be specific.")

_RETRYABLE_EXCEPTIONS_CLASSES = tuple(
    exc for exc in (AnthropicRateLimitError, OpenAIRateLimitError, GoogleResourceExhausted) if exc is not None
)

if not _RETRYABLE_EXCEPTIONS_CLASSES:
    logger.warning(
        "No specific retryable exception classes were successfully imported. "
        "Retries might not trigger as expected or might catch too broadly if fallback to general Exception is used."
    )
    EFFECTIVE_RETRYABLE_EXCEPTIONS = (Exception,)
else:
    EFFECTIVE_RETRYABLE_EXCEPTIONS = _RETRYABLE_EXCEPTIONS_CLASSES

# Default values
DEFAULT_RATE_LIMIT_RATE = 400
DEFAULT_RATE_LIMIT_PERIOD = 60

# --- Configuration ---
# Default model configuration to test if not provided via CLI.
# This is a name from your models.yml file.
DEFAULT_MODEL_CONFIG = "gpt-4o-2024-11-20"

DEFAULT_DATA_DIR = "data/sample/tasks"
DEFAULT_SAVE_SUBMISSION_DIR = "submissions"
DEFAULT_OVERWRITE_SUBMISSION = False
DEFAULT_PRINT_SUBMISSION = False # ARCTester specific: whether it logs submission content
DEFAULT_NUM_ATTEMPTS = 2
DEFAULT_RETRY_ATTEMPTS = 2
# DEFAULT_PRINT_LOGS = False # This is now controlled by the global log level

# --- Globals for Orchestrator ---
PROVIDER_RATE_LIMITERS: Dict[str, AsyncRequestRateLimiter] = {}
MODEL_CONFIG_CACHE: Dict[str, Any] = {}

def get_model_config(config_name: str):
    if config_name not in MODEL_CONFIG_CACHE:
        MODEL_CONFIG_CACHE[config_name] = read_models_config(config_name)
    return MODEL_CONFIG_CACHE[config_name]

def get_or_create_rate_limiter(provider_name: str, all_provider_limits: Dict) -> AsyncRequestRateLimiter:
    if provider_name not in PROVIDER_RATE_LIMITERS:
        if provider_name not in all_provider_limits:
            logger.warning(f"No rate limit configuration found for provider '{provider_name}' in provider_config.yml. Using default ({DEFAULT_RATE_LIMIT_RATE} req/{DEFAULT_RATE_LIMIT_PERIOD}s).")
            default_config_rate = DEFAULT_RATE_LIMIT_RATE
            default_config_period = DEFAULT_RATE_LIMIT_PERIOD
            actual_rate_for_limiter = default_config_rate / default_config_period
            actual_capacity_for_limiter = max(1.0, actual_rate_for_limiter)
        else:
            limits = all_provider_limits[provider_name]
            config_rate = limits['rate']
            config_period = limits['period']
            if config_period <= 0:
                actual_rate_for_limiter = float('inf')
                actual_capacity_for_limiter = float('inf')
                logger.warning(f"Provider '{provider_name}' has period <= 0 in config. Treating as unconstrained.")
            else:
                calculated_rps = config_rate / config_period
                actual_rate_for_limiter = calculated_rps
                actual_capacity_for_limiter = max(1.0, calculated_rps)
        logger.info(f"Initializing rate limiter for provider '{provider_name}' with rate={actual_rate_for_limiter:.2f} req/s, capacity={actual_capacity_for_limiter:.2f}.")
        PROVIDER_RATE_LIMITERS[provider_name] = AsyncRequestRateLimiter(rate=actual_rate_for_limiter, capacity=actual_capacity_for_limiter)
    return PROVIDER_RATE_LIMITERS[provider_name]

async def run_single_test_wrapper(config_name: str, task_id: str, limiter: AsyncRequestRateLimiter,
                                  data_dir: str, save_submission_dir: str,
                                  overwrite_submission: bool, print_submission: bool,
                                  num_attempts: int, retry_attempts: int,
                                  logs_base_dir: Path) -> bool: # removed print_logs
    logger.info(f"[Orchestrator] Queuing task: {task_id}, config: {config_name}")

    # Apply tenacity retry decorator directly to the synchronous function
    # The logger passed to before_sleep_log is the module-level logger of cli.run_all
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(EFFECTIVE_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _synchronous_task_execution_attempt_with_tenacity():
        logger.debug(f"[Thread-{task_id}-{config_name}] Spawning ARCTester (Executing attempt)...")

        # Configure per-task JSONL file logging: <logs_base_dir>/<config>/<task_id>/openai.jsonl
        log_dir = logs_base_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{task_id}.jsonl"

        # Ensure only records for this task/config reach this file handler
        class _TaskFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return record.config_name == config_name and record.task_id == task_id

        # Set context vars so every log record (including library logs) carries config/task ids
        config_token = LOG_CONFIG_CTX.set(config_name)
        task_token = LOG_TASK_CTX.set(task_id)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        file_handler.addFilter(_TaskFilter())
        logging.getLogger().addHandler(file_handler)
        logging.getLogger("openai").setLevel(logging.INFO)
        logger.info(f"[Thread-{task_id}-{config_name}] OpenAI SDK logs will be written to {log_path}")

        arc_solver = ARCTester(
            config=config_name,
            save_submission_dir=save_submission_dir,
            overwrite_submission=overwrite_submission,
            print_submission=print_submission, # This ARCTester arg controls if it logs submission content
            num_attempts=num_attempts,
            retry_attempts=retry_attempts # ARCTester's internal retries
            # print_logs removed from ARCTester instantiation
        )
        logger.debug(f"[Thread-{task_id}-{config_name}] Starting generate_task_solution...")
        try:
            arc_solver.generate_task_solution(
                data_dir=data_dir,
                task_id=task_id
            )
            logger.debug(f"[Thread-{task_id}-{config_name}] Task attempt completed successfully.")
        finally:
            LOG_CONFIG_CTX.reset(config_token)
            LOG_TASK_CTX.reset(task_token)
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

    try:
        async with limiter:
            logger.info(f"[Orchestrator] Rate limiter acquired for: {config_name}. Executing task {task_id} with tenacity retries...")
            await asyncio.to_thread(_synchronous_task_execution_attempt_with_tenacity)
        
        logger.info(f"[Orchestrator] Successfully processed (with tenacity retries if any): {config_name} / {task_id}")
        return True
    except Exception as e:
        logger.error(f"[Orchestrator] Failed to process (after all tenacity retries or due to non-retryable error): {config_name} / {task_id}. Error: {type(e).__name__} - {e}", exc_info=True)
        return False

async def main(task_list_file: Optional[str],
               config_to_test: str,
               data_dir: str, save_submission_dir: str,
               overwrite_submission: bool, print_submission: bool,
               num_attempts: int, retry_attempts: int,
               logs_base_dir: Path) -> int: # Added return type hint
    # Basic logging setup is now done in if __name__ == "__main__"
    
    start_time = time.perf_counter()
    logger.info("Starting ARC Test Orchestrator...")
    logger.info(f"Testing with model configuration: {config_to_test}")

    task_ids: List[str] = []
    try:
        if task_list_file:
            logger.info(f"Using task list file: {task_list_file}")
            with open(task_list_file, 'r') as f:
                task_ids = [line.strip() for line in f if line.strip()]
            if not task_ids:
                logger.error(f"No task IDs found in {task_list_file}. Exiting.")
                return 1 # Return an error code
            logger.info(f"Loaded {len(task_ids)} task IDs from {task_list_file}.")
        else:
            logger.info(f"No task list file provided. Inferring task list from data directory: {data_dir}")
            task_ids = [
                os.path.splitext(fname)[0] 
                for fname in os.listdir(data_dir) 
                if os.path.isfile(os.path.join(data_dir, fname)) and fname.endswith('.json')
            ]
            if not task_ids:
                logger.error(f"No task files (.json) found in {data_dir}. Exiting.")
                return 1 # Return an error code
            logger.info(f"Found {len(task_ids)} task IDs in {data_dir}.")

    except FileNotFoundError:
        if task_list_file:
            logger.error(f"Task list file not found: {task_list_file}. Exiting.")
        else: # Should not happen if data_dir is validated by argparse, but as a safeguard
            logger.error(f"Data directory not found: {data_dir}. Exiting.")
        return 1 # Return an error code
    except Exception as e:
        logger.error(f"Error loading tasks: {e}", exc_info=True)
        return 1 # Return an error code

    all_jobs_to_run: List[Tuple[str, str]] = []
    for task_id in task_ids:
        all_jobs_to_run.append((config_to_test, task_id))
    
    if not all_jobs_to_run:
        logger.warning("No jobs to run (check config_to_test and task list). Exiting.")
        return 1 # Return an error code
    
    logger.info(f"Total jobs to process: {len(all_jobs_to_run)}")

    try:
        all_provider_limits = read_provider_rate_limits()
        logger.info(f"Loaded rate limits from provider_config.yml for providers: {list(all_provider_limits.keys())}")
    except FileNotFoundError:
        logger.warning("provider_config.yml not found. Using default rate limits (400 req/60s per provider).")
        all_provider_limits = {}
    except Exception as e:
        logger.warning(f"Error reading or parsing provider_config.yml: {e}. Using default rate limits.")
        all_provider_limits = {}

    async_tasks_to_execute = []
    for config_name, task_id in all_jobs_to_run:
        try:
            model_config_obj = get_model_config(config_name)
            provider_name = model_config_obj.provider
            limiter = get_or_create_rate_limiter(provider_name, all_provider_limits)
            async_tasks_to_execute.append(run_single_test_wrapper(
                config_name, task_id, limiter,
                data_dir, save_submission_dir,
                overwrite_submission, print_submission, 
                num_attempts, retry_attempts,
                logs_base_dir
            ))
        except ValueError as e: # Specific error for model config issues
            logger.error(f"Skipping config '{config_name}' for task '{task_id}' due to model config error: {e}")
        except Exception as e: # General error for other setup issues
            logger.error(f"Unexpected error setting up task for '{config_name}', '{task_id}': {e}", exc_info=True)

    if not async_tasks_to_execute:
        logger.warning("No tasks could be prepared for execution. Exiting.")
        return 1 # Return an error code

    logger.info(f"Executing {len(async_tasks_to_execute)} tasks concurrently...")
    results = await asyncio.gather(*async_tasks_to_execute, return_exceptions=True)

    successful_runs = sum(1 for r in results if r is True)
    orchestrator_level_failures = sum(1 for r in results if r is False or isinstance(r, Exception))

    logger.info("--- Orchestrator Summary ---")
    exit_code = 0 # Default to success
    if orchestrator_level_failures == 0:
        logger.info(f"✨ All {successful_runs} test configurations completed successfully by the orchestrator.")
    else:
        logger.error(f"💥 {orchestrator_level_failures} out of {len(results)} test configurations failed or encountered errors during orchestration.")
        for i, res in enumerate(results):
            original_job_config, original_job_task_id = all_jobs_to_run[i] # Get original job details
            if isinstance(res, Exception):
                logger.error(f"  - Error for {original_job_config}/{original_job_task_id}: {type(res).__name__} - {str(res)}", exc_info=True)
            elif res is False: # Wrapper reported failure
                logger.warning(f"  - Failure reported by wrapper for {original_job_config}/{original_job_task_id} (check ARCTester logs for this task/config)")
        exit_code = 1 # Indicate failure
    
    logger.info("Note: Individual task success/failure is logged by ARCTester within its own logger (main.py's logger).")
    logger.info("Orchestrator failure indicates an issue with running the ARCTester task itself or an unhandled exception in the wrapper.")
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    logger.info("--- Orchestrator Timing ---")
    logger.info(f"Total execution time for cli/run_all.py: {total_duration:.2f} seconds")
    
    return exit_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC tasks concurrently. Tasks can be specified via a task list file or inferred from a data directory.")
    parser.add_argument(
        "--task_list_file", 
        type=str, 
        default=None, # Default to None, indicating it's optional
        required=False,
        help="Optional path to a .txt file containing task IDs, one per line. If not provided, tasks are inferred from all .json files in --data_dir."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_MODEL_CONFIG,
        help=f"Model configuration name to test (from models.yml). Defaults to: {DEFAULT_MODEL_CONFIG}"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Data set directory to run. If --task_list_file is not used, .json task files are inferred from here. Defaults to {DEFAULT_DATA_DIR}"
    )
    parser.add_argument(
        "--save_submission_dir", "--submissions-root",
        dest="save_submission_dir",
        type=str,
        default=DEFAULT_SAVE_SUBMISSION_DIR,
        help=f"Folder to save submissions under (alias: --submissions-root for backward compatibility). Defaults to {DEFAULT_SAVE_SUBMISSION_DIR}"
    )
    parser.add_argument(
        "--overwrite_submission",
        action="store_true", # Defaults to False if not present
        help=f"Overwrite submissions if they already exist. Defaults to {DEFAULT_OVERWRITE_SUBMISSION}"
    )
    parser.add_argument(
        "--print_submission", # This flag is for ARCTester to log submission content
        action="store_true", # Defaults to False if not present
        help=f"Enable ARCTester to log final submission content (at INFO level). Defaults to {DEFAULT_PRINT_SUBMISSION}"
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=DEFAULT_NUM_ATTEMPTS,
        help=f"Number of attempts for each prediction by ARCTester. Defaults to {DEFAULT_NUM_ATTEMPTS}"
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help=f"Number of internal retry attempts by ARCTester for failed predictions. Defaults to {DEFAULT_RETRY_ATTEMPTS}"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        help="Set the logging level for the orchestrator and ARCTester (default: INFO). Use NONE to disable logging."
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true", # Defaults to False if not present
        help="Enable metrics collection and dumping (disabled by default)."
    )
    parser.add_argument(
        "--logs-base-dir",
        type=str,
        default="logs",
        help="Base directory for JSONL logs. Per-task logs go to <base>/<config>/<task_id>/openai.jsonl (default: logs)."
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight validation checks (not recommended for production runs)."
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=None,
        help="Maximum estimated cost in USD. Abort if estimated cost exceeds this limit."
    )

    args = parser.parse_args()

    # Set metrics enabled status based on CLI arg
    set_metrics_enabled(args.enable_metrics)

    # Configure structured logging for the entire application
    setup_logging(level=args.log_level, quiet_libraries=True)

    config_name = args.config.strip() if args.config else DEFAULT_MODEL_CONFIG
    if not config_name:
        config_name = DEFAULT_MODEL_CONFIG
        logger.info(f"No config provided or empty, using default: {config_name}")
    if "," in config_name:
        logger.error("run_all supports one model config per invocation. Please invoke cli/run_all.py separately for each config.")
        sys.exit(1)

    # --- Set metrics filename prefix based on the model config being run --- 
    if args.enable_metrics:
        provider_name = "unknown_provider"
        try:
            first_config_obj = get_model_config(config_name)
            provider_name = first_config_obj.provider
        except Exception: 
            logger.warning(f"Could not determine provider for metrics filename from config: {config_name or 'N/A'}")
        
        prefix = f"{provider_name}_{config_name}"
        set_metrics_filename_prefix(prefix)
        logger.info(f"Metrics enabled. Filename prefix set to: {prefix}")
    # ----------------------------------------------------------------------------

    # Resolve logs base dir; if relative, anchor to project root for consistency
    logs_base_dir = Path(args.logs_base_dir)
    if not logs_base_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        logs_base_dir = (project_root / logs_base_dir).resolve()

    # --- Preflight validation ---
    if not args.skip_preflight:
        logger.info("Running preflight validation...")
        preflight_report = run_preflight(
            config_name=config_name,
            data_dir=args.data_dir,
            output_dir=args.save_submission_dir,
            num_attempts=args.num_attempts,
        )
        print(preflight_report)

        if not preflight_report.all_passed:
            logger.error("Preflight validation failed. Use --skip-preflight to bypass (not recommended).")
            sys.exit(1)

        # Check cost limit if specified
        if args.cost_limit is not None and preflight_report.cost_estimate:
            if preflight_report.cost_estimate.estimated_cost > args.cost_limit:
                logger.error(
                    f"Estimated cost (${preflight_report.cost_estimate.estimated_cost:.2f}) "
                    f"exceeds limit (${args.cost_limit:.2f}). Aborting."
                )
                sys.exit(1)
            logger.info(
                f"Cost check passed: ${preflight_report.cost_estimate.estimated_cost:.2f} "
                f"<= ${args.cost_limit:.2f} limit"
            )
    else:
        logger.warning("Preflight validation skipped (--skip-preflight flag set)")
    # --- End preflight validation ---

    # Ensure `main` returns an exit code which is then used by sys.exit
    exit_code_from_main = asyncio.run(main(
        task_list_file=args.task_list_file,
        config_to_test=config_name,
        data_dir=args.data_dir,
        save_submission_dir=args.save_submission_dir,
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts,
        retry_attempts=args.retry_attempts,
        logs_base_dir=logs_base_dir
    ))
    
    sys.exit(exit_code_from_main) 
