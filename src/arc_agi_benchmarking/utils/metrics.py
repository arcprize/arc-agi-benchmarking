import time
import functools
import atexit
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Callable
import datetime # Added for timestamp

# --- Global Storage & Control ---
_timing_data: List[Dict[str, Any]] = []
_output_dir = Path(os.environ.get("METRICS_OUTPUT_DIR", "metrics_output"))
METRICS_ENABLED = False  # Metrics are disabled by default
_filename_prefix = "" # Added global for dynamic filename prefix


def set_metrics_enabled(enabled: bool):
    """Enable or disable metrics collection globally."""
    global METRICS_ENABLED
    METRICS_ENABLED = enabled

# --- Timing ---

def timeit(func: Callable) -> Callable:
    """Decorator to measure execution time of a function, if METRICS_ENABLED is True."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not METRICS_ENABLED:
            return func(*args, **kwargs)

        start_time = time.perf_counter()
        start_timestamp = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if METRICS_ENABLED: # Re-check in case it was disabled during func execution
                end_time = time.perf_counter()
                end_timestamp = time.time()
                duration = end_time - start_time
                _timing_data.append({
                    "function_name": func.__name__,
                    "module": func.__module__,
                    "start_time_ns": int(start_time * 1e9),
                    "end_time_ns": int(end_time * 1e9),
                    "duration_ms": duration * 1000,
                    "start_timestamp_utc": start_timestamp,
                    "end_timestamp_utc": end_timestamp,
                })
    return wrapper

def get_timing_data() -> List[Dict[str, Any]]:
    """Returns a copy of the collected timing data."""
    return list(_timing_data)

def dump_timing(filename: str = "metrics_timing.csv"):
    """Saves the collected timing data to a CSV file in the configured output directory, if METRICS_ENABLED is True."""
    if not METRICS_ENABLED:
        return
    if not _timing_data:
        print("No timing data collected.")
        return

    # Ensure the main output directory exists
    try:
        _output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating metrics directory {_output_dir.resolve()}: {e}")
        return # Cannot proceed if directory can't be created

    output_path = _output_dir / filename # Construct full path using _output_dir and filename

    # Check if _timing_data is not empty before accessing keys
    if not _timing_data:
        print("Internal state error: _timing_data became empty before processing.")
        return

    fieldnames = list(_timing_data[0].keys()) # Ensure it's a list
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_timing_data)
        # print(f"Timing metrics saved to {output_path.resolve()}") # Optional: for debugging
    except Exception as e:
        print(f"Error saving timing metrics to {output_path.resolve()}: {e}")

# --- Automatic Dumping ---
def _dump_all():
    """Function called by atexit to dump timing metrics, if METRICS_ENABLED is True."""
    if not METRICS_ENABLED:
        return
    
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_filename_suffix = "_timing.csv" # Define a common suffix

    if _filename_prefix:
        # Ensure prefix is filesystem-safe (basic example, might need more robust cleaning)
        safe_prefix = "".join(c for c in _filename_prefix if c.isalnum() or c in ('_', '-')).rstrip()
        filename_for_atexit = f"{safe_prefix}_{timestamp_str}{base_filename_suffix}"
    else:
        filename_for_atexit = f"{timestamp_str}_default{base_filename_suffix}"
        
    dump_timing(filename=filename_for_atexit) # Pass only the filename

atexit.register(_dump_all)

# --- Setter for Filename Prefix ---
def set_metrics_filename_prefix(prefix: str):
    """Set a prefix for the automatically generated metrics filename."""
    global _filename_prefix
    _filename_prefix = prefix

# --- Optional: Resetting (useful for testing) ---
def reset_metrics():
    """Clears timing metrics and resets METRICS_ENABLED to its default (False)."""
    global _timing_data, METRICS_ENABLED
    _timing_data = []
    METRICS_ENABLED = False 