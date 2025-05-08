import pytest
import time
import csv
import json
import os
from pathlib import Path
from arc_agi_benchmarking.utils import metrics

# --- Fixtures ---

@pytest.fixture(autouse=True)
def reset_metrics_fixture():
    """Automatically reset metrics and other relevant global state before each test."""
    metrics.reset_metrics()
    # Also reset rate limiter cache from cli/run_all.py
    try:
        from cli import run_all
        run_all.PROVIDER_RATE_LIMITERS.clear()
        run_all.MODEL_CONFIG_CACHE.clear()
    except ImportError:
        pass # Ignore if cli.run_all not available (e.g. running only metrics tests)
    # No yield needed, just run before

@pytest.fixture
def set_metrics_output_dir(tmp_path, monkeypatch):
    """Fixture to set the metrics output directory to a temporary path."""
    temp_dir = tmp_path / "test_metrics_output"
    monkeypatch.setattr(metrics, "_output_dir", temp_dir)
    # Also set the env var in case it's read again, though direct patch is better
    monkeypatch.setenv("METRICS_OUTPUT_DIR", str(temp_dir))
    return temp_dir

# --- Test Functions ---

@metrics.timeit
def _dummy_timed_function(delay: float = 0.01):
    """A simple function to test the timeit decorator."""
    time.sleep(delay)
    return "done"

@metrics.timeit
def _dummy_timed_function_error():
    """A timed function that raises an error."""
    raise ValueError("Something went wrong")

def test_timeit_decorator():
    """Test that the @timeit decorator records function execution."""
    metrics.set_metrics_enabled(True)
    result = _dummy_timed_function(delay=0.02)
    assert result == "done"

    timing_data = metrics.get_timing_data()
    assert len(timing_data) == 1
    record = timing_data[0]

    assert record["function_name"] == "_dummy_timed_function"
    assert record["module"] == __name__ # Should be the module where the func is defined
    assert isinstance(record["duration_ms"], float)
    assert record["duration_ms"] > 15 # Should be roughly 20ms, allow some buffer
    assert record["duration_ms"] < 50 # Sanity check upper bound
    assert isinstance(record["start_time_ns"], int)
    assert isinstance(record["end_time_ns"], int)
    assert record["end_time_ns"] > record["start_time_ns"]
    assert isinstance(record["start_timestamp_utc"], float)
    assert isinstance(record["end_timestamp_utc"], float)

    # Call again
    _dummy_timed_function(delay=0.01)
    timing_data = metrics.get_timing_data()
    assert len(timing_data) == 2

def test_timeit_decorator_exception():
    """Test that @timeit still records timing info if the function errors."""
    metrics.set_metrics_enabled(True)
    with pytest.raises(ValueError, match="Something went wrong"):
        _dummy_timed_function_error()

    timing_data = metrics.get_timing_data()
    assert len(timing_data) == 1
    record = timing_data[0]
    assert record["function_name"] == "_dummy_timed_function_error"
    assert record["module"] == __name__
    assert record["duration_ms"] >= 0

def test_get_timing_data_returns_copy():
    """Test that get_timing_data returns a copy, not the internal list."""
    metrics.set_metrics_enabled(True)
    _dummy_timed_function(delay=0.001)
    data1 = metrics.get_timing_data()
    assert len(data1) == 1
    data1.append({"fake": "data"})

    data2 = metrics.get_timing_data()
    assert len(data2) == 1 # Should not have the appended fake data
    assert data1 != data2

def test_dump_timing(set_metrics_output_dir):
    """Test dumping timing data to a CSV file."""
    metrics.set_metrics_enabled(True)
    _dummy_timed_function(delay=0.01)
    _dummy_timed_function(delay=0.02)

    output_dir = set_metrics_output_dir
    expected_file = output_dir / "metrics_timing.csv"

    metrics.dump_timing()

    assert expected_file.exists()
    with open(expected_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) > 5 # Check header exists and has expected columns
        assert "function_name" in header
        assert "duration_ms" in header
        rows = list(reader)
        assert len(rows) == 2 # Two function calls were made
        assert rows[0][header.index("function_name")] == "_dummy_timed_function"
        assert float(rows[0][header.index("duration_ms")]) > 5

def test_dump_no_timing_data(set_metrics_output_dir, capsys):
    """Test dumping when no timing data has been collected."""
    metrics.set_metrics_enabled(True)
    output_dir = set_metrics_output_dir
    expected_file = output_dir / "metrics_timing.csv"

    metrics.dump_timing()

    assert not expected_file.exists()
    captured = capsys.readouterr()
    assert "No timing data collected." in captured.out

def test_reset_metrics():
    """Test the reset_metrics function."""
    metrics.set_metrics_enabled(True)
    _dummy_timed_function(delay=0.001)

    assert len(metrics.get_timing_data()) == 1

    metrics.reset_metrics() # Called by fixture, but test explicitly too

    assert len(metrics.get_timing_data()) == 0

def test_metrics_disabled():
    """Test that no metrics are collected when METRICS_ENABLED is False."""
    # Ensure metrics are disabled (should be by default fixture, but double-check)
    metrics.set_metrics_enabled(False)

    # Call timed function and counter
    result = _dummy_timed_function(delay=0.01)
    assert result == "done" # Function should still run

    # Assert that no data was collected
    timing_data = metrics.get_timing_data()
    assert len(timing_data) == 0

# Note: Testing the actual execution via atexit is complex and often skipped
# in favor of directly testing the dump functions. 