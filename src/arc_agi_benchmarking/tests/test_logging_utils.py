"""Tests for structured logging utilities."""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from arc_agi_benchmarking.utils.logging_utils import (
    HumanReadableFormatter,
    LogContext,
    StructuredFormatter,
    get_logger,
    log_with_context,
    setup_logging,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_args(self):
        """Test formatting with message arguments."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Task %s completed in %dms",
            args=("abc123", 500),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "Task abc123 completed in 500ms"

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="An error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError: Test error" in data["exception"]

    def test_format_with_context_fields(self):
        """Test formatting with extra context fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing",
            args=(),
            exc_info=None,
        )
        record.task_id = "task123"
        record.config = "gpt-4o"
        record.duration_ms = 1500
        record.tokens_used = 2000
        record.cost_usd = 0.0045

        output = formatter.format(record)
        data = json.loads(output)

        assert data["task_id"] == "task123"
        assert data["config"] == "gpt-4o"
        assert data["duration_ms"] == 1500
        assert data["tokens_used"] == 2000
        assert data["cost_usd"] == 0.0045

    def test_format_excludes_none_values(self):
        """Test that None values are excluded from output."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.task_id = "task123"
        record.config = None  # Should be excluded

        output = formatter.format(record)
        data = json.loads(output)

        assert data["task_id"] == "task123"
        assert "config" not in data


class TestHumanReadableFormatter:
    """Tests for HumanReadableFormatter."""

    def test_basic_format(self):
        """Test basic human-readable formatting."""
        formatter = HumanReadableFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "test.logger" in output
        assert "Test message" in output

    def test_format_with_context(self):
        """Test formatting with context fields."""
        formatter = HumanReadableFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Task completed",
            args=(),
            exc_info=None,
        )
        record.task_id = "task123"
        record.duration_ms = 1500
        record.cost_usd = 0.0045
        record.tokens_used = 2000

        output = formatter.format(record)

        assert "task_id=task123" in output
        assert "1500ms" in output
        assert "$0.0045" in output
        assert "2000 tokens" in output


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_default(self):
        """Test default logging setup."""
        logger = setup_logging(level="INFO")

        assert logger.name == "arc_agi_benchmarking"
        # Level is set on root logger, app logger inherits via getEffectiveLevel
        assert logger.getEffectiveLevel() == logging.INFO
        # Handlers are on root logger
        assert len(logging.getLogger().handlers) == 1

    def test_setup_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.jsonl"
            logger = setup_logging(level="DEBUG", log_file=log_file)

            # Handlers are on root logger
            assert len(logging.getLogger().handlers) == 2  # Console + file

            # Log something and check file
            logger.info("Test message")

            assert log_file.exists()
            with open(log_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["message"] == "Test message"

    def test_setup_none_level(self):
        """Test that NONE level disables logging."""
        logger = setup_logging(level="NONE")

        # Should not raise but logging should be disabled
        logger.critical("This should not appear")

        # Re-enable for other tests
        logging.disable(logging.NOTSET)

    def test_setup_json_console(self):
        """Test JSON console output."""
        logger = setup_logging(level="INFO", json_console=True)

        # Check that formatter is StructuredFormatter (on root logger)
        assert isinstance(logging.getLogger().handlers[0].formatter, StructuredFormatter)

    def test_quiet_libraries(self):
        """Test that noisy libraries are quieted."""
        setup_logging(level="DEBUG", quiet_libraries=True)

        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level == logging.WARNING

        openai_logger = logging.getLogger("openai")
        assert openai_logger.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_module_name(self):
        """Test getting logger with module name."""
        logger = get_logger("arc_agi_benchmarking.adapters.openai")
        assert logger.name == "arc_agi_benchmarking.adapters.openai"

    def test_get_logger_adds_prefix(self):
        """Test that prefix is added for non-app loggers."""
        logger = get_logger("my_module")
        assert logger.name == "arc_agi_benchmarking.my_module"


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_context_adds_fields(self):
        """Test that context adds fields to log records."""
        setup_logging(level="DEBUG")
        logger = get_logger("test")

        records = []

        class RecordCapture(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger.addHandler(RecordCapture())

        with LogContext(task_id="task123", config="gpt-4o"):
            logger.info("Test message")

        assert len(records) == 1
        assert records[0].task_id == "task123"
        assert records[0].config == "gpt-4o"

    def test_context_restored_after_exit(self):
        """Test that context is properly restored after exiting."""
        setup_logging(level="DEBUG")
        logger = get_logger("test2")

        records = []

        class RecordCapture(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger.addHandler(RecordCapture())

        with LogContext(task_id="task123"):
            logger.info("Inside context")

        logger.info("Outside context")

        assert len(records) == 2
        assert hasattr(records[0], "task_id")
        # Outside context should not have task_id
        assert not hasattr(records[1], "task_id") or records[1].task_id is None


class TestLogWithContext:
    """Tests for log_with_context function."""

    def test_log_with_context(self):
        """Test logging with context fields."""
        setup_logging(level="DEBUG")
        logger = get_logger("test3")

        records = []

        class RecordCapture(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger.addHandler(RecordCapture())

        log_with_context(
            logger,
            logging.INFO,
            "Task completed",
            task_id="task123",
            duration_ms=1500,
            tokens_used=2000,
        )

        assert len(records) == 1
        assert records[0].task_id == "task123"
        assert records[0].duration_ms == 1500
        assert records[0].tokens_used == 2000
