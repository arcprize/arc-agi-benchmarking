"""Tests for progress tracking utilities."""

import time

import pytest

from arc_agi_benchmarking.utils.progress import (
    ProgressStats,
    ProgressTracker,
    format_cost,
    format_duration,
)


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_seconds(self):
        """Test formatting seconds only."""
        assert format_duration(45) == "45s"

    def test_format_minutes_seconds(self):
        """Test formatting minutes and seconds."""
        assert format_duration(125) == "2m 5s"

    def test_format_hours_minutes_seconds(self):
        """Test formatting hours, minutes, seconds."""
        assert format_duration(3725) == "1h 2m 5s"

    def test_format_days(self):
        """Test formatting days."""
        assert format_duration(90061) == "1d 1h 1m 1s"

    def test_format_none(self):
        """Test formatting None value."""
        assert format_duration(None) == "calculating..."

    def test_format_negative(self):
        """Test formatting negative value."""
        assert format_duration(-1) == "unknown"

    def test_format_zero(self):
        """Test formatting zero."""
        assert format_duration(0) == "0s"


class TestFormatCost:
    """Tests for format_cost function."""

    def test_format_dollars(self):
        """Test formatting dollar amounts."""
        assert format_cost(12.50) == "$12.50"

    def test_format_cents(self):
        """Test formatting cent amounts."""
        assert format_cost(0.50) == "$0.50"

    def test_format_small_amounts(self):
        """Test formatting very small amounts."""
        assert format_cost(0.0045) == "$0.0045"

    def test_format_zero(self):
        """Test formatting zero."""
        assert format_cost(0) == "$0.0000"


class TestProgressStats:
    """Tests for ProgressStats dataclass."""

    def test_percent_complete_empty(self):
        """Test percent complete with no tasks done."""
        stats = ProgressStats(total_tasks=100)
        assert stats.percent_complete == 0.0

    def test_percent_complete_partial(self):
        """Test percent complete with some tasks done."""
        stats = ProgressStats(total_tasks=100, completed_tasks=25)
        assert stats.percent_complete == 25.0

    def test_percent_complete_full(self):
        """Test percent complete with all tasks done."""
        stats = ProgressStats(total_tasks=100, completed_tasks=100)
        assert stats.percent_complete == 100.0

    def test_percent_complete_zero_total(self):
        """Test percent complete with zero total tasks."""
        stats = ProgressStats(total_tasks=0)
        assert stats.percent_complete == 100.0

    def test_tasks_per_second(self):
        """Test tasks per second calculation."""
        stats = ProgressStats(total_tasks=100, completed_tasks=10)
        # Manually set start time to 10 seconds ago
        stats.start_time = time.time() - 10
        assert 0.9 <= stats.tasks_per_second <= 1.1  # ~1 task/second

    def test_tasks_per_second_zero_completed(self):
        """Test tasks per second with no completed tasks."""
        stats = ProgressStats(total_tasks=100)
        assert stats.tasks_per_second == 0.0

    def test_eta_calculation(self):
        """Test ETA calculation."""
        stats = ProgressStats(total_tasks=100, completed_tasks=50)
        stats.start_time = time.time() - 50  # 50 seconds elapsed
        # Should take about 50 more seconds for 50 remaining tasks
        assert stats.eta_seconds is not None
        assert 45 <= stats.eta_seconds <= 55

    def test_eta_no_progress(self):
        """Test ETA with no progress."""
        stats = ProgressStats(total_tasks=100)
        assert stats.eta_seconds is None


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_basic_update(self):
        """Test basic progress update."""
        tracker = ProgressTracker(total_tasks=10, show_progress=False)
        tracker.update(success=True)

        assert tracker.stats.completed_tasks == 1
        assert tracker.stats.successful_tasks == 1
        assert tracker.stats.failed_tasks == 0

    def test_update_failure(self):
        """Test updating with failure."""
        tracker = ProgressTracker(total_tasks=10, show_progress=False)
        tracker.update(success=False)

        assert tracker.stats.completed_tasks == 1
        assert tracker.stats.successful_tasks == 0
        assert tracker.stats.failed_tasks == 1

    def test_update_skipped(self):
        """Test updating with skipped task."""
        tracker = ProgressTracker(total_tasks=10, show_progress=False)
        tracker.update(success=True, skipped=True)

        assert tracker.stats.completed_tasks == 1
        assert tracker.stats.skipped_tasks == 1
        assert tracker.stats.successful_tasks == 0  # Skipped doesn't count as success

    def test_cost_tracking(self):
        """Test cost tracking."""
        tracker = ProgressTracker(total_tasks=10, show_progress=False)
        tracker.update(success=True, cost=0.50)
        tracker.update(success=True, cost=0.75)

        assert tracker.stats.estimated_cost == 1.25

    def test_token_tracking(self):
        """Test token tracking."""
        tracker = ProgressTracker(total_tasks=10, show_progress=False)
        tracker.update(success=True, tokens=1000)
        tracker.update(success=True, tokens=2000)

        assert tracker.stats.tokens_used == 3000

    def test_get_summary(self):
        """Test summary generation."""
        tracker = ProgressTracker(total_tasks=5, show_progress=False)
        tracker.update(success=True, cost=0.10)
        tracker.update(success=True, cost=0.10)
        tracker.update(success=False, cost=0.05)
        tracker.update(success=True, skipped=True)

        summary = tracker.get_summary()

        assert "Total tasks:     5" in summary
        assert "Successful:      2" in summary
        assert "Failed:          1" in summary
        assert "Skipped:         1" in summary
        assert "$0.25" in summary

    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        tracker = ProgressTracker(total_tasks=100, show_progress=False)

        for i in range(100):
            tracker.update(success=(i % 10 != 0))  # 10% failure rate

        assert tracker.stats.completed_tasks == 100
        assert tracker.stats.successful_tasks == 90
        assert tracker.stats.failed_tasks == 10
        assert tracker.stats.percent_complete == 100.0
