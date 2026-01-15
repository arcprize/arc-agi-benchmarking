"""Progress tracking for batch benchmark runs.

Provides real-time progress updates with ETA and cost tracking.
"""

import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional


@dataclass
class ProgressStats:
    """Statistics for a batch run."""

    total_tasks: int
    completed_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_cost: float = 0.0
    tokens_used: int = 0

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def tasks_per_second(self) -> float:
        """Return average tasks completed per second."""
        if self.completed_tasks == 0 or self.elapsed_seconds == 0:
            return 0.0
        return self.completed_tasks / self.elapsed_seconds

    @property
    def eta_seconds(self) -> Optional[float]:
        """Return estimated time remaining in seconds."""
        if self.tasks_per_second == 0:
            return None
        remaining = self.total_tasks - self.completed_tasks
        return remaining / self.tasks_per_second

    @property
    def percent_complete(self) -> float:
        """Return percentage complete."""
        if self.total_tasks == 0:
            return 100.0
        return (self.completed_tasks / self.total_tasks) * 100


def format_duration(seconds: Optional[float]) -> str:
    """Format seconds as human-readable duration."""
    if seconds is None:
        return "calculating..."
    if seconds < 0:
        return "unknown"

    td = timedelta(seconds=int(seconds))
    parts = []

    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if td.days > 0:
        parts.append(f"{td.days}d")
    if hours > 0 or td.days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or td.days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def format_cost(cost: float) -> str:
    """Format cost as USD string."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


class ProgressTracker:
    """Track and display progress for batch runs.

    Usage:
        tracker = ProgressTracker(total_tasks=100, show_progress=True)
        for task in tasks:
            result = process_task(task)
            tracker.update(success=result.success, cost=result.cost)
        tracker.finish()
    """

    def __init__(
        self,
        total_tasks: int,
        show_progress: bool = True,
        cost_per_task: float = 0.0,
    ):
        """Initialize progress tracker.

        Args:
            total_tasks: Total number of tasks to process.
            show_progress: Whether to display progress to stderr.
            cost_per_task: Estimated cost per task for projection.
        """
        self.stats = ProgressStats(total_tasks=total_tasks)
        self.show_progress = show_progress
        self.cost_per_task = cost_per_task
        self._last_line_length = 0

    def update(
        self,
        success: bool = True,
        skipped: bool = False,
        cost: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Update progress after a task completes.

        Args:
            success: Whether the task succeeded.
            skipped: Whether the task was skipped (already exists).
            cost: Cost of this task in USD.
            tokens: Tokens used by this task.
        """
        self.stats.completed_tasks += 1

        if skipped:
            self.stats.skipped_tasks += 1
        elif success:
            self.stats.successful_tasks += 1
        else:
            self.stats.failed_tasks += 1

        self.stats.estimated_cost += cost
        self.stats.tokens_used += tokens

        if self.show_progress:
            self._display_progress()

    def _display_progress(self) -> None:
        """Display current progress to stderr."""
        stats = self.stats

        # Build progress bar
        bar_width = 20
        filled = int(bar_width * stats.percent_complete / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build status line
        eta_str = format_duration(stats.eta_seconds)
        elapsed_str = format_duration(stats.elapsed_seconds)
        cost_str = format_cost(stats.estimated_cost)

        # Project total cost
        if stats.completed_tasks > 0 and stats.estimated_cost > 0:
            projected_cost = (stats.estimated_cost / stats.completed_tasks) * stats.total_tasks
            cost_str += f" (proj: {format_cost(projected_cost)})"

        line = (
            f"\r[{bar}] {stats.completed_tasks}/{stats.total_tasks} "
            f"({stats.percent_complete:.0f}%) | "
            f"ETA: {eta_str} | "
            f"Elapsed: {elapsed_str} | "
            f"Cost: {cost_str}"
        )

        # Add success/fail counts if there are failures
        if stats.failed_tasks > 0:
            line += f" | ✓{stats.successful_tasks} ✗{stats.failed_tasks}"

        if stats.skipped_tasks > 0:
            line += f" | Skipped: {stats.skipped_tasks}"

        # Clear previous line and write new one
        padding = " " * max(0, self._last_line_length - len(line))
        sys.stderr.write(line + padding)
        sys.stderr.flush()
        self._last_line_length = len(line)

    def finish(self) -> None:
        """Mark progress as complete and print final summary."""
        if self.show_progress:
            # Move to new line after progress bar
            sys.stderr.write("\n")
            sys.stderr.flush()

    def get_summary(self) -> str:
        """Get a summary string of the run."""
        stats = self.stats
        lines = [
            "=" * 50,
            "Run Summary",
            "=" * 50,
            f"Total tasks:     {stats.total_tasks}",
            f"Successful:      {stats.successful_tasks}",
            f"Failed:          {stats.failed_tasks}",
            f"Skipped:         {stats.skipped_tasks}",
            f"Duration:        {format_duration(stats.elapsed_seconds)}",
            f"Total cost:      {format_cost(stats.estimated_cost)}",
        ]

        if stats.tokens_used > 0:
            lines.append(f"Tokens used:     {stats.tokens_used:,}")

        if stats.successful_tasks > 0:
            avg_time = stats.elapsed_seconds / stats.completed_tasks
            lines.append(f"Avg time/task:   {avg_time:.2f}s")

        lines.append("=" * 50)
        return "\n".join(lines)
