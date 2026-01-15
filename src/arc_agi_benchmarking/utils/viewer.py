"""
Terminal viewer for ARC-AGI tasks.

Displays colored grids in the terminal for visualizing tasks and submissions.
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

# ARC color palette (0-9) mapped to ANSI 256-color codes
# These colors match the official ARC visualization
ARC_COLORS = {
    0: 0,    # Black
    1: 21,   # Blue
    2: 196,  # Red
    3: 46,   # Green
    4: 226,  # Yellow
    5: 244,  # Gray
    6: 201,  # Magenta/Pink
    7: 208,  # Orange
    8: 51,   # Cyan/Light blue
    9: 88,   # Brown/Maroon
}

# Block character for rendering cells
BLOCK = "██"
HALF_BLOCK = "▀"

# Regex to strip ANSI codes for width calculation
import re
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def visible_len(s: str) -> int:
    """Return the visible length of a string (excluding ANSI codes)."""
    return len(ANSI_ESCAPE.sub('', s))


def pad_to_visible_width(s: str, width: int) -> str:
    """Pad a string to a visible width, accounting for ANSI codes."""
    visible = visible_len(s)
    if visible < width:
        return s + " " * (width - visible)
    return s


def ansi_bg(color_code: int) -> str:
    """Return ANSI escape code for background color."""
    return f"\033[48;5;{color_code}m"


def ansi_fg(color_code: int) -> str:
    """Return ANSI escape code for foreground color."""
    return f"\033[38;5;{color_code}m"


def ansi_reset() -> str:
    """Return ANSI reset code."""
    return "\033[0m"


def render_grid(grid: List[List[int]], label: Optional[str] = None) -> List[str]:
    """
    Render a single grid as colored terminal output.

    Args:
        grid: 2D list of integers (0-9)
        label: Optional label to display above the grid

    Returns:
        List of strings (lines) to print
    """
    lines = []

    if label:
        lines.append(f"  {label}")

    # Top border
    width = len(grid[0]) if grid else 0
    lines.append("  ┌" + "──" * width + "┐")

    for row in grid:
        line = "  │"
        for cell in row:
            color = ARC_COLORS.get(cell, 0)
            line += f"{ansi_fg(color)}{BLOCK}{ansi_reset()}"
        line += "│"
        lines.append(line)

    # Bottom border
    lines.append("  └" + "──" * width + "┘")

    return lines


def render_pair(
    input_grid: List[List[int]],
    output_grid: Optional[List[List[int]]] = None,
    predicted_grid: Optional[List[List[int]]] = None,
    pair_label: str = "",
    correct: Optional[bool] = None
) -> List[str]:
    """
    Render an input/output pair side by side.

    Args:
        input_grid: Input grid
        output_grid: Expected output grid (optional)
        predicted_grid: Model's predicted grid (optional)
        pair_label: Label for this pair
        correct: Whether prediction matches expected (optional)

    Returns:
        List of strings (lines) to print
    """
    lines = []

    if pair_label:
        lines.append(f"\n{pair_label}")
        lines.append("─" * 60)

    # Render grids
    input_lines = render_grid(input_grid, "Input")

    output_lines = []
    if output_grid is not None:
        output_lines = render_grid(output_grid, "Expected Output")

    predicted_lines = []
    if predicted_grid is not None:
        status = ""
        if correct is True:
            status = " ✓"
        elif correct is False:
            status = " ✗"
        predicted_lines = render_grid(predicted_grid, f"Predicted{status}")

    # Calculate the max visible width for each column
    def max_visible_width(lst: List[str]) -> int:
        if not lst:
            return 0
        return max(visible_len(line) for line in lst)

    input_width = max_visible_width(input_lines)
    output_width = max_visible_width(output_lines) if output_lines else 0
    predicted_width = max_visible_width(predicted_lines) if predicted_lines else 0

    # Combine side by side
    max_lines = max(len(input_lines), len(output_lines), len(predicted_lines))

    # Pad shorter lists with empty strings
    def pad_list(lst: List[str], length: int) -> List[str]:
        result = lst.copy()
        while len(result) < length:
            result.append("")
        return result

    input_lines = pad_list(input_lines, max_lines)
    output_lines = pad_list(output_lines, max_lines) if output_lines else [""] * max_lines
    predicted_lines = pad_list(predicted_lines, max_lines) if predicted_lines else []

    for i in range(max_lines):
        # Pad each cell to its column width
        input_cell = pad_to_visible_width(input_lines[i], input_width)

        line = input_cell
        if output_lines:
            output_cell = pad_to_visible_width(output_lines[i], output_width)
            line += "  →  " + output_cell
        if predicted_lines:
            predicted_cell = pad_to_visible_width(predicted_lines[i], predicted_width) if i < len(predicted_lines) else ""
            line += "  |  " + predicted_cell
        lines.append(line)

    return lines


def view_task(task_path: str) -> None:
    """
    View a task file (train and test pairs).

    Args:
        task_path: Path to the task JSON file
    """
    with open(task_path, 'r') as f:
        task = json.load(f)

    task_id = Path(task_path).stem
    print(f"\n{'=' * 60}")
    print(f"TASK: {task_id}")
    print(f"{'=' * 60}")

    # Show training pairs
    print(f"\n📚 TRAINING EXAMPLES ({len(task['train'])} pairs)")
    for i, pair in enumerate(task['train']):
        lines = render_pair(
            input_grid=pair['input'],
            output_grid=pair.get('output'),
            pair_label=f"Train Pair {i + 1}"
        )
        for line in lines:
            print(line)

    # Show test pairs
    print(f"\n🧪 TEST ({len(task['test'])} pairs)")
    for i, pair in enumerate(task['test']):
        lines = render_pair(
            input_grid=pair['input'],
            output_grid=pair.get('output'),
            pair_label=f"Test Pair {i + 1}"
        )
        for line in lines:
            print(line)


def view_submission(task_path: str, submission_path: str) -> None:
    """
    View a submission compared to the expected output.

    Args:
        task_path: Path to the task JSON file (ground truth)
        submission_path: Path to the submission JSON file
    """
    with open(task_path, 'r') as f:
        task = json.load(f)

    with open(submission_path, 'r') as f:
        submission = json.load(f)

    task_id = Path(task_path).stem
    print(f"\n{'=' * 60}")
    print(f"SUBMISSION: {task_id}")
    print(f"{'=' * 60}")

    # Iterate through test pairs
    for pair_idx, test_pair in enumerate(task['test']):
        expected_output = test_pair.get('output')

        # Get predictions for this pair
        if pair_idx < len(submission):
            pair_attempts = submission[pair_idx]

            # Handle different submission formats
            if isinstance(pair_attempts, dict) and 'attempts' in pair_attempts:
                attempts = pair_attempts['attempts']
            elif isinstance(pair_attempts, list):
                attempts = pair_attempts
            else:
                attempts = [pair_attempts]

            for attempt_idx, attempt in enumerate(attempts):
                if attempt is None:
                    continue

                # Extract prediction
                predicted = None
                correct = None

                if isinstance(attempt, dict):
                    predicted = attempt.get('answer')
                    correct = attempt.get('correct')

                    # Handle string answers (unparsed)
                    if isinstance(predicted, str):
                        print(f"\n⚠️  Test {pair_idx + 1}, Attempt {attempt_idx + 1}: Raw string output (not parsed)")
                        print(f"    {predicted[:100]}..." if len(predicted) > 100 else f"    {predicted}")
                        continue

                if predicted:
                    lines = render_pair(
                        input_grid=test_pair['input'],
                        output_grid=expected_output,
                        predicted_grid=predicted,
                        pair_label=f"Test {pair_idx + 1}, Attempt {attempt_idx + 1}",
                        correct=correct
                    )
                    for line in lines:
                        print(line)


def view_directory(task_dir: str, limit: int = 5) -> None:
    """
    View multiple tasks from a directory.

    Args:
        task_dir: Path to directory containing task JSON files
        limit: Maximum number of tasks to show
    """
    task_path = Path(task_dir)
    task_files = sorted(task_path.glob("*.json"))[:limit]

    print(f"\nShowing {len(task_files)} of {len(list(task_path.glob('*.json')))} tasks")

    for task_file in task_files:
        view_task(str(task_file))
        print("\n")


def print_color_legend() -> None:
    """Print the color legend for reference."""
    print("\n📊 COLOR LEGEND")
    print("─" * 40)
    for value, color in ARC_COLORS.items():
        print(f"  {ansi_fg(color)}{BLOCK}{ansi_reset()} = {value}")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Terminal viewer for ARC-AGI tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View a single task
  python -m arc_agi_benchmarking.utils.viewer --task data/sample/tasks/00576224.json

  # View a submission compared to ground truth
  python -m arc_agi_benchmarking.utils.viewer --task data/tasks/00576224.json --submission submissions/model/00576224.json

  # View multiple tasks from a directory
  python -m arc_agi_benchmarking.utils.viewer --dir data/sample/tasks --limit 3

  # Show color legend
  python -m arc_agi_benchmarking.utils.viewer --legend
"""
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Path to a task JSON file"
    )
    parser.add_argument(
        "--submission",
        type=str,
        help="Path to a submission JSON file (requires --task)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to a directory of task files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of tasks to show when using --dir (default: 5)"
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Show color legend"
    )

    args = parser.parse_args()

    if args.legend:
        print_color_legend()
        return

    if args.task and args.submission:
        view_submission(args.task, args.submission)
    elif args.task:
        view_task(args.task)
    elif args.dir:
        view_directory(args.dir, limit=args.limit)
    else:
        parser.print_help()
        print("\n💡 Try: python -m arc_agi_benchmarking.utils.viewer --legend")


if __name__ == "__main__":
    main()
