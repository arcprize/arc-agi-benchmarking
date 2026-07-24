"""Score multiple configs across datasets and display as a table."""

import argparse
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arc_agi_benchmarking.scoring.scoring import ARCScorer
from arc_agi_benchmarking.utils.task_utils import read_models_config

DATASETS = [
    ("public-v1/evaluation", "Public Eval v1"),
    ("public-v2/evaluation", "Public Eval v2"),
    ("semiprivate-v1/evaluation", "Semi-Private v1"),
    ("semiprivate-v2/evaluation", "Semi-Private v2"),
]


def score_config_dataset(config: str, dataset_path: str, base_dir: Path) -> dict | None:
    task_dir = base_dir / "data" / dataset_path
    submission_dir = base_dir / "submissions" / config / dataset_path

    if not submission_dir.exists() or not any(submission_dir.glob("*.json")):
        return None

    if not task_dir.exists():
        return None

    scorer = ARCScorer(
        task_dir=str(task_dir),
        submission_dir=str(submission_dir),
        print_logs=False,
    )
    # Suppress the scorer's built-in print output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        total_score, total_tasks = scorer.score_submission()
    finally:
        sys.stdout = old_stdout
    return {
        "score": total_score,
        "total_tasks": total_tasks,
        "cost": scorer.total_cost,
        "cost_per_task": scorer.total_cost / total_tasks,
    }


def format_cell(result: dict | None) -> str:
    if result is None:
        return "-"
    pct = (
        (result["score"] / result["total_tasks"] * 100)
        if result["total_tasks"] > 0
        else 0
    )
    return f"{pct:.2f}% / {result['score']:.4g}/{result['total_tasks']} / ${result['cost_per_task']:.2f}"


DISPLAY_PARAMS = ["reasoning_effort", "thinking", "thinking_config", "reasoning", "stream", "max_tokens"]


def get_model_params(config: str) -> str:
    """Get key model params from models.yml for display."""
    try:
        model_config = read_models_config(config)
        parts = [f"provider={model_config.provider}"]
        for key in DISPLAY_PARAMS:
            if key in model_config.kwargs:
                val = model_config.kwargs[key]
                parts.append(f"{key}={val}")
        return ", ".join(parts)
    except (ValueError, Exception):
        return ""


def main():
    parser = argparse.ArgumentParser(description="Score multiple configs in a table")
    parser.add_argument(
        "configs",
        type=lambda s: [c.strip() for c in s.split(",")],
        help="Comma-separated model config names to score",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Base directory of arc-agi-benchmarking",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Collect results
    rows = []
    for config in args.configs:
        row = {"config": config, "params": get_model_params(config)}
        for dataset_path, _ in DATASETS:
            row[dataset_path] = score_config_dataset(config, dataset_path, base_dir)
        rows.append(row)

    # Build table
    headers = ["Model Config"] + [label for _, label in DATASETS] + ["Params"]
    config_col_width = max(len(headers[0]), max(len(r["config"]) for r in rows))
    col_widths = [config_col_width]
    for i, (dataset_path, label) in enumerate(DATASETS):
        cells = [format_cell(r[dataset_path]) for r in rows]
        col_widths.append(max(len(label), max(len(c) for c in cells)))
    params_width = max(len("Params"), max(len(r["params"]) for r in rows))
    col_widths.append(params_width)

    def fmt_row(values):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, col_widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    print(fmt_row(headers))
    print(sep)
    for row in rows:
        cells = [row["config"]]
        for dataset_path, _ in DATASETS:
            cells.append(format_cell(row[dataset_path]))
        cells.append(row["params"])
        print(fmt_row(cells))


if __name__ == "__main__":
    main()
