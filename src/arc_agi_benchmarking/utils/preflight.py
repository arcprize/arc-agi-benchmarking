"""
Pre-flight validation and cost estimation for ARC-AGI benchmarking runs.

Run this before expensive batch operations to catch configuration errors early
and estimate costs before spending money.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from arc_agi_benchmarking.utils.task_utils import read_models_config
from arc_agi_benchmarking.schemas import ModelConfig

logger = logging.getLogger(__name__)

# Provider to environment variable mapping
PROVIDER_API_KEYS: Dict[str, List[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "claude_agent_sdk": ["ANTHROPIC_API_KEY"],
    "gemini": ["GOOGLE_API_KEY"],
    "google": ["GOOGLE_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "grok": ["XAI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "codex": ["OPENAI_API_KEY", "CODEX_API_KEY"],  # Either works
    "random": [],  # No API key needed
}

# Average tokens per ARC task (empirically estimated)
# This is a rough estimate based on typical task sizes
DEFAULT_AVG_INPUT_TOKENS_PER_TASK = 2500
DEFAULT_AVG_OUTPUT_TOKENS_PER_TASK = 500


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    message: str
    details: Optional[str] = None


@dataclass
class CostEstimate:
    """Estimated cost breakdown for a benchmark run."""
    num_tasks: int
    num_attempts_per_task: int
    total_attempts: int
    input_price_per_1m: float
    output_price_per_1m: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float

    def __str__(self) -> str:
        return (
            f"Cost Estimate:\n"
            f"  Tasks: {self.num_tasks}\n"
            f"  Attempts per task: {self.num_attempts_per_task}\n"
            f"  Total attempts: {self.total_attempts}\n"
            f"  Estimated input tokens: {self.estimated_input_tokens:,}\n"
            f"  Estimated output tokens: {self.estimated_output_tokens:,}\n"
            f"  Input price: ${self.input_price_per_1m:.2f}/1M tokens\n"
            f"  Output price: ${self.output_price_per_1m:.2f}/1M tokens\n"
            f"  Estimated cost: ${self.estimated_cost:.2f}"
        )


@dataclass
class PreflightReport:
    """Complete preflight validation report."""
    config_name: str
    validations: List[ValidationResult]
    cost_estimate: Optional[CostEstimate]
    all_passed: bool

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "PREFLIGHT VALIDATION REPORT",
            "=" * 60,
            f"Config: {self.config_name}",
            "",
            "Validations:",
        ]

        for v in self.validations:
            status = "✓" if v.passed else "✗"
            lines.append(f"  {status} {v.message}")
            if v.details and not v.passed:
                lines.append(f"    └─ {v.details}")

        lines.append("")

        if self.cost_estimate:
            lines.append(str(self.cost_estimate))

        lines.append("")
        lines.append("=" * 60)

        if self.all_passed:
            lines.append("✓ All preflight checks PASSED")
        else:
            lines.append("✗ Preflight checks FAILED - fix issues before running")

        lines.append("=" * 60)

        return "\n".join(lines)


def validate_config_exists(config_name: str) -> ValidationResult:
    """Check if the model configuration exists in models.yml."""
    try:
        config = read_models_config(config_name)
        return ValidationResult(
            passed=True,
            message=f"Config '{config_name}' found",
            details=f"Model: {config.model_name}, Provider: {config.provider}"
        )
    except ValueError as e:
        return ValidationResult(
            passed=False,
            message=f"Config '{config_name}' not found",
            details=str(e)
        )
    except Exception as e:
        return ValidationResult(
            passed=False,
            message=f"Error reading config '{config_name}'",
            details=str(e)
        )


def validate_api_key(provider: str) -> ValidationResult:
    """Check if the required API key exists for the provider."""
    if provider not in PROVIDER_API_KEYS:
        return ValidationResult(
            passed=False,
            message=f"Unknown provider '{provider}'",
            details=f"Known providers: {', '.join(PROVIDER_API_KEYS.keys())}"
        )

    required_keys = PROVIDER_API_KEYS[provider]

    if not required_keys:
        return ValidationResult(
            passed=True,
            message=f"No API key required for '{provider}'"
        )

    # Check if any of the valid keys exist
    for key_name in required_keys:
        if os.environ.get(key_name):
            # Mask the key for security
            key_value = os.environ.get(key_name, "")
            masked = key_value[:4] + "..." + key_value[-4:] if len(key_value) > 8 else "***"
            return ValidationResult(
                passed=True,
                message=f"API key '{key_name}' found",
                details=f"Value: {masked}"
            )

    return ValidationResult(
        passed=False,
        message=f"API key not found for '{provider}'",
        details=f"Set one of: {', '.join(required_keys)}"
    )


def validate_data_dir(data_dir: str) -> Tuple[ValidationResult, List[str]]:
    """Check if the data directory exists and contains valid task files."""
    path = Path(data_dir)
    task_ids = []

    if not path.exists():
        return ValidationResult(
            passed=False,
            message=f"Data directory not found",
            details=str(path.absolute())
        ), task_ids

    if not path.is_dir():
        return ValidationResult(
            passed=False,
            message=f"Data path is not a directory",
            details=str(path.absolute())
        ), task_ids

    # Find all JSON files
    json_files = list(path.glob("*.json"))

    if not json_files:
        return ValidationResult(
            passed=False,
            message=f"No task files found in data directory",
            details=str(path.absolute())
        ), task_ids

    # Validate each file
    valid_count = 0
    invalid_files = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check for required keys
            if 'train' in data and 'test' in data:
                valid_count += 1
                task_ids.append(json_file.stem)
            else:
                invalid_files.append(json_file.name)
        except json.JSONDecodeError:
            invalid_files.append(f"{json_file.name} (invalid JSON)")
        except Exception as e:
            invalid_files.append(f"{json_file.name} ({str(e)})")

    if invalid_files:
        return ValidationResult(
            passed=valid_count > 0,  # Partial pass if some files are valid
            message=f"Found {valid_count} valid tasks, {len(invalid_files)} invalid",
            details=f"Invalid: {', '.join(invalid_files[:5])}" +
                    (f" (+{len(invalid_files)-5} more)" if len(invalid_files) > 5 else "")
        ), task_ids

    return ValidationResult(
        passed=True,
        message=f"Found {valid_count} valid task files",
        details=str(path.absolute())
    ), task_ids


def validate_output_dir(output_dir: str) -> ValidationResult:
    """Check if the output directory is writable."""
    path = Path(output_dir)

    # If it doesn't exist, check if parent is writable
    if not path.exists():
        parent = path.parent
        if parent.exists() and os.access(parent, os.W_OK):
            return ValidationResult(
                passed=True,
                message=f"Output directory will be created",
                details=str(path.absolute())
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"Cannot create output directory",
                details=f"Parent not writable: {parent.absolute()}"
            )

    if not path.is_dir():
        return ValidationResult(
            passed=False,
            message=f"Output path exists but is not a directory",
            details=str(path.absolute())
        )

    if not os.access(path, os.W_OK):
        return ValidationResult(
            passed=False,
            message=f"Output directory not writable",
            details=str(path.absolute())
        )

    return ValidationResult(
        passed=True,
        message=f"Output directory exists and is writable",
        details=str(path.absolute())
    )


def estimate_cost(
    model_config: ModelConfig,
    num_tasks: int,
    num_attempts: int = 2,
    avg_input_tokens: int = DEFAULT_AVG_INPUT_TOKENS_PER_TASK,
    avg_output_tokens: int = DEFAULT_AVG_OUTPUT_TOKENS_PER_TASK,
) -> CostEstimate:
    """Estimate the cost of a benchmark run."""
    total_attempts = num_tasks * num_attempts

    estimated_input_tokens = total_attempts * avg_input_tokens
    estimated_output_tokens = total_attempts * avg_output_tokens

    input_cost = (estimated_input_tokens / 1_000_000) * model_config.pricing.input
    output_cost = (estimated_output_tokens / 1_000_000) * model_config.pricing.output
    total_cost = input_cost + output_cost

    return CostEstimate(
        num_tasks=num_tasks,
        num_attempts_per_task=num_attempts,
        total_attempts=total_attempts,
        input_price_per_1m=model_config.pricing.input,
        output_price_per_1m=model_config.pricing.output,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_cost=total_cost,
    )


def run_preflight(
    config_name: str,
    data_dir: str,
    output_dir: str,
    num_attempts: int = 2,
) -> PreflightReport:
    """
    Run all preflight validations and return a comprehensive report.

    Args:
        config_name: Name of the model configuration from models.yml
        data_dir: Directory containing task JSON files
        output_dir: Directory where submissions will be saved
        num_attempts: Number of attempts per task

    Returns:
        PreflightReport with all validation results and cost estimate
    """
    validations: List[ValidationResult] = []
    cost_estimate = None
    model_config = None
    num_tasks = 0

    # 1. Validate config exists
    config_result = validate_config_exists(config_name)
    validations.append(config_result)

    if config_result.passed:
        try:
            model_config = read_models_config(config_name)
        except Exception:
            pass

    # 2. Validate API key (only if config was found)
    if model_config:
        api_result = validate_api_key(model_config.provider)
        validations.append(api_result)
    else:
        validations.append(ValidationResult(
            passed=False,
            message="Skipping API key validation (config not found)"
        ))

    # 3. Validate data directory
    data_result, task_ids = validate_data_dir(data_dir)
    validations.append(data_result)
    num_tasks = len(task_ids)

    # 4. Validate output directory
    output_result = validate_output_dir(output_dir)
    validations.append(output_result)

    # 5. Calculate cost estimate (only if we have valid config and tasks)
    if model_config and num_tasks > 0:
        cost_estimate = estimate_cost(
            model_config=model_config,
            num_tasks=num_tasks,
            num_attempts=num_attempts,
        )

    all_passed = all(v.passed for v in validations)

    return PreflightReport(
        config_name=config_name,
        validations=validations,
        cost_estimate=cost_estimate,
        all_passed=all_passed,
    )


def main():
    """CLI entry point for preflight validation."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run preflight validation before benchmark runs"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model configuration name from models.yml"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sample/tasks",
        help="Directory containing task JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="submissions",
        help="Directory for saving submissions"
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=2,
        help="Number of attempts per task"
    )

    args = parser.parse_args()

    report = run_preflight(
        config_name=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_attempts=args.num_attempts,
    )

    print(report)

    # Exit with error code if validation failed
    exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
