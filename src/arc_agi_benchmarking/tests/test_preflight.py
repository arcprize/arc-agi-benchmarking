"""Tests for preflight validation module."""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from arc_agi_benchmarking.utils.preflight import (
    validate_config_exists,
    validate_api_key,
    validate_data_dir,
    validate_output_dir,
    estimate_cost,
    run_preflight,
    ValidationResult,
    CostEstimate,
    PreflightReport,
    PROVIDER_API_KEYS,
)
from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing


class TestValidateConfigExists:
    """Tests for validate_config_exists function."""

    def test_valid_config(self):
        """Test that a valid config is found."""
        result = validate_config_exists("gpt-4o-2024-11-20")
        assert result.passed is True
        assert "gpt-4o-2024-11-20" in result.message

    def test_invalid_config(self):
        """Test that an invalid config returns failure."""
        result = validate_config_exists("nonexistent-model-xyz")
        assert result.passed is False
        assert "not found" in result.message.lower()


class TestValidateApiKey:
    """Tests for validate_api_key function."""

    def test_known_provider_with_key(self):
        """Test that a known provider with an API key passes."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test12345678"}):
            result = validate_api_key("openai")
            assert result.passed is True
            assert "OPENAI_API_KEY" in result.message

    def test_known_provider_without_key(self):
        """Test that a known provider without an API key fails."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing DEEPSEEK_API_KEY
            if "DEEPSEEK_API_KEY" in os.environ:
                del os.environ["DEEPSEEK_API_KEY"]
            result = validate_api_key("deepseek")
            assert result.passed is False
            assert "not found" in result.message.lower()

    def test_random_provider_no_key_needed(self):
        """Test that the random provider doesn't need an API key."""
        result = validate_api_key("random")
        assert result.passed is True
        assert "no api key required" in result.message.lower()

    def test_unknown_provider(self):
        """Test that an unknown provider returns failure."""
        result = validate_api_key("unknown_provider_xyz")
        assert result.passed is False
        assert "unknown provider" in result.message.lower()

    def test_codex_with_either_key(self):
        """Test that codex accepts either OPENAI_API_KEY or CODEX_API_KEY."""
        # Test with OPENAI_API_KEY
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test12345678"}, clear=True):
            result = validate_api_key("codex")
            assert result.passed is True

        # Test with CODEX_API_KEY
        with patch.dict(os.environ, {"CODEX_API_KEY": "codex-test12345678"}, clear=True):
            result = validate_api_key("codex")
            assert result.passed is True


class TestValidateDataDir:
    """Tests for validate_data_dir function."""

    def test_valid_data_dir(self):
        """Test validation of a directory with valid task files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid task file
            task_file = Path(tmpdir) / "task1.json"
            task_file.write_text(json.dumps({
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[4]]}]
            }))

            result, task_ids = validate_data_dir(tmpdir)
            assert result.passed is True
            assert len(task_ids) == 1
            assert "task1" in task_ids

    def test_nonexistent_dir(self):
        """Test validation of a nonexistent directory."""
        result, task_ids = validate_data_dir("/nonexistent/path/xyz")
        assert result.passed is False
        assert len(task_ids) == 0
        assert "not found" in result.message.lower()

    def test_empty_dir(self):
        """Test validation of an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, task_ids = validate_data_dir(tmpdir)
            assert result.passed is False
            assert len(task_ids) == 0
            assert "no task files" in result.message.lower()

    def test_invalid_json_file(self):
        """Test validation with an invalid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / "invalid.json"
            invalid_file.write_text("not valid json")

            result, task_ids = validate_data_dir(tmpdir)
            assert result.passed is False
            assert "invalid" in result.message.lower()

    def test_missing_required_keys(self):
        """Test validation with a JSON file missing required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a JSON file without 'train' and 'test' keys
            task_file = Path(tmpdir) / "bad_task.json"
            task_file.write_text(json.dumps({"data": [1, 2, 3]}))

            result, task_ids = validate_data_dir(tmpdir)
            assert result.passed is False
            assert len(task_ids) == 0


class TestValidateOutputDir:
    """Tests for validate_output_dir function."""

    def test_existing_writable_dir(self):
        """Test validation of an existing writable directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_output_dir(tmpdir)
            assert result.passed is True
            assert "writable" in result.message.lower()

    def test_nonexistent_dir_with_writable_parent(self):
        """Test validation of a nonexistent directory with writable parent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_subdir")
            result = validate_output_dir(new_dir)
            assert result.passed is True
            assert "will be created" in result.message.lower()

    def test_file_instead_of_dir(self):
        """Test validation when path is a file, not a directory."""
        with tempfile.NamedTemporaryFile() as tmpfile:
            result = validate_output_dir(tmpfile.name)
            assert result.passed is False
            assert "not a directory" in result.message.lower()


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_basic_cost_estimate(self):
        """Test basic cost estimation."""
        mock_config = ModelConfig(
            name="test-model",
            model_name="test-model",
            provider="test",
            pricing=ModelPricing(date="2024-01-01", input=1.0, output=2.0)
        )

        estimate = estimate_cost(
            model_config=mock_config,
            num_tasks=10,
            num_attempts=2,
            avg_input_tokens=1000,
            avg_output_tokens=500
        )

        assert estimate.num_tasks == 10
        assert estimate.num_attempts_per_task == 2
        assert estimate.total_attempts == 20
        assert estimate.estimated_input_tokens == 20000  # 10 * 2 * 1000
        assert estimate.estimated_output_tokens == 10000  # 10 * 2 * 500

        # Cost: (20000/1M) * $1 + (10000/1M) * $2 = $0.02 + $0.02 = $0.04
        assert estimate.estimated_cost == pytest.approx(0.04, rel=0.01)

    def test_zero_tasks(self):
        """Test cost estimation with zero tasks."""
        mock_config = ModelConfig(
            name="test-model",
            model_name="test-model",
            provider="test",
            pricing=ModelPricing(date="2024-01-01", input=10.0, output=30.0)
        )

        estimate = estimate_cost(
            model_config=mock_config,
            num_tasks=0,
            num_attempts=2
        )

        assert estimate.estimated_cost == 0.0


class TestRunPreflight:
    """Tests for run_preflight function."""

    def test_full_preflight_with_valid_inputs(self):
        """Test full preflight with all valid inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid task file
            task_file = Path(tmpdir) / "task1.json"
            task_file.write_text(json.dumps({
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[4]]}]
            }))

            output_dir = os.path.join(tmpdir, "output")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test12345678"}):
                report = run_preflight(
                    config_name="gpt-4o-2024-11-20",
                    data_dir=tmpdir,
                    output_dir=output_dir,
                    num_attempts=2
                )

                assert report.all_passed is True
                assert report.cost_estimate is not None
                assert len(report.validations) == 4  # config, api key, data dir, output dir

    def test_preflight_with_invalid_config(self):
        """Test preflight with an invalid config name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_preflight(
                config_name="nonexistent-model",
                data_dir=tmpdir,
                output_dir=tmpdir,
                num_attempts=2
            )

            assert report.all_passed is False
            # Should have failed on config validation
            config_results = [v for v in report.validations if "config" in v.message.lower()]
            assert any(not v.passed for v in config_results)


class TestPreflightReport:
    """Tests for PreflightReport string formatting."""

    def test_report_string_format(self):
        """Test that the report string is properly formatted."""
        report = PreflightReport(
            config_name="test-config",
            validations=[
                ValidationResult(passed=True, message="Check 1 passed"),
                ValidationResult(passed=False, message="Check 2 failed", details="Some error"),
            ],
            cost_estimate=CostEstimate(
                num_tasks=10,
                num_attempts_per_task=2,
                total_attempts=20,
                input_price_per_1m=1.0,
                output_price_per_1m=2.0,
                estimated_input_tokens=10000,
                estimated_output_tokens=5000,
                estimated_cost=0.02
            ),
            all_passed=False
        )

        report_str = str(report)
        assert "test-config" in report_str
        assert "✓" in report_str  # Passed check
        assert "✗" in report_str  # Failed check
        assert "FAILED" in report_str  # Overall status


class TestProviderApiKeyMapping:
    """Tests for PROVIDER_API_KEYS mapping."""

    def test_all_major_providers_covered(self):
        """Test that all major providers are in the mapping."""
        expected_providers = [
            "openai", "anthropic", "gemini", "deepseek",
            "fireworks", "xai", "groq", "openrouter", "random"
        ]
        for provider in expected_providers:
            assert provider in PROVIDER_API_KEYS, f"Provider {provider} not in PROVIDER_API_KEYS"

    def test_random_provider_has_no_keys(self):
        """Test that random provider has empty key list."""
        assert PROVIDER_API_KEYS["random"] == []
