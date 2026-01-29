"""Tests for detect_config_changes script."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from detect_config_changes import (
    parse_models_yaml,
    detect_changed_configs,
    compare_configs_direct,
    configs_equal,
    get_file_at_ref,
    YamlParseError,
    NULL_SHA,
)


class TestParseModelsYaml:
    """Tests for parse_models_yaml function."""

    def test_parse_valid_yaml(self):
        """Test parsing valid YAML content."""
        content = """
gpt-4o:
  model: gpt-4o
  provider: openai
  temperature: 0.0

claude-3-opus:
  model: claude-3-opus-20240229
  provider: anthropic
  temperature: 0.0
"""
        result = parse_models_yaml(content)
        assert "gpt-4o" in result
        assert "claude-3-opus" in result
        assert result["gpt-4o"]["provider"] == "openai"

    def test_parse_none_content(self):
        """Test parsing None returns empty dict."""
        result = parse_models_yaml(None)
        assert result == {}

    def test_parse_empty_content(self):
        """Test parsing empty string returns empty dict."""
        result = parse_models_yaml("")
        assert result == {}

    def test_parse_invalid_yaml_lenient(self):
        """Test parsing invalid YAML returns empty dict in lenient mode."""
        result = parse_models_yaml("{{invalid yaml::", strict=False)
        assert result == {}

    def test_parse_invalid_yaml_strict_raises(self):
        """Test parsing invalid YAML raises in strict mode."""
        with pytest.raises(YamlParseError) as exc_info:
            parse_models_yaml("{{invalid yaml::", strict=True)
        assert "Invalid YAML" in str(exc_info.value)


class TestDetectChangedConfigs:
    """Tests for detect_changed_configs function."""

    @patch("detect_config_changes.get_file_at_ref")
    def test_detect_added_config(self, mock_get_file):
        """Test detecting a newly added config."""
        base_content = """
gpt-4o:
  model: gpt-4o
  provider: openai
"""
        head_content = """
gpt-4o:
  model: gpt-4o
  provider: openai

claude-3-opus:
  model: claude-3-opus
  provider: anthropic
"""
        mock_get_file.side_effect = [base_content, head_content]

        result = detect_changed_configs("base", "head")

        assert result == ["claude-3-opus"]

    @patch("detect_config_changes.get_file_at_ref")
    def test_detect_modified_config(self, mock_get_file):
        """Test detecting a modified config."""
        base_content = """
gpt-4o:
  model: gpt-4o
  provider: openai
  temperature: 0.0
"""
        head_content = """
gpt-4o:
  model: gpt-4o
  provider: openai
  temperature: 0.5
"""
        mock_get_file.side_effect = [base_content, head_content]

        result = detect_changed_configs("base", "head")

        assert result == ["gpt-4o"]

    @patch("detect_config_changes.get_file_at_ref")
    def test_detect_removed_config_not_included(self, mock_get_file):
        """Test that removed configs are not included in results."""
        base_content = """
gpt-4o:
  model: gpt-4o
  provider: openai

old-config:
  model: old-model
  provider: openai
"""
        head_content = """
gpt-4o:
  model: gpt-4o
  provider: openai
"""
        mock_get_file.side_effect = [base_content, head_content]

        result = detect_changed_configs("base", "head")

        # Removed configs are not included (no benchmark to run)
        assert result == []

    @patch("detect_config_changes.get_file_at_ref")
    def test_detect_no_changes(self, mock_get_file):
        """Test detecting no changes."""
        content = """
gpt-4o:
  model: gpt-4o
  provider: openai
"""
        mock_get_file.side_effect = [content, content]

        result = detect_changed_configs("base", "head")

        assert result == []

    @patch("detect_config_changes.get_file_at_ref")
    def test_detect_multiple_changes(self, mock_get_file):
        """Test detecting multiple changed configs."""
        base_content = """
gpt-4o:
  model: gpt-4o
  temperature: 0.0

claude-3-opus:
  model: claude-3-opus
  temperature: 0.0
"""
        head_content = """
gpt-4o:
  model: gpt-4o-2024-11-20
  temperature: 0.0

claude-3-opus:
  model: claude-3-opus
  temperature: 0.5

gemini-pro:
  model: gemini-pro
  temperature: 0.0
"""
        mock_get_file.side_effect = [base_content, head_content]

        result = detect_changed_configs("base", "head")

        assert sorted(result) == ["claude-3-opus", "gemini-pro", "gpt-4o"]

    @patch("detect_config_changes.get_file_at_ref")
    def test_new_file(self, mock_get_file):
        """Test when models.yml is newly created."""
        head_content = """
gpt-4o:
  model: gpt-4o
"""
        mock_get_file.side_effect = [None, head_content]  # Base doesn't exist

        result = detect_changed_configs("base", "head")

        assert result == ["gpt-4o"]

    @patch("detect_config_changes.get_file_at_ref")
    def test_deleted_file(self, mock_get_file):
        """Test when models.yml is deleted."""
        base_content = """
gpt-4o:
  model: gpt-4o
"""
        mock_get_file.side_effect = [base_content, None]  # Head doesn't exist

        result = detect_changed_configs("base", "head")

        # No configs to run if file is deleted
        assert result == []


class TestConfigsEqual:
    """Tests for configs_equal function."""

    def test_equal_simple_configs(self):
        """Test equality of simple configs."""
        config1 = {"model": "gpt-4o", "temperature": 0.0}
        config2 = {"model": "gpt-4o", "temperature": 0.0}
        assert configs_equal(config1, config2) is True

    def test_unequal_configs(self):
        """Test inequality when values differ."""
        config1 = {"model": "gpt-4o", "temperature": 0.0}
        config2 = {"model": "gpt-4o", "temperature": 0.5}
        assert configs_equal(config1, config2) is False

    def test_unequal_keys(self):
        """Test inequality when keys differ."""
        config1 = {"model": "gpt-4o"}
        config2 = {"model": "gpt-4o", "temperature": 0.0}
        assert configs_equal(config1, config2) is False

    def test_nested_configs(self):
        """Test equality of nested configs."""
        config1 = {"model": "gpt-4o", "params": {"temp": 0.0, "top_p": 1.0}}
        config2 = {"model": "gpt-4o", "params": {"temp": 0.0, "top_p": 1.0}}
        assert configs_equal(config1, config2) is True

    def test_nested_configs_differ(self):
        """Test inequality of nested configs."""
        config1 = {"model": "gpt-4o", "params": {"temp": 0.0}}
        config2 = {"model": "gpt-4o", "params": {"temp": 0.5}}
        assert configs_equal(config1, config2) is False


class TestCompareConfigsDirect:
    """Tests for compare_configs_direct function."""

    def test_detect_added(self):
        """Test detecting added config."""
        base = {"gpt-4o": {"model": "gpt-4o"}}
        head = {
            "gpt-4o": {"model": "gpt-4o"},
            "claude": {"model": "claude-3"},
        }
        result = compare_configs_direct(base, head)
        assert result == ["claude"]

    def test_detect_modified(self):
        """Test detecting modified config."""
        base = {"gpt-4o": {"model": "gpt-4o", "temp": 0.0}}
        head = {"gpt-4o": {"model": "gpt-4o", "temp": 0.5}}
        result = compare_configs_direct(base, head)
        assert result == ["gpt-4o"]

    def test_no_changes(self):
        """Test no changes detected."""
        config = {"gpt-4o": {"model": "gpt-4o"}}
        result = compare_configs_direct(config, config)
        assert result == []

    def test_removed_not_included(self):
        """Test that removed configs are not in result."""
        base = {"gpt-4o": {"model": "gpt-4o"}, "old": {"model": "old"}}
        head = {"gpt-4o": {"model": "gpt-4o"}}
        result = compare_configs_direct(base, head)
        assert result == []

    def test_multiple_changes(self):
        """Test multiple added and modified configs."""
        base = {"a": {"v": 1}, "b": {"v": 2}}
        head = {"a": {"v": 1}, "b": {"v": 3}, "c": {"v": 4}}
        result = compare_configs_direct(base, head)
        assert result == ["b", "c"]


class TestGetFileAtRef:
    """Tests for get_file_at_ref function."""

    def test_returns_none_for_nonexistent_file(self):
        """Test that nonexistent file returns None."""
        # This test actually calls git, so it tests real behavior
        result = get_file_at_ref("nonexistent/path/file.yml", "HEAD")
        assert result is None

    @patch("subprocess.run")
    def test_successful_file_retrieval(self, mock_run):
        """Test successful file content retrieval."""
        mock_run.return_value = MagicMock(stdout="file content", returncode=0)

        result = get_file_at_ref("some/file.yml", "HEAD")

        assert result == "file content"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_subprocess_error(self, mock_run):
        """Test handling of subprocess error."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")

        result = get_file_at_ref("some/file.yml", "HEAD")

        assert result is None

    def test_null_sha_returns_none(self):
        """Test that null SHA (first push) returns None without calling git."""
        result = get_file_at_ref("some/file.yml", NULL_SHA)
        assert result is None

    @patch("subprocess.run")
    def test_partial_zeros_sha_is_valid(self, mock_run):
        """Test that a SHA starting with zeros is treated as valid (not null)."""
        # A legitimate commit can start with zeros - should not be treated as null
        mock_run.return_value = MagicMock(stdout="file content", returncode=0)
        
        result = get_file_at_ref("some/file.yml", "0000000abc123")
        
        # Should call git, not return None
        assert result == "file content"
        mock_run.assert_called_once()


class TestDetectChangedConfigsStrict:
    """Tests for strict mode in detect_changed_configs."""

    @patch("detect_config_changes.get_file_at_ref")
    def test_strict_mode_raises_on_invalid_head_yaml(self, mock_get_file):
        """Test that strict mode raises when head YAML is invalid."""
        mock_get_file.side_effect = [
            "gpt-4o:\n  model: gpt-4o",  # valid base
            "{{invalid yaml::",  # invalid head
        ]

        with pytest.raises(YamlParseError):
            detect_changed_configs("base", "head", strict=True)

    @patch("detect_config_changes.get_file_at_ref")
    def test_strict_mode_tolerates_invalid_base_yaml(self, mock_get_file):
        """Test that strict mode tolerates invalid base YAML (old commits)."""
        mock_get_file.side_effect = [
            "{{invalid yaml::",  # invalid base (treated as empty)
            "gpt-4o:\n  model: gpt-4o",  # valid head
        ]

        # Should not raise, treats base as empty
        result = detect_changed_configs("base", "head", strict=True)
        assert result == ["gpt-4o"]

    @patch("detect_config_changes.get_file_at_ref")
    def test_lenient_mode_tolerates_invalid_head_yaml(self, mock_get_file):
        """Test that lenient mode tolerates invalid head YAML."""
        mock_get_file.side_effect = [
            "gpt-4o:\n  model: gpt-4o",
            "{{invalid yaml::",
        ]

        # Should not raise in lenient mode
        result = detect_changed_configs("base", "head", strict=False)
        # Treats head as empty, so all base configs are "removed" (not reported)
        assert result == []

    @patch("detect_config_changes.get_file_at_ref")
    def test_first_push_null_sha(self, mock_get_file):
        """Test handling of first push (null SHA for base)."""

        # Simulate first push where base is null SHA
        def get_file_side_effect(filepath, ref):
            if ref == NULL_SHA:
                return None  # First push - no previous commit
            return "gpt-4o:\n  model: gpt-4o"

        mock_get_file.side_effect = get_file_side_effect

        result = detect_changed_configs(NULL_SHA, "HEAD", strict=True)
        # All configs in head are "added"
        assert result == ["gpt-4o"]


class TestMainScript:
    """Integration tests for the main script."""

    @pytest.fixture
    def repo_root(self):
        """Get the repository root directory."""
        # Navigate from tests/ -> arc_agi_benchmarking/ -> src/ -> repo_root
        return Path(__file__).parent.parent.parent.parent

    def test_script_outputs_json(self, repo_root):
        """Test that the script outputs valid JSON."""
        script_path = repo_root / "scripts" / "detect_config_changes.py"

        # Run the actual script
        result = subprocess.run(
            [sys.executable, str(script_path), "--base", "HEAD", "--head", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )

        # Should output valid JSON (even if empty list)
        output = result.stdout.strip()
        parsed = json.loads(output)
        assert isinstance(parsed, list)

    def test_script_with_max_configs(self, repo_root):
        """Test --max-configs flag."""
        script_path = repo_root / "scripts" / "detect_config_changes.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--base",
                "HEAD",
                "--head",
                "HEAD",
                "--fallback-all",
                "--max-configs",
                "2",
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )

        output = result.stdout.strip()
        parsed = json.loads(output)
        assert len(parsed) <= 2

    @patch("detect_config_changes.get_file_at_ref")
    def test_script_exits_with_error_on_invalid_yaml(self, mock_get_file, repo_root):
        """Test that script exits with error code 1 on invalid YAML."""
        # Note: We can't easily test the actual script exit code with mocking,
        # so we test the function behavior instead
        mock_get_file.side_effect = [
            "gpt-4o:\n  model: gpt-4o",  # valid base
            "{{invalid yaml::",  # invalid head
        ]

        with pytest.raises(YamlParseError):
            detect_changed_configs("base", "head", strict=True)

    def test_script_lenient_flag(self, repo_root):
        """Test --lenient flag exists and doesn't crash."""
        script_path = repo_root / "scripts" / "detect_config_changes.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--base",
                "HEAD",
                "--head",
                "HEAD",
                "--lenient",
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )

        assert result.returncode == 0
        output = result.stdout.strip()
        parsed = json.loads(output)
        assert isinstance(parsed, list)
