"""Tests for detect_config_changes script."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from detect_config_changes import (
    parse_models_yaml,
    configs_equal,
    compare_configs_direct,
    YamlParseError,
)


class TestParseModelsYaml:
    """Tests for parse_models_yaml function."""

    def test_none_content_returns_empty_dict(self):
        assert parse_models_yaml(None) == {}

    def test_empty_string_returns_empty_dict(self):
        assert parse_models_yaml("") == {}

    def test_list_format(self):
        content = """
models:
  - name: gpt-4o
    provider: openai
    model: gpt-4o
  - name: claude-3-opus
    provider: anthropic
    model: claude-3-opus
"""
        result = parse_models_yaml(content)
        assert "gpt-4o" in result
        assert "claude-3-opus" in result
        assert result["gpt-4o"]["provider"] == "openai"

    def test_dict_format(self):
        content = """
gpt-4o:
  provider: openai
  model: gpt-4o
claude-3-opus:
  provider: anthropic
  model: claude-3-opus
"""
        result = parse_models_yaml(content)
        assert "gpt-4o" in result
        assert "claude-3-opus" in result
        assert result["gpt-4o"]["provider"] == "openai"

    def test_invalid_yaml_lenient(self):
        content = "invalid: yaml: content: ["
        result = parse_models_yaml(content, strict=False)
        assert result == {}

    def test_invalid_yaml_strict(self):
        content = "invalid: yaml: content: ["
        with pytest.raises(YamlParseError):
            parse_models_yaml(content, strict=True)

    def test_list_format_skips_invalid_entries(self):
        content = """
models:
  - name: valid-config
    provider: openai
  - invalid_entry_no_name: true
  - name: another-valid
    provider: anthropic
"""
        result = parse_models_yaml(content)
        assert "valid-config" in result
        assert "another-valid" in result
        assert len(result) == 2


class TestConfigsEqual:
    """Tests for configs_equal function."""

    def test_identical_configs(self):
        config = {"provider": "openai", "model": "gpt-4o"}
        assert configs_equal(config, config.copy())

    def test_different_configs(self):
        config1 = {"provider": "openai", "model": "gpt-4o"}
        config2 = {"provider": "openai", "model": "gpt-4-turbo"}
        assert not configs_equal(config1, config2)

    def test_nested_configs(self):
        config1 = {"kwargs": {"temperature": 0.7, "max_tokens": 100}}
        config2 = {"kwargs": {"temperature": 0.7, "max_tokens": 100}}
        assert configs_equal(config1, config2)

    def test_nested_configs_different(self):
        config1 = {"kwargs": {"temperature": 0.7}}
        config2 = {"kwargs": {"temperature": 0.8}}
        assert not configs_equal(config1, config2)


class TestCompareConfigsDirect:
    """Tests for compare_configs_direct function."""

    def test_no_changes(self):
        base = {"config-a": {"value": 1}, "config-b": {"value": 2}}
        head = {"config-a": {"value": 1}, "config-b": {"value": 2}}
        assert compare_configs_direct(base, head) == []

    def test_added_config(self):
        base = {"config-a": {"value": 1}}
        head = {"config-a": {"value": 1}, "config-b": {"value": 2}}
        assert compare_configs_direct(base, head) == ["config-b"]

    def test_modified_config(self):
        base = {"config-a": {"value": 1}}
        head = {"config-a": {"value": 2}}
        assert compare_configs_direct(base, head) == ["config-a"]

    def test_removed_config_not_included(self):
        base = {"config-a": {"value": 1}, "config-b": {"value": 2}}
        head = {"config-a": {"value": 1}}
        assert compare_configs_direct(base, head) == []

    def test_multiple_changes(self):
        base = {"config-a": {"value": 1}, "config-b": {"value": 2}}
        head = {"config-a": {"value": 999}, "config-c": {"value": 3}}
        result = compare_configs_direct(base, head)
        assert result == ["config-a", "config-c"]

    def test_results_sorted(self):
        base = {}
        head = {"zebra": {}, "apple": {}, "mango": {}}
        result = compare_configs_direct(base, head)
        assert result == ["apple", "mango", "zebra"]

    def test_empty_base(self):
        base = {}
        head = {"config-a": {"value": 1}}
        assert compare_configs_direct(base, head) == ["config-a"]

    def test_empty_head(self):
        base = {"config-a": {"value": 1}}
        head = {}
        assert compare_configs_direct(base, head) == []
