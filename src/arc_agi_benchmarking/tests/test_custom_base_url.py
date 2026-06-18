"""Tests for OpenAI-compatible custom endpoints via `base_url` / `api_key_env`.

These verify that the standard `openai` adapter can target any OpenAI-compatible
provider by setting `base_url` and `api_key_env` in the model config, without a
dedicated adapter class. Baseten is used as the running example.
"""
from unittest.mock import patch

import pytest

from arc_agi_benchmarking.adapters.open_ai import OpenAIAdapter
from arc_agi_benchmarking.schemas import APIType, ModelConfig, ModelPricing


def _make_config(**overrides) -> ModelConfig:
    """Build a ModelConfig, allowing extra/override fields per test."""
    base = dict(
        name="baseten-glm-5-2",
        model_name="zai-org/GLM-5.2",
        provider="openai",
        pricing=ModelPricing(date="2026-06-18", input=0.0, output=0.0),
        api_type=APIType.CHAT_COMPLETIONS,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _init_client_with_config(config: ModelConfig):
    """Run OpenAIAdapter.init_client() against a config, mocking the OpenAI SDK.

    Returns the (args, kwargs) the OpenAI client was constructed with.
    """
    # Bypass ProviderAdapter.__init__ so no real config-name lookup happens.
    with patch(
        "arc_agi_benchmarking.adapters.provider.ProviderAdapter.__init__",
        return_value=None,
    ):
        adapter = OpenAIAdapter(config=config.name)
        adapter.model_config = config

        with patch("arc_agi_benchmarking.adapters.open_ai.OpenAI") as mock_openai:
            adapter.init_client()

    mock_openai.assert_called_once()
    return mock_openai.call_args


class TestCustomBaseUrlSchema:
    """The new fields must be first-class, not swept into kwargs by the validator."""

    def test_base_url_and_api_key_env_are_first_class_fields(self):
        config = _make_config(
            base_url="https://inference.baseten.co/v1",
            api_key_env="BASETEN_API_KEY",
        )
        assert config.base_url == "https://inference.baseten.co/v1"
        assert config.api_key_env == "BASETEN_API_KEY"
        # Critical: these must NOT leak into kwargs, or they'd be passed as API
        # request params to chat.completions.create() and rejected by the SDK.
        assert "base_url" not in config.kwargs
        assert "api_key_env" not in config.kwargs

    def test_fields_default_to_none(self):
        config = _make_config()
        assert config.base_url is None
        assert config.api_key_env is None


class TestCustomBaseUrlClientInit:
    """init_client() should honor base_url / api_key_env from the config."""

    def test_uses_custom_base_url_and_api_key_env(self, monkeypatch):
        monkeypatch.setenv("BASETEN_API_KEY", "baseten-secret")
        config = _make_config(
            base_url="https://inference.baseten.co/v1",
            api_key_env="BASETEN_API_KEY",
        )

        call_args = _init_client_with_config(config)

        assert call_args.kwargs["base_url"] == "https://inference.baseten.co/v1"
        assert call_args.kwargs["api_key"] == "baseten-secret"

    def test_defaults_preserve_standard_openai_behavior(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        monkeypatch.delenv("BASETEN_API_KEY", raising=False)
        config = _make_config(provider="openai")  # no base_url / api_key_env

        call_args = _init_client_with_config(config)

        # No base_url override should be passed when not configured.
        assert "base_url" not in call_args.kwargs
        assert call_args.kwargs["api_key"] == "openai-secret"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("BASETEN_API_KEY", raising=False)
        config = _make_config(
            base_url="https://inference.baseten.co/v1",
            api_key_env="BASETEN_API_KEY",
        )

        with pytest.raises(ValueError, match="BASETEN_API_KEY"):
            _init_client_with_config(config)
