"""Compatibility tests for the pinned Google Gen AI SDK."""

from unittest.mock import patch

from google.genai import types

from arc_agi_benchmarking.adapters.gemini import GeminiAdapter
from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing


def test_gemini_sdk_supports_thinking_level():
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="medium")
    )

    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


def test_gemini_init_uses_configured_api_key_env(monkeypatch):
    config = ModelConfig(
        name="test-gemini",
        model_name="gemini-test",
        provider="gemini",
        api_key_env="CUSTOM_GEMINI_KEY",
        pricing=ModelPricing(date="2026-07-27", input=1.0, output=2.0),
    )
    monkeypatch.setenv("CUSTOM_GEMINI_KEY", "custom-gemini-secret")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.model_config = config
    with patch("arc_agi_benchmarking.adapters.gemini.genai.Client") as client:
        adapter.init_client()

    client.assert_called_once_with(api_key="custom-gemini-secret")
