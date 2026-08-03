"""Compatibility tests for the pinned Google Gen AI SDK."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from arc_agi_benchmarking.adapters.gemini import GeminiAdapter
from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing


def test_gemini_sdk_supports_thinking_level():
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="medium")
    )

    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


def test_minimal_thinking_treats_missing_reasoning_tokens_as_zero():
    adapter = object.__new__(GeminiAdapter)
    adapter.model_config = SimpleNamespace(
        model_name="gemini-3.6-flash",
        provider="gemini",
        kwargs={"thinking_config": {"thinking_level": "minimal"}},
        pricing=SimpleNamespace(input=1.5, output=7.5),
    )
    adapter.chat_completion = MagicMock(
        return_value=SimpleNamespace(
            text="[[1]]",
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=20,
                thoughts_token_count=None,
                total_token_count=120,
            ),
        )
    )

    attempt = adapter.make_prediction("Solve this task")

    assert attempt.metadata.usage.completion_tokens_details.reasoning_tokens == 0
    assert attempt.metadata.cost.reasoning_cost == 0
    assert attempt.metadata.cost.total_cost == pytest.approx(
        (100 * 1.5 / 1_000_000) + (20 * 7.5 / 1_000_000)
    )


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

    client.assert_called_once()
    call_kwargs = client.call_args.kwargs
    assert call_kwargs["api_key"] == "custom-gemini-secret"
    assert call_kwargs["http_options"].retry_options.attempts == 1
