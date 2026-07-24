"""Compatibility tests for the pinned Google Gen AI SDK."""

from google.genai import types


def test_gemini_sdk_supports_thinking_level():
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="medium")
    )

    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
