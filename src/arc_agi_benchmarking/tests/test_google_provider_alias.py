"""Google provider names must share adapter routing and provider budgets."""

from unittest.mock import MagicMock

import pytest

import main
from cli import run_all
from arc_agi_benchmarking.adapters.gemini import GeminiAdapter
from arc_agi_benchmarking.schemas import ModelConfig
from arc_agi_benchmarking.utils.task_utils import get_provider_timeout_config


def config(provider):
    return ModelConfig(
        name=f"test-{provider}",
        model_name="test-model",
        provider=provider,
        api_key_env="GOOGLE_API_KEY",
        pricing={"date": "2026-01-01", "input": 0, "output": 0},
    )


@pytest.mark.parametrize("provider", ["google", "gemini"])
def test_provider_routes_to_google_adapter(provider, monkeypatch):
    assert main.PROVIDER_ADAPTERS[provider] is GeminiAdapter
    adapter = MagicMock()
    monkeypatch.setitem(main.PROVIDER_ADAPTERS, provider, adapter)
    tester = object.__new__(main.ARCTester)
    tester.config = "test-config"
    tester.request_limiter = MagicMock()
    tester.raw_api_logger = MagicMock()
    assert tester.init_provider(provider) is adapter.return_value
    adapter.assert_called_once_with(
        "test-config", request_limiter=tester.request_limiter,
        raw_api_logger=tester.raw_api_logger,
    )


def test_aliases_share_provider_limits_and_timeouts(monkeypatch):
    monkeypatch.setattr(run_all, "PROVIDER_RATE_LIMITERS", {})
    google, gemini = config("google"), config("gemini")
    assert google.provider == gemini.provider == "gemini"
    limits = {"gemini": {"rate": 12, "period": 60, "request_timeout": 123}}
    first = run_all.get_or_create_rate_limiter(google.provider, limits)
    assert run_all.get_or_create_rate_limiter(gemini.provider, limits) is first
    assert get_provider_timeout_config(google.provider, limits)["request_timeout"] == 123
    assert google.model_name == "test-model"
    assert google.api_key_env == "GOOGLE_API_KEY"


def test_other_provider_names_are_unchanged():
    assert config("openai").provider == "openai"
