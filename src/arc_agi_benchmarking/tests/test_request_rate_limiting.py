from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from main import ARCTester
from arc_agi_benchmarking.adapters.open_ai import OpenAIAdapter
from arc_agi_benchmarking.adapters.random import RandomAdapter


def test_provider_request_acquires_before_calling_sdk():
    adapter = RandomAdapter.__new__(RandomAdapter)
    adapter.model_config = SimpleNamespace(provider="test-provider")
    events = []
    adapter.request_limiter = MagicMock(
        acquire=MagicMock(side_effect=lambda: events.append("acquire") or 0)
    )

    result = adapter._request(
        "test.create", lambda: events.append("request") or "response"
    )

    assert result == "response"
    assert events == ["acquire", "request"]


def test_openai_request_boundary_consumes_one_allowance():
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.model_config = SimpleNamespace(
        provider="openai",
        model_name="test-model",
        kwargs={},
    )
    adapter.request_limiter = MagicMock()
    adapter.request_limiter.acquire.return_value = 0
    adapter.client = MagicMock()
    adapter.client.chat.completions.create.return_value = "response"

    assert adapter._chat_completion([{"role": "user", "content": "hello"}]) == "response"
    adapter.request_limiter.acquire.assert_called_once_with()
    adapter.client.chat.completions.create.assert_called_once()


def test_openai_background_control_requests_each_consume_allowance():
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.model_config = SimpleNamespace(
        provider="openai",
        model_name="test-model",
        kwargs={"background": True},
    )
    adapter.request_limiter = MagicMock()
    adapter.request_limiter.acquire.return_value = 0
    adapter.client = MagicMock()
    queued = SimpleNamespace(id="response-id", status="queued")
    completed = SimpleNamespace(id="response-id", status="completed")
    adapter.client.responses.create.return_value = queued
    adapter.client.responses.retrieve.return_value = completed

    with patch("arc_agi_benchmarking.adapters.openai_base.sleep"):
        assert adapter._responses([{"role": "user", "content": "hello"}]) is completed

    assert adapter.request_limiter.acquire.call_count == 3
    adapter.client.responses.create.assert_called_once()
    adapter.client.responses.retrieve.assert_called_once_with("response-id")
    adapter.client.responses.delete.assert_called_once_with("response-id")


def test_arc_tester_passes_shared_limiter_to_adapter():
    limiter = MagicMock()
    model_config = SimpleNamespace(provider="test-provider")
    captured = {}

    class FakeAdapter:
        def __init__(self, config, request_limiter=None):
            captured["config"] = config
            captured["request_limiter"] = request_limiter

    with (
        patch("main.utils.read_models_config", return_value=model_config),
        patch.dict("main.PROVIDER_ADAPTERS", {"test-provider": FakeAdapter}),
    ):
        ARCTester(
            config="test-config",
            save_submission_dir="submissions",
            overwrite_submission=False,
            print_submission=False,
            num_attempts=1,
            retry_attempts=1,
            request_limiter=limiter,
        )

    assert captured == {
        "config": "test-config",
        "request_limiter": limiter,
    }
