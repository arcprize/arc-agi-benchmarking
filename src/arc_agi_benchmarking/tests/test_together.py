from unittest.mock import MagicMock, patch

import pytest
from together.types.chat.chat_completion import ChatCompletion, Choice, ChoiceMessage
from together.types.chat.chat_completion_usage import ChatCompletionUsage

from arc_agi_benchmarking.errors import TokenMismatchError
from arc_agi_benchmarking.schemas import APIType, ModelConfig, ModelPricing


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="test-together-model",
        model_name="deepseek-ai/DeepSeek-V4-Pro",
        provider="together",
        pricing=ModelPricing(date="2026-04-27", input=1.0, output=2.0),
        api_type=APIType.CHAT_COMPLETIONS,
        kwargs={"temperature": 0.0},
    )


def _raw_together_response(
    content: str = "[[1]]",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int = 150,
    reasoning: str | None = None,
    usage: ChatCompletionUsage | None = None,
) -> ChatCompletion:
    return ChatCompletion(
        id="resp_1",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChoiceMessage(
                    content=content,
                    role="assistant",
                    reasoning=reasoning,
                ),
            )
        ],
        created=1,
        model="deepseek-ai/DeepSeek-V4-Pro",
        object="chat.completion",
        prompt=[],
        usage=usage
        if usage is not None
        else ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


@pytest.fixture
def adapter(monkeypatch):
    from arc_agi_benchmarking.adapters.together import TogetherAdapter

    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")

    with patch(
        "arc_agi_benchmarking.adapters.provider.read_models_config",
        return_value=_model_config(),
    ), patch("arc_agi_benchmarking.adapters.together.Together"):
        return TogetherAdapter("test-together-model")


def test_together_sdk_exposes_chat_completions_shape():
    from together import Together

    client = Together(api_key="test-key")

    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")


def test_together_adapter_can_be_imported():
    from arc_agi_benchmarking.adapters.together import TogetherAdapter

    assert TogetherAdapter.__name__ == "TogetherAdapter"


def test_together_adapter_is_exported_from_adapters_package():
    from arc_agi_benchmarking.adapters import TogetherAdapter

    assert TogetherAdapter.__name__ == "TogetherAdapter"


def test_together_provider_is_registered_in_main():
    from arc_agi_benchmarking.adapters.together import TogetherAdapter
    from main import PROVIDER_ADAPTERS

    assert PROVIDER_ADAPTERS["together"] is TogetherAdapter


def test_arctester_initializes_together_provider(monkeypatch):
    from arc_agi_benchmarking.adapters.together import TogetherAdapter
    from main import ARCTester

    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")

    with patch("main.utils.read_models_config", return_value=_model_config()), patch(
        "arc_agi_benchmarking.adapters.provider.read_models_config",
        return_value=_model_config(),
    ), patch("arc_agi_benchmarking.adapters.together.Together"):
        tester = ARCTester(
            config="test-together-model",
            save_submission_dir="submissions/test",
            overwrite_submission=True,
            print_submission=False,
            num_attempts=1,
            retry_attempts=1,
        )

    assert isinstance(tester.provider, TogetherAdapter)
    assert tester.model_config.provider == "together"


def test_together_adapter_initializes_with_api_key(monkeypatch):
    from arc_agi_benchmarking.adapters.together import TogetherAdapter

    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")

    mock_client = MagicMock()
    with patch(
        "arc_agi_benchmarking.adapters.provider.read_models_config",
        return_value=_model_config(),
    ), patch(
        "arc_agi_benchmarking.adapters.together.Together",
        return_value=mock_client,
    ) as mock_together:
        adapter = TogetherAdapter("test-together-model")

    mock_together.assert_called_once_with(api_key="test-key")
    assert adapter.client is mock_client
    assert adapter.model_config.provider == "together"


def test_together_adapter_requires_api_key(monkeypatch):
    from arc_agi_benchmarking.adapters.together import TogetherAdapter

    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    with patch(
        "arc_agi_benchmarking.adapters.provider.read_models_config",
        return_value=_model_config(),
    ), pytest.raises(ValueError, match="TOGETHER_API_KEY not found"):
        TogetherAdapter("test-together-model")


def test_call_together_model_uses_chat_completions_and_filters_config_kwargs(adapter):
    adapter.model_config.kwargs = {
        "temperature": 0.0,
        "max_tokens": 128,
        "rate_limit": {"rate": 1, "period": 60},
        "enable_thinking": True,
    }
    adapter.client = MagicMock()
    adapter.client.chat.completions.create.return_value = _raw_together_response()

    response = adapter._call_together_model("prompt text")

    assert response.choices[0].message.content == "[[1]]"
    adapter.client.chat.completions.create.assert_called_once_with(
        model="deepseek-ai/DeepSeek-V4-Pro",
        messages=[{"role": "user", "content": "prompt text"}],
        temperature=0.0,
        max_tokens=128,
    )


def test_call_together_model_rejects_streaming_until_supported(adapter):
    adapter.model_config.kwargs = {"stream": True}

    with pytest.raises(ValueError, match="streaming is not supported"):
        adapter._call_together_model("prompt text")


def test_make_prediction_normalizes_raw_together_response_and_costs(adapter):
    raw_response = _raw_together_response(
        content="Here is the answer: [[1]]",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=160,
        reasoning="short reasoning summary",
    )

    with patch.object(adapter, "_call_together_model", return_value=raw_response):
        attempt = adapter.make_prediction(
            "prompt text",
            task_id="task-1",
            test_id="test-config",
            pair_index=2,
        )

    assert attempt.answer == [[1]]
    assert attempt.metadata.model == "deepseek-ai/DeepSeek-V4-Pro"
    assert attempt.metadata.provider == "together"
    assert attempt.metadata.task_id == "task-1"
    assert attempt.metadata.test_id == "test-config"
    assert attempt.metadata.pair_index == 2
    assert attempt.metadata.reasoning_summary == "short reasoning summary"
    assert attempt.metadata.choices[0].message.content == "prompt text"
    assert attempt.metadata.choices[1].message.content == "Here is the answer: [[1]]"

    usage = attempt.metadata.usage
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 160
    assert usage.completion_tokens_details.reasoning_tokens == 10

    assert attempt.metadata.cost.prompt_cost == pytest.approx(100 * 1.0 / 1_000_000)
    assert attempt.metadata.cost.completion_cost == pytest.approx(
        50 * 2.0 / 1_000_000
    )
    assert attempt.metadata.cost.reasoning_cost == pytest.approx(
        10 * 2.0 / 1_000_000
    )
    assert attempt.metadata.cost.total_cost == pytest.approx(
        (100 * 1.0 / 1_000_000)
        + (50 * 2.0 / 1_000_000)
        + (10 * 2.0 / 1_000_000)
    )


def test_make_prediction_defaults_missing_usage_to_zero_cost(adapter):
    raw_response = _raw_together_response(content="[[2]]", usage=None)
    raw_response.usage = None

    with patch.object(adapter, "_call_together_model", return_value=raw_response):
        attempt = adapter.make_prediction("prompt text")

    assert attempt.answer == [[2]]
    assert attempt.metadata.usage.total_tokens == 0
    assert attempt.metadata.cost.total_cost == 0.0


def test_make_prediction_raises_when_total_tokens_are_inconsistent(adapter):
    raw_response = _raw_together_response(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=140,
    )

    with patch.object(adapter, "_call_together_model", return_value=raw_response):
        with pytest.raises(TokenMismatchError, match="Token count mismatch"):
            adapter.make_prediction("prompt text")


def test_cost_calculation_splits_explicit_reasoning_inside_completion(adapter):
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 60,
            "total_tokens": 160,
            "completion_tokens_details": {"reasoning_tokens": 10},
        }
    }

    usage = adapter._get_usage(response)
    cost = adapter._calculate_cost(usage)

    assert usage.completion_tokens_details.reasoning_tokens == 10
    assert cost.prompt_cost == pytest.approx(100 * 1.0 / 1_000_000)
    assert cost.completion_cost == pytest.approx(50 * 2.0 / 1_000_000)
    assert cost.reasoning_cost == pytest.approx(10 * 2.0 / 1_000_000)
    assert cost.total_cost == pytest.approx(
        (100 * 1.0 / 1_000_000)
        + (50 * 2.0 / 1_000_000)
        + (10 * 2.0 / 1_000_000)
    )


def test_usage_handles_together_direct_reasoning_tokens(adapter):
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 60,
            "total_tokens": 160,
            "reasoning_tokens": 10,
        }
    }

    usage = adapter._get_usage(response)
    cost = adapter._calculate_cost(usage)

    assert usage.completion_tokens_details.reasoning_tokens == 10
    assert cost.completion_cost == pytest.approx(50 * 2.0 / 1_000_000)
    assert cost.reasoning_cost == pytest.approx(10 * 2.0 / 1_000_000)


def test_usage_raises_when_explicit_reasoning_conflicts_with_total(adapter):
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 160,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
    }

    with pytest.raises(TokenMismatchError, match="Token count mismatch"):
        adapter._get_usage(response)


def test_extract_json_from_response_uses_shared_parser(adapter):
    assert adapter.extract_json_from_response("final answer: [[3, 4]]") == [[3, 4]]
    assert adapter.extract_json_from_response("not json") is None
