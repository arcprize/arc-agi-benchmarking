"""Exercise continuation through the real SDK with an in-memory HTTP server."""
import copy
import json
from unittest.mock import Mock, patch

import httpx
import pytest
from google import genai
from google.genai import types

from arc_agi_benchmarking.adapters.gemini import GeminiAdapter
from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing


def slice_response(reason, parts=None, token=None, usage=None):
    candidate = {"finishReason": reason, "content": {"role": "model", "parts": parts or []}}
    if token is not None:
        candidate["continuationToken"] = token
    return {"candidates": [candidate], "usageMetadata": usage or {}}


@pytest.fixture
def adapter_factory(monkeypatch):
    clients = []
    monkeypatch.setenv("TEST_GEMINI_KEY", "test-key")

    def make(responses, **overrides):
        requests = []
        responses = iter(responses)

        def handle(request):
            requests.append(request)
            result = next(responses)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, httpx.Response):
                return result
            return httpx.Response(200, json=result)

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
            httpx_client=http_client, retry_options=types.HttpRetryOptions(attempts=1)))
        clients.append(client)
        config = ModelConfig(
            name="test-continuation", model_name="models/test-model", provider="gemini",
            api_key_env="TEST_GEMINI_KEY", pricing=ModelPricing(date="2026-09-29", input=2, output=4),
            kwargs={"max_output_tokens": 1000000,
                    "http_options": {"timeout": 3600000}, **overrides},
        )
        with patch("arc_agi_benchmarking.adapters.provider.read_models_config", return_value=config), patch(
            "arc_agi_benchmarking.adapters.gemini.genai.Client", return_value=client
        ):
            adapter = GeminiAdapter("test-continuation", request_limiter=Mock(acquire=Mock(return_value=0)), raw_api_logger=Mock())
        return adapter, requests

    yield make
    for client in clients:
        client.close()


def test_multislice_wire_protocol_accounting_and_local_grid(adapter_factory):
    thought = {"text": "reasoning", "thought": True, "thoughtSignature": "opaque"}
    responses = [
        slice_response("CONTINUATION", [thought], "token-one", {"promptTokenCount": 10, "thoughtsTokenCount": 30}),
        slice_response("CONTINUATION", [{"text": "[["}], "token-two", {"promptTokenCount": 20, "candidatesTokenCount": 2, "thoughtsTokenCount": 40}),
        slice_response("STOP", [{"text": "1]]"}], usage={"promptTokenCount": 30, "candidatesTokenCount": 3, "thoughtsTokenCount": None}),
    ]
    adapter, requests = adapter_factory(responses)
    original = copy.deepcopy(adapter.model_config.kwargs)
    attempt = adapter.make_prediction("Solve the grid", task_id="test", pair_index=0)
    assert attempt.answer == [[1]]
    assert attempt.metadata.choices[-1].message.content == "[[1]]"
    usage = attempt.metadata.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (60, 5, 135)
    assert usage.completion_tokens_details.reasoning_tokens == 70
    assert attempt.metadata.cost.total_cost == pytest.approx((60 * 2 + 75 * 4) / 1000000)
    bodies = [json.loads(request.content) for request in requests]
    assert 'continuationToken' not in bodies[0]
    assert bodies[1]['continuationToken'] == 'token-one'
    assert bodies[2]['continuationToken'] == 'token-two'
    assert bodies[1]['contents'] == bodies[0]['contents'] + [{"role": "model", "parts": [thought]}]
    assert bodies[2]['contents'][-1]['parts'] == [thought, {"text": "[["}]
    for request, body in zip(requests, bodies):
        assert body['generationConfig']['maxOutputTokens'] == 1000000
        assert 'continuation' not in body['generationConfig']
        assert request.extensions['timeout']['read'] == 3600
    assert adapter.model_config.kwargs == original
    assert adapter.request_limiter.acquire.call_count == 3
    assert adapter.raw_api_logger.record_success.call_count == 3
    assert adapter.raw_api_logger.record_success.call_args_list[0].args[1] == responses[0]
    assert adapter.extract_json_from_response('```json\n[[1]]\n```') == [[1]]
    assert adapter.extract_json_from_response('no grid') is None
    assert len(requests) == 3


@pytest.mark.parametrize('reason', ['STOP', 'MAX_TOKENS'])
def test_single_terminal_response(adapter_factory, reason):
    adapter, requests = adapter_factory([slice_response(reason, [{"text": "[[2]]"}, {"inlineData": {"data": "AA=="}}])])
    response = adapter.chat_completion([{"role": "user", "content": "test"}])
    assert response.text == '[[2]]'
    assert response.finish_reason == reason
    assert response.usage_metadata.total_token_count == 0
    assert len(requests) == 1


@pytest.mark.parametrize('response, message', [
    (slice_response('CONTINUATION', [{"text": "[[1]]"}]), 'no continuation token'),
    ({'candidates': []}, 'no candidates'),
    (slice_response('SAFETY', [{"text": "[[1]]"}]), 'Unexpected Gemini finish reason'),
    (slice_response(None), 'Unexpected Gemini finish reason'),
])
def test_invalid_response_never_returns_partial_answer(adapter_factory, response, message):
    adapter, _ = adapter_factory([response])
    with pytest.raises(ValueError, match=message):
        adapter.chat_completion([{"role": "user", "content": "test"}])


def test_repeated_token_stops_loop(adapter_factory):
    adapter, requests = adapter_factory([slice_response('CONTINUATION', token='same')] * 2)
    with pytest.raises(ValueError, match='repeated continuation token'):
        adapter.chat_completion([{"role": "user", "content": "test"}])
    assert len(requests) == 2


def test_http_failure_after_slice_is_logged_and_propagated(adapter_factory):
    adapter, requests = adapter_factory([
        slice_response('CONTINUATION', [{"text": "[[1]]"}], 'token'),
        httpx.Response(504, json={"error": {"code": 504, "message": "deadline exceeded", "status": "DEADLINE_EXCEEDED"}}),
    ])
    with pytest.raises(Exception, match='504'):
        adapter.chat_completion([{"role": "user", "content": "test"}])
    assert len(requests) == 2
    assert adapter.raw_api_logger.record_success.call_count == 1
    assert adapter.raw_api_logger.record_failure.call_count == 1


@pytest.mark.parametrize('budget', [None, 0])
def test_continuation_requires_explicit_budget(adapter_factory, budget):
    adapter, requests = adapter_factory([slice_response('CONTINUATION', token='token')], max_output_tokens=budget)
    with pytest.raises(ValueError, match='explicit positive'):
        adapter.chat_completion([{'role': 'user', 'content': 'test'}])
    assert len(requests) == 1


def test_single_response_allows_default_budget(adapter_factory):
    adapter, requests = adapter_factory([slice_response('STOP', [{'text': '[[3]]'}])], max_output_tokens=None)
    assert adapter.make_prediction('test', pair_index=0).answer == [[3]]
    assert len(requests) == 1


def test_normal_gemini_path_unchanged(adapter_factory):
    adapter, requests = adapter_factory([slice_response('STOP', [{"text": "[[3]]"}])])
    assert adapter.make_prediction('test', pair_index=0).answer == [[3]]
    assert len(requests) == 1
    assert 'continuationToken' not in json.loads(requests[0].content)


def test_legacy_false_flag_does_not_disable_continuation(adapter_factory):
    adapter, requests = adapter_factory([
        slice_response('CONTINUATION', token='token'),
        slice_response('STOP', [{'text': '[[1]]'}]),
    ], continuation=False)
    assert adapter.make_prediction('test', pair_index=0).answer == [[1]]
    assert len(requests) == 2
