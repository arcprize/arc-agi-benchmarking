"""Exercise real SDK parsing, including usage omitted by its stream accumulator."""

import json
from unittest.mock import Mock

import anthropic
from anthropic.lib.streaming import BetaMessageStream, MessageStream
import httpx
import pytest

from arc_agi_benchmarking.adapters.anthropic import AnthropicAdapter
from arc_agi_benchmarking.schemas import ModelConfig
from arc_agi_benchmarking.utils.logging_utils import RawAPILogger


def _message(usage):
    return {
        "id": "msg_usage_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "content": [{"type": "text", "text": "[[1]]"}],
        "usage": usage,
    }


def _stream_response(deltas):
    events = [
        {
            "type": "message_start",
            "message": {
                **_message(
                    {
                        "input_tokens": 101,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 12,
                        "cache_read_input_tokens": 5,
                    }
                ),
                "content": [],
                "stop_reason": None,
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "[[1]]"},
        },
        {"type": "content_block_stop", "index": 0},
        *[
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": usage,
            }
            for usage in deltas
        ],
        {"type": "message_stop"},
    ]
    body = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _predict(handler, streaming, beta):
    config = ModelConfig.model_validate(
        {
            "name": "sdk-usage-test",
            "provider": "anthropic",
            "model_name": "claude-opus-4-7",
            "api_key_env": "ANTHROPIC_API_KEY",
            "stream": streaming,
            "max_tokens": 1024,
            "pricing": {"date": "2026-10-21", "input": 5.0, "output": 25.0},
            **({"betas": ["thinking-token-count-2026-05-13"]} if beta else {}),
        }
    )
    with anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.model_config = config
        adapter.client = client
        adapter.raw_api_logger = RawAPILogger()
        adapter.raw_api_logger._emit = Mock()
        attempt = adapter.make_prediction("Return a grid", task_id="synthetic")

    events = [call.args[0] for call in adapter.raw_api_logger._emit.call_args_list]
    assert [event["event"] for event in events] == [
        "request_started",
        "request_succeeded",
    ]
    assert events[0]["operation"] == (
        ("beta." if beta else "")
        + ("messages.stream" if streaming else "messages.create")
    )
    return json.loads(attempt.model_dump_json()), events[1]["response"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("thinking_tokens", [None, 0, 100])
def test_sdk_usage_reaches_saved_attempt_and_raw_log(streaming, beta, thinking_tokens):
    usage = {
        "input_tokens": 101,
        "output_tokens": 200,
        "cache_creation_input_tokens": 12,
        "cache_read_input_tokens": 5,
    }
    if thinking_tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}

    def handler(request):
        assert json.loads(request.content).get("stream", False) == streaming
        if beta:
            assert (
                "thinking-token-count-2026-05-13" in request.headers["anthropic-beta"]
            )
        if streaming:
            return _stream_response([usage])
        return httpx.Response(200, json=_message(usage))

    attempt, raw_response = _predict(handler, streaming, beta)
    saved_usage = attempt["metadata"]["usage"]
    assert attempt["answer"] == [[1]]
    assert saved_usage["prompt_tokens"] == 101
    assert saved_usage["completion_tokens"] == 200
    assert saved_usage["total_tokens"] == 301
    assert saved_usage["completion_tokens_details"]["reasoning_tokens"] == (
        thinking_tokens or 0
    )
    assert saved_usage["completion_tokens_details"]["accepted_prediction_tokens"] == 200
    assert saved_usage["prompt_tokens_details"]["cache_creation_tokens"] == 12
    assert saved_usage["prompt_tokens_details"]["cache_read_tokens"] == 5
    assert attempt["metadata"]["cost"]["total_cost"] == pytest.approx(0.005505)
    assert raw_response["usage"]["input_tokens"] == 101
    assert raw_response["usage"]["output_tokens"] == 200
    assert len(raw_response["content"]) == 1
    assert raw_response["content"][0]["type"] == "text"
    assert raw_response["content"][0]["text"] == "[[1]]"
    if thinking_tokens is not None:
        assert (
            raw_response["usage"]["output_tokens_details"]["thinking_tokens"]
            == thinking_tokens
        )


@pytest.mark.parametrize("beta", [False, True])
def test_sdk_keeps_thinking_from_partial_deltas_with_final_totals(beta):
    deltas = [
        {"output_tokens": 150, "output_tokens_details": {"thinking_tokens": 100}},
        {"output_tokens": 200, "output_tokens_details": {"thinking_tokens": None}},
    ]
    attempt, raw_response = _predict(
        lambda request: _stream_response(deltas), True, beta
    )
    usage = attempt["metadata"]["usage"]
    assert usage["prompt_tokens"] == 101
    assert usage["completion_tokens"] == 200
    assert usage["total_tokens"] == 301
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 100
    assert usage["prompt_tokens_details"]["cache_creation_tokens"] == 12
    assert usage["prompt_tokens_details"]["cache_read_tokens"] == 5
    assert raw_response["usage"]["output_tokens_details"]["thinking_tokens"] == 100


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("final_thinking,expected", [(None, 100), (0, 0), (80, 80)])
def test_stream_final_usage_is_authoritative_when_present(
    monkeypatch, beta, final_thinking, expected
):
    # Simulate an SDK that preserves some or all final usage details. A missing
    # nested count needs the stream fallback, but an explicit zero must win.
    stream_class = BetaMessageStream if beta else MessageStream
    original = stream_class.get_final_message

    def get_final_message(stream):
        message = original(stream)
        return message.model_copy(
            update={
                "usage": message.usage.model_copy(
                    update={
                        "output_tokens": 250,
                        "output_tokens_details": {
                            "thinking_tokens": final_thinking,
                            "text_tokens": 150,
                        },
                    }
                ),
            }
        )

    monkeypatch.setattr(stream_class, "get_final_message", get_final_message)
    deltas = [{"output_tokens": 200, "output_tokens_details": {"thinking_tokens": 100}}]
    attempt, raw_response = _predict(
        lambda request: _stream_response(deltas), True, beta
    )
    usage = attempt["metadata"]["usage"]
    assert usage["completion_tokens"] == 250
    assert usage["total_tokens"] == 351
    assert usage["completion_tokens_details"]["reasoning_tokens"] == expected
    assert attempt["metadata"]["cost"]["total_cost"] == pytest.approx(0.006755)
    assert raw_response["usage"]["output_tokens_details"] == {
        "thinking_tokens": expected,
        "text_tokens": 150,
    }
