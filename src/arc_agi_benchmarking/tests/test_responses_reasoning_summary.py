import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI
from openai.types.responses import Response, ResponseReasoningSummaryTextDeltaEvent

from arc_agi_benchmarking.adapters.open_ai import OpenAIAdapter
from arc_agi_benchmarking.schemas import APIType, ModelConfig, ModelPricing


def response_payload(summary=True):
    output = []
    if summary:
        output.append(
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "First paragraph."},
                    {"type": "summary_text", "text": "Second paragraph."},
                ],
            }
        )
    output.append(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "[[1]]", "annotations": []}],
        }
    )
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "test-model",
        "output": output,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 5},
        },
    }


@pytest.mark.parametrize("mode", ["regular", "background", "stream"])
@pytest.mark.parametrize("has_summary", [True, False])
def test_summary_persisted_without_changing_request(monkeypatch, mode, has_summary):
    """Exercise real SDK decoding and the full prediction metadata path offline."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    kwargs = {"reasoning": {"effort": "medium", "summary": "auto"}}
    if mode == "stream":
        kwargs["stream"] = True
    if mode == "background":
        kwargs["background"] = True
    config = ModelConfig(
        name="test-model",
        model_name="test-model",
        provider="openai",
        api_type=APIType.RESPONSES,
        api_key_env="OPENAI_API_KEY",
        pricing=ModelPricing(date="2026-01-01", input=1, output=2),
        kwargs=deepcopy(kwargs),
    )
    payload = response_payload(has_summary)
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "resp_1", "deleted": True})
        if mode == "stream":
            events = [
                {
                    "type": "response.created",
                    "response": {
                        **payload,
                        "status": "in_progress",
                        "output": [],
                        "usage": None,
                    },
                    "sequence_number": 0,
                },
                {
                    "type": "response.output_text.delta",
                    "delta": "[[1]]",
                    "item_id": "msg_1",
                    "output_index": 1,
                    "content_index": 0,
                    "sequence_number": 1,
                },
            ]
            if has_summary:
                # Completed output must take precedence over streamed fragments.
                events.append(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "delta": "First ",
                        "item_id": "rs_1",
                        "output_index": 0,
                        "summary_index": 0,
                        "sequence_number": 2,
                    }
                )
            events.append(
                {
                    "type": "response.completed",
                    "response": payload,
                    "sequence_number": 3,
                }
            )
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            return httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        if mode == "background" and request.method == "POST":
            return httpx.Response(
                200, json={**payload, "status": "queued", "output": []}
            )
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        "arc_agi_benchmarking.adapters.provider.read_models_config", lambda _: config
    )
    monkeypatch.setattr(
        "arc_agi_benchmarking.adapters.openai_base.sleep", lambda _: None
    )
    adapter = OpenAIAdapter("test-model")
    adapter.client.close()
    with OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        adapter.client = client
        attempt = adapter.make_prediction("Solve this grid", pair_index=0)

    saved = json.loads(attempt.model_dump_json())
    expected = "First paragraph.\n\nSecond paragraph." if has_summary else None
    assert saved["metadata"]["reasoning_summary"] == expected
    assert attempt.answer == [[1]]
    assert attempt.metadata.choices[-1].message.content == "[[1]]"
    assert attempt.metadata.usage.total_tokens == 18
    assert attempt.metadata.usage.completion_tokens_details.reasoning_tokens == 5
    assert config.kwargs == kwargs
    body = json.loads(requests[0].content)
    assert body["reasoning"] == kwargs["reasoning"]
    assert "text" not in body
    assert [r.method for r in requests] == (
        ["POST", "GET", "DELETE"] if mode == "background" else ["POST"]
    )


def test_stream_summary_deltas_keep_item_and_paragraph_order(monkeypatch):
    """Streams without final output still retain summaries, without extra requests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ModelConfig(
        name="test-model",
        model_name="test-model",
        provider="openai",
        api_type=APIType.RESPONSES,
        api_key_env="OPENAI_API_KEY",
        pricing=ModelPricing(date="2026-01-01", input=1, output=2),
    )
    monkeypatch.setattr(
        "arc_agi_benchmarking.adapters.provider.read_models_config", lambda _: config
    )
    adapter = OpenAIAdapter("test-model")
    adapter.client.close()
    events = [
        ResponseReasoningSummaryTextDeltaEvent(
            type="response.reasoning_summary_text.delta",
            sequence_number=index,
            item_id=f"rs_{item}",
            output_index=item,
            summary_index=part,
            delta=text,
        )
        for index, (item, part, text) in enumerate(
            [
                (1, 0, "Later item."),
                (0, 1, "Second paragraph."),
                (0, 0, "First "),
                (0, 0, "paragraph."),
            ]
        )
    ]
    adapter.client = Mock()
    adapter.client.responses.create.return_value = iter(events)
    response = adapter._responses_stream([])
    assert adapter._get_reasoning_summary(response) == (
        "First paragraph.\n\nSecond paragraph.\n\nLater item."
    )
    adapter.client.responses.retrieve.assert_not_called()


def test_summary_extraction_ignores_options_and_non_reasoning_output():
    adapter = object.__new__(OpenAIAdapter)
    adapter.model_config = SimpleNamespace(api_type=APIType.RESPONSES)
    payload = response_payload()
    payload["output"].append(
        {
            "id": "rs_2",
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": ""},
                {"type": "summary_text", "text": "Later item."},
            ],
        }
    )
    response = Response.model_validate(payload)
    assert adapter._get_reasoning_summary(response) == (
        "First paragraph.\n\nSecond paragraph.\n\nLater item."
    )
    adapter.model_config.api_type = APIType.CHAT_COMPLETIONS
    assert adapter._get_reasoning_summary(response) is None
