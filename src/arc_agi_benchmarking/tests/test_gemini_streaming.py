"""Streaming wire-protocol regressions through the real SDK, without API calls."""

import base64
import copy
import json
import logging
from contextlib import nullcontext
from io import StringIO

import httpx
import pytest

from arc_agi_benchmarking.adapters.gemini_streaming import (
    MAX_STREAM_RECOVERY_RETRIES,
    GeminiStreamError,
)
from arc_agi_benchmarking.tests.test_gemini_continuation import (
    adapter_factory as adapter_factory,
)
from arc_agi_benchmarking.tests.test_gemini_continuation import (
    slice_response,
)
from arc_agi_benchmarking.utils.logging_utils import RawAPILogger, StructuredFormatter


def token(value):
    return base64.b64encode(value.encode()).decode()


def usage(prompt=10, output=1, thoughts=2):
    return {
        "promptTokenCount": prompt,
        "candidatesTokenCount": output,
        "thoughtsTokenCount": thoughts,
        "totalTokenCount": prompt + output + thoughts,
    }


class EventStream(httpx.SyncByteStream):
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield ("data: " + json.dumps(event) + "\n\n").encode()

    def close(self):
        self.closed = True


def sse(*events):
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=EventStream(events)
    )


def test_stream_continuation_before_grid_keeps_same_prediction_and_counts_usage(
    adapter_factory,
):
    thought = {
        "text": "reasoning",
        "thought": True,
        "thoughtSignature": token("signature"),
    }
    responses = [
        sse(
            slice_response(None, [thought], usage=usage(10, 0, 3)),
            slice_response(
                "CONTINUATION", [{"text": "[["}], token("one"), usage(10, 1, 5)
            ),
            {"usageMetadata": usage(10, 1, 5)},  # A trailing usage-only event.
        ),
        sse(
            slice_response(
                "CONTINUATION", [{"text": "1"}], token("two"), usage(20, 1, 4)
            )
        ),
        sse(slice_response("STOP", [{"text": "]]"}], usage=usage(30, 2, 1))),
    ]
    adapter, requests = adapter_factory(
        responses, stream=True, automatic_continuation=True
    )
    original = copy.deepcopy(adapter.model_config.kwargs)
    attempt = adapter.make_prediction(
        "Solve the grid", task_id="mock-task", pair_index=0
    )
    assert attempt.answer == [[1]]
    assert attempt.metadata.choices[-1].message.content == "[[1]]"
    counts = attempt.metadata.usage
    assert (counts.prompt_tokens, counts.completion_tokens, counts.total_tokens) == (
        60,
        4,
        74,
    )
    assert counts.completion_tokens_details.reasoning_tokens == 10
    assert attempt.metadata.cost.total_cost == pytest.approx(
        (60 * 2 + 14 * 4) / 1000000
    )
    bodies = [json.loads(r.content) for r in requests]
    assert len(bodies) == 3  # No hidden SDK continuation bypassing _request.
    assert "continuationToken" not in bodies[0]
    assert bodies[1]["continuationToken"] == token("one")
    assert bodies[2]["continuationToken"] == token("two")
    assert bodies[1]["contents"] == bodies[0]["contents"] + [
        {"role": "model", "parts": [thought, {"text": "[["}]}
    ]
    assert bodies[2]["contents"][-1]["parts"] == [
        thought,
        {"text": "[["},
        {"text": "1"},
    ]
    for request, body in zip(requests, bodies):
        assert body["generationConfig"]["maxOutputTokens"] == 1000000
        assert request.extensions["timeout"]["read"] == 3600
    assert adapter.request_limiter.acquire.call_count == 3
    assert adapter.raw_api_logger.start_request.call_count == 3
    assert adapter.raw_api_logger.record_success.call_count == 3
    assert adapter.raw_api_logger.record_failure.call_count == 0
    assert adapter.model_config.kwargs == original
    assert adapter._deferred_raw_api_requests == {}


@pytest.mark.parametrize("disconnect", [False, True])
def test_midstream_checkpoint_recovery_discards_uncommitted_text_but_not_usage(
    adapter_factory,
    disconnect,
):
    thought = {
        "text": "reasoning",
        "thought": True,
        "thoughtSignature": token("signature"),
    }
    events = [
        slice_response(None, [thought, {"text": "[["}], token("one"), usage(10, 1, 3)),
        slice_response(None, [{"text": "discard me"}], usage=usage(10, 2, 4)),
    ]
    if disconnect:
        events.append(httpx.ReadError("connection lost"))
    first = sse(*events)
    adapter, requests = adapter_factory(
        [
            first,
            sse(slice_response("STOP", [{"text": "2]]"}], usage=usage(20, 2, 5))),
        ],
        stream=True,
    )
    attempt = adapter.make_prediction("test", pair_index=0)
    assert attempt.answer == [[2]]
    assert len(requests) == 2
    resumed = json.loads(requests[1].content)
    assert resumed["continuationToken"] == token("one")
    assert resumed["contents"][-1]["parts"] == [thought, {"text": "[["}]
    assert attempt.metadata.usage.total_tokens == 43
    assert attempt.metadata.usage.completion_tokens_details.reasoning_tokens == 9
    assert adapter.raw_api_logger.record_failure.call_count == int(disconnect)
    if disconnect:
        error = adapter.raw_api_logger.record_failure.call_args.args[1]
        assert error.usage_metadata.candidates_token_count == 2
    assert adapter._deferred_raw_api_requests == {}


def test_interrupted_resumed_hop_reuses_previous_checkpoint_without_duplicate_output(
    adapter_factory,
):
    adapter, requests = adapter_factory(
        [
            sse(
                slice_response("CONTINUATION", [{"text": "[["}], token("one"), usage())
            ),
            sse(
                slice_response(None, [{"text": "discard"}], usage=usage(20, 3, 4)),
                httpx.ReadError("connection lost"),
            ),
            sse(slice_response("STOP", [{"text": "3]]"}], usage=usage(30, 2, 5))),
        ],
        stream=True,
    )
    attempt = adapter.make_prediction("test", pair_index=0)
    assert attempt.answer == [[3]]
    second, third = [json.loads(r.content) for r in requests[1:]]
    assert second["continuationToken"] == third["continuationToken"] == token("one")
    assert second["contents"] == third["contents"]
    assert third["contents"][-1]["parts"] == [{"text": "[["}]
    assert attempt.metadata.usage.total_tokens == 77


@pytest.mark.parametrize(
    "reason", ["SAFETY", "MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE", None]
)
def test_failed_or_unfinished_stream_never_accepts_provisional_grid(
    adapter_factory, reason
):
    adapter, requests = adapter_factory(
        [
            sse(slice_response(reason, [{"text": "[[1]]"}], usage=usage())),
        ],
        stream=True,
    )
    warning = (
        pytest.warns(
            UserWarning, match="MALFORMED_RESPONSE is not a valid FinishReason"
        )
        if reason == "MALFORMED_RESPONSE"
        else nullcontext()
    )
    with (
        warning,
        pytest.raises(
            GeminiStreamError,
            match="Unexpected Gemini stream finish reason",
        ) as exc,
    ):
        adapter.make_prediction("test", pair_index=0)
    assert len(requests) == 1
    assert exc.value.usage_metadata.total_token_count == 13


def test_continuation_without_token_is_error_even_with_parseable_grid(adapter_factory):
    adapter, requests = adapter_factory(
        [
            sse(slice_response("CONTINUATION", [{"text": "[[1]]"}], usage=usage())),
        ],
        stream=True,
    )
    with pytest.raises(GeminiStreamError, match="no continuation token"):
        adapter.make_prediction("test", pair_index=0)
    assert len(requests) == 1


def test_repeated_token_stops_without_a_third_request(adapter_factory):
    adapter, requests = adapter_factory(
        [
            sse(slice_response("CONTINUATION", token=token("same"), usage=usage())),
            sse(slice_response("CONTINUATION", token=token("same"), usage=usage(20))),
        ],
        stream=True,
    )
    with pytest.raises(GeminiStreamError, match="repeated continuation token") as exc:
        adapter.make_prediction("test", pair_index=0)
    assert len(requests) == 2
    assert exc.value.usage_metadata.total_token_count == 36


@pytest.mark.parametrize("budget", [None, 0, -1, True])
def test_stream_continuation_requires_explicit_positive_budget(adapter_factory, budget):
    adapter, requests = adapter_factory(
        [
            sse(slice_response("CONTINUATION", token=token("one"), usage=usage())),
        ],
        stream=True,
        max_output_tokens=budget,
    )
    with pytest.raises(GeminiStreamError, match="explicit positive"):
        adapter.make_prediction("test", pair_index=0)
    assert len(requests) == 1


@pytest.mark.parametrize("reason", ["STOP", "MAX_TOKENS"])
def test_single_stream_terminal_response_does_not_need_explicit_budget(
    adapter_factory, reason
):
    adapter, requests = adapter_factory(
        [
            sse(slice_response(reason, [{"text": "[[4]]"}], usage=usage())),
        ],
        stream=True,
        max_output_tokens=None,
    )
    assert adapter.make_prediction("test", pair_index=0).answer == [[4]]
    assert len(requests) == 1


@pytest.mark.parametrize(
    "reason,checkpoint,message",
    [
        ("STOP", None, "without usage metadata"),
        ("CONTINUATION", token("one"), "without usage metadata"),
        (None, token("one"), "checkpoint has no usage metadata"),
    ],
)
def test_missing_usage_is_not_silently_treated_as_free(
    adapter_factory, reason, checkpoint, message
):
    adapter, requests = adapter_factory(
        [
            sse(slice_response(reason, [{"text": "[[1]]"}], checkpoint)),
        ],
        stream=True,
    )
    with pytest.raises(GeminiStreamError, match=message):
        adapter.make_prediction("test", pair_index=0)
    assert len(requests) == 1


def test_completed_reasoning_only_stream_has_explicit_error(adapter_factory):
    adapter, _ = adapter_factory(
        [
            sse(
                slice_response(
                    "STOP", [{"text": "thinking", "thought": True}], usage=usage()
                )
            ),
        ],
        stream=True,
    )
    with pytest.raises(GeminiStreamError, match="without visible answer text") as exc:
        adapter.make_prediction("test", pair_index=0)
    assert exc.value.usage_metadata.total_token_count == 13


def test_initial_transport_recovery_is_bounded_and_preserves_original_contents(
    adapter_factory,
):
    adapter, requests = adapter_factory(
        [
            sse(httpx.ReadError("disconnected"))
            for _ in range(MAX_STREAM_RECOVERY_RETRIES + 1)
        ],
        stream=True,
    )
    with pytest.raises(GeminiStreamError, match="recovery retries exhausted"):
        adapter.make_prediction("test", pair_index=0)
    assert len(requests) == MAX_STREAM_RECOVERY_RETRIES + 1
    bodies = [json.loads(r.content) for r in requests]
    assert all(body["contents"] == bodies[0]["contents"] for body in bodies)
    assert all("continuationToken" not in body for body in bodies)
    assert adapter.raw_api_logger.record_failure.call_count == len(requests)
    assert adapter._deferred_raw_api_requests == {}


def test_partial_stream_without_checkpoint_restarts_with_original_input(adapter_factory):
    adapter, requests = adapter_factory(
        [
            sse(
                slice_response(None, [{"text": "discard"}], usage=usage()),
                httpx.ReadError("lost"),
            ),
            sse(slice_response("STOP", [{"text": "[[4]]"}], usage=usage())),
        ],
        stream=True,
    )
    attempt = adapter.make_prediction("test", pair_index=0)
    assert attempt.answer == [[4]]
    bodies = [json.loads(request.content) for request in requests]
    assert bodies[1]["contents"] == bodies[0]["contents"]
    assert "continuationToken" not in bodies[1]
    assert attempt.metadata.usage.total_tokens == 26


def test_interrupted_resumed_hop_without_usage_does_not_complete_with_zero_cost(
    adapter_factory,
):
    adapter, requests = adapter_factory(
        [
            sse(
                slice_response("CONTINUATION", [{"text": "[["}], token("one"), usage())
            ),
            sse(slice_response(None, [{"text": "discard"}]), httpx.ReadError("lost")),
        ],
        stream=True,
    )
    with pytest.raises(
        GeminiStreamError, match="Interrupted Gemini stream has no usage"
    ) as exc:
        adapter.make_prediction("test", pair_index=0)
    assert len(requests) == 2
    assert exc.value.usage_metadata.total_token_count == 13
    assert exc.value.response["slices"][-1]["usageMetadata"] == {}


def test_provider_error_after_completed_slice_retains_usage_and_is_not_retried(
    adapter_factory,
):
    adapter, requests = adapter_factory(
        [
            sse(
                slice_response("CONTINUATION", [{"text": "[["}], token("one"), usage())
            ),
            httpx.Response(
                429,
                json={
                    "error": {
                        "code": 429,
                        "message": "quota",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                },
            ),
        ],
        stream=True,
    )
    with pytest.raises(GeminiStreamError, match="429") as exc:
        adapter.make_prediction("test", pair_index=0)
    assert len(requests) == 2
    assert exc.value.status_code == 429
    assert exc.value.usage_metadata.total_token_count == 13
    assert adapter.raw_api_logger.record_success.call_count == 1
    assert adapter.raw_api_logger.record_failure.call_count == 1


def test_raw_api_events_persist_each_hop_and_partial_failure_usage(adapter_factory):
    adapter, _ = adapter_factory(
        [
            sse(
                slice_response(None, [{"text": "[["}], token("one"), usage()),
                httpx.ReadError("disconnected"),
            ),
            sse(slice_response("STOP", [{"text": "5]]"}], usage=usage(20, 2, 3))),
        ],
        stream=True,
    )
    output = StringIO()
    logger = logging.Logger("gemini-stream-test", level=logging.INFO)
    handler = logging.StreamHandler(output)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    adapter.raw_api_logger = RawAPILogger(run_id="mock-run", event_logger=logger)
    try:
        with adapter.request_context(task_id="mock-task", attempt=1, retry=1):
            assert adapter.make_prediction("test", pair_index=0).answer == [[5]]
    finally:
        handler.close()
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [e["event"] for e in events] == [
        "request_started",
        "request_failed",
        "request_started",
        "request_succeeded",
    ]
    assert all(e["context"]["retry"] == 1 for e in events)
    assert events[1]["error"]["response"]["slices"][0]["usageMetadata"] == usage()
    assert events[3]["response"]["usageMetadata"] == usage(20, 2, 3)
    assert "test-key" not in output.getvalue()


@pytest.mark.parametrize(
    "overrides",
    [
        {"continuation_token": b"caller-token"},
        {"http_options": {"extra_body": {"continuationToken": "caller-token"}}},
        {"http_options": {"extra_body": {"contents": []}}},
        {"http_options": {"extra_body": {"generationConfig": {}}}},
        {"should_return_http_response": True},
    ],
)
def test_caller_cannot_override_managed_stream_fields(adapter_factory, overrides):
    adapter, requests = adapter_factory([], stream=True, **overrides)
    with pytest.raises(ValueError):
        adapter.make_prediction("test", pair_index=0)
    assert requests == []


def test_empty_usage_event_does_not_erase_last_reported_cumulative_counts(
    adapter_factory,
):
    adapter, _ = adapter_factory(
        [
            sse(
                slice_response("STOP", [{"text": "[[1]]"}], usage=usage()),
                {"usageMetadata": {}},
            ),
        ],
        stream=True,
    )
    assert (
        adapter.make_prediction("test", pair_index=0).metadata.usage.total_tokens == 13
    )
