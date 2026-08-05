import json
import logging
from dataclasses import dataclass
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arc_agi_benchmarking.adapters.provider import ProviderAdapter
from arc_agi_benchmarking.utils.logging_utils import (
    RawAPILogger,
    StructuredFormatter,
    serialize_for_raw_log,
)


class DummyAdapter(ProviderAdapter):
    def init_client(self):
        return None

    def make_prediction(self, prompt, task_id=None, test_id=None, pair_index=None):
        raise NotImplementedError

    def extract_json_from_response(self, input_response):
        raise NotImplementedError


@pytest.fixture
def event_output():
    output = StringIO()
    event_logger = logging.getLogger(f"test.raw_api.{id(output)}")
    event_logger.handlers.clear()
    event_logger.propagate = False
    event_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(output)
    handler.setFormatter(StructuredFormatter())
    event_logger.addHandler(handler)
    try:
        yield event_logger, output
    finally:
        handler.close()
        event_logger.handlers.clear()


def make_adapter(monkeypatch, api_logger=None):
    model_config = SimpleNamespace(
        provider="dummy",
        model_name="dummy-model",
        api_key_env="DUMMY_API_KEY",
    )
    monkeypatch.setattr(
        "arc_agi_benchmarking.adapters.provider.read_models_config",
        lambda _config: model_config,
    )
    return DummyAdapter("dummy-config", raw_api_logger=api_logger)


def read_events(output):
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_request_success_emits_correlated_sanitized_events(
    monkeypatch,
    event_output,
):
    event_logger, output = event_output
    api_logger = RawAPILogger(run_id="run-1", event_logger=event_logger)
    adapter = make_adapter(monkeypatch, api_logger)

    with adapter.request_context(
        config="dummy-config",
        provider="dummy",
        model="dummy-model",
        task_id="task-a",
        pair_index=0,
        attempt=2,
        retry=3,
        orchestrator_attempt=1,
    ):
        response = adapter._request(
            "responses.create",
            lambda **_kwargs: {"id": "response-1", "output": "answer"},
            model="dummy-model",
            input=[{"role": "user", "content": "prompt"}],
            api_key="must-not-appear",
            extra_headers={"Authorization": "Bearer must-not-appear", "x-trace": "ok"},
        )

    assert response["id"] == "response-1"
    events = read_events(output)
    assert [event["event"] for event in events] == [
        "request_started",
        "request_succeeded",
    ]
    assert "request_id" not in events[0]
    assert "request_id" not in events[1]
    assert events[0]["run_id"] == "run-1"
    assert events[0]["context"]["attempt"] == 2
    assert events[0]["context"]["retry"] == 3
    assert events[0]["request"]["kwargs"]["api_key"] == "[REDACTED]"
    assert (
        events[0]["request"]["kwargs"]["extra_headers"]["Authorization"]
        == "[REDACTED]"
    )
    assert "must-not-appear" not in output.getvalue()
    assert events[1]["response"]["output"] == "answer"


def test_request_failure_emits_error_and_reraises(monkeypatch, event_output):
    event_logger, output = event_output
    api_logger = RawAPILogger(run_id="run-2", event_logger=event_logger)
    adapter = make_adapter(monkeypatch, api_logger)

    class ProviderError(RuntimeError):
        status_code = 429
        request_id = "provider-request-1"
        body = {"error": "rate limited", "token": "must-not-appear"}

    def fail():
        raise ProviderError("rate limited")

    with adapter.request_context(task_id="task-b"):
        with pytest.raises(ProviderError):
            adapter._request("messages.create", fail)

    events = read_events(output)
    assert [event["event"] for event in events] == [
        "request_started",
        "request_failed",
    ]
    assert events[1]["error"]["status_code"] == 429
    assert events[1]["error"]["request_id"] == "provider-request-1"
    assert events[1]["error"]["body"]["token"] == "[REDACTED]"


def test_deferred_stream_emits_final_response(monkeypatch, event_output):
    event_logger, output = event_output
    api_logger = RawAPILogger(event_logger=event_logger)
    adapter = make_adapter(monkeypatch, api_logger)
    stream = iter(["chunk-1", "chunk-2"])

    with adapter.request_context(task_id="stream-task"):
        returned_stream = adapter._request(
            "responses.create_stream",
            lambda: stream,
            _raw_log_deferred=True,
        )
        adapter._record_deferred_raw_api_success(
            returned_stream,
            {"output_text": "complete"},
        )

    events = read_events(output)
    assert [event["event"] for event in events] == [
        "request_started",
        "request_succeeded",
    ]
    assert events[1]["response"] == {"output_text": "complete"}


def test_task_timeout_and_repeated_runs_share_existing_handler(event_output):
    event_logger, output = event_output
    first = RawAPILogger(run_id="first-run", event_logger=event_logger)
    second = RawAPILogger(run_id="second-run", event_logger=event_logger)

    first.record_task_timeout(
        task_id="timeout-task",
        config="config-a",
        elapsed=12.5,
        timeout=10,
    )
    second.record_task_timeout(
        task_id="timeout-task",
        config="config-a",
        elapsed=22.5,
        timeout=20,
    )

    events = read_events(output)
    assert [event["run_id"] for event in events] == ["first-run", "second-run"]
    assert all(event["event"] == "task_timed_out" for event in events)


def test_application_and_api_events_use_same_structured_handler(
    monkeypatch,
    event_output,
):
    event_logger, output = event_output
    event_logger.info("application event")
    api_logger = RawAPILogger(run_id="combined-run", event_logger=event_logger)
    adapter = make_adapter(monkeypatch, api_logger)

    with adapter.request_context(task_id="task-a"):
        adapter._request("responses.create", lambda: {"id": "response-1"})

    events = read_events(output)
    assert events[0]["message"] == "application event"
    assert [event["event"] for event in events[1:]] == [
        "request_started",
        "request_succeeded",
    ]


def test_serializer_supports_dataclasses_and_environment_redaction():
    @dataclass
    class Payload:
        prompt: str
        env: dict[str, str]

    serialized = serialize_for_raw_log(
        Payload(prompt="hello", env={"API_KEY": "must-not-appear"})
    )

    assert serialized == {"prompt": "hello", "env": "[REDACTED]"}


def test_no_api_logger_emits_no_raw_events(monkeypatch, event_output):
    _event_logger, output = event_output
    adapter = make_adapter(monkeypatch)
    with adapter.request_context(task_id="no-log"):
        assert adapter._request("operation", lambda: "ok") == "ok"
    assert output.getvalue() == ""


def test_api_logger_failure_does_not_change_provider_result(monkeypatch):
    api_logger = Mock()
    api_logger.start_request.side_effect = RuntimeError("logging unavailable")
    adapter = make_adapter(monkeypatch, api_logger)

    assert adapter._request("operation", lambda: "provider-result") == "provider-result"
