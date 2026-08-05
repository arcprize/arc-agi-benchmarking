import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arc_agi_benchmarking.adapters.provider import ProviderAdapter
from arc_agi_benchmarking.utils.raw_api_logging import (
    RawAPIRecorder,
    serialize_for_raw_log,
)


class DummyAdapter(ProviderAdapter):
    def init_client(self):
        return None

    def make_prediction(self, prompt, task_id=None, test_id=None, pair_index=None):
        raise NotImplementedError

    def extract_json_from_response(self, input_response):
        raise NotImplementedError


def make_adapter(monkeypatch, recorder=None):
    model_config = SimpleNamespace(
        provider="dummy",
        model_name="dummy-model",
        api_key_env="DUMMY_API_KEY",
    )
    monkeypatch.setattr(
        "arc_agi_benchmarking.adapters.provider.read_models_config",
        lambda _config: model_config,
    )
    return DummyAdapter("dummy-config", raw_api_recorder=recorder)


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_request_success_records_correlated_sanitized_events(tmp_path, monkeypatch):
    recorder = RawAPIRecorder(tmp_path, run_id="run-1")
    adapter = make_adapter(monkeypatch, recorder)

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
    events = read_events(tmp_path / "task-a.jsonl")
    assert [event["event"] for event in events] == [
        "request_started",
        "request_succeeded",
    ]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert events[0]["run_id"] == "run-1"
    assert events[0]["context"]["attempt"] == 2
    assert events[0]["context"]["retry"] == 3
    assert events[0]["request"]["kwargs"]["api_key"] == "[REDACTED]"
    assert (
        events[0]["request"]["kwargs"]["extra_headers"]["Authorization"]
        == "[REDACTED]"
    )
    assert "must-not-appear" not in (tmp_path / "task-a.jsonl").read_text()
    assert events[1]["response"]["output"] == "answer"


def test_request_failure_records_error_and_reraises(tmp_path, monkeypatch):
    recorder = RawAPIRecorder(tmp_path, run_id="run-2")
    adapter = make_adapter(monkeypatch, recorder)

    class ProviderError(RuntimeError):
        status_code = 429
        request_id = "provider-request-1"
        body = {"error": "rate limited", "token": "must-not-appear"}

    def fail():
        raise ProviderError("rate limited")

    with adapter.request_context(task_id="task-b"):
        with pytest.raises(ProviderError):
            adapter._request("messages.create", fail)

    events = read_events(tmp_path / "task-b.jsonl")
    assert [event["event"] for event in events] == [
        "request_started",
        "request_failed",
    ]
    assert events[1]["error"]["status_code"] == 429
    assert events[1]["error"]["request_id"] == "provider-request-1"
    assert events[1]["error"]["body"]["token"] == "[REDACTED]"


def test_deferred_stream_records_final_response(tmp_path, monkeypatch):
    recorder = RawAPIRecorder(tmp_path)
    adapter = make_adapter(monkeypatch, recorder)
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

    events = read_events(tmp_path / "stream-task.jsonl")
    assert [event["event"] for event in events] == [
        "request_started",
        "request_succeeded",
    ]
    assert events[1]["response"] == {"output_text": "complete"}


def test_task_timeout_and_repeated_runs_append(tmp_path):
    first = RawAPIRecorder(tmp_path, run_id="first-run")
    second = RawAPIRecorder(tmp_path, run_id="second-run")

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

    events = read_events(tmp_path / "timeout-task.jsonl")
    assert [event["run_id"] for event in events] == ["first-run", "second-run"]
    assert all(event["event"] == "task_timed_out" for event in events)


def test_concurrent_writes_remain_valid_jsonl(tmp_path):
    recorder = RawAPIRecorder(tmp_path, run_id="concurrent-run")

    def write_call(index):
        handle = recorder.start_request(
            "operation",
            (),
            {"index": index},
            {"task_id": "shared-task"},
        )
        recorder.record_success(handle, {"index": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_call, range(40)))

    events = read_events(tmp_path / "shared-task.jsonl")
    assert len(events) == 80
    assert len({event["request_id"] for event in events}) == 40


def test_serializer_supports_dataclasses_and_environment_redaction():
    @dataclass
    class Payload:
        prompt: str
        env: dict[str, str]

    serialized = serialize_for_raw_log(
        Payload(prompt="hello", env={"API_KEY": "must-not-appear"})
    )

    assert serialized == {"prompt": "hello", "env": "[REDACTED]"}


def test_no_recorder_creates_no_files(tmp_path, monkeypatch):
    adapter = make_adapter(monkeypatch)
    with adapter.request_context(task_id="no-log"):
        assert adapter._request("operation", lambda: "ok") == "ok"
    assert list(tmp_path.iterdir()) == []


def test_recorder_failure_does_not_change_provider_result(monkeypatch):
    recorder = Mock()
    recorder.start_request.side_effect = OSError("disk unavailable")
    adapter = make_adapter(monkeypatch, recorder)

    assert adapter._request("operation", lambda: "provider-result") == "provider-result"
