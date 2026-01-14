import io
import json
import os
from unittest.mock import patch

import pytest

from arc_agi_benchmarking.adapters.codexcli import CodexcliAdapter
from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing


class FakeProcess:
    def __init__(self, stdout_lines, stderr="", returncode=0):
        self.stdin = io.StringIO()
        joined = "\n".join(stdout_lines) + ("\n" if stdout_lines else "")
        self.stdout = io.StringIO(joined)
        self.stderr = io.StringIO(stderr)
        self._returncode = returncode

    def wait(self):
        return self._returncode


@pytest.fixture
def mock_model_config(tmp_path):
    return ModelConfig(
        name="test-codexcli-model",
        model_name="codex-mini-latest",
        provider="codexcli",
        pricing=ModelPricing(date="2025-05-16", input=2.0, output=4.0),
        kwargs={"scratchpad_root": str(tmp_path / "scratchpad")},
    )


def test_init_client_accepts_openai_api_key(mock_model_config, monkeypatch):
    adapter = CodexcliAdapter.__new__(CodexcliAdapter)
    adapter.model_config = mock_model_config

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    assert adapter.init_client() is None


def test_init_client_requires_openai_api_key(mock_model_config, monkeypatch):
    adapter = CodexcliAdapter.__new__(CodexcliAdapter)
    adapter.model_config = mock_model_config

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        adapter.init_client()


def test_run_codex_exec_parses_events_and_usage(mock_model_config, monkeypatch):
    adapter = CodexcliAdapter.__new__(CodexcliAdapter)
    adapter.model_config = mock_model_config

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    stdout_lines = [
        json.dumps({"type": "turn.started"}),
        json.dumps({"type": "item.completed", "item": {"id": "r1", "type": "reasoning", "text": "reasoning"}}),
        json.dumps({"type": "item.completed", "item": {"id": "a1", "type": "agent_message", "text": "final"}}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "cmd1",
                    "type": "command_execution",
                    "command": "ls",
                    "aggregated_output": "out",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "file1",
                    "type": "file_change",
                    "changes": [{"path": "a.txt", "kind": "add"}],
                    "status": "completed",
                },
            }
        ),
        json.dumps({"type": "item.completed", "item": {"id": "web1", "type": "web_search", "query": "abc"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 5}}),
    ]

    captured = {}

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess(stdout_lines)

    scratchpad_dir = os.path.join(mock_model_config.kwargs["scratchpad_root"], "run-1")

    with patch.object(CodexcliAdapter, "_resolve_codex_executable", return_value="codex"):
        with patch("subprocess.Popen", side_effect=fake_popen):
            result = adapter._run_codex_exec("prompt", working_directory=scratchpad_dir)

    assert result["final_response"] == "final"
    assert result["assistant_messages"] == ["final"]
    assert result["reasoning_summary"] == "reasoning"
    assert result["num_turns"] == 1
    assert result["usage"]["prompt_tokens"] == 12
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["cache_read_tokens"] == 2

    tool_names = [tool.tool_name for tool in result["tool_calls"]]
    assert tool_names == ["command_execution", "file_change", "web_search"]

    env = captured["kwargs"]["env"]
    assert env["CODEX_API_KEY"] == "test-openai-key"

    args = captured["args"][0]
    assert "--sandbox" in args
    assert "workspace-write" in args
    assert "--skip-git-repo-check" in args
    assert "--cd" in args
    assert "approval_policy=\"never\"" in args
    assert "sandbox_workspace_write.network_access=false" in args
    assert f"sandbox_workspace_write.writable_roots=[\"{scratchpad_dir}\"]" in args
    assert "features.shell_tool=true" in args
    assert "features.unified_exec=false" in args
    assert "features.web_search_request=false" in args
    assert "mcp_servers={}" in args
    cd_index = args.index("--cd")
    assert args[cd_index + 1] == scratchpad_dir


def test_make_prediction_populates_attempt_metadata(mock_model_config):
    adapter = CodexcliAdapter.__new__(CodexcliAdapter)
    adapter.model_config = mock_model_config

    run_result = {
        "final_response": "[[1]]",
        "assistant_messages": ["[[1]]"],
        "reasoning_summary": None,
        "tool_calls": None,
        "num_turns": 1,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        },
    }

    with patch.object(adapter, "_run_codex_exec", return_value=run_result) as mock_run:
        attempt = adapter.make_prediction("prompt", task_id="t1", test_id="test1", pair_index=2)
    assert mock_run.call_count == 1
    assert "scratchpad" in mock_run.call_args[0][0].lower()
    assert mock_run.call_args.kwargs["working_directory"].startswith(mock_model_config.kwargs["scratchpad_root"])

    assert attempt.answer == [[1]]
    assert attempt.metadata.task_id == "t1"
    assert attempt.metadata.test_id == "test1"
    assert attempt.metadata.pair_index == 2
    assert attempt.metadata.choices[0].message.content == "prompt"
    assert attempt.metadata.choices[1].message.content == "[[1]]"

    expected_prompt_cost = 100 * (2.0 / 1_000_000)
    expected_completion_cost = 50 * (4.0 / 1_000_000)
    assert attempt.metadata.cost.prompt_cost == pytest.approx(expected_prompt_cost)
    assert attempt.metadata.cost.completion_cost == pytest.approx(expected_completion_cost)


def test_mcp_tool_call_is_rejected(mock_model_config, monkeypatch):
    adapter = CodexcliAdapter.__new__(CodexcliAdapter)
    adapter.model_config = mock_model_config

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    stdout_lines = [
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "mcp1",
                    "type": "mcp_tool_call",
                    "server": "srv",
                    "tool": "mcp.tool",
                    "arguments": {"x": 1},
                    "status": "completed",
                },
            }
        ),
    ]

    def fake_popen(*args, **kwargs):
        return FakeProcess(stdout_lines)

    with patch.object(CodexcliAdapter, "_resolve_codex_executable", return_value="codex"):
        with patch("subprocess.Popen", side_effect=fake_popen):
            with pytest.raises(RuntimeError, match="MCP tool calls are disabled"):
                adapter._run_codex_exec("prompt", working_directory=mock_model_config.kwargs["scratchpad_root"])
