from .provider import ProviderAdapter
from arc_agi_benchmarking.schemas import (
    AttemptMetadata,
    Choice,
    Message,
    Usage,
    Cost,
    CompletionTokensDetails,
    PromptTokensDetails,
    Attempt,
    ToolCall,
)
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import subprocess
import logging
import json
import os
import shutil

logger = logging.getLogger(__name__)


class CodexcliAdapter(ProviderAdapter):
    SCRATCHPAD_ROOT = "/tmp/arc_agi_scratchpad"
    SCRATCHPAD_INSTRUCTIONS = """
IMPORTANT: You have access to a scratchpad directory at {scratchpad_dir} for your working notes.

Use this scratchpad to work through the ARC task:
1. Create files to analyze the input/output patterns you observe
2. Write down hypotheses about the transformation rules
3. Test hypotheses against each training example
4. Refine before producing the final answer

Write any notes in files under the scratchpad directory (e.g., analysis.txt, hypotheses.txt).
Web search is disabled.

---

"""

    def init_client(self):
        """
        Initialize the Codex CLI adapter.
        Requires OPENAI_API_KEY (or api_key/openai_api_key in model kwargs).
        """
        api_key = (
            self.model_config.kwargs.get("openai_api_key")
            or self.model_config.kwargs.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("CODEX_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables (or api_key/openai_api_key in model config)"
            )
        return None

    def make_prediction(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        test_id: Optional[str] = None,
        pair_index: int = None,
    ) -> Attempt:
        start_time = datetime.now(timezone.utc)

        scratchpad_dir = self._create_scratchpad_dir(task_id, pair_index, start_time)
        augmented_prompt = self.SCRATCHPAD_INSTRUCTIONS.format(scratchpad_dir=scratchpad_dir) + prompt

        run_result = self._run_codex_exec(augmented_prompt, working_directory=scratchpad_dir)

        end_time = datetime.now(timezone.utc)

        usage = run_result["usage"]
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]

        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000

        prompt_cost = prompt_tokens * input_cost_per_token
        completion_cost = completion_tokens * output_cost_per_token
        total_cost = prompt_cost + completion_cost

        input_choices = [
            Choice(
                index=0,
                message=Message(
                    role="user",
                    content=prompt,
                ),
            )
        ]

        response_choices = [
            Choice(
                index=len(input_choices) + i,
                message=Message(
                    role="assistant",
                    content=text,
                ),
            )
            for i, text in enumerate(run_result["assistant_messages"])
        ]

        all_choices = input_choices + response_choices

        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            reasoning_summary=run_result["reasoning_summary"],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cache_creation_tokens=usage["cache_creation_tokens"],
                    cache_read_tokens=usage["cache_read_tokens"],
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,
                    accepted_prediction_tokens=completion_tokens,
                    rejected_prediction_tokens=0,
                ),
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=total_cost,
            ),
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id,
            tool_calls=run_result["tool_calls"],
            num_turns=run_result["num_turns"],
        )

        attempt = Attempt(
            metadata=metadata,
            answer=run_result["final_response"],
        )

        return attempt

    def _run_codex_exec(self, prompt: str, working_directory: Optional[str] = None) -> Dict[str, Any]:
        executable = self._resolve_codex_executable()
        args = [executable, "exec", "--experimental-json"]

        if self.model_config.model_name:
            args.extend(["--model", self.model_config.model_name])

        self._apply_cli_options(args, working_directory=working_directory)

        env = os.environ.copy()
        api_key = (
            self.model_config.kwargs.get("openai_api_key")
            or self.model_config.kwargs.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("CODEX_API_KEY")
        )
        if api_key:
            env["OPENAI_API_KEY"] = api_key
            env["CODEX_API_KEY"] = api_key
        if "base_url" in self.model_config.kwargs:
            env["OPENAI_BASE_URL"] = self.model_config.kwargs["base_url"]
        env.setdefault("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", "arc_agi_benchmarking_codexcli")

        logger.debug(f"Running Codex CLI: {' '.join(args)}")

        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        assert process.stdin is not None
        process.stdin.write(prompt)
        process.stdin.close()

        assert process.stdout is not None
        assert process.stderr is not None

        items: List[Dict[str, Any]] = []
        assistant_messages: List[str] = []
        reasoning_messages: List[str] = []
        tool_calls: List[ToolCall] = []
        final_response = ""
        usage_data: Optional[Dict[str, Any]] = None
        num_turns = 0
        error_message: Optional[str] = None

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse Codex CLI output: {line}") from exc

            event_type = event.get("type")
            if event_type == "turn.started":
                num_turns += 1
            elif event_type == "item.completed":
                item = event.get("item", {})
                items.append(item)
                item_type = item.get("type")
                if item_type == "agent_message":
                    text = item.get("text", "")
                    assistant_messages.append(text)
                    final_response = text
                elif item_type == "reasoning":
                    reasoning_messages.append(item.get("text", ""))
                elif item_type == "mcp_tool_call":
                    error_message = "MCP tool calls are disabled for Codex CLI runs."
                    break
                tool_call = self._item_to_tool_call(item)
                if tool_call:
                    tool_calls.append(tool_call)
            elif event_type == "turn.completed":
                usage_data = event.get("usage")
            elif event_type == "turn.failed":
                error = event.get("error") or {}
                error_message = error.get("message", "Codex turn failed")
                break
            elif event_type == "error":
                error_message = event.get("message", "Codex stream error")
                break

        stderr_output = process.stderr.read()
        return_code = process.wait()

        if error_message:
            raise RuntimeError(error_message)
        if return_code != 0:
            raise RuntimeError(f"Codex CLI exited with code {return_code}: {stderr_output.strip()}")

        if usage_data is None:
            usage_data = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}

        prompt_tokens = usage_data.get("input_tokens", 0) + usage_data.get("cached_input_tokens", 0)
        completion_tokens = usage_data.get("output_tokens", 0)

        return {
            "items": items,
            "final_response": final_response,
            "assistant_messages": assistant_messages,
            "reasoning_summary": "\n\n".join([m for m in reasoning_messages if m]) or None,
            "tool_calls": tool_calls or None,
            "num_turns": num_turns or 1,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cache_creation_tokens": 0,
                "cache_read_tokens": usage_data.get("cached_input_tokens", 0),
            },
        }

    def _apply_cli_options(self, args: List[str], working_directory: Optional[str] = None) -> None:
        kwargs = self.model_config.kwargs

        args.extend(["--sandbox", "workspace-write"])

        if working_directory:
            args.extend(["--cd", working_directory])

        args.append("--skip-git-repo-check")

        output_schema_file = kwargs.get("output_schema_file")
        if output_schema_file:
            args.extend(["--output-schema", output_schema_file])

        model_reasoning_effort = kwargs.get("model_reasoning_effort")
        if model_reasoning_effort:
            args.extend(["--config", f'model_reasoning_effort="{model_reasoning_effort}"'])

        args.extend(["--config", "approval_policy=\"never\""])
        args.extend(["--config", "sandbox_workspace_write.network_access=false"])
        if working_directory:
            writable_roots = json.dumps([working_directory])
        else:
            writable_roots = "[]"
        args.extend(["--config", f"sandbox_workspace_write.writable_roots={writable_roots}"])
        args.extend(["--config", "features.shell_tool=true"])
        args.extend(["--config", "features.unified_exec=false"])
        args.extend(["--config", "features.web_search_request=false"])
        args.extend(["--config", "mcp_servers={}"])

        thread_id = kwargs.get("thread_id")
        if thread_id:
            args.extend(["resume", thread_id])

    def _item_to_tool_call(self, item: Dict[str, Any]) -> Optional[ToolCall]:
        item_type = item.get("type")
        if item_type == "command_execution":
            output = {
                "aggregated_output": item.get("aggregated_output"),
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
            }
            is_error = item.get("status") == "failed"
            if item.get("exit_code") not in (None, 0):
                is_error = True
            return ToolCall(
                tool_name="command_execution",
                tool_use_id=item.get("id", ""),
                input={"command": item.get("command")},
                output=output,
                is_error=is_error,
            )
        if item_type == "file_change":
            return ToolCall(
                tool_name="file_change",
                tool_use_id=item.get("id", ""),
                input={"changes": item.get("changes", [])},
                output={"status": item.get("status")},
                is_error=item.get("status") == "failed",
            )
        if item_type == "mcp_tool_call":
            return None
        if item_type == "web_search":
            return ToolCall(
                tool_name="web_search",
                tool_use_id=item.get("id", ""),
                input={"query": item.get("query")},
                output=None,
                is_error=False,
            )
        return None

    def _resolve_codex_executable(self) -> str:
        override = self.model_config.kwargs.get("codex_cli_path") or os.environ.get("CODEX_CLI_PATH")
        if override:
            return override

        executable = shutil.which("codex")
        if not executable:
            raise FileNotFoundError("codex executable not found in PATH (set CODEX_CLI_PATH or codex_cli_path)")
        return executable

    def _create_scratchpad_dir(
        self,
        task_id: Optional[str],
        pair_index: Optional[int],
        start_time: datetime,
    ) -> str:
        root = self.model_config.kwargs.get("scratchpad_root")
        if not root:
            root = self.SCRATCHPAD_ROOT
        os.makedirs(root, exist_ok=True)

        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        task_label = task_id or "unknown"
        pair_label = pair_index if pair_index is not None else 0
        scratchpad_dir = os.path.join(root, f"{task_label}_{pair_label}_{timestamp}")
        os.makedirs(scratchpad_dir, exist_ok=True)
        return scratchpad_dir

    def chat_completion(self, messages, tools=[]):
        raise NotImplementedError("Codex CLI adapter uses codex exec, not chat_completion")

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        import re

        pattern = r'\[\s*\[[\d\s,\[\]]+\]\s*\]'
        matches = re.findall(pattern, input_response)

        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list) and all(
                    isinstance(row, list) and all(isinstance(x, int) for x in row)
                    for row in parsed
                ):
                    return parsed
            except json.JSONDecodeError:
                continue

        try:
            parsed = json.loads(input_response)
            if isinstance(parsed, list) and all(
                isinstance(row, list) and all(isinstance(x, int) for x in row)
                for row in parsed
            ):
                return parsed
        except json.JSONDecodeError:
            pass

        return None
