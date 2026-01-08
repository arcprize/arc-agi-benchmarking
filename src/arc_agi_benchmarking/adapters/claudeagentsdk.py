from .provider import ProviderAdapter
from arc_agi_benchmarking.schemas import AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, PromptTokensDetails, Attempt, ToolCall
import os
from dotenv import load_dotenv
import json
from typing import List, Optional, Any
from datetime import datetime, timezone
import logging
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)


class ClaudeagentsdkAdapter(ProviderAdapter):
    # Scratchpad directory for agentic reasoning
    SCRATCHPAD_DIR = "/tmp/arc_agi_scratchpad"

    # Instructions to prepend to the prompt encouraging tool use
    SCRATCHPAD_INSTRUCTIONS = """
IMPORTANT: You have access to a scratchpad directory at {scratchpad_dir} for your working notes.

You MUST use this scratchpad to work through this problem iteratively:
1. First, create a file to analyze the input/output patterns you observe
2. Write down your hypotheses about the transformation rules
3. Test your hypotheses against each training example
4. Refine your understanding before producing the final answer

Use the Write tool to create files in the scratchpad. For example:
- {scratchpad_dir}/analysis.txt - your pattern analysis
- {scratchpad_dir}/hypotheses.txt - your transformation hypotheses

This iterative approach will help you solve the puzzle more accurately.

---

"""

    def init_client(self):
        """
        Initialize the Claude Agent SDK.
        The SDK uses ANTHROPIC_API_KEY from environment.
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        # Import here to avoid import errors if SDK not installed
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions
            self._query = query
            self._ClaudeAgentOptions = ClaudeAgentOptions
        except ImportError:
            raise ImportError(
                "claude_agent_sdk not installed. Install with: pip install claude-agent-sdk"
            )

        # Ensure scratchpad directory exists
        os.makedirs(self.SCRATCHPAD_DIR, exist_ok=True)

        return None  # No persistent client needed for query() method

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Claude Agent SDK using query() method.
        Each call creates a new session.

        Allowed tools (for iterative reasoning):
        - TodoWrite: in-memory scratchpad for organizing thoughts
        - Write/Read/Edit: file-based scratchpad in /tmp/arc_agi_scratchpad/
        - Task: spawn subagents for deeper analysis

        Blocked tools:
        - Bash, Glob, Grep: no shell or file search
        - WebFetch, WebSearch: no web access (prevents info leakage)

        The prompt is augmented with instructions to use the scratchpad.
        """
        return asyncio.run(self._make_prediction_async(prompt, task_id, test_id, pair_index))

    async def _make_prediction_async(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Async implementation of make_prediction.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            ToolResultBlock,
        )

        start_time = datetime.now(timezone.utc)

        # Create task-specific scratchpad directory
        task_scratchpad = os.path.join(
            self.SCRATCHPAD_DIR,
            f"{task_id or 'unknown'}_{pair_index or 0}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(task_scratchpad, exist_ok=True)

        # Prepend scratchpad instructions to the prompt
        augmented_prompt = self.SCRATCHPAD_INSTRUCTIONS.format(scratchpad_dir=task_scratchpad) + prompt

        # Build options for ARC-AGI testing:
        # - TodoWrite: scratchpad for organizing thoughts and iterating on analysis
        # - Task: spawn subagents for deeper analysis (optional, can be removed if not wanted)
        # - No file writes, no web access, no bash - safe for benchmarking
        options = self._ClaudeAgentOptions(
            model=self.model_config.model_name,
            allowed_tools=[
                "TodoWrite",  # In-memory scratchpad
                "Write", "Read", "Edit",  # File-based scratchpad for a /tmp/* entry per task
                "Task",  # Subagents for deeper analysis
            ],
            disallowed_tools=[
                "Bash", "Glob", "Grep",  # No shell or file search
                "WebFetch", "WebSearch",  # No web access (prevents info leakage)
                "NotebookEdit", "KillShell", "BashOutput",  # No other risky tools
            ],
            permission_mode="bypassPermissions",  # Auto-approve tool usage for benchmarking
            cwd=task_scratchpad,  # Set working directory to scratchpad
        )

        # Collect all messages
        all_messages = []
        result_message = None
        text_contents = []
        thinking_contents = []

        # Track tool calls for the iteration log
        tool_calls_dict = {}  # tool_use_id -> ToolCall (mutable, update with results)
        tool_calls_order = []  # Preserve order of tool calls

        try:
            async for message in self._query(prompt=augmented_prompt, options=options):
                all_messages.append(message)
                logger.debug(f"Message type: {type(message).__name__}")

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        block_type = type(block).__name__
                        logger.debug(f"  Block type: {block_type}")

                        if isinstance(block, TextBlock):
                            text_contents.append(block.text)
                        elif isinstance(block, ThinkingBlock):
                            thinking_contents.append(block.thinking)
                        elif isinstance(block, ToolUseBlock):
                            # Capture tool use immediately
                            logger.info(f"Tool use: {block.name} (id: {block.id})")
                            tool_call = ToolCall(
                                tool_name=block.name,
                                tool_use_id=block.id,
                                input=block.input,
                                output=None,  # Will be updated if we get a result
                                is_error=False,
                            )
                            tool_calls_dict[block.id] = tool_call
                            tool_calls_order.append(block.id)
                        elif isinstance(block, ToolResultBlock):
                            # Update the corresponding tool call with its result
                            logger.info(f"Tool result for id: {block.tool_use_id}")
                            if block.tool_use_id in tool_calls_dict:
                                # Parse output content
                                output_data = None
                                if block.content:
                                    if isinstance(block.content, str):
                                        output_data = {"text": block.content}
                                    elif isinstance(block.content, list):
                                        output_data = {"content": block.content}
                                    else:
                                        output_data = {"raw": str(block.content)}

                                # Update the existing tool call
                                tool_calls_dict[block.tool_use_id].output = output_data
                                tool_calls_dict[block.tool_use_id].is_error = block.is_error or False

                elif isinstance(message, ResultMessage):
                    result_message = message

        except Exception as e:
            logger.error(f"Error during Claude Agent SDK query: {e}")
            raise

        end_time = datetime.now(timezone.utc)

        # Extract usage and cost from result message
        usage_data = result_message.usage if result_message and result_message.usage else {}
        total_cost_from_sdk = result_message.total_cost_usd if result_message else None

        # Get token counts from usage - include cache tokens in total input
        base_input_tokens = usage_data.get('input_tokens', 0)
        cache_creation_tokens = usage_data.get('cache_creation_input_tokens', 0)
        cache_read_tokens = usage_data.get('cache_read_input_tokens', 0)
        output_tokens = usage_data.get('output_tokens', 0)

        # Total input tokens includes all cache tokens
        total_input_tokens = base_input_tokens + cache_creation_tokens + cache_read_tokens

        # Calculate costs using model config pricing (same as other adapters)
        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000

        prompt_cost = total_input_tokens * input_cost_per_token
        completion_cost = output_tokens * output_cost_per_token
        total_cost = prompt_cost + completion_cost

        # Log SDK cost for comparison (they may use different cache pricing)
        if total_cost_from_sdk is not None:
            logger.debug(f"Cost comparison: SDK=${total_cost_from_sdk:.6f}, our pricing=${total_cost:.6f}")

        # Build reasoning summary from thinking blocks
        reasoning_summary = "\n\n".join(thinking_contents) if thinking_contents else None

        # Get the final answer text
        answer_text = "\n".join(text_contents) if text_contents else ""

        # Build input choices (the user prompt)
        input_choices = [
            Choice(
                index=0,
                message=Message(
                    role="user",
                    content=prompt
                )
            )
        ]

        # Build response choices from all text blocks
        response_choices = [
            Choice(
                index=len(input_choices) + i,
                message=Message(
                    role="assistant",
                    content=text
                )
            )
            for i, text in enumerate(text_contents)
        ]

        all_choices = input_choices + response_choices

        # Get num_turns from result message
        num_turns = result_message.num_turns if result_message else 1

        # Build tool calls list in order
        tool_calls_list = [tool_calls_dict[tid] for tid in tool_calls_order] if tool_calls_order else None

        # Debug: warn if we have turns but no tool calls captured
        if num_turns > 1 and not tool_calls_list:
            logger.warning(f"num_turns={num_turns} but no tool calls captured. Check message parsing.")
            # Log all message types we received for debugging
            msg_types = [type(m).__name__ for m in all_messages]
            logger.warning(f"Message types received: {msg_types}")

        # Create metadata
        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            reasoning_summary=reasoning_summary,
            usage=Usage(
                prompt_tokens=total_input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_input_tokens + output_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens,
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,  # SDK doesn't break this out separately
                    accepted_prediction_tokens=output_tokens,
                    rejected_prediction_tokens=0
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=total_cost
            ),
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id,
            tool_calls=tool_calls_list,
            num_turns=num_turns,
        )

        attempt = Attempt(
            metadata=metadata,
            answer=answer_text
        )

        return attempt

    def chat_completion(self, messages, tools=[]):
        """
        Raw chat completion - not used for this adapter since we use query().
        """
        raise NotImplementedError("ClaudeAgentSDK adapter uses query() method, not chat_completion")

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """
        Extract JSON from the response. Uses a simple extraction method
        since we can't use tools with the agent SDK in this configuration.
        """
        import re

        # Try to find JSON arrays in the response
        # Look for patterns like [[1, 2], [3, 4]]
        pattern = r'\[\s*\[[\d\s,\[\]]+\]\s*\]'
        matches = re.findall(pattern, input_response)

        for match in matches:
            try:
                parsed = json.loads(match)
                # Validate it's a list of lists of integers
                if isinstance(parsed, list) and all(
                    isinstance(row, list) and all(isinstance(x, int) for x in row)
                    for row in parsed
                ):
                    return parsed
            except json.JSONDecodeError:
                continue

        # Fallback: try to parse the entire response as JSON
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


if __name__ == "__main__":
    # Simple test
    adapter = ClaudeagentsdkAdapter("claudeagentsdk-sonnet-4")
    print("Adapter initialized successfully")
