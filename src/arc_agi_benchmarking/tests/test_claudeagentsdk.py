import pytest
from unittest.mock import Mock, MagicMock, patch
from arc_agi_benchmarking.adapters.claudeagentsdk import ClaudeagentsdkAdapter
from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing, PromptTokensDetails
import os
import sys
import asyncio
from datetime import datetime, timezone

# Suppress warning about unawaited coroutines from mock framework
# This occurs because mocking async generators interacts with unittest.mock internals
pytestmark = pytest.mark.filterwarnings(
    "ignore:coroutine 'ClaudeagentsdkAdapter._make_prediction_async' was never awaited:RuntimeWarning"
)


# Create mock SDK module and classes before importing the adapter
class MockTextBlock:
    def __init__(self, text=""):
        self.text = text


class MockThinkingBlock:
    def __init__(self, thinking=""):
        self.thinking = thinking


class MockToolUseBlock:
    def __init__(self, name="", id="", input=None):
        self.name = name
        self.id = id
        self.input = input or {}


class MockToolResultBlock:
    def __init__(self, tool_use_id="", content="", is_error=False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class MockAssistantMessage:
    def __init__(self, content=None):
        self.content = content or []


class MockResultMessage:
    def __init__(self, usage=None, total_cost_usd=0.0, num_turns=1):
        self.usage = usage or {}
        self.total_cost_usd = total_cost_usd
        self.num_turns = num_turns


# Create mock SDK module
mock_sdk_module = MagicMock()
mock_sdk_module.AssistantMessage = MockAssistantMessage
mock_sdk_module.ResultMessage = MockResultMessage
mock_sdk_module.TextBlock = MockTextBlock
mock_sdk_module.ThinkingBlock = MockThinkingBlock
mock_sdk_module.ToolUseBlock = MockToolUseBlock
mock_sdk_module.ToolResultBlock = MockToolResultBlock


@pytest.fixture
def mock_model_config():
    """Provides a mock ModelConfig for Claude Agent SDK."""
    return ModelConfig(
        name="test-claudeagentsdk-model",
        model_name="claude-sonnet-4-5-20250929",
        provider="claudeagentsdk",
        pricing=ModelPricing(date="2026-01-02", input=1.0, output=5.0),
        kwargs={}
    )


@pytest.fixture
def adapter_instance(mock_model_config):
    """Creates a ClaudeagentsdkAdapter instance with mocked config."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch.object(ClaudeagentsdkAdapter, 'init_client') as mock_init:
            mock_init.return_value = None
            adapter = ClaudeagentsdkAdapter.__new__(ClaudeagentsdkAdapter)
            adapter.model_config = mock_model_config
            adapter._query = None  # Each test overrides this with a generator
            adapter._ClaudeAgentOptions = Mock(return_value=Mock())
            return adapter


class TestTokenTracking:
    """Test token tracking and consolidation for Claude Agent SDK."""

    @pytest.mark.asyncio
    async def test_token_consolidation_with_cache(self, adapter_instance):
        """Test that cache tokens are included in total input tokens."""
        mock_text = MockTextBlock(text="[[1, 2], [3, 4]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={
                'input_tokens': 1000,
                'cache_creation_input_tokens': 500,
                'cache_read_input_tokens': 200,
                'output_tokens': 300,
            },
            total_cost_usd=0.005,
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Verify token consolidation: 1000 + 500 + 200 = 1700
        assert attempt.metadata.usage.prompt_tokens == 1700
        assert attempt.metadata.usage.completion_tokens == 300
        assert attempt.metadata.usage.total_tokens == 2000

    @pytest.mark.asyncio
    async def test_prompt_tokens_details_populated(self, adapter_instance):
        """Test that prompt_tokens_details contains cache breakdown."""
        mock_text = MockTextBlock(text="[[0]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={
                'input_tokens': 800,
                'cache_creation_input_tokens': 300,
                'cache_read_input_tokens': 150,
                'output_tokens': 200,
            },
            total_cost_usd=0.003,
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Verify prompt_tokens_details
        assert attempt.metadata.usage.prompt_tokens_details is not None
        assert attempt.metadata.usage.prompt_tokens_details.cache_creation_tokens == 300
        assert attempt.metadata.usage.prompt_tokens_details.cache_read_tokens == 150

    @pytest.mark.asyncio
    async def test_token_tracking_without_cache(self, adapter_instance):
        """Test token tracking when no cache tokens are present."""
        mock_text = MockTextBlock(text="[[1]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={
                'input_tokens': 500,
                'output_tokens': 100,
            },
            total_cost_usd=0.001,
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Without cache tokens, prompt_tokens should just be input_tokens
        assert attempt.metadata.usage.prompt_tokens == 500
        assert attempt.metadata.usage.prompt_tokens_details.cache_creation_tokens == 0
        assert attempt.metadata.usage.prompt_tokens_details.cache_read_tokens == 0


class TestCostCalculation:
    """Test cost calculation using model_config.pricing."""

    @pytest.mark.asyncio
    async def test_cost_uses_model_config_pricing(self, adapter_instance):
        """Test that costs are calculated using model_config.pricing, not SDK pricing."""
        mock_text = MockTextBlock(text="[[1, 2]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={
                'input_tokens': 1000,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'output_tokens': 500,
            },
            total_cost_usd=0.999,  # Should be ignored
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Pricing: input=1.0/1M, output=5.0/1M
        # Expected: (1000 * 1.0 / 1M) + (500 * 5.0 / 1M) = 0.001 + 0.0025 = 0.0035
        expected_prompt_cost = 1000 * (1.0 / 1_000_000)  # 0.001
        expected_completion_cost = 500 * (5.0 / 1_000_000)  # 0.0025
        expected_total_cost = expected_prompt_cost + expected_completion_cost  # 0.0035

        assert attempt.metadata.cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert attempt.metadata.cost.completion_cost == pytest.approx(expected_completion_cost)
        assert attempt.metadata.cost.total_cost == pytest.approx(expected_total_cost)

    @pytest.mark.asyncio
    async def test_cost_includes_cache_tokens(self, adapter_instance):
        """Test that cache tokens are included in cost calculation."""
        mock_text = MockTextBlock(text="[[0]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={
                'input_tokens': 1000,
                'cache_creation_input_tokens': 500,
                'cache_read_input_tokens': 200,
                'output_tokens': 300,
            },
            total_cost_usd=0.0,
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Total input tokens: 1000 + 500 + 200 = 1700
        expected_prompt_cost = 1700 * (1.0 / 1_000_000)
        assert attempt.metadata.cost.prompt_cost == pytest.approx(expected_prompt_cost)


class TestToolCallCapture:
    """Test tool call capture from ToolUseBlock and ToolResultBlock."""

    @pytest.mark.asyncio
    async def test_tool_calls_captured(self, adapter_instance):
        """Test that tool calls are captured from ToolUseBlock."""
        mock_tool_use = MockToolUseBlock(
            name="Write",
            id="tool_123",
            input={"file_path": "/tmp/test.txt", "content": "hello"}
        )
        mock_tool_result = MockToolResultBlock(
            tool_use_id="tool_123",
            content="File written successfully",
            is_error=False
        )
        mock_text = MockTextBlock(text="[[1, 2]]")

        mock_assistant1 = MockAssistantMessage(content=[mock_tool_use])
        mock_assistant2 = MockAssistantMessage(content=[mock_tool_result, mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 1000, 'output_tokens': 500},
            total_cost_usd=0.0,
            num_turns=2
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant1
            yield mock_assistant2
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Verify tool calls were captured
        assert attempt.metadata.tool_calls is not None
        assert len(attempt.metadata.tool_calls) == 1

        tool_call = attempt.metadata.tool_calls[0]
        assert tool_call.tool_name == "Write"
        assert tool_call.tool_use_id == "tool_123"
        assert tool_call.input == {"file_path": "/tmp/test.txt", "content": "hello"}
        assert tool_call.output == {"text": "File written successfully"}
        assert tool_call.is_error == False

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_captured_in_order(self, adapter_instance):
        """Test that multiple tool calls are captured in order."""
        mock_tool_use1 = MockToolUseBlock(name="Write", id="tool_1", input={"file_path": "/tmp/analysis.txt"})
        mock_tool_use2 = MockToolUseBlock(name="Read", id="tool_2", input={"file_path": "/tmp/analysis.txt"})
        mock_result1 = MockToolResultBlock(tool_use_id="tool_1", content="Written", is_error=False)
        mock_result2 = MockToolResultBlock(tool_use_id="tool_2", content="Content read", is_error=False)
        mock_text = MockTextBlock(text="[[0]]")

        mock_msg1 = MockAssistantMessage(content=[mock_tool_use1])
        mock_msg2 = MockAssistantMessage(content=[mock_result1, mock_tool_use2])
        mock_msg3 = MockAssistantMessage(content=[mock_result2, mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 1000, 'output_tokens': 500},
            num_turns=3
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_msg1
            yield mock_msg2
            yield mock_msg3
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Verify tool calls in order
        assert len(attempt.metadata.tool_calls) == 2
        assert attempt.metadata.tool_calls[0].tool_use_id == "tool_1"
        assert attempt.metadata.tool_calls[1].tool_use_id == "tool_2"

    @pytest.mark.asyncio
    async def test_tool_call_error_captured(self, adapter_instance):
        """Test that tool call errors are captured."""
        mock_tool_use = MockToolUseBlock(name="Write", id="tool_err", input={"file_path": "/forbidden/path.txt"})
        mock_tool_result = MockToolResultBlock(tool_use_id="tool_err", content="Permission denied", is_error=True)
        mock_text = MockTextBlock(text="[[0]]")

        mock_msg1 = MockAssistantMessage(content=[mock_tool_use])
        mock_msg2 = MockAssistantMessage(content=[mock_tool_result, mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 500, 'output_tokens': 200},
            num_turns=2
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_msg1
            yield mock_msg2
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        assert attempt.metadata.tool_calls[0].is_error == True
        assert attempt.metadata.tool_calls[0].output == {"text": "Permission denied"}

    @pytest.mark.asyncio
    async def test_tool_call_without_result(self, adapter_instance):
        """Test tool call captured even if result not received (edge case)."""
        mock_tool_use = MockToolUseBlock(name="TodoWrite", id="tool_no_result", input={"todos": []})
        mock_text = MockTextBlock(text="[[1]]")

        mock_msg1 = MockAssistantMessage(content=[mock_tool_use])
        mock_msg2 = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 300, 'output_tokens': 100},
            num_turns=2
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_msg1
            yield mock_msg2
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Tool call should be captured even without explicit result
        assert len(attempt.metadata.tool_calls) == 1
        assert attempt.metadata.tool_calls[0].tool_name == "TodoWrite"
        assert attempt.metadata.tool_calls[0].output is None


class TestNumTurns:
    """Test num_turns tracking."""

    @pytest.mark.asyncio
    async def test_num_turns_tracked(self, adapter_instance):
        """Test that num_turns is captured from ResultMessage."""
        mock_text = MockTextBlock(text="[[1]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 500, 'output_tokens': 200},
            num_turns=5
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        assert attempt.metadata.num_turns == 5

    @pytest.mark.asyncio
    async def test_num_turns_default(self, adapter_instance):
        """Test num_turns defaults to 1 if not present."""
        mock_text = MockTextBlock(text="[[1]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 500, 'output_tokens': 200},
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        assert attempt.metadata.num_turns == 1


class TestThinkingCapture:
    """Test thinking/reasoning capture."""

    @pytest.mark.asyncio
    async def test_thinking_blocks_captured(self, adapter_instance):
        """Test that thinking blocks are captured in reasoning_summary."""
        mock_thinking1 = MockThinkingBlock(thinking="Let me analyze the pattern...")
        mock_thinking2 = MockThinkingBlock(thinking="I see the transformation rule.")
        mock_text = MockTextBlock(text="[[1, 2]]")

        mock_assistant = MockAssistantMessage(content=[mock_thinking1, mock_thinking2, mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 500, 'output_tokens': 200},
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        assert attempt.metadata.reasoning_summary is not None
        assert "Let me analyze the pattern..." in attempt.metadata.reasoning_summary
        assert "I see the transformation rule." in attempt.metadata.reasoning_summary

    @pytest.mark.asyncio
    async def test_no_thinking_blocks(self, adapter_instance):
        """Test that reasoning_summary is None when no thinking blocks."""
        mock_text = MockTextBlock(text="[[0]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 500, 'output_tokens': 200},
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        assert attempt.metadata.reasoning_summary is None


class TestScratchpadConfiguration:
    """Test scratchpad directory configuration."""

    def test_scratchpad_instructions_in_prompt(self):
        """Test that scratchpad instructions are prepended to prompt."""
        assert "scratchpad" in ClaudeagentsdkAdapter.SCRATCHPAD_INSTRUCTIONS.lower()
        assert "{scratchpad_dir}" in ClaudeagentsdkAdapter.SCRATCHPAD_INSTRUCTIONS

    def test_scratchpad_dir_constant(self):
        """Test scratchpad directory constant."""
        assert ClaudeagentsdkAdapter.SCRATCHPAD_DIR == "/tmp/arc_agi_scratchpad"


class TestJsonExtraction:
    """Test JSON extraction from response."""

    def test_extract_json_from_simple_response(self, adapter_instance):
        """Test extracting JSON array from simple response."""
        response = "[[1, 2], [3, 4]]"
        result = adapter_instance.extract_json_from_response(response)
        assert result == [[1, 2], [3, 4]]

    def test_extract_json_from_text_with_array(self, adapter_instance):
        """Test extracting JSON from response with surrounding text."""
        response = "Based on my analysis, the answer is [[1, 2], [3, 4]]. This represents..."
        result = adapter_instance.extract_json_from_response(response)
        assert result == [[1, 2], [3, 4]]

    def test_extract_json_invalid_returns_none(self, adapter_instance):
        """Test that invalid JSON returns None."""
        response = "I'm not sure about this"
        result = adapter_instance.extract_json_from_response(response)
        assert result is None

    def test_extract_json_nested_array(self, adapter_instance):
        """Test extracting deeply nested array."""
        response = "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]"
        result = adapter_instance.extract_json_from_response(response)
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_extract_json_single_row(self, adapter_instance):
        """Test extracting single row array."""
        response = "[[5]]"
        result = adapter_instance.extract_json_from_response(response)
        assert result == [[5]]


class TestMakePredictionSync:
    """Test synchronous make_prediction wrapper."""

    def test_make_prediction_calls_async(self, adapter_instance):
        """Test that make_prediction wraps async method."""
        mock_attempt = Mock()

        with patch('asyncio.run', return_value=mock_attempt) as mock_run:
            result = adapter_instance.make_prediction("Test prompt", "task_1", "test_1", 0)

            mock_run.assert_called_once()
            assert result == mock_attempt


class TestAnswerExtraction:
    """Test answer extraction from response."""

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_in_choices(self, adapter_instance):
        """Test that multiple text blocks are captured as choices."""
        mock_text1 = MockTextBlock(text="Let me think...")
        mock_text2 = MockTextBlock(text="[[1, 2]]")
        mock_assistant = MockAssistantMessage(content=[mock_text1, mock_text2])
        mock_result = MockResultMessage(
            usage={'input_tokens': 500, 'output_tokens': 200},
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Both text blocks should be captured in choices
        # (answer is parsed by Attempt model to extract JSON)
        choice_contents = [c.message.content for c in attempt.metadata.choices]
        assert "Let me think..." in choice_contents
        assert "[[1, 2]]" in choice_contents

    @pytest.mark.asyncio
    async def test_answer_parsed_to_json(self, adapter_instance):
        """Test that answer containing JSON is parsed to array."""
        mock_text = MockTextBlock(text="The answer is [[1, 2], [3, 4]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 500, 'output_tokens': 200},
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        # Answer is parsed to JSON by Attempt model
        assert attempt.answer == [[1, 2], [3, 4]]


class TestMetadataPopulation:
    """Test metadata is properly populated."""

    @pytest.mark.asyncio
    async def test_metadata_task_info(self, adapter_instance):
        """Test that task metadata is populated."""
        mock_text = MockTextBlock(text="[[0]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 100, 'output_tokens': 50},
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async(
                "Test prompt",
                task_id="abc123",
                test_id="test_0",
                pair_index=2
            )

        assert attempt.metadata.task_id == "abc123"
        assert attempt.metadata.test_id == "test_0"
        assert attempt.metadata.pair_index == 2

    @pytest.mark.asyncio
    async def test_metadata_model_info(self, adapter_instance):
        """Test that model info is in metadata."""
        mock_text = MockTextBlock(text="[[0]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 100, 'output_tokens': 50},
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        assert attempt.metadata.model == "claude-sonnet-4-5-20250929"
        assert attempt.metadata.provider == "claudeagentsdk"

    @pytest.mark.asyncio
    async def test_metadata_timestamps(self, adapter_instance):
        """Test that timestamps are populated."""
        mock_text = MockTextBlock(text="[[0]]")
        mock_assistant = MockAssistantMessage(content=[mock_text])
        mock_result = MockResultMessage(
            usage={'input_tokens': 100, 'output_tokens': 50},
            num_turns=1
        )

        async def mock_query_gen(*args, **kwargs):
            yield mock_assistant
            yield mock_result

        adapter_instance._query = mock_query_gen

        with patch.dict(sys.modules, {'claude_agent_sdk': mock_sdk_module}):
            attempt = await adapter_instance._make_prediction_async("Test prompt")

        assert attempt.metadata.start_timestamp is not None
        assert attempt.metadata.end_timestamp is not None
        assert attempt.metadata.end_timestamp >= attempt.metadata.start_timestamp
