import pytest
from unittest.mock import Mock, MagicMock, patch
import inspect # Added for subclass discovery
import sys # Added for subclass discovery

from arc_agi_benchmarking.adapters.openai_base import OpenAIBaseAdapter
from arc_agi_benchmarking.adapters.open_ai import OpenAIAdapter
from arc_agi_benchmarking.adapters.grok import GrokAdapter
from arc_agi_benchmarking.adapters.deepseek import DeepseekAdapter
from arc_agi_benchmarking.adapters.fireworks import FireworksAdapter
from arc_agi_benchmarking.adapters.openrouter import OpenRouterAdapter
# Import all adapters to ensure they are available for discovery
import arc_agi_benchmarking.adapters 

from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing, Usage, CompletionTokensDetails, Cost, APIType
from openai.types import CompletionUsage
from arc_agi_benchmarking.errors import TokenMismatchError
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables for potential API key checks during init

# --- Helper Function for Subclass Discovery ---
def find_concrete_subclasses(cls):
    """Finds all concrete subclasses of a given class currently imported."""
    concrete_subclasses = []
    for name, obj in inspect.getmembers(sys.modules[cls.__module__.split('.')[0]]): # Look within the 'arc_agi_benchmarking' module tree
        if inspect.ismodule(obj):
             for sub_name, sub_obj in inspect.getmembers(obj):
                  if inspect.isclass(sub_obj) and \
                     issubclass(sub_obj, cls) and \
                     sub_obj != cls and \
                     not inspect.isabstract(sub_obj):
                      # Check specifically within adapters submodule
                      if 'adapters' in sub_obj.__module__:
                           concrete_subclasses.append(sub_obj)
                           
    # Need to also check adapters imported directly
    for name, obj in inspect.getmembers(arc_agi_benchmarking.adapters):
         if inspect.isclass(obj) and issubclass(obj, cls) and obj != cls and not inspect.isabstract(obj):
             if obj not in concrete_subclasses:
                  concrete_subclasses.append(obj)
                  
    return concrete_subclasses

# Dynamically find all concrete subclasses of OpenAIBaseAdapter
openai_adapter_subclasses = find_concrete_subclasses(OpenAIBaseAdapter)

# --- Fixtures ---

@pytest.fixture
def mock_model_config(adapter_class):
    """Provides a mock ModelConfig, adapting provider based on the adapter class."""
    # Infer provider name from class name (e.g., OpenAIAdapter -> openai)
    provider_name = adapter_class.__name__.lower().replace("adapter", "")
    config_name = f"test-{provider_name}-model"
    model_name = f"{provider_name}-test-model"
    
    # Handle potential variations if needed (e.g., fireworks vs fireworks-ai)
    # For now, simple lowercase name works for openai, grok, deepseek, fireworks

    return ModelConfig(
        name=config_name,
        model_name=model_name,
        provider=provider_name, 
        pricing=ModelPricing(date="2024-01-01", input=1.0, output=2.0), 
        api_type=APIType.CHAT_COMPLETIONS, # Assuming chat completions for simplicity
        kwargs={"temperature": 0.5}
    )

# --- Fixtures for Different Cost Calculation Scenarios ---

@pytest.fixture
def mock_response_case_a_no_reasoning():
    """Case A: pt + ct == tt. No explicit reasoning provided."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 200 # ct_raw
    mock_response.usage.total_tokens = 300      # tt_raw
    # Simulate provider not giving reasoning details or details object
    mock_response.usage.completion_tokens_details = Mock()
    mock_response.usage.completion_tokens_details.reasoning_tokens = 0 # rt_explicit = 0
    return mock_response

@pytest.fixture
def mock_response_case_a_explicit_reasoning():
    """Case A: pt + ct == tt. Explicit reasoning provided."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 220 # ct_raw (includes the 20 reasoning)
    mock_response.usage.total_tokens = 320      # tt_raw
    mock_response.usage.completion_tokens_details = Mock()
    mock_response.usage.completion_tokens_details.reasoning_tokens = 20 # rt_explicit = 20
    return mock_response
    
@pytest.fixture
def mock_response_case_b_explicit_reasoning():
    """Case B: pt + ct < tt. Explicit reasoning provided."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 150 # ct_raw (separate from reasoning)
    mock_response.usage.total_tokens = 210      # tt_raw (implies 10 reasoning, matches explicit)
    mock_response.usage.completion_tokens_details = Mock()
    mock_response.usage.completion_tokens_details.reasoning_tokens = 10 # rt_explicit = 10
    return mock_response

@pytest.fixture
def mock_response_case_b_inferred_reasoning():
    """Case B: pt + ct < tt. Reasoning must be inferred."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 200 # ct_raw (separate from reasoning)
    mock_response.usage.total_tokens = 320      # tt_raw (implies 20 reasoning)
    # Simulate provider not giving details or reasoning_tokens=0
    mock_response.usage.completion_tokens_details = Mock()
    mock_response.usage.completion_tokens_details.reasoning_tokens = 0 # rt_explicit = 0
    return mock_response

@pytest.fixture
def mock_openai_response_mismatch():
    """Provides a mock OpenAI API response with mismatched token counts that cannot be reconciled."""
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 200
    mock_response.usage.total_tokens = 350  # Provider claims 350
    # Explicitly claims some reasoning tokens, but not enough to reconcile
    mock_response.usage.completion_tokens_details = Mock()
    # Even with explicit rt=10, Case B logic applies (pt+ct < tt). 
    # computed_total = pt(100) + ct(200) + rt(10) = 310. Mismatch with tt(350).
    mock_response.usage.completion_tokens_details.reasoning_tokens = 10  
    return mock_response
    

# --- Test Class ---
# Parameterize the whole class to run tests for each adapter subclass
@pytest.mark.parametrize("adapter_class", openai_adapter_subclasses)
class TestOpenAIBaseProviderLogic:

    @pytest.fixture(autouse=True)
    def patch_env_vars(self, monkeypatch):
        """Ensure necessary API keys are mocked for adapter initialization."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    @pytest.fixture
    def adapter_instance(self, adapter_class, mock_model_config):
        """Provides an instance of the parameterized adapter class, bypassing real init."""
        
        # Patch the base ProviderAdapter.__init__ to prevent it from running
        # We will manually set the necessary attributes (model_config, client)
        with patch("arc_agi_benchmarking.adapters.provider.ProviderAdapter.__init__", return_value=None) as mock_provider_init:
            
            # Instantiate the parameterized adapter class (init is mocked)
            adapter = adapter_class(config=mock_model_config.name) 
            
            # Manually set the attributes that ProviderAdapter.__init__ would normally set
            adapter.config = mock_model_config.name
            adapter.model_config = mock_model_config 
            adapter.client = MagicMock() # Mock the client instance directly
            
            return adapter

    # --- Test _get_usage (Ensures input to cost calculation is correct) ---
    def test_get_usage_case_a_no_reasoning(self, adapter_instance, mock_response_case_a_no_reasoning):
        """Test _get_usage extracts tokens correctly for Case A, no explicit reasoning."""
        usage = adapter_instance._get_usage(mock_response_case_a_no_reasoning)
        assert isinstance(usage, Usage)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 200 # raw ct
        assert usage.total_tokens == 300
        assert usage.completion_tokens_details.reasoning_tokens == 0 # rt_explicit

    def test_get_usage_case_a_explicit_reasoning(self, adapter_instance, mock_response_case_a_explicit_reasoning):
        """Test _get_usage extracts tokens correctly for Case A, explicit reasoning."""
        usage = adapter_instance._get_usage(mock_response_case_a_explicit_reasoning)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 220 # raw ct
        assert usage.total_tokens == 320
        assert usage.completion_tokens_details.reasoning_tokens == 20 # rt_explicit

    def test_get_usage_case_b_explicit_reasoning(self, adapter_instance, mock_response_case_b_explicit_reasoning):
        """Test _get_usage extracts tokens correctly for Case B, explicit reasoning."""
        usage = adapter_instance._get_usage(mock_response_case_b_explicit_reasoning)
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 150 # raw ct
        assert usage.total_tokens == 210
        assert usage.completion_tokens_details.reasoning_tokens == 10 # rt_explicit

    def test_get_usage_case_b_inferred_reasoning(self, adapter_instance, mock_response_case_b_inferred_reasoning):
        """Test _get_usage extracts tokens and infers reasoning correctly for Case B."""
        usage = adapter_instance._get_usage(mock_response_case_b_inferred_reasoning)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 200 # raw ct
        assert usage.total_tokens == 320
        # IMPORTANT: _get_usage itself infers reasoning here when rt_explicit=0 and pt+ct < tt
        assert usage.completion_tokens_details.reasoning_tokens == 20 # inferred rt

    # --- Test _calculate_cost for different cases ---
    def test_calculate_cost_case_a_no_reasoning(self, adapter_instance, mock_response_case_a_no_reasoning):
        """Test cost calculation for Case A (pt+ct=tt), no explicit reasoning."""
        cost = adapter_instance._calculate_cost(mock_response_case_a_no_reasoning)
        # pt=100, ct_raw=200, rt_explicit=0
        # -> ct_for_cost = 200, rt_for_cost = 0
        expected_prompt_cost = 100 * (1.0 / 1_000_000)
        expected_completion_cost = 200 * (2.0 / 1_000_000) # Based on ct_for_cost
        expected_reasoning_cost = 0 * (2.0 / 1_000_000)    # Based on rt_for_cost
        expected_total_cost = expected_prompt_cost + expected_completion_cost + expected_reasoning_cost
        
        assert isinstance(cost, Cost)
        assert cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert cost.completion_cost == pytest.approx(expected_completion_cost)
        assert cost.reasoning_cost == pytest.approx(expected_reasoning_cost)
        assert cost.total_cost == pytest.approx(expected_total_cost)

    def test_calculate_cost_case_a_explicit_reasoning(self, adapter_instance, mock_response_case_a_explicit_reasoning):
        """Test cost calculation for Case A (pt+ct=tt), explicit reasoning."""
        cost = adapter_instance._calculate_cost(mock_response_case_a_explicit_reasoning)
        # pt=100, ct_raw=220, rt_explicit=20
        # -> ct_for_cost = 200 (220-20), rt_for_cost = 20
        expected_prompt_cost = 100 * (1.0 / 1_000_000)
        expected_completion_cost = 200 * (2.0 / 1_000_000) # Based on ct_for_cost
        expected_reasoning_cost = 20 * (2.0 / 1_000_000)   # Based on rt_for_cost
        expected_total_cost = expected_prompt_cost + expected_completion_cost + expected_reasoning_cost

        assert cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert cost.completion_cost == pytest.approx(expected_completion_cost)
        assert cost.reasoning_cost == pytest.approx(expected_reasoning_cost)
        assert cost.total_cost == pytest.approx(expected_total_cost)
        
    def test_calculate_cost_case_b_explicit_reasoning(self, adapter_instance, mock_response_case_b_explicit_reasoning):
        """Test cost calculation for Case B (pt+ct<tt), explicit reasoning."""
        cost = adapter_instance._calculate_cost(mock_response_case_b_explicit_reasoning)
        # pt=50, ct_raw=150, tt=210, rt_explicit=10
        # -> ct_for_cost = 150 (ct_raw), rt_for_cost = 10 (rt_explicit)
        expected_prompt_cost = 50 * (1.0 / 1_000_000)
        expected_completion_cost = 150 * (2.0 / 1_000_000) # Based on ct_for_cost
        expected_reasoning_cost = 10 * (2.0 / 1_000_000)   # Based on rt_for_cost
        expected_total_cost = expected_prompt_cost + expected_completion_cost + expected_reasoning_cost

        assert cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert cost.completion_cost == pytest.approx(expected_completion_cost)
        assert cost.reasoning_cost == pytest.approx(expected_reasoning_cost)
        assert cost.total_cost == pytest.approx(expected_total_cost)
        
    def test_calculate_cost_case_b_inferred_reasoning(self, adapter_instance, mock_response_case_b_inferred_reasoning):
        """Test cost calculation for Case B (pt+ct<tt), inferred reasoning."""
        cost = adapter_instance._calculate_cost(mock_response_case_b_inferred_reasoning)
        # pt=100, ct_raw=200, tt=320, rt_explicit=0
        # -> inferred rt = 20 (320 - (100+200))
        # -> ct_for_cost = 200 (ct_raw), rt_for_cost = 20 (inferred)
        expected_prompt_cost = 100 * (1.0 / 1_000_000)
        expected_completion_cost = 200 * (2.0 / 1_000_000) # Based on ct_for_cost
        expected_reasoning_cost = 20 * (2.0 / 1_000_000)   # Based on rt_for_cost (inferred)
        expected_total_cost = expected_prompt_cost + expected_completion_cost + expected_reasoning_cost

        assert cost.prompt_cost == pytest.approx(expected_prompt_cost)
        assert cost.completion_cost == pytest.approx(expected_completion_cost)
        assert cost.reasoning_cost == pytest.approx(expected_reasoning_cost)
        assert cost.total_cost == pytest.approx(expected_total_cost)

    def test_calculate_cost_token_mismatch_error(self, adapter_instance, mock_openai_response_mismatch):
        """Test _calculate_cost raises TokenMismatchError on mismatch."""
        with pytest.raises(TokenMismatchError) as excinfo:
            adapter_instance._calculate_cost(mock_openai_response_mismatch)
        # Check the specific error message format from the latest code
        # pt=100, ct_raw=200, tt=350, rt_explicit=10 -> Case B
        # ct_for_cost=200, rt_for_cost=10, pt=100 -> computed_total=310
        expected_msg_part1 = "Token count mismatch: API reports total 350"
        expected_msg_part2 = "but computed P:100 + C:200 + R:10 = 310"
        assert expected_msg_part1 in str(excinfo.value)
        assert expected_msg_part2 in str(excinfo.value)

    # --- Test make_prediction flow for each subclass --- 
    # Use Case A (no reasoning) for the flow test for simplicity
    def test_make_prediction_flow(self, adapter_class, adapter_instance, mock_response_case_a_no_reasoning):
        """Test the overall flow of make_prediction using Case A (no reasoning) fixture."""
        
        # Patch the methods on the *specific class* being tested
        with patch.object(adapter_class, '_call_ai_model') as mock_call_ai, \
             patch.object(adapter_class, '_get_content', return_value='[[1]]') as mock_get_content, \
             patch.object(adapter_class, '_get_role', return_value="assistant") as mock_get_role, \
             patch.object(adapter_class, '_get_reasoning_summary', return_value="Reasoning summary") as mock_get_reasoning_summary:

            mock_call_ai.return_value = mock_response_case_a_no_reasoning # Use the Case A fixture
            
            prompt = "Test prompt"
            attempt = adapter_instance.make_prediction(prompt)

            mock_call_ai.assert_called_once_with(prompt)
            assert mock_get_content.called
            assert mock_get_role.called
            
            # Check the final attempt object metadata
            assert attempt.answer == [[1]]
            assert attempt.metadata.model == adapter_instance.model_config.model_name
            assert attempt.metadata.provider == adapter_instance.model_config.provider
            assert attempt.metadata.usage.prompt_tokens == 100
            assert attempt.metadata.usage.completion_tokens == 200 # raw ct
            assert attempt.metadata.usage.total_tokens == 300
            assert attempt.metadata.usage.completion_tokens_details.reasoning_tokens == 0 # rt_explicit
            
            # Verify cost calculation based on Case A (no reasoning) logic
            # ct_for_cost = 200, rt_for_cost = 0
            expected_prompt_cost = 100 * (1.0 / 1_000_000)
            expected_completion_cost = 200 * (2.0 / 1_000_000)
            expected_reasoning_cost = 0 * (2.0 / 1_000_000)
            expected_total_cost = expected_prompt_cost + expected_completion_cost + expected_reasoning_cost
            
            assert attempt.metadata.cost.prompt_cost == pytest.approx(expected_prompt_cost)
            assert attempt.metadata.cost.completion_cost == pytest.approx(expected_completion_cost)
            assert attempt.metadata.cost.reasoning_cost == pytest.approx(expected_reasoning_cost)
            assert attempt.metadata.cost.total_cost == pytest.approx(expected_total_cost)
            
            assert attempt.metadata.kwargs == {"temperature": 0.5}
            assert len(attempt.metadata.choices) == 2
            assert attempt.metadata.choices[0].message.role == "user"
            assert attempt.metadata.choices[0].message.content == prompt
            assert attempt.metadata.choices[1].message.role == "assistant"
            assert attempt.metadata.choices[1].message.content == "[[1]]"

    # --- Test streaming functionality ---
    def test_call_ai_model_streaming_enabled(self, adapter_instance, mock_response_case_a_no_reasoning):
        """Test that _call_ai_model uses streaming when stream=True in config."""
        # Set streaming to True in model config
        adapter_instance.model_config.stream = True
        adapter_instance.model_config.api_type = APIType.CHAT_COMPLETIONS
        adapter_instance.model_config.kwargs = {'stream': True}
        
        # Mock the _chat_completion_stream method to check if streaming is used
        with patch.object(adapter_instance, '_chat_completion_stream') as mock_completion_stream:
            mock_completion_stream.return_value = mock_response_case_a_no_reasoning
            
            prompt = "Test streaming prompt"
            
            result = adapter_instance._call_ai_model(prompt)
            
            # Verify _chat_completion_stream was called for streaming
            mock_completion_stream.assert_called_once()
            # Verify the result is our expected response
            assert result == mock_response_case_a_no_reasoning

    def test_call_ai_model_streaming_disabled(self, adapter_instance, mock_response_case_a_no_reasoning):
        """Test that _call_ai_model uses regular completion when stream=False or not set."""
        # Ensure streaming is False or not set
        adapter_instance.model_config.stream = False
        adapter_instance.model_config.api_type = APIType.CHAT_COMPLETIONS
        
        with patch.object(adapter_instance, '_chat_completion') as mock_completion, \
             patch.object(adapter_instance, '_chat_completion_stream') as mock_stream:
            
            mock_completion.return_value = mock_response_case_a_no_reasoning
            
            prompt = "Test non-streaming prompt"
            messages = [{"role": "user", "content": prompt}]
            
            result = adapter_instance._call_ai_model(prompt)
            
            # Verify regular completion was called
            mock_completion.assert_called_once_with(messages)
            # Verify streaming was NOT called
            mock_stream.assert_not_called()
            # Verify the result is our expected response
            assert result == mock_response_case_a_no_reasoning

    def test_chat_completion_stream_functionality(self, adapter_instance, mock_response_case_a_no_reasoning):
        """Test the streaming functionality works correctly with proper chunk handling."""
        # Create mock chunks for streaming
        mock_chunks = []
        for i in range(3):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = f"chunk{i}"
            chunk.choices[0].finish_reason = "stop" if i == 2 else None  # Only set finish_reason on last chunk
            chunk.id = "test-id"
            chunk.created = 12345
            chunk.model = "test-model"
            # Mock usage data on the last chunk
            if i == 2:
                chunk.usage = CompletionUsage(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30
                )
            mock_chunks.append(chunk)
        
        # Mock the client's chat.completions.create to return an iterable
        adapter_instance.client.chat.completions.create = MagicMock(return_value=iter(mock_chunks))
        adapter_instance.model_config.kwargs = {'stream': True}
        
        messages = [{"role": "user", "content": "Test streaming"}]
        
        # Capture print output to verify status messages
        with patch('builtins.print') as mock_print:
            result = adapter_instance._chat_completion_stream(messages)
        
        # Verify the client was called with correct parameters
        adapter_instance.client.chat.completions.create.assert_called_once()
        call_args = adapter_instance.client.chat.completions.create.call_args
        
        # Check that model and messages were passed correctly
        assert call_args[1]['model'] == adapter_instance.model_config.model_name
        assert call_args[1]['messages'] == messages
        assert call_args[1]['stream'] is True
        
        # Verify result contains the concatenated content
        assert result.choices[0].message.content == "chunk0chunk1chunk2"
        
        # Verify the response has the expected structure
        assert result.id == "test-id"
        assert result.model == adapter_instance.model_config.model_name
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    def test_streaming_config_attribute_access(self, adapter_instance):
        """Test that streaming config is properly accessed from model config."""
        # Test when stream attribute doesn't exist (should default to False)
        assert not hasattr(adapter_instance.model_config, 'stream')
        stream_enabled = getattr(adapter_instance.model_config, 'stream', False)
        assert stream_enabled is False
        
        # Test when stream attribute is explicitly set to True
        adapter_instance.model_config.stream = True
        stream_enabled = getattr(adapter_instance.model_config, 'stream', False)
        assert stream_enabled is True
        
        # Test when stream attribute is explicitly set to False
        adapter_instance.model_config.stream = False
        stream_enabled = getattr(adapter_instance.model_config, 'stream', False)
        assert stream_enabled is False
    
    def test_responses_streaming_enabled(self, adapter_instance):
        """Test that responses streaming is properly handled."""
        # Set up for responses API with streaming
        adapter_instance.model_config.api_type = APIType.RESPONSES
        adapter_instance.model_config.stream = True
        adapter_instance.model_config.kwargs = {'stream': True}
        
        # Mock the _responses_stream method
        with patch.object(adapter_instance, '_responses_stream') as mock_responses_stream:
            mock_response = MagicMock()
            mock_responses_stream.return_value = mock_response
            
            prompt = "Test responses streaming prompt"
            
            result = adapter_instance._call_ai_model(prompt)
            
            # Verify _responses_stream was called
            mock_responses_stream.assert_called_once()
            # Verify the result is our mock response
            assert result == mock_response
    
    def test_responses_streaming_with_background_raises_error(self, adapter_instance):
        """Test that enabling both streaming and background for responses API raises ValueError."""
        # Set up for responses API with both streaming and background enabled
        adapter_instance.model_config.api_type = APIType.RESPONSES
        adapter_instance.model_config.stream = True
        adapter_instance.model_config.background = True
        adapter_instance.model_config.kwargs = {'stream': True, 'background': True}

        prompt = "Test prompt"

        # Should raise ValueError when both streaming and background are enabled
        with pytest.raises(ValueError, match="Cannot enable both streaming and background for the responses API type"):
            adapter_instance._call_ai_model(prompt)

    # --- Tests for extract_json_from_response ---

    def test_extract_json_direct_array(self, adapter_class, adapter_instance):
        """Test extracting a direct JSON array of arrays."""
        with patch.object(adapter_class, '_call_ai_model') as mock_call, \
             patch.object(adapter_class, '_get_content') as mock_get_content:
            mock_get_content.return_value = "[[1, 2], [3, 4]]"

            result = adapter_instance.extract_json_from_response("some response")

            assert result == [[1, 2], [3, 4]]

    def test_extract_json_dict_with_response_key(self, adapter_class, adapter_instance):
        """Test extracting JSON from a dict with 'response' key."""
        with patch.object(adapter_class, '_call_ai_model') as mock_call, \
             patch.object(adapter_class, '_get_content') as mock_get_content:
            mock_get_content.return_value = '{"response": [[5, 6], [7, 8]]}'

            result = adapter_instance.extract_json_from_response("some response")

            assert result == [[5, 6], [7, 8]]

    def test_extract_json_code_block_wrapped(self, adapter_class, adapter_instance):
        """Test extracting JSON wrapped in markdown code blocks."""
        with patch.object(adapter_class, '_call_ai_model') as mock_call, \
             patch.object(adapter_class, '_get_content') as mock_get_content:
            mock_get_content.return_value = "```json\n[[1, 2], [3, 4]]\n```"

            result = adapter_instance.extract_json_from_response("some response")

            # After code block removal, should find the array via regex or bracket search
            assert result == [[1, 2], [3, 4]]

    def test_extract_json_with_surrounding_text(self, adapter_class, adapter_instance):
        """Test extracting JSON array embedded in surrounding text."""
        with patch.object(adapter_class, '_call_ai_model') as mock_call, \
             patch.object(adapter_class, '_get_content') as mock_get_content:
            mock_get_content.return_value = "The answer is [[1, 2], [3, 4]] as shown above."

            result = adapter_instance.extract_json_from_response("some response")

            assert result == [[1, 2], [3, 4]]

    def test_extract_json_invalid_returns_none(self, adapter_class, adapter_instance):
        """Test that invalid/unparseable responses return None."""
        with patch.object(adapter_class, '_call_ai_model') as mock_call, \
             patch.object(adapter_class, '_get_content') as mock_get_content:
            mock_get_content.return_value = "I don't know the answer"

            result = adapter_instance.extract_json_from_response("some response")

            assert result is None

    def test_extract_json_ai_call_failure_fallback(self, adapter_class, adapter_instance):
        """Test fallback to input when AI call fails."""
        with patch.object(adapter_class, '_call_ai_model') as mock_call:
            mock_call.side_effect = Exception("API Error")

            # Even with AI failure, if input contains valid JSON it should parse
            result = adapter_instance.extract_json_from_response("[[9, 9], [8, 8]]")

            assert result == [[9, 9], [8, 8]]

    def test_extract_json_nested_in_explanation(self, adapter_class, adapter_instance):
        """Test regex extraction when array is nested in explanation text."""
        with patch.object(adapter_class, '_call_ai_model') as mock_call, \
             patch.object(adapter_class, '_get_content') as mock_get_content:
            mock_get_content.return_value = "Based on the pattern, the output should be [[0, 1, 2], [3, 4, 5]] because of the transformation rules."

            result = adapter_instance.extract_json_from_response("some response")

            assert result == [[0, 1, 2], [3, 4, 5]]
