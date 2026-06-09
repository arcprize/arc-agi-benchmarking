import pytest
from unittest.mock import Mock, MagicMock, patch
from arc_agi_benchmarking.adapters.anthropic import AnthropicAdapter
from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def mock_model_config():
    """Provides a mock ModelConfig for Anthropic."""
    return ModelConfig(
        name="test-claude-model",
        model_name="claude-3-7-sonnet-20250219",
        provider="anthropic",
        pricing=ModelPricing(date="2025-03-12", input=3.0, output=15.0),
        kwargs={"max_tokens": 8192}
    )


@pytest.fixture
def adapter_instance(mock_model_config):
    """Creates an AnthropicAdapter instance with mocked config."""
    with patch.object(AnthropicAdapter, 'init_client') as mock_init:
        mock_init.return_value = Mock()
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.model_config = mock_model_config
        adapter.client = mock_init.return_value
        return adapter


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic message response."""
    mock_response = Mock()
    mock_response.id = "msg_test123"
    mock_response.model = "claude-3-7-sonnet-20250219"
    mock_response.role = "assistant"

    # Mock content blocks with valid JSON array response
    mock_content = Mock()
    mock_content.type = "text"
    mock_content.text = "[[1, 2], [3, 4]]"
    mock_response.content = [mock_content]

    # Mock usage
    mock_response.usage = Mock()
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 25

    return mock_response


class TestAnthropicStreaming:
    """Test streaming functionality for Anthropic adapter."""

    def test_streaming_enabled(self, adapter_instance, mock_anthropic_response):
        """Test that streaming is used when stream=True in config."""
        # Enable streaming
        adapter_instance.model_config.stream = True
        adapter_instance.model_config.kwargs = {'stream': True, 'max_tokens': 8192}

        # Mock the chat_completion_stream method
        with patch.object(adapter_instance, 'chat_completion_stream') as mock_stream:
            mock_stream.return_value = mock_anthropic_response

            messages = [{"role": "user", "content": "Hello"}]

            # Call make_prediction which should route to streaming
            attempt = adapter_instance.make_prediction("Hello")

            # Verify chat_completion_stream was called
            mock_stream.assert_called_once()
            assert attempt.answer == [[1, 2], [3, 4]]

    def test_streaming_disabled(self, adapter_instance, mock_anthropic_response):
        """Test that regular completion is used when stream=False."""
        # Disable streaming
        adapter_instance.model_config.kwargs = {'max_tokens': 8192}

        # Mock the chat_completion method
        with patch.object(adapter_instance, 'chat_completion') as mock_completion:
            mock_completion.return_value = mock_anthropic_response

            # Call make_prediction which should route to regular completion
            attempt = adapter_instance.make_prediction("Hello")

            # Verify chat_completion was called (not streaming)
            mock_completion.assert_called_once()
            assert attempt.answer == [[1, 2], [3, 4]]

    def test_chat_completion_stream_method(self, adapter_instance, mock_anthropic_response):
        """Test the chat_completion_stream method directly."""
        messages = [{"role": "user", "content": "Test message"}]

        # Mock the Anthropic client's messages.stream context manager
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value.get_final_message.return_value = mock_anthropic_response

        with patch.object(adapter_instance.client.messages, 'stream', return_value=mock_stream_context):
            result = adapter_instance.chat_completion_stream(messages)

            # Verify the result is the final message
            assert result == mock_anthropic_response
            assert result.id == "msg_test123"

            # Verify stream was called with correct parameters
            adapter_instance.client.messages.stream.assert_called_once_with(
                model="claude-3-7-sonnet-20250219",
                messages=messages,
                tools=[],
                max_tokens=8192
            )

    def test_streaming_with_tools(self, adapter_instance, mock_anthropic_response):
        """Test that streaming works with tools parameter."""
        messages = [{"role": "user", "content": "Test with tools"}]
        tools = [{"name": "test_tool", "description": "A test tool"}]

        # Mock the stream context
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value.get_final_message.return_value = mock_anthropic_response

        with patch.object(adapter_instance.client.messages, 'stream', return_value=mock_stream_context):
            result = adapter_instance.chat_completion_stream(messages, tools=tools)

            # Verify tools were passed correctly
            adapter_instance.client.messages.stream.assert_called_once_with(
                model="claude-3-7-sonnet-20250219",
                messages=messages,
                tools=tools,
                max_tokens=8192
            )
            assert result == mock_anthropic_response


def _make_batch_handle(batch_id="msgbatch_test", status_sequence=("ended",)):
    """
    Build a mock batches API namespace whose retrieve() returns successive
    batches with the given processing_status values, then sticks on the last one.
    """
    status_iter = iter(status_sequence)

    def _next_batch():
        try:
            status = next(status_iter)
        except StopIteration:
            status = status_sequence[-1]
        b = Mock()
        b.id = batch_id
        b.processing_status = status
        b.request_counts = Mock()
        return b

    batches = Mock()
    batches.create = Mock(side_effect=lambda *a, **kw: _next_batch())
    batches.retrieve = Mock(side_effect=lambda _id: _next_batch())
    batches.results = Mock()
    batches.delete = Mock()
    return batches


def _make_succeeded_result(custom_id, message):
    """Build a single succeeded batch result entry."""
    result = Mock()
    result.custom_id = custom_id
    result.result = Mock()
    result.result.type = "succeeded"
    result.result.message = message
    return result


class TestAnthropicBatch:
    """Test Message Batches API integration for the Anthropic adapter."""

    def test_batch_and_stream_together_raises(self, adapter_instance):
        """batch=True + stream=True should be rejected before any API call."""
        adapter_instance.model_config.kwargs = {
            'batch': True,
            'stream': True,
            'max_tokens': 8192,
        }

        with pytest.raises(ValueError, match="Cannot enable both 'stream' and 'batch'"):
            adapter_instance.make_prediction("Hello")

    def test_make_prediction_routes_to_batch(self, adapter_instance, mock_anthropic_response):
        """When batch=True, make_prediction must call chat_completion_batch."""
        adapter_instance.model_config.kwargs = {'batch': True, 'max_tokens': 8192}

        with patch.object(adapter_instance, 'chat_completion_batch') as mock_batch:
            mock_batch.return_value = mock_anthropic_response

            attempt = adapter_instance.make_prediction("Hello")

            mock_batch.assert_called_once()
            assert attempt.answer == [[1, 2], [3, 4]]

    def test_chat_completion_batch_create_poll_results_delete(self, adapter_instance, mock_anthropic_response):
        """
        chat_completion_batch must:
          1. create a batch with a single Request,
          2. poll until processing_status == 'ended',
          3. fetch results and return the message for our custom_id,
          4. delete the batch afterwards.
        """
        adapter_instance.model_config.kwargs = {'batch': True, 'max_tokens': 8192}

        batches = _make_batch_handle(
            batch_id="msgbatch_abc",
            status_sequence=("in_progress", "in_progress", "ended"),
        )
        adapter_instance.client.messages.batches = batches

        # Capture custom_id from the create() call so we can return a matching result
        captured = {}
        original_create = batches.create.side_effect

        def capture_create(*args, **kwargs):
            captured['requests'] = kwargs.get('requests') or (args[0] if args else None)
            return original_create(*args, **kwargs)

        batches.create.side_effect = capture_create

        # Patch sleep so the poll loop doesn't actually wait
        with patch('arc_agi_benchmarking.adapters.anthropic.time.sleep') as mock_sleep:
            # results() will be called after the batch ends; return our message
            def results_side_effect(_batch_id):
                custom_id = captured['requests'][0]['custom_id']
                return iter([_make_succeeded_result(custom_id, mock_anthropic_response)])

            batches.results.side_effect = results_side_effect

            messages = [{"role": "user", "content": "Hi"}]
            result = adapter_instance.chat_completion_batch(messages)

        # Returned message is the underlying Anthropic message object
        assert result is mock_anthropic_response

        # create called exactly once, with one Request, no betas kwarg
        batches.create.assert_called_once()
        create_kwargs = batches.create.call_args.kwargs
        assert 'betas' not in create_kwargs
        assert len(create_kwargs['requests']) == 1
        req = create_kwargs['requests'][0]
        assert req['custom_id'].startswith('arc-')
        # 'batch' and 'stream' must NOT be passed through into the per-request params
        params = req['params']
        assert 'batch' not in params
        assert 'stream' not in params
        assert params['model'] == "claude-3-7-sonnet-20250219"
        assert params['max_tokens'] == 8192

        # Poll loop ran (in_progress -> in_progress -> ended = 2 retrieves)
        assert batches.retrieve.call_count == 2
        # And it slept between polls
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10)

        # Delete was called with the batch id
        batches.delete.assert_called_once_with("msgbatch_abc")

    def test_chat_completion_batch_deletes_even_on_errored_result(self, adapter_instance):
        """If the result type is 'errored', we must still delete the batch."""
        adapter_instance.model_config.kwargs = {'batch': True, 'max_tokens': 8192}

        batches = _make_batch_handle(batch_id="msgbatch_err", status_sequence=("ended",))
        adapter_instance.client.messages.batches = batches

        captured = {}
        original_create = batches.create.side_effect

        def capture_create(*args, **kwargs):
            captured['requests'] = kwargs.get('requests') or (args[0] if args else None)
            return original_create(*args, **kwargs)

        batches.create.side_effect = capture_create

        def results_side_effect(_batch_id):
            custom_id = captured['requests'][0]['custom_id']
            errored = Mock()
            errored.custom_id = custom_id
            errored.result = Mock()
            errored.result.type = "errored"
            errored.result.error = "some_error"
            return iter([errored])

        batches.results.side_effect = results_side_effect

        with patch('arc_agi_benchmarking.adapters.anthropic.time.sleep'):
            with pytest.raises(RuntimeError, match="errored"):
                adapter_instance.chat_completion_batch([{"role": "user", "content": "Hi"}])

        # Delete is still called even though we raised
        batches.delete.assert_called_once_with("msgbatch_err")

    def test_chat_completion_batch_routes_to_beta_when_betas_set(self, adapter_instance, mock_anthropic_response):
        """When 'betas' is in kwargs, requests must go through client.beta.messages.batches."""
        adapter_instance.model_config.kwargs = {
            'batch': True,
            'max_tokens': 8192,
            'betas': ['some-beta-flag'],
        }

        beta_batches = _make_batch_handle(batch_id="msgbatch_beta", status_sequence=("ended",))
        adapter_instance.client.beta.messages.batches = beta_batches

        captured = {}
        original_create = beta_batches.create.side_effect

        def capture_create(*args, **kwargs):
            captured['requests'] = kwargs.get('requests') or (args[0] if args else None)
            captured['betas'] = kwargs.get('betas')
            return original_create(*args, **kwargs)

        beta_batches.create.side_effect = capture_create

        def results_side_effect(_batch_id):
            custom_id = captured['requests'][0]['custom_id']
            return iter([_make_succeeded_result(custom_id, mock_anthropic_response)])

        beta_batches.results.side_effect = results_side_effect

        with patch('arc_agi_benchmarking.adapters.anthropic.time.sleep'):
            result = adapter_instance.chat_completion_batch([{"role": "user", "content": "Hi"}])

        assert result is mock_anthropic_response
        # Went through beta endpoint with the betas kwarg
        beta_batches.create.assert_called_once()
        assert captured['betas'] == ['some-beta-flag']
        # 'betas' was NOT leaked into per-request params
        assert 'betas' not in captured['requests'][0]['params']
        beta_batches.delete.assert_called_once_with("msgbatch_beta")

    def test_chat_completion_batch_inlines_extra_body_into_params(self, adapter_instance, mock_anthropic_response):
        """
        Regression test for the 'extra_body: Extra inputs are not permitted' error.

        The Anthropic batches API sends per-request `params` verbatim. The SDK-level
        `extra_body` escape hatch only works on the sync Messages API (where the SDK
        merges it into the JSON body). For batch requests we must inline its contents
        into the params dict ourselves.
        """
        adapter_instance.model_config.kwargs = {
            'batch': True,
            'max_tokens': 8192,
            'extra_body': {'output_config': {'effort': 'low'}},
        }

        batches = _make_batch_handle(batch_id="msgbatch_eb", status_sequence=("ended",))
        adapter_instance.client.messages.batches = batches

        captured = {}
        original_create = batches.create.side_effect

        def capture_create(*args, **kwargs):
            captured['requests'] = kwargs.get('requests') or (args[0] if args else None)
            return original_create(*args, **kwargs)

        batches.create.side_effect = capture_create

        def results_side_effect(_batch_id):
            custom_id = captured['requests'][0]['custom_id']
            return iter([_make_succeeded_result(custom_id, mock_anthropic_response)])

        batches.results.side_effect = results_side_effect

        with patch('arc_agi_benchmarking.adapters.anthropic.time.sleep'):
            adapter_instance.chat_completion_batch([{"role": "user", "content": "Hi"}])

        params = captured['requests'][0]['params']
        # extra_body must NOT appear as a literal key in params
        assert 'extra_body' not in params, \
            "'extra_body' leaked into batch params and will be rejected by Anthropic"
        # Its contents must be merged into params at the top level
        assert params.get('output_config') == {'effort': 'low'}, \
            "extra_body contents were not inlined into params"

    def test_chat_completion_batch_delete_failure_does_not_raise(self, adapter_instance, mock_anthropic_response):
        """If delete fails, we should log a warning but not raise (matches OpenAI bg-mode behavior)."""
        adapter_instance.model_config.kwargs = {'batch': True, 'max_tokens': 8192}

        batches = _make_batch_handle(batch_id="msgbatch_del", status_sequence=("ended",))
        batches.delete.side_effect = Exception("delete failed")
        adapter_instance.client.messages.batches = batches

        captured = {}
        original_create = batches.create.side_effect

        def capture_create(*args, **kwargs):
            captured['requests'] = kwargs.get('requests') or (args[0] if args else None)
            return original_create(*args, **kwargs)

        batches.create.side_effect = capture_create

        def results_side_effect(_batch_id):
            custom_id = captured['requests'][0]['custom_id']
            return iter([_make_succeeded_result(custom_id, mock_anthropic_response)])

        batches.results.side_effect = results_side_effect

        with patch('arc_agi_benchmarking.adapters.anthropic.time.sleep'):
            # Should NOT raise even though delete blew up
            result = adapter_instance.chat_completion_batch([{"role": "user", "content": "Hi"}])

        assert result is mock_anthropic_response
        batches.delete.assert_called_once_with("msgbatch_del")