import abc
from .provider import ProviderAdapter
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types import CompletionUsage
from arc_agi_benchmarking.schemas import APIType, Cost, Attempt, Usage, CompletionTokensDetails
from typing import Optional, Any, List, Dict
from time import sleep
import logging
import time

load_dotenv()

logger = logging.getLogger(__name__)


class StreamingResponseBuilder:
    """Helper class to build structured responses from streaming data."""
    
    @staticmethod
    def build_chat_completion_response(content: str, usage_data: CompletionUsage, 
                                     response_id: str, model_name: str, 
                                     finish_reason: str = "stop") -> ChatCompletion:
        """Build a ChatCompletion response from streaming data."""
        return ChatCompletion(
            id=response_id,
            choices=[
                OpenAIChoice(
                    finish_reason=finish_reason,
                    index=0,
                    message=ChatCompletionMessage(
                        content=content,
                        role='assistant'
                    ),
                    logprobs=None
                )
            ],
            created=int(time.time()),
            model=model_name,
            object='chat.completion',
            usage=usage_data
        )
    
    @staticmethod
    def build_responses_response(content: str, usage_data: Any, 
                               response_id: str, model_name: str, 
                               finish_reason: str = "stop") -> Any:
        """Build a responses API response from streaming data."""
        class MockContent:
            def __init__(self, text):
                self.text = text
                self.type = "output_text"
        
        class MockOutput:
            def __init__(self, text):
                self.content = [MockContent(text)]
                self.role = "assistant"
                self.type = "message"
        
        class MockResponse:
            def __init__(self, model_name, full_content, usage_data, response_id, finish_reason=None):
                self.id = response_id or "stream-response"
                self.model = model_name
                self.object = "response"
                self.output = [MockOutput(full_content)]
                self.finish_reason = finish_reason
                self.usage = usage_data
        
        return MockResponse(model_name, content, usage_data, response_id, finish_reason)


class OpenAIBaseAdapter(ProviderAdapter, abc.ABC):


    @abc.abstractmethod
    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the model and return an Attempt object.
        Subclasses must implement this to handle provider-specific response parsing.
        """
        pass

    def _call_ai_model(self, prompt: str) -> Any:
        """
        Call the appropriate OpenAI API based on the api_type
        """
        
        # Validate that background and stream are not both enabled for responses API
        stream_enabled = self.model_config.kwargs.get('stream', False) or getattr(self.model_config, 'stream', False)
        messages = [{"role": "user", "content": prompt}]
        if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
            if stream_enabled:
                return self._chat_completion_stream(messages)
            return self._chat_completion(messages)
        
        else:  # APIType.RESPONSES
            # account for different parameter names between chat completions and responses APIs
            self._normalize_to_responses_kwargs()
            background_enabled = getattr(self.model_config, 'background', False)
            if stream_enabled and background_enabled:
                raise ValueError("Cannot enable both streaming and background for the responses API type.")
            if stream_enabled:
                return self._responses_stream(messages)
            return self._responses(messages)
    
    def _chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make a call to the OpenAI Chat Completions API
        """
        logger.debug(f"Calling OpenAI API with model: {self.model_config.model_name} and kwargs: {self.model_config.kwargs}")
    
        
        return self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            **self.model_config.kwargs
        )
    
    def _chat_completion_stream(self, messages: List[Dict[str, str]]) -> ChatCompletion:
        """
        Make a streaming call to the OpenAI Chat Completions API and return the final response.
        """
        logger.debug(f"Starting streaming chat completion for model: {self.model_config.model_name}")
        
        # Prepare kwargs for streaming, removing 'stream' to avoid duplication
        stream_kwargs = {k: v for k, v in self.model_config.kwargs.items() if k != 'stream'}
        
        try:
            # Create the stream with usage tracking
            stream = self.client.chat.completions.create(
                model=self.model_config.model_name,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **stream_kwargs
            )
            
            # Process the stream and collect data
            content_chunks = []
            last_chunk = None
            finish_reason = "stop"
            
            for chunk in stream:
                last_chunk = chunk
                
                # Extract content from chunk
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
                
                # Track finish reason
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            
            # Build final response
            final_content = ''.join(content_chunks)
            
            # Get usage data and metadata from last chunk
            usage_data = last_chunk.usage if last_chunk and hasattr(last_chunk, 'usage') else None
            response_id = last_chunk.id if last_chunk else f"stream-{int(time.time())}"
            
            if not usage_data:
                logger.warning("No usage data received from streaming response")
                usage_data = CompletionUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                )
            
            logger.debug(f"Streaming complete. Content length: {len(final_content)}")
            
            return StreamingResponseBuilder.build_chat_completion_response(
                content=final_content,
                usage_data=usage_data,
                response_id=response_id,
                model_name=self.model_config.model_name,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise
    
    def _delete_response(self, response_id: str) -> None:
        """
        Delete a response from the OpenAI API
        """
        self.client.responses.delete(response_id)
    
    def _responses(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make a call to the OpenAI Responses API
        """

        resp = self.client.responses.create(
            model=self.model_config.model_name,
            input=messages,
            **self.model_config.kwargs
        )

        # For background mode
        if "background" in self.model_config.kwargs and self.model_config.kwargs["background"]:
            while resp.status in {"queued", "in_progress"}:
                sleep(10)
                resp = self.client.responses.retrieve(resp.id)

            # Delete the background response after we're done with it
            try:
                self._delete_response(resp.id)
            except Exception as e:
                logger.warning(f"Error deleting response: {e}")

        return resp
    
    

    def _responses_stream(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make a streaming call to the OpenAI Responses API and return the final response.
        """
        logger.debug(f"Starting streaming responses for model: {self.model_config.model_name}")
        
        # Prepare kwargs for streaming, removing 'stream' to avoid duplication
        stream_kwargs = {k: v for k, v in self.model_config.kwargs.items() if k != 'stream'}
        
        try:
            # Create the stream
            stream = self.client.responses.create(
                model=self.model_config.model_name,
                input=messages,
                stream=True,
                **stream_kwargs
            )
            
            # Process the stream and collect data
            content_chunks = []
            response_id = None
            finish_reason = "stop"
            usage_data = None
            
            for chunk in stream:
                # Extract response ID
                if chunk.type == 'response.created':
                    response_id = chunk.response.id
                
                # Extract content deltas
                if chunk.type == 'response.output_text.delta':
                    content_chunks.append(chunk.delta)
                
                # Track finish reason
                if hasattr(chunk, 'finish_reason') and chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                
                # Extract usage data
                if hasattr(chunk, 'response') and chunk.response.usage:
                    raw_usage = chunk.response.usage
                    # Normalize token field names
                    prompt_tokens = getattr(raw_usage, 'prompt_tokens', getattr(raw_usage, 'input_tokens', 0))
                    completion_tokens = getattr(raw_usage, 'completion_tokens', getattr(raw_usage, 'output_tokens', 0))
                    total_tokens = getattr(raw_usage, 'total_tokens', prompt_tokens + completion_tokens)
                    
                    # Extract reasoning tokens if available
                    reasoning_tokens = 0
                    if hasattr(raw_usage, 'completion_tokens_details') and raw_usage.completion_tokens_details:
                        reasoning_tokens = getattr(raw_usage.completion_tokens_details, 'reasoning_tokens', 0)
                    
                    # Create proper Usage object from raw usage data
                    usage_data = Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        completion_tokens_details=CompletionTokensDetails(
                            reasoning_tokens=reasoning_tokens,
                            accepted_prediction_tokens=completion_tokens - reasoning_tokens,
                            rejected_prediction_tokens=0
                        )
                    )
            
            # Build final response
            final_content = ''.join(content_chunks)
            response_id = response_id or f"stream-{int(time.time())}"
            
            if not usage_data:
                logger.warning("No usage data received from streaming response")
                # Create proper Usage object with minimal data
                usage_data = Usage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=0,
                        accepted_prediction_tokens=0,
                        rejected_prediction_tokens=0
                    )
                )
            
            logger.debug(f"Streaming complete. Content length: {len(final_content)}")
            
            return StreamingResponseBuilder.build_responses_response(
                content=final_content,
                usage_data=usage_data,
                response_id=response_id,
                model_name=self.model_config.model_name,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise

    @abc.abstractmethod
    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        """
        Extract JSON from the provider's response.
        Subclasses must implement this based on expected response format.
        """
        pass
        
    def _get_usage(self, response: Any) -> Any:
        """
        Get the usage from the response
        """
        return response.usage

    def _get_reasoning_summary(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Extract reasoning summary from the response if available (primarily for Responses API).
        """
        reasoning_summary = None
        if self.model_config.api_type == APIType.RESPONSES:
            # Safely access potential reasoning summary
            if hasattr(response, 'reasoning') and response.reasoning and hasattr(response.reasoning, 'summary'):
                reasoning_summary = response.reasoning.summary # Will be None if not present
        # Chat Completions API does not currently provide a separate summary field
        return reasoning_summary

    def _get_content(self, response: Any) -> str:
        """
        Get the content from the response
        """
        api_type = self.model_config.api_type
        if api_type == APIType.CHAT_COMPLETIONS:
            return response.choices[0].message.content
        elif api_type == APIType.RESPONSES:
            # The structure for responses is response.output[0].content[0].text
            return response.output[0].content[0].text
        else:
            # Fallback for other potential types, though not expected
            return response.choices[0].text

    def _get_role(self, response: Any) -> str:
        """Extract role from a standard OpenAI-like response object."""
        # Implementation copied from OpenAIAdapter
        role = "assistant" # Default role
        if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
                 role = getattr(response.choices[0].message, 'role', "assistant") or "assistant"
        # Responses API implies assistant role for the main output
        return role
        
    def _normalize_to_responses_kwargs(self):
        """
        Normalize kwargs based on API type to handle different parameter names between chat completions and responses APIs
        """
        if self.model_config.api_type == APIType.RESPONSES:
            # Convert max_tokens and max_completion_tokens to max_output_tokens for responses API
            if "max_tokens" in self.model_config.kwargs:
                self.model_config.kwargs["max_output_tokens"] = self.model_config.kwargs.pop("max_tokens")
            if "max_completion_tokens" in self.model_config.kwargs:
                self.model_config.kwargs["max_output_tokens"] = self.model_config.kwargs.pop("max_completion_tokens") 

    def _calculate_cost(self, response: Any) -> Cost:
        """Calculate usage costs, validate token counts, and return a Cost object."""
        usage = self._get_usage(response)
        
        # Raw token counts from provider response (via _get_usage)
        pt_raw = usage.prompt_tokens
        ct_raw = usage.completion_tokens
        tt_raw = usage.total_tokens or 0
        rt_explicit = 0 # Not available from OpenAI

        # For OpenAI, we assume the raw tokens are correct and can be used directly for billing.
        # Determine effective token counts for cost calculation based on the two cases
        prompt_tokens_for_cost = pt_raw
        completion_tokens_for_cost = 0
        reasoning_tokens_for_cost = 0

        # Case A: Completion includes Reasoning (pt + ct == tt)
        # Here, ct_raw contains both reasoning and actual completion.
        if tt_raw == 0 or (pt_raw + ct_raw == tt_raw): 
            reasoning_tokens_for_cost = rt_explicit # Use explicit reasoning count if provided
            # Subtract explicit reasoning from raw completion to get actual completion
            completion_tokens_for_cost = max(0, ct_raw - reasoning_tokens_for_cost) 
            # Safety check: ensure computed total matches raw total if tt_raw was provided
            computed_total = pt_raw + ct_raw # In this case, ct_raw represents the full assistant output
        
        # Case B: Reasoning is Separate or Inferred (pt + ct < tt)
        # Here, ct_raw likely represents only the final answer tokens.
        else: 
            # Use explicit reasoning if provided, otherwise infer it
            reasoning_tokens_for_cost = rt_explicit if rt_explicit else tt_raw - (pt_raw + ct_raw)
            completion_tokens_for_cost = ct_raw # Raw completion is assumed to be separate
            # Calculate computed total based on the parts
            computed_total = pt_raw + completion_tokens_for_cost + reasoning_tokens_for_cost

        # Final Sanity Check: Compare computed total against provider's total (if provider gave one)
        if tt_raw and computed_total != tt_raw:
            from arc_agi_benchmarking.errors import TokenMismatchError # Local import
            raise TokenMismatchError(
                f"Token count mismatch: API reports total {tt_raw}, "
                f"but computed P:{prompt_tokens_for_cost} + C:{completion_tokens_for_cost} + R:{reasoning_tokens_for_cost} = {computed_total}"
            )

        # Determine costs per token
        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000
        
        # Calculate costs based on the derived token counts
        prompt_cost = prompt_tokens_for_cost * input_cost_per_token
        # Cost for the 'actual' completion tokens (excluding reasoning in Case A)
        completion_cost = completion_tokens_for_cost * output_cost_per_token
        # Cost for the reasoning tokens
        reasoning_cost = reasoning_tokens_for_cost * output_cost_per_token
        # Total cost is the sum of all components
        total_cost = prompt_cost + completion_cost + reasoning_cost

        from arc_agi_benchmarking.schemas import Cost  # Local import (avoids circular issues in some environments)
        return Cost(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost, # Cost of 'actual' completion
            reasoning_cost=reasoning_cost,   # Cost of reasoning part
            total_cost=total_cost,           # True total expenditure
        ) 