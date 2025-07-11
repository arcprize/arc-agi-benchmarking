# This comment is just to trigger a rename operation via the edit tool
# The actual content change will happen in the next step. 

import abc
from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types import CompletionUsage
from datetime import datetime, timezone
from arc_agi_benchmarking.schemas import APIType, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
from typing import Optional, Any, List, Dict # Added List, Dict
from time import sleep
import logging
import time

from openai.types.responses import Response as OpenAIResponse

from openai.types import CompletionUsage

from ..schemas import Usage, CompletionTokensDetails

load_dotenv()

logger = logging.getLogger(__name__)


# Helper classes to mock the structure of an OpenAI Response object for streaming
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

        prompt_tokens = getattr(usage_data, 'prompt_tokens', 0)
        completion_tokens = getattr(usage_data, 'completion_tokens', 0)
        total_tokens = getattr(usage_data, 'total_tokens', 0) or (prompt_tokens + completion_tokens)
        
        # Create a mock CompletionTokensDetails since it's not provided by the stream
        mock_details = CompletionTokensDetails(
            reasoning_tokens=0,
            accepted_prediction_tokens=completion_tokens,
            rejected_prediction_tokens=0
        )
        
        self.usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=mock_details
        )


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
    
    def _chat_completion_stream(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make a streaming call to the OpenAI Chat Completions API and return the final response
        """
        logger.debug(f"Calling OpenAI API with streaming for model: {self.model_config.model_name} and kwargs: {self.model_config.kwargs}")
        
        # Create the stream with stream=True and include_usage for token tracking
        stream = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},  # Include usage stats in stream
            **{k: v for k, v in self.model_config.kwargs.items() if k != 'stream'}  # Remove stream from kwargs to avoid duplication
        )
        
        logger.debug("Starting streaming response...")
        chunk_count = 0
        
        # Collect all chunks and track metadata
        collected_messages = []
        last_chunk = None
        finish_reason = None
        
        try:
            for chunk in stream:
                chunk_count += 1
                if chunk_count % 100 == 0:  # Only print every 100 chunks to reduce verbosity
                    logger.debug(f"Received {chunk_count} chunks...", end="\r")
                
                # Keep track of the last chunk for metadata
                last_chunk = chunk
                
                # Collect content from each chunk
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        collected_messages.append(chunk.choices[0].delta.content)
                    # Track finish reason if available
                    if chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise
        
        logger.debug(f"\nStreaming complete. Total chunks: {chunk_count}")
        
        # Build a complete response object that looks like a non-streaming response
        final_content = ''.join(collected_messages)
        
        # Extract usage data from the last chunk if available
        usage_data = None
        if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
            logger.debug(f"Usage data: {last_chunk.usage}")
            usage_data = last_chunk.usage
        
        # Create response that matches non-streaming format
        if usage_data:
            # Use actual usage data from stream
            mock_response = ChatCompletion(
                id=last_chunk.id if last_chunk else 'stream-response',
                choices=[
                    OpenAIChoice(
                        finish_reason=finish_reason or 'stop',
                        index=0,
                        message=ChatCompletionMessage(
                            content=final_content,
                            role='assistant'
                        ),
                        logprobs=None
                    )
                ],
                created=last_chunk.created if last_chunk else 0,
                model=last_chunk.model if last_chunk else self.model_config.model_name,
                object='chat.completion',
                usage=usage_data  # Use the actual usage data from stream
            )
        else:
            # Fallback if no usage data (shouldn't happen with include_usage=True)
            logger.warning("No usage data received in streaming response")
            mock_response = ChatCompletion(
                id=last_chunk.id if last_chunk else 'stream-response',
                choices=[
                    OpenAIChoice(
                        finish_reason=finish_reason or 'stop',
                        index=0,
                        message=ChatCompletionMessage(
                            content=final_content,
                            role='assistant'
                        ),
                        logprobs=None
                    )
                ],
                created=last_chunk.created if last_chunk else 0,
                model=last_chunk.model if last_chunk else self.model_config.model_name,
                object='chat.completion',
                usage=CompletionUsage(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0
                )
            )
        
        return mock_response
    
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
        logger.debug(f"Calling OpenAI Responses API with streaming for model: {self.model_config.model_name} and kwargs: {self.model_config.kwargs}")
        
        stream = self.client.responses.create(
            model=self.model_config.model_name,
            input=messages,
            stream=True,
            **{k: v for k, v in self.model_config.kwargs.items() if k != 'stream'}
        )
        
        logger.debug("Starting streaming response...")
        chunk_count = 0
        collected_content = []
        response_id = None
        finish_reason = None
        usage_data = None
        
        try:
            for chunk in stream:
                chunk_count += 1
                if chunk_count % 100 == 0:
                    logger.debug(f"Received {chunk_count} chunks...")
                
                if chunk.type == 'response.created':
                    response_id = chunk.response.id
                
                if chunk.type == 'response.output_text.delta':
                    collected_content.append(chunk.delta)
                
                if hasattr(chunk, 'finish_reason') and chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_data = chunk.usage
                    # Standardize and compute total_tokens using correct fields
                    prompt_toks = getattr(usage_data, 'prompt_tokens', getattr(usage_data, 'input_tokens', 0))
                    completion_toks = getattr(usage_data, 'completion_tokens', getattr(usage_data, 'output_tokens', 0))
                    usage_data.total_tokens = prompt_toks + completion_toks
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise

        logger.debug(f"Streaming complete. Total chunks: {chunk_count}")
        
        full_content = ''.join(collected_content)
        
        # Construct a mock response object that mimics the non-streaming response
        mock_response = MockResponse(
            model_name=self.model_config.model_name,
            full_content=full_content,
            usage_data=usage_data,
            response_id=response_id,
            finish_reason=finish_reason,
        )

        return mock_response

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