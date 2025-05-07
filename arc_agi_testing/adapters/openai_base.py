# This comment is just to trigger a rename operation via the edit tool
# The actual content change will happen in the next step. 

import abc
from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from datetime import datetime, timezone
from arc_agi_testing.schemas import APIType, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
from typing import Optional, Any, List, Dict # Added List, Dict

load_dotenv()


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
        messages = [{"role": "user", "content": prompt}]
        if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
            return self._chat_completion(messages)
        else:  # APIType.RESPONSES
            # account for different parameter names between chat completions and responses APIs
            self._normalize_to_responses_kwargs()
            return self._responses(messages)
    
    def _chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make a call to the OpenAI Chat Completions API
        """
        return self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            **self.model_config.kwargs
        )
    
    def _responses(self, messages: List[Dict[str, str]]) -> Any:
        """
        Make a call to the OpenAI Responses API
        """
        return self.client.responses.create(
            model=self.model_config.model_name,
            input=messages,
            **self.model_config.kwargs
        )

    @abc.abstractmethod
    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        """
        Extract JSON from the provider's response.
        Subclasses must implement this based on expected response format.
        """
        pass
        
    def _get_usage(self, response: Any) -> Usage:
        """Extract usage information from a standard OpenAI-like response object."""
        # Implementation copied from OpenAIAdapter
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0 # Default - subclasses can override if needed

        if hasattr(response, 'usage') and response.usage:
            if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                total_tokens = getattr(response.usage, 'total_tokens', 0)
                # Safely access potential reasoning tokens (still needs verification based on actual Grok/updated OpenAI responses)
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
            
            else: # APIType.RESPONSES (Assume this structure if not CHAT_COMPLETIONS)
                prompt_tokens = getattr(response.usage, 'input_tokens', 0)
                completion_tokens = getattr(response.usage, 'output_tokens', 0)
                total_tokens = prompt_tokens + completion_tokens # Responses API doesn't always return total, calculate here
                # Safely access potential reasoning tokens
                if hasattr(response.usage, 'output_tokens_details') and response.usage.output_tokens_details and hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens or 0

            # ---- Infer reasoning tokens when provider does not break them out ----
            # Note: This inference step is done *before* the main cost calculation logic below, 
            # providing the necessary reasoning_tokens value for that logic.
            if total_tokens and (prompt_tokens + completion_tokens) < total_tokens and reasoning_tokens == 0:
                # Provider counted extra tokens that must correspond to reasoning
                reasoning_tokens = total_tokens - (prompt_tokens + completion_tokens)
            # If provider's explicit reasoning makes the sum exceed total, we keep as-is; the mismatch will be caught later.

        else:
            # Handle cases where usage might be missing (should log this appropriately)
            print(f"Warning: Usage information missing or incomplete in response for model {self.model_config.model_name}") 
            # Attempt basic calculation if possible (e.g., if we have token counts elsewhere)
            # For now, just return zeros or defaults
            pass # Keep defaults

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
                # Assume all completion tokens are accepted/rejected for now unless overridden
                accepted_prediction_tokens=completion_tokens,
                rejected_prediction_tokens=0,
            ),
        )

    def _get_reasoning_summary(self, response: Any) -> Optional[str]:
        """Extract reasoning summary from the response if available (primarily for Responses API)."""
        reasoning_summary = None
        if self.model_config.api_type == APIType.RESPONSES:
            # Safely access potential reasoning summary
            if hasattr(response, 'reasoning') and response.reasoning and hasattr(response.reasoning, 'summary'):
                reasoning_summary = response.reasoning.summary # Will be None if not present
        # Chat Completions API does not currently provide a separate summary field
        return reasoning_summary

    def _get_content(self, response: Any) -> str:
        """Extract content from a standard OpenAI-like response object."""
        # Implementation copied from OpenAIAdapter
        content = ""
        if self.model_config.api_type == APIType.CHAT_COMPLETIONS:
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
                content = getattr(response.choices[0].message, 'content', "") or ""
        else: # APIType.RESPONSES
            # Check standard attribute first
            content = getattr(response, 'output_text', "")
            # Fallback: Sometimes it might be in a 'choices' structure even for non-chat
            if not content and hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'text'):
                 content = getattr(response.choices[0], 'text', "") or ""
        return content.strip()

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
        rt_explicit = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

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
            from src.errors import TokenMismatchError # Local import
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

        from src.schemas import Cost  # Local import (avoids circular issues in some environments)
        return Cost(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost, # Cost of 'actual' completion
            reasoning_cost=reasoning_cost,   # Cost of reasoning part
            total_cost=total_cost,           # True total expenditure
        ) 