from .provider import ProviderAdapter
import os
import json
from openai import OpenAI
from datetime import datetime, timezone
from arc_agi_benchmarking.schemas import APIType, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import logging
from typing import Optional, Any, List, Dict
import re

# Import the base class
from .openai_base import OpenAIBaseAdapter

logger = logging.getLogger(__name__)

class GrokAdapter(OpenAIBaseAdapter):
    """Adapter specific to Grok API endpoints and response structures."""

    def init_client(self):
        """
        Initialize the OpenAI client configured for Grok API.
        """
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in environment variables")
        
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        return client

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction using the Grok model.
        Relies on OpenAIBaseAdapter for API calls and standard parsing.
        Specific reasoning token handling might require overriding _get_usage later.
        """
        start_time = datetime.now(timezone.utc)
        
        # Use the inherited call_ai_model
        # Assumes Grok uses CHAT_COMPLETIONS or RESPONSES API type defined in config
        response = self._call_ai_model(prompt)
        
        end_time = datetime.now(timezone.utc)

        # Centralised cost calculation (includes sanity-check & calls _get_usage internally)
        cost = self._calculate_cost(response)
        
        # Retrieve usage *after* cost calculation, as cost calc might infer/update reasoning tokens
        usage = self._get_usage(response)
        
        prompt_cost = usage.prompt_tokens * self.model_config.pricing.input / 1_000_000
        completion_cost = usage.completion_tokens * self.model_config.pricing.output / 1_000_000

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=0,
                message=Message(role="user", content=prompt)
            )
        ]

        # Convert Grok response (assumed OpenAI-compatible) using inherited helpers
        response_choices = [
            Choice(
                index=1,
                message=Message(
                    role=self._get_role(response),       # Inherited
                    content=self._get_content(response)  # Inherited
                )
            )
        ]

        all_choices = input_choices + response_choices

        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            usage=usage,
            cost=cost,
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id
        )

        attempt = Attempt(
            metadata=metadata,
            answer=self._get_content(response) # Inherited
        )

        return attempt

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        """Placeholder for Grok-specific JSON extraction. Assumes standard OpenAI format for now."""

        prompt = f"""
You are a helpful assistant. Extract only the JSON of the test output from the following response. 
Do not include any explanation or additional text; only return valid JSON.

Response:
{input_response}

The JSON should be in this format:
{{
"response": [
    [1, 2, 3],
    [4, 5, 6]
]
}}
"""
        try:
            # Use inherited chat_completion or call_ai_model
            completion = self._chat_completion(
                messages=[{"role": "user", "content": prompt}],
            )
            assistant_content = self._get_content(completion) # Inherited
        except Exception as e:
            print(f"Error during AI-based JSON extraction via Grok: {e}")
            assistant_content = input_response

        # Parsing logic adapted from Deepseek/Fireworks
        assistant_content = assistant_content.strip()
        if assistant_content.startswith("```json"):
            assistant_content = "\n".join(assistant_content.split("\n")[1:])
        if assistant_content.endswith("```"):
            assistant_content = "\n".join(assistant_content.split("\n")[:-1])
        assistant_content = assistant_content.strip()

        try:
            json_entities = json.loads(assistant_content)
            potential_list = json_entities.get("response")
            if isinstance(potential_list, list) and all(isinstance(item, list) for item in potential_list):
                 if all(isinstance(num, int) for sublist in potential_list for num in sublist):
                     return potential_list
            return None
        except (json.JSONDecodeError, AttributeError):
            return None
