from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from datetime import datetime, timezone
from arc_agi_testing.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import logging
from typing import Optional, Any, List, Dict
from .openai_base import OpenAIBaseAdapter
import re

load_dotenv()
logger = logging.getLogger(__name__)

class DeepseekAdapter(OpenAIBaseAdapter): # Inherit from OpenAIBaseAdapter
    """Adapter specific to Deepseek API endpoints and response structures."""

    def init_client(self):
        """
        Initialize the OpenAI client configured for Deepseek.
        """
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return client

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction using the Deepseek model.
        Relies on OpenAIBaseAdapter for API calls and standard parsing.
        """
        start_time = datetime.now(timezone.utc)
        
        # Use the inherited call_ai_model
        response = self._call_ai_model(prompt)
        
        end_time = datetime.now(timezone.utc)

        # Centralised cost calculation (includes sanity-check & calls _get_usage internally)
        cost = self._calculate_cost(response)

        # Retrieve usage *after* cost calculation, as cost calc might infer/update reasoning tokens
        usage = self._get_usage(response)

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000
        
        prompt_cost = usage.prompt_tokens * input_cost_per_token
        completion_cost = usage.completion_tokens * output_cost_per_token

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=0,
                message=Message(role="user", content=prompt)
            )
        ]

        # Convert Deepseek response (assumed OpenAI-compatible) using inherited helpers
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
        """Extract JSON using Deepseek-specific prompting and parsing (Original Implementation)."""
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
            completion = self._chat_completion(
                messages=[{"role": "user", "content": prompt}],
            )
            # Use inherited _get_content
            assistant_content = self._get_content(completion)
        except Exception as e:
             print(f"Error during AI-based JSON extraction via Deepseek: {e}")
             # Fallback: Try to parse the original input_response directly if AI fails
             assistant_content = input_response

        assistant_content = assistant_content.strip()
        # Some models like to wrap the response in a code block
        if assistant_content.startswith("```json"):
            assistant_content = "\n".join(assistant_content.split("\n")[1:])
        
        if assistant_content.endswith("```"):
            assistant_content = "\n".join(assistant_content.split("\n")[:-1])
        
        assistant_content = assistant_content.strip() # Strip again after potential ``` removal

        try:
            json_entities = json.loads(assistant_content)
            # Original logic specifically looked for the 'response' key
            potential_list = json_entities.get("response")
            if isinstance(potential_list, list) and all(isinstance(item, list) for item in potential_list):
                 # Optional: Add validation for inner list types if needed
                 if all(isinstance(num, int) for sublist in potential_list for num in sublist):
                     return potential_list
            return None # Return None if 'response' key doesn't contain the expected list of lists
        except (json.JSONDecodeError, AttributeError):
             # Catch potential errors if parsing fails or .get returns None
            return None

