from .provider import ProviderAdapter
import os
import json
from openai import OpenAI
from datetime import datetime, timezone
from arc_agi_benchmarking.schemas import APIType, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import logging
from typing import Optional, Any, List, Dict
import re

# Import the base class we will now inherit from
from .openai_base import OpenAIBaseAdapter

logger = logging.getLogger(__name__)

class GroqAdapter(OpenAIBaseAdapter): # Inherit from OpenAIBaseAdapter
    def init_client(self):
        """
        Initialize the Groq client using GROQ_API_KEY and hardcoded base URL.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        return client

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction using the Groq model.
        Relies on OpenAIBaseAdapter for API calls and standard parsing.
        """
        start_time = datetime.now(timezone.utc)
        
        response = self._call_ai_model(prompt)
        print (f"Response: {response}")
        
        end_time = datetime.now(timezone.utc)

        # Centralised cost calculation (includes sanity-check & calls _get_usage internally)
        cost = self._calculate_cost(response)

        # Retrieve usage *after* cost calculation, as cost calc might infer/update reasoning tokens
        usage = self._get_usage(response)
        
        input_choices = [
            Choice(
                index=0,
                message=Message(role="user", content=prompt)
            )
        ]

        response_choices = [
            Choice(
                index=1,
                message=Message(
                    role=self._get_role(response),
                    content=self._get_content(response)
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
            answer=self._get_content(response)
        )

        print (f"Attempt: {attempt}")
        return attempt

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        """Extract JSON specifically for Groq's potential response formats."""
        prompt = f"""
You are a helpful assistant. Extract only the JSON array of arrays from the following response. 
Do not include any explanation, formatting, or additional text.
Return ONLY the valid JSON array of arrays with integers.

Response:
{input_response}

Example of expected output format:
[[1, 2, 3], [4, 5, 6]]

IMPORTANT: Return ONLY the array, with no additional text, quotes, or formatting.
"""
        try:
            completion = self._call_ai_model(prompt=prompt) 
            assistant_content = self._get_content(completion) 
        except Exception as e:
            print(f"Error during AI-based JSON extraction via Groq: {e}")
            assistant_content = input_response
        
        assistant_content = assistant_content.strip()
        if assistant_content.startswith("```json"):
            assistant_content = assistant_content[7:]
        if assistant_content.startswith("```"):
            assistant_content = assistant_content[3:]
        if assistant_content.endswith("```"):
            assistant_content = assistant_content[:-3]
        assistant_content = assistant_content.strip()

        try:
            json_result = json.loads(assistant_content)
            if isinstance(json_result, list) and all(isinstance(item, list) for item in json_result):
                return json_result
            if isinstance(json_result, dict) and "response" in json_result:
                 json_response = json_result.get("response")
                 if isinstance(json_response, list) and all(isinstance(item, list) for item in json_response):
                    return json_response
            return None
        except json.JSONDecodeError:
             try:
                array_pattern = r'\[\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\s*\]'
                match = re.search(array_pattern, assistant_content)
                if match:
                    parsed_match = json.loads(match.group(0))
                    if isinstance(parsed_match, list) and all(isinstance(item, list) for item in parsed_match):
                        return parsed_match
             except:
                 pass
             return None