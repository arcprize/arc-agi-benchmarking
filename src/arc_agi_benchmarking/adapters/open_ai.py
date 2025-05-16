from .provider import ProviderAdapter
from .openai_base import OpenAIBaseAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from datetime import datetime, timezone
from arc_agi_benchmarking.schemas import APIType, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
from typing import Optional, Any, List, Dict

import re

load_dotenv()


class OpenAIAdapter(OpenAIBaseAdapter):


    def init_client(self):
        """
        Initialize the OpenAI client
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        client = OpenAI()
        return client


    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the OpenAI model and return an Attempt object
        
        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
        """
        start_time = datetime.now(timezone.utc)
        

        response = self._call_ai_model(prompt)
        
        end_time = datetime.now(timezone.utc)

        # Centralised usage & cost calculation (includes sanity-check)
        cost = self._calculate_cost(response)

        # Retrieve usage *after* cost calculation, as cost calc might infer reasoning tokens
        usage = self._get_usage(response)

        # Get reasoning summary (will be None if not available or not Responses API)
        reasoning_summary = self._get_reasoning_summary(response)

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=0,
                message=Message(
                    role="user",
                    content=prompt
                )
            )
        ]

        # Convert OpenAI response to our schema
        response_choices = [
            Choice(
                index=1,
                message=Message(
                    role=self._get_role(response),
                    content=self._get_content(response)
                )
            )
        ]

        # Combine input and response choices
        all_choices = input_choices + response_choices

        # Create metadata
        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            reasoning_summary=reasoning_summary,
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

        return attempt

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
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
        completion = self._call_ai_model(
            prompt=prompt
        )

        assistant_content = self._get_content(completion)

        # Try to extract JSON from various formats
        # Remove markdown code blocks if present
        if "```" in assistant_content:
            # Extract content between code blocks
            code_blocks = assistant_content.split("```")
            for block in code_blocks:
                if block.strip() and not block.strip().startswith("json"):
                    assistant_content = block.strip()
                    break
        
        # Remove any leading/trailing text that's not part of the JSON
        assistant_content = assistant_content.strip()
        
        # Try to find array start/end if there's surrounding text
        if assistant_content and not assistant_content.startswith("["):
            start_idx = assistant_content.find("[[")
            if start_idx >= 0:
                end_idx = assistant_content.rfind("]]") + 2
                if end_idx > start_idx:
                    assistant_content = assistant_content[start_idx:end_idx]

        try:
            # Try direct parsing first
            json_result = json.loads(assistant_content)
            if isinstance(json_result, list) and all(isinstance(item, list) for item in json_result):
                return json_result
            
            # If we got a dict with a response key, use that
            if isinstance(json_result, dict) and "response" in json_result:
                return json_result.get("response")
                
            return None
        except json.JSONDecodeError:
            # If direct parsing fails, try to find and extract just the array part
            try:
                # Look for array pattern and extract it
                array_pattern = r'\[\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\s*\]'
                match = re.search(array_pattern, assistant_content)
                if match:
                    return json.loads(match.group(0))
            except:
                pass
            
            return None