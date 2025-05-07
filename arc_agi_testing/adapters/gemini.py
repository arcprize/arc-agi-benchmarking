from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
import google.generativeai as genai
from typing import List, Optional
from datetime import datetime, timezone
from arc_agi_testing.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the Gemini model
        """
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        
        # Use kwargs directly as generation config
        self.generation_config = self.model_config.kwargs
        
        return genai.GenerativeModel(self.model_config.model_name)

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Gemini model and return an Attempt object
        
        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
        """
        start_time = datetime.now(timezone.utc)
        
        # Get input token count before making the request
        input_tokens = self.client.count_tokens(prompt)
        logger.debug(f"Input tokens count: {input_tokens}")
        
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages)
        
        end_time = datetime.now(timezone.utc)

        # Get token counts from response metadata
        usage_metadata = response.usage_metadata
        logger.debug(f"Response usage metadata: {usage_metadata}")
        
        input_tokens = usage_metadata.prompt_token_count
        output_tokens = usage_metadata.candidates_token_count
        total_tokens = usage_metadata.total_token_count

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000  # Convert from per 1M tokens
        output_cost_per_token = self.model_config.pricing.output / 1_000_000  # Convert from per 1M tokens
        
        prompt_cost = input_tokens * input_cost_per_token
        completion_cost = output_tokens * output_cost_per_token

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=i,
                message=Message(
                    role=msg["role"],
                    content=msg["content"]
                )
            )
            for i, msg in enumerate(messages)
        ]

        # Convert Gemini response to our schema
        response_choices = [
            Choice(
                index=len(input_choices),
                message=Message(
                    role="assistant",
                    content=response.text
                )
            )
        ]

        # Combine input and response choices
        all_choices = input_choices + response_choices

        # Create metadata using our Pydantic models
        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,  # Gemini doesn't provide this breakdown
                    accepted_prediction_tokens=output_tokens,
                    rejected_prediction_tokens=0  # Gemini doesn't provide this
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            ),
            task_id=task_id,  # Add task_id to metadata
            pair_index=pair_index,  # Add pair_index to metadata
            test_id=test_id  # Add test_id to metadata
        )

        attempt = Attempt(
            metadata=metadata,
            answer=response.text
        )

        return attempt

    def chat_completion(self, messages: list) -> str:
        # Convert to Gemini's message format
        history = [{"parts": [msg["content"]], "role": msg["role"]} for msg in messages]
        
        response = self.client.generate_content(
            contents=history,
            generation_config=self.generation_config,
            stream=False
        )
        return response

    def extract_json_from_response(self, input_response: str) -> Optional[List[List[int]]]:
        prompt = f"""
        Extract only the JSON of the test output from the following response.
        Remove any markdown code blocks and return only valid JSON.

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

        response = self.client.generate_content(prompt)
        content = response.text.strip()

        # Handle possible code block formatting
        if content.startswith("```json"):
            content = content[7:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        try:
            json_data = json.loads(content)
            return json_data.get("response")
        except json.JSONDecodeError:
            return None