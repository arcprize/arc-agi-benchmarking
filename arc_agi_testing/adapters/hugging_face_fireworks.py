from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from huggingface_hub import InferenceClient
from datetime import datetime
from arc_agi_testing.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import logging
from typing import Optional

load_dotenv()
logger = logging.getLogger(__name__)

class HuggingFaceFireworksAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the Hugging Face Fireworks model
        """
        if not os.environ.get("HUGGING_FACE_API_KEY"):
            raise ValueError("HUGGING_FACE_API_KEY not found in environment variables")
        
        client = InferenceClient(
            provider="fireworks-ai",
            api_key=os.environ.get("HUGGING_FACE_API_KEY")
        )
        return client

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Hugging Face Fireworks model and return an Attempt object
        """
        start_time = datetime.utcnow()
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.chat_completion(messages)
        
        end_time = datetime.utcnow()

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000  # Convert from per 1M tokens
        output_cost_per_token = self.model_config.pricing.output / 1_000_000  # Convert from per 1M tokens
        
        prompt_cost = response.usage.prompt_tokens * input_cost_per_token
        completion_cost = response.usage.completion_tokens * output_cost_per_token

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

        # Convert Hugging Face Fireworks response to our schema
        response_choices = [
            Choice(
                index=len(input_choices),
                message=Message(
                    role=response.choices[0].message.role,
                    content=response.choices[0].message.content.strip()
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
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,  # Hugging Face Fireworks doesn't provide this breakdown
                    accepted_prediction_tokens=response.usage.completion_tokens,
                    rejected_prediction_tokens=0  # Hugging Face Fireworks doesn't provide this
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            ),
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id
        )

        print(f"Response (response.choices[0].message): {response.choices[0].message.content.strip()}")
        attempt = Attempt(
            metadata=metadata,
            answer=response.choices[0].message.content.strip()
        )

        return attempt

    def chat_completion(self, messages: str) -> str:
        return self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            **self.model_config.kwargs
        )

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
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
        completion = self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )

        assistant_content = completion.choices[0].message.content.strip()

        # Some models like to wrap the response in a code block
        if assistant_content.startswith("```json"):
            assistant_content = "\n".join(assistant_content.split("\n")[1:])
        
        if assistant_content.endswith("```"):
            assistant_content = "\n".join(assistant_content.split("\n")[:-1])

        try:
            json_entities = json.loads(assistant_content)
            return json_entities.get("response")
        except json.JSONDecodeError:
            return None