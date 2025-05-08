from .provider import ProviderAdapter
from arc_agi_benchmarking.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import anthropic
import os
from dotenv import load_dotenv
import json
from typing import List, Optional
from datetime import datetime, timezone
load_dotenv()

class AnthropicAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the Anthropic model
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        return client
    
    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Anthropic model and return an Attempt object
        
        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
        """
        start_time = datetime.now(timezone.utc)
        
        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(messages)
        end_time = datetime.now(timezone.utc)

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000  # Convert from per 1M tokens
        output_cost_per_token = self.model_config.pricing.output / 1_000_000  # Convert from per 1M tokens
        
        prompt_cost = response.usage.input_tokens * input_cost_per_token
        completion_cost = response.usage.output_tokens * output_cost_per_token

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

        # Convert Anthropic response to our schema
        response_choices = [
            Choice(
                index=len(input_choices),
                message=Message(
                    role="assistant",
                    content=content.text if content.type == "text" else json.dumps(content.input)
                )
            )
            for content in response.content
            if content.type in ["text", "tool_use"]
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
            kwargs=self.model_config.kwargs,  # Use kwargs from model config
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,  # Anthropic doesn't provide this breakdown
                    accepted_prediction_tokens=response.usage.output_tokens,
                    rejected_prediction_tokens=0  # Anthropic doesn't provide this
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

        # Incase there is a thinking block
        answer = ""
        for content in response.content:
            if content.type == "text":
                answer = content.text
                break

        attempt = Attempt(
            metadata=metadata,
            answer=answer
        )

        return attempt

    def chat_completion(self, messages, tools=[]):
        """
        Make a raw API call to Anthropic and return the response
        """
        
        return self.client.messages.create(
            model=self.model_config.model_name,
            messages=messages,
            tools=tools,
            **self.model_config.kwargs
        )
    
    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        tools = [
            {
                "name": "extract_json",
                "description": "Extracts JSON from the response.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                }
                            },
                            "description": "A list of lists of integers extracted from the response."
                        }
                    },
                    "required": ["response"]
                }
            }
        ]

        text = f"Extract JSON of the test output from the following response: {input_response}"

        query = f"""
        <document>
        {text}
        </document>

        Use the extract_json tool.
        """

        response = self.chat_completion(
            messages=[{"role": "user", "content": query}],
            tools=tools
        )

        json_response = None
        for content in response.content:
            if content.type == "tool_use" and content.name == "extract_json":
                json_entities = content.input
                break

        if json_entities:
            return json_entities['response']
        else:
            return None
        
if __name__ == "__main__":
    adapter = AnthropicAdapter("claude-3-5-sonnet-20240620")
    print(type(adapter.extract_json_from_response("[[1, 2, 3], [4, 5, 6]]")))