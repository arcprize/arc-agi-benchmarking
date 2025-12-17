from .provider import ProviderAdapter
from arc_agi_benchmarking.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import anthropic
import os
import re
from dotenv import load_dotenv
import json
from typing import List, Optional, Any
from datetime import datetime, timezone
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class AnthropicAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the Anthropic model.
        
        Supports two modes:
        1. Azure: Set AZURE_ANTHROPIC_API_KEY and AZURE_ANTHROPIC_ENDPOINT
        2. Direct Anthropic: Set ANTHROPIC_API_KEY
        """
        # Default to non-Azure mode
        self._use_azure = False
        
        azure_api_key = os.environ.get("AZURE_ANTHROPIC_API_KEY")
        azure_endpoint = os.environ.get("AZURE_ANTHROPIC_ENDPOINT")
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        if azure_api_key:
            if not azure_endpoint:
                raise ValueError(
                    "AZURE_ANTHROPIC_ENDPOINT is required when using AZURE_ANTHROPIC_API_KEY"
                )
            # Use Azure endpoint
            self._use_azure = True
            client = anthropic.Anthropic(
                api_key=azure_api_key,
                base_url=azure_endpoint,
            )
        elif anthropic_api_key:
            # Use direct Anthropic API
            self._use_azure = False
            client = anthropic.Anthropic(
                api_key=anthropic_api_key,
            )
        else:
            raise ValueError(
                "No API key found. Set either AZURE_ANTHROPIC_API_KEY and AZURE_ANTHROPIC_ENDPOINT "
                "(for Azure) or ANTHROPIC_API_KEY (for direct Anthropic API)"
            )

        return client
    
    def _get_model_name(self) -> str:
        """
        Get the model name to use for API calls.
        Azure deployments don't use date suffixes, so strip them for Azure.
        e.g., 'claude-opus-4-5-20251101-thinking-none' -> 'claude-opus-4-5' for Azure
        """
        model_name = self.model_config.model_name
        if getattr(self, '_use_azure', False):
            # Strip date suffix (format: -YYYYMMDD) and anything after it
            model_name = re.sub(r'-\d{8}.*$', '', model_name)
        return model_name
    
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

        # Check if streaming is enabled
        stream_enabled = self.model_config.kwargs.get('stream', False) or getattr(self.model_config, 'stream', False)

        if stream_enabled:
            response = self.chat_completion_stream(messages)
        else:
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

        # Thinking blocks from Anthropic
        reasoning_summary = self._get_reasoning_summary(response)

        # Create metadata using our Pydantic models
        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,  # Use kwargs from model config
            reasoning_summary=reasoning_summary,
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
            model=self._get_model_name(),
            messages=messages,
            tools=tools,
            **self.model_config.kwargs
        )

    def chat_completion_stream(self, messages, tools=[]):
        """
        Make a streaming API call to Anthropic and return the final complete response.
        Only the final message is returned; intermediate deltas are ignored.
        """
        logger.debug(f"Starting streaming for Anthropic model: {self._get_model_name()}")

        # Prepare kwargs for streaming, removing 'stream' to avoid duplication
        stream_kwargs = {k: v for k, v in self.model_config.kwargs.items() if k != 'stream'}

        try:
            # Create the stream
            with self.client.messages.stream(
                model=self._get_model_name(),
                messages=messages,
                tools=tools,
                **stream_kwargs
            ) as stream:
                # Accumulate the complete message
                # The stream context manager handles all the event processing
                # and gives us the final message when done
                final_message = stream.get_final_message()

            logger.debug(f"Streaming complete for message ID: {final_message.id}")
            return final_message

        except Exception as e:
            logger.error(f"Error during Anthropic streaming: {e}")
            raise

    def _get_reasoning_summary(self, response: Any) -> str:
        """Get the reasoning summary from the response."""
        reasoning_summary = None
        thinking_texts: List[str] = []
        try:
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "thinking" and hasattr(block, 'thinking'):
                        if isinstance(block.thinking, str): # Ensure it's a string
                            thinking_texts.append(block.thinking)
            if thinking_texts:
                reasoning_summary = "\n\n".join(thinking_texts)
        except Exception as e:
            logger.warning(f"Error extracting thinking blocks from Anthropic response: {e}", exc_info=True)
        return reasoning_summary

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