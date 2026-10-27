from .provider import ProviderAdapter
from arc_agi_benchmarking.schemas import AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import anthropic
import os
from dotenv import load_dotenv
import json
from typing import List, Optional, Any
from datetime import datetime, timezone
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Timeout for Azure API calls (30 minutes)
AZURE_TIMEOUT = 1800

class AzureAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the Azure client for Anthropic models.
        Azure acts as a hosting platform for Claude models.
        """
        return self._init_anthropic_client()
    
    def _init_anthropic_client(self):
        """Initialize Anthropic client for Claude models hosted on Azure"""
        if not os.environ.get("AZURE_ANTHROPIC_API_KEY"):
            raise ValueError("AZURE_ANTHROPIC_API_KEY not found in environment variables")
        
        if not os.environ.get("AZURE_ANTHROPIC_ENDPOINT"):
            raise ValueError("AZURE_ANTHROPIC_ENDPOINT not found in environment variables")
        
        client = anthropic.Anthropic(
            api_key=os.environ.get("AZURE_ANTHROPIC_API_KEY"),
            base_url=os.environ.get("AZURE_ANTHROPIC_ENDPOINT"),
            timeout=AZURE_TIMEOUT,
        )
        
        return client
    
    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Azure-hosted Anthropic model and return an Attempt object.

        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
            pair_index: Optional pair index to include in metadata
        """
        return self._make_prediction_anthropic(prompt, task_id, test_id, pair_index)
    
    def _make_prediction_anthropic(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """Make prediction using Anthropic format (for Claude models on Azure)"""
        start_time = datetime.now(timezone.utc)

        messages = [
            {"role": "user", "content": prompt}
        ]

        # Check if streaming is enabled
        stream_enabled = self.model_config.kwargs.get('stream', False) or getattr(self.model_config, 'stream', False)

        if stream_enabled:
            response = self._chat_completion_stream_anthropic(messages)
        else:
            response = self._chat_completion_anthropic(messages)

        end_time = datetime.now(timezone.utc)

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000
        
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

        all_choices = input_choices + response_choices
        reasoning_summary = self._get_reasoning_summary_anthropic(response)

        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            reasoning_summary=reasoning_summary,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,
                    accepted_prediction_tokens=response.usage.output_tokens,
                    rejected_prediction_tokens=0
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

        # Extract answer from thinking block if present
        answer = ""
        for content in response.content:
            if content.type == "text":
                answer = content.text
                break

        return Attempt(metadata=metadata, answer=answer)

    def chat_completion(self, messages, tools=[]):
        """Make a raw API call using Anthropic format"""
        return self._chat_completion_anthropic(messages, tools)
    
    def _chat_completion_anthropic(self, messages, tools=[]):
        """Make API call using Anthropic format"""
        return self.client.messages.create(
            model=self.model_config.model_name,
            messages=messages,
            tools=tools,
            **self.model_config.kwargs
        )
    
    def chat_completion_stream(self, messages, tools=[]):
        """Make a streaming API call using Anthropic format"""
        return self._chat_completion_stream_anthropic(messages, tools)
    
    def _chat_completion_stream_anthropic(self, messages, tools=[]):
        """Streaming API call using Anthropic format"""
        logger.debug(f"Starting streaming for Anthropic model on Azure: {self.model_config.model_name}")

        stream_kwargs = {k: v for k, v in self.model_config.kwargs.items() if k != 'stream'}

        try:
            with self.client.messages.stream(
                model=self.model_config.model_name,
                messages=messages,
                tools=tools,
                **stream_kwargs
            ) as stream:
                final_message = stream.get_final_message()

            logger.debug(f"Streaming complete for message ID: {final_message.id}")
            return final_message

        except Exception as e:
            logger.error(f"Error during Anthropic streaming on Azure: {e}")
            raise
    
    def _get_reasoning_summary_anthropic(self, response: Any) -> str:
        """Get reasoning summary from Anthropic response"""
        reasoning_summary = None
        thinking_texts: List[str] = []
        try:
            if hasattr(response, 'content') and response.content:
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "thinking" and hasattr(block, 'thinking'):
                        if isinstance(block.thinking, str):
                            thinking_texts.append(block.thinking)
            if thinking_texts:
                reasoning_summary = "\n\n".join(thinking_texts)
        except Exception as e:
            logger.warning(f"Error extracting thinking blocks from Anthropic response: {e}", exc_info=True)
        return reasoning_summary
    
    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """Extract JSON using Anthropic tool calling format"""
        return self._extract_json_anthropic(input_response)
    
    def _extract_json_anthropic(self, input_response: str) -> List[List[int]]:
        """Extract JSON using Anthropic tool calling format"""
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

        response = self._chat_completion_anthropic(
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
