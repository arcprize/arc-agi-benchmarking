from .provider import ProviderAdapter
from arc_agi_benchmarking.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import os
import json
import uuid
import time
from typing import List, Optional, Any
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

# Poll batch status every 10 seconds (matches OpenAI background mode polling cadence)
_BATCH_POLL_INTERVAL_SECONDS = 10

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

        # Check if streaming / batch mode are enabled
        stream_enabled = self.model_config.kwargs.get('stream', False) or getattr(self.model_config, 'stream', False)
        batch_enabled = self.model_config.kwargs.get('batch', False) or getattr(self.model_config, 'batch', False)

        if batch_enabled and stream_enabled:
            raise ValueError("Cannot enable both 'stream' and 'batch' for Anthropic. Batch requests do not support streaming.")

        if batch_enabled:
            response = self.chat_completion_batch(messages)
        elif stream_enabled:
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
        betas = self.model_config.kwargs.get('betas')
        api_kwargs = {k: v for k, v in self.model_config.kwargs.items() if k != 'betas'}
        if betas:
            return self.client.beta.messages.create(
                model=self.model_config.model_name,
                betas=betas,
                messages=messages,
                tools=tools,
                **api_kwargs
            )
        return self.client.messages.create(
            model=self.model_config.model_name,
            messages=messages,
            tools=tools,
            **api_kwargs
        )

    def chat_completion_stream(self, messages, tools=[]):
        """
        Make a streaming API call to Anthropic and return the final complete response.
        Only the final message is returned; intermediate deltas are ignored.
        """
        logger.debug(f"Starting streaming for Anthropic model: {self.model_config.model_name}")

        # Prepare kwargs for streaming, removing 'stream' and 'betas'
        betas = self.model_config.kwargs.get('betas')
        stream_kwargs = {k: v for k, v in self.model_config.kwargs.items() if k not in ('stream', 'betas')}

        try:
            if betas:
                with self.client.beta.messages.stream(
                    model=self.model_config.model_name,
                    betas=betas,
                    messages=messages,
                    tools=tools,
                    **stream_kwargs
                ) as stream:
                    final_message = stream.get_final_message()
            else:
                with self.client.messages.stream(
                    model=self.model_config.model_name,
                    messages=messages,
                    tools=tools,
                    **stream_kwargs
                ) as stream:
                    final_message = stream.get_final_message()

            logger.debug(f"Streaming complete for message ID: {final_message.id}")
            logger.debug(f"Final message: {final_message}")
            return final_message

        except Exception as e:
            logger.error(f"Error during Anthropic streaming: {e}")
            logger.error(f"Error details: {e.response}")
            raise

    def chat_completion_batch(self, messages, tools=[]):
        """
        Submit a single Messages request via Anthropic's Message Batches API,
        poll until processing has ended, fetch the result, and delete the batch.

        Mirrors OpenAI's background-mode pattern: the batch is always deleted
        afterwards so the request doesn't sit in Anthropic's batch storage.
        Returns the Anthropic Message object so the caller can treat it the
        same as a synchronous Messages API response.
        """
        # Build params for the single batched request. Strip flags that aren't
        # valid inside batch params: our internal 'batch' flag, 'stream' (not
        # supported in batches), and 'betas' (passed at the batch level below).
        betas = self.model_config.kwargs.get('betas')
        batch_kwargs = {
            k: v for k, v in self.model_config.kwargs.items()
            if k not in ('batch', 'stream', 'betas')
        }

        # `extra_body` is an SDK-level escape hatch the synchronous Messages
        # SDK merges into the final HTTP body. In batch mode, per-request
        # `params` are sent verbatim (no SDK merging step), so a literal
        # `extra_body` key would be rejected as "Extra inputs are not
        # permitted". Mirror the sync behavior here by inlining extra_body's
        # contents directly into the per-request params.
        extra_body = batch_kwargs.pop('extra_body', None) or {}

        custom_id = f"arc-{uuid.uuid4().hex[:16]}"

        params: dict = MessageCreateParamsNonStreaming(
            model=self.model_config.model_name,
            messages=messages,
            tools=tools,
            **batch_kwargs,
        )
        # Inline extra_body fields (e.g. output_config, context_management)
        # into the top level of params, matching how the sync SDK serializes them.
        for k, v in extra_body.items():
            params[k] = v

        request = Request(custom_id=custom_id, params=params)

        # Route through beta batches endpoint if a betas list is configured
        batches_api = self.client.beta.messages.batches if betas else self.client.messages.batches

        if betas:
            batch = batches_api.create(requests=[request], betas=betas)
        else:
            batch = batches_api.create(requests=[request])

        batch_id = batch.id
        logger.debug(f"Created Anthropic batch {batch_id} with custom_id={custom_id}")

        try:
            # Poll until processing ends. No hard cap: matches OpenAI background
            # mode, which also loops until the response leaves queued/in_progress.
            while batch.processing_status != "ended":
                time.sleep(_BATCH_POLL_INTERVAL_SECONDS)
                batch = batches_api.retrieve(batch_id)
                logger.debug(
                    f"Anthropic batch {batch_id} status={batch.processing_status} "
                    f"counts={batch.request_counts}"
                )

            # Stream results back; we only submitted one request so we expect one entry.
            matched = None
            for result in batches_api.results(batch_id):
                if result.custom_id == custom_id:
                    matched = result
                    break

            if matched is None:
                raise RuntimeError(
                    f"Anthropic batch {batch_id} returned no result for custom_id={custom_id}"
                )

            result_type = matched.result.type
            if result_type == "succeeded":
                return matched.result.message
            if result_type == "errored":
                err = getattr(matched.result, "error", None)
                raise RuntimeError(
                    f"Anthropic batch request {custom_id} errored: {err}"
                )
            if result_type == "canceled":
                raise RuntimeError(f"Anthropic batch request {custom_id} was canceled")
            if result_type == "expired":
                raise RuntimeError(f"Anthropic batch request {custom_id} expired")
            raise RuntimeError(
                f"Anthropic batch request {custom_id} returned unknown result type: {result_type}"
            )
        finally:
            # Always delete the batch from Anthropic's storage, regardless of
            # success or failure above. Failures here are logged but do not
            # override the original exception (if any).
            try:
                batches_api.delete(batch_id)
                logger.debug(f"Deleted Anthropic batch {batch_id}")
            except Exception as delete_err:
                logger.warning(f"Failed to delete Anthropic batch {batch_id}: {delete_err}")

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