from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from google import genai
from google.genai import types
from typing import List, Optional
from datetime import datetime, timezone
from arc_agi_benchmarking.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class _StreamResponse:
    """Wrapper for streaming response to hold accumulated text and metadata."""
    def __init__(self, text: str, usage_metadata):
        self.text = text
        self.usage_metadata = usage_metadata

class GeminiAdapter(ProviderAdapter):
    def init_client(self):
        """Initialize the Gemini client."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.generation_config_dict = self.model_config.kwargs
        
        client = genai.Client(api_key=api_key)
        return client

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Gemini model and return an Attempt object.

        Args:
            prompt: The prompt to send to the model.
            task_id: Optional task ID.
            test_id: Optional test ID.
            pair_index: Optional index for paired data.
        """
        start_time = datetime.now(timezone.utc)

        messages = [{"role": "user", "content": prompt}]

        # Check if streaming is enabled
        stream_enabled = self.model_config.kwargs.get('stream', False) or getattr(self.model_config, 'stream', False)

        if stream_enabled:
            response = self.chat_completion_stream(messages)
        else:
            response = self.chat_completion(messages)
        
        if response is None:
            logger.error(f"Failed to get response from chat_completion for task {task_id}")
            # Create a default Attempt object to signify failure
            default_usage = Usage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0, accepted_prediction_tokens=0, rejected_prediction_tokens=0
                )
            )
            default_cost = Cost(prompt_cost=0.0, completion_cost=0.0, total_cost=0.0)
            
            return Attempt(
                metadata=AttemptMetadata(
                    model=self.model_config.model_name,
                    provider=self.model_config.provider,
                    start_timestamp=start_time,
                    end_timestamp=datetime.now(timezone.utc),
                    choices=[], 
                    kwargs=self.model_config.kwargs, 
                    usage=default_usage,
                    cost=default_cost,
                    error_message="Failed to get valid response from provider",
                    task_id=task_id, pair_index=pair_index, test_id=test_id
                ),
                answer=""
            )

        end_time = datetime.now(timezone.utc)

        usage_metadata = getattr(response, 'usage_metadata', None)
        logger.debug(f"Response usage metadata: {usage_metadata}")
        
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0
        reasoning_tokens = getattr(usage_metadata, 'thoughts_token_count', 0) if usage_metadata else 0
        total_tokens = getattr(usage_metadata, 'total_token_count', 0) if usage_metadata else 0
        
        response_text = getattr(response, 'text', "")

        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000

        prompt_cost = input_tokens * input_cost_per_token
        completion_cost = output_tokens * output_cost_per_token
        reasoning_cost = reasoning_tokens * output_cost_per_token

        input_choices = [
            Choice(index=i, message=Message(role=msg["role"], content=msg["content"]))
            for i, msg in enumerate(messages)
        ]
        response_choices = [
            Choice(index=len(input_choices), message=Message(role="assistant", content=response_text))
        ]
        all_choices = input_choices + response_choices

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
                    reasoning_tokens=reasoning_tokens,
                    accepted_prediction_tokens=output_tokens,
                    rejected_prediction_tokens=0
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                reasoning_cost=reasoning_cost,
                total_cost=prompt_cost + completion_cost + reasoning_cost
            ),
            task_id=task_id, pair_index=pair_index, test_id=test_id
        )
        attempt = Attempt(metadata=metadata, answer=response_text)
        return attempt

    def chat_completion(self, messages: list):
        contents_list = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant":
                role = "model"  # Gemini uses 'model' for assistant responses

            if role in ["user", "model"]:
                contents_list.append(types.Content(role=role, parts=[types.Part(text=content)]))
            elif role == "system" and content:
                # System messages from the input 'messages' list are not directly added to Gemini's 'contents'.
                # They should be provided via the 'system_instruction' key within self.model_config.kwargs,
                # which populates self.generation_config_dict for types.GenerateContentConfig.
                # If 'system_instruction' is not in model_config.kwargs, this message will be effectively ignored
                # by the Gemini API unless explicitly handled elsewhere or if this adapter's behavior changes.
                if 'system_instruction' not in self.generation_config_dict:
                     logger.info(
                         f"System message found in chat history: '{content}'. "
                         "This will not be used unless 'system_instruction' is set in model_config.kwargs "
                         "or this adapter is modified to handle it directly in 'contents'."
                     )
                # `pass` ensures these are not added to `contents_list` at this stage.
                pass

        config_params = self.generation_config_dict.copy()

        try:
            response = self.client.models.generate_content(
                model=self.model_config.model_name,
                contents=contents_list,
                config=types.GenerateContentConfig(**config_params)
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat_completion with google.genai: {e}")
            if hasattr(e, 'response') and e.response:
                 logger.error(f"API Error details: {e.response}")
            return None

    def chat_completion_stream(self, messages: list):
        """
        Make a streaming API call to Gemini and return the final complete response.
        Only the final message is returned after streaming all chunks.
        """
        logger.debug(f"Starting streaming for Gemini model: {self.model_config.model_name}")

        contents_list = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant":
                role = "model"  # Gemini uses 'model' for assistant responses

            if role in ["user", "model"]:
                contents_list.append(types.Content(role=role, parts=[types.Part(text=content)]))
            elif role == "system" and content:
                if 'system_instruction' not in self.generation_config_dict:
                     logger.info(
                         f"System message found in chat history: '{content}'. "
                         "This will not be used unless 'system_instruction' is set in model_config.kwargs "
                         "or this adapter is modified to handle it directly in 'contents'."
                     )
                pass

        # Prepare config for streaming, removing 'stream' to avoid duplication
        config_params = {k: v for k, v in self.generation_config_dict.items() if k != 'stream'}

        try:
            # Stream the content - we only care about the final result
            stream = self.client.models.generate_content_stream(
                model=self.model_config.model_name,
                contents=contents_list,
                config=types.GenerateContentConfig(**config_params)
            )

            # Iterate through all chunks to get to the final response
            # This prevents connection hangs while we don't care about intermediate results
            # We need to accumulate text chunks since streaming responses may not have .text on individual chunks
            text_chunks = []
            final_chunk = None
            chunk_count = 0
            for chunk in stream:
                final_chunk = chunk
                chunk_count += 1

                # Accumulate text from each chunk
                chunk_text = getattr(chunk, 'text', '')
                if chunk_text:
                    text_chunks.append(chunk_text)

                # Log progress every 100 chunks to show it's working
                if chunk_count % 1 == 0:
                    logger.debug(f"Streaming progress: {chunk_count} chunks received")

            # Combine all text chunks into final text
            final_text = ''.join(text_chunks)

            # Get usage metadata from the final chunk
            usage_metadata = getattr(final_chunk, 'usage_metadata', None) if final_chunk else None

            logger.debug(f"Streaming complete for Gemini model: {self.model_config.model_name}. Total chunks: {chunk_count}")

            # Return a wrapper object with the accumulated text and metadata
            return _StreamResponse(text=final_text, usage_metadata=usage_metadata)

        except Exception as e:
            logger.error(f"Error in chat_completion_stream with google.genai: {e}")
            if hasattr(e, 'response') and e.response:
                 logger.error(f"API Error details: {e.response}")
            return None

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
        
        # Filter config for extraction, using common generation parameters.
        # System instructions are generally not needed for this type of extraction.
        extract_config_params = {
            k: v for k, v in self.generation_config_dict.items() 
            if k in ['temperature', 'top_p', 'top_k', 'max_output_tokens', 'stop_sequences']
        }

        try:
            response = self.client.models.generate_content(
                model=self.model_config.model_name,
                contents=prompt, 
                config=types.GenerateContentConfig(**extract_config_params) if extract_config_params else None
            )
            content = response.text.strip()

            if content.startswith("```json"):
                content = content[7:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            try:
                json_data = json.loads(content)
                return json_data.get("response")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from extraction response: {content}")
                return None
        except Exception as e:
            logger.error(f"Error in extract_json_from_response with google.genai: {e}")
            return None